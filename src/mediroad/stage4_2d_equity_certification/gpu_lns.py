from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from mediroad.stage4_2_certification.formulation import CertificationFormulation

from .types import EvidenceSeed


@dataclass
class LnsResult:
    backend: str
    proposals_generated: int
    proposals_hard_feasible: int
    proposals_metric_feasible: int
    elapsed_sec: float
    elites: list[EvidenceSeed]
    best_before: float
    best_after: float


class HybridLargeNeighborhoodSearch:
    """CUDA-batched ruin/recreate search for primal incumbents.

    This component never produces a certificate. Every GPU result promoted to the
    elite pool is recomputed and revalidated on CPU against the frozen formulation.
    """

    def __init__(self, formulation: CertificationFormulation, config: dict[str, Any]) -> None:
        self.f = formulation
        self.cfg = config.get("search", {})
        self.B_np = formulation._candidate_pattern.toarray().astype(np.bool_, copy=False)
        self.population_np = np.asarray(formulation.population, dtype=np.float64)
        self.high_np = np.asarray(formulation.high_need, dtype=np.float64)
        self.sigungu_weights_np = np.zeros((formulation.ns, formulation.ny), dtype=np.float64)
        self.sigungu_totals = np.empty(formulation.ns, dtype=np.float64)
        for s_idx, sigungu in enumerate(formulation.sigungu_values):
            idx = formulation.sigungu_pattern_indices[sigungu]
            self.sigungu_weights_np[s_idx, idx] = formulation.population[idx]
            self.sigungu_totals[s_idx] = formulation.sigungu_population_total[sigungu]
        frame = formulation.candidates
        self.admin = frame["admin_code"].astype(str).to_numpy()
        self.sigungu = frame["sigungu"].astype(str).to_numpy()
        self.cluster = frame["cluster_id"].fillna(frame["venue_id"]).astype(str).to_numpy()
        self.backend = "NUMPY_CPU"
        self.xp = np
        self.B = self.B_np
        self.population = self.population_np
        self.high = self.high_np
        self.sigungu_weights = self.sigungu_weights_np
        if bool(config.get("hardware", {}).get("gpu_lns", True)):
            try:
                import cupy as cp  # type: ignore

                cp.cuda.Device(0).use()
                cp.zeros(1, dtype=cp.float32)
                fraction = float(config.get("hardware", {}).get("gpu_memory_fraction", 0.85))
                if not 0.1 <= fraction <= 0.95:
                    raise ValueError("hardware.gpu_memory_fraction must be in [0.1, 0.95]")
                total_vram = int(cp.cuda.runtime.memGetInfo()[1])
                try:
                    cp.get_default_memory_pool().set_limit(size=int(total_vram * fraction))
                except Exception:
                    pass
                self.xp = cp
                self.B = cp.asarray(self.B_np)
                self.population = cp.asarray(self.population_np)
                self.high = cp.asarray(self.high_np)
                self.sigungu_weights = cp.asarray(self.sigungu_weights_np)
                self.backend = "CUPY_CUDA"
            except Exception:
                self.xp = np
                self.backend = "NUMPY_CPU"

    def _hard_feasible(self, selected: np.ndarray) -> bool:
        selected = np.asarray(selected, dtype=int)
        if selected.size != self.f.visit_count or np.unique(selected).size != selected.size:
            return False
        if np.unique(self.admin[selected], return_counts=True)[1].max(initial=0) > self.f.max_admin:
            return False
        if np.unique(self.sigungu[selected], return_counts=True)[1].max(initial=0) > self.f.max_sigungu:
            return False
        if self.f.one_cluster and np.unique(self.cluster[selected]).size != selected.size:
            return False
        return True

    def _objective_value(self, metrics: dict[str, Any], objective: str) -> float:
        if objective == "min_sigungu_coverage":
            return float(metrics["min_sigungu_coverage_ratio"])
        if objective == "high_need_population":
            return float(metrics["high_need_population"])
        raise ValueError(objective)

    def _exact_seed(self, source: str, selected: np.ndarray) -> EvidenceSeed | None:
        selected = np.asarray(selected, dtype=int)
        valid, _ = self.f.validate_selection(selected)
        if not valid:
            return None
        raw = self.f.metrics(selected)
        return EvidenceSeed(
            source=source,
            selected_indices=selected,
            metrics={
                "total_population": float(raw["unique_elderly_population"]),
                "min_sigungu_coverage": float(raw["min_sigungu_coverage_ratio"]),
                "high_need_population": float(raw["high_need_population"]),
                "need_weighted": float(raw["need_weighted_population"]),
                "cost": float(raw["cost"]),
            },
        )

    def _evaluation_chunk(self) -> int:
        configured = int(self.cfg.get("gpu_batch_size", 4096))
        if self.backend != "CUPY_CUDA":
            return min(configured, 256)
        try:
            free_bytes = int(self.xp.cuda.runtime.memGetInfo()[0])
            # Advanced indexing creates approximately batch*visit_count*patterns bytes.
            estimated = max(1, self.f.visit_count * self.f.ny * 2)
            safe = max(128, int((free_bytes * 0.35) // estimated))
            return max(128, min(configured, safe))
        except Exception:
            return configured

    def _evaluate_batch(self, plans: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xp = self.xp
        output_total: list[np.ndarray] = []
        output_high: list[np.ndarray] = []
        output_min: list[np.ndarray] = []
        chunk = self._evaluation_chunk()
        for start in range(0, len(plans), chunk):
            raw = plans[start : start + chunk]
            idx = xp.asarray(raw, dtype=xp.int32) if self.backend == "CUPY_CUDA" else raw
            covered = xp.any(self.B[idx], axis=1)
            total = covered @ self.population
            high = covered @ self.high
            sigungu_cov = covered @ self.sigungu_weights.T
            ratios = sigungu_cov / xp.asarray(self.sigungu_totals)[None, :]
            min_ratio = xp.min(ratios, axis=1)
            if self.backend == "CUPY_CUDA":
                output_total.append(xp.asnumpy(total))
                output_high.append(xp.asnumpy(high))
                output_min.append(xp.asnumpy(min_ratio))
                del idx, covered, total, high, sigungu_cov, ratios, min_ratio
            else:
                output_total.append(np.asarray(total))
                output_high.append(np.asarray(high))
                output_min.append(np.asarray(min_ratio))
        if self.backend == "CUPY_CUDA":
            self.xp.cuda.Stream.null.synchronize()
        return np.concatenate(output_total), np.concatenate(output_high), np.concatenate(output_min)

    def _unique_contribution(self, selected: np.ndarray, objective: str) -> np.ndarray:
        counts = self.B_np[selected].sum(axis=0)
        weights = self.high_np if objective == "high_need_population" else self.population_np
        return np.asarray((self.B_np[selected] & (counts[None, :] == 1)) @ weights, dtype=np.float64)

    def _candidate_priority(
        self,
        base: np.ndarray,
        *,
        objective: str,
        min_sigungu_floor: float | None,
    ) -> np.ndarray:
        if self.backend == "CUPY_CUDA":
            xp = self.xp
            base_gpu = xp.asarray(base, dtype=xp.int32)
            covered_gpu = xp.any(self.B[base_gpu], axis=0) if base.size else xp.zeros(self.f.ny, dtype=xp.bool_)
            uncovered_gpu = ~covered_gpu
            if objective == "high_need_population":
                score = xp.asnumpy(self.B[:, uncovered_gpu] @ self.high[uncovered_gpu]).astype(np.float64)
            else:
                score = np.zeros(self.f.nx, dtype=np.float64)
            sigungu_covered = xp.asnumpy(self.sigungu_weights[:, covered_gpu].sum(axis=1)).astype(np.float64)
            ratios = sigungu_covered / self.sigungu_totals
            if min_sigungu_floor is None:
                deficits = np.maximum(ratios.max(initial=0.0) - ratios, 0.0)
            else:
                deficits = np.maximum(float(min_sigungu_floor) - ratios, 0.0)
            if deficits.max(initial=0.0) > 0:
                deficit_weights = deficits / deficits.max()
                marginal_by_sigungu = xp.asnumpy(
                    self.B[:, uncovered_gpu] @ self.sigungu_weights[:, uncovered_gpu].T
                ).astype(np.float64)
                score += float(self.cfg.get("sigungu_deficit_boost", 4.0)) * (marginal_by_sigungu @ deficit_weights)
            xp.cuda.Stream.null.synchronize()
            del base_gpu, covered_gpu, uncovered_gpu
        else:
            covered = self.B_np[base].any(axis=0) if base.size else np.zeros(self.f.ny, dtype=bool)
            if objective == "high_need_population":
                score = np.asarray(self.B_np[:, ~covered] @ self.high_np[~covered], dtype=np.float64)
            else:
                score = np.zeros(self.f.nx, dtype=np.float64)
            sigungu_covered = self.sigungu_weights_np[:, covered].sum(axis=1)
            ratios = sigungu_covered / self.sigungu_totals
            if min_sigungu_floor is None:
                deficits = np.maximum(ratios.max(initial=0.0) - ratios, 0.0)
            else:
                deficits = np.maximum(float(min_sigungu_floor) - ratios, 0.0)
            if deficits.max(initial=0.0) > 0:
                deficit_weights = deficits / deficits.max()
                uncovered = ~covered
                marginal_by_sigungu = self.B_np[:, uncovered] @ self.sigungu_weights_np[:, uncovered].T
                score += float(self.cfg.get("sigungu_deficit_boost", 4.0)) * (marginal_by_sigungu @ deficit_weights)
        score += 1e-9 * np.arange(self.f.nx, 0, -1)
        score[base] = -np.inf
        return score

    def _recreate(
        self,
        base: np.ndarray,
        radius: int,
        priority: np.ndarray,
        rng: np.random.Generator,
        pool_size: int,
    ) -> np.ndarray | None:
        selected = list(map(int, base))
        top = np.argsort(priority)[::-1]
        top = top[np.isfinite(priority[top])][:pool_size]
        random_pool = np.flatnonzero(np.isfinite(priority))
        for _ in range(radius):
            # A selected candidate is marked -inf below. Re-filter on every
            # addition; otherwise a radius>1 recreation keeps stale entries,
            # produces inf/nan weights, and silently degenerates to uniform
            # random selection after the first addition.
            candidates = top[np.isfinite(priority[top])]
            if random_pool.size and rng.random() < float(self.cfg.get("random_addition_probability", 0.18)):
                candidates = random_pool[np.isfinite(priority[random_pool])]
            if candidates.size == 0:
                return None
            candidate_priority = priority[candidates]
            if not np.isfinite(candidate_priority).all():
                raise RuntimeError("LNS recreation retained non-finite candidate priorities")
            weights = np.maximum(candidate_priority - candidate_priority.min(), 0.0) + 1e-9
            trial_order: list[int] = []
            for _attempt in range(min(64, candidates.size)):
                if weights.sum() > 0 and np.isfinite(weights.sum()):
                    idx = int(rng.choice(candidates.size, p=weights / weights.sum()))
                else:
                    idx = int(rng.integers(0, candidates.size))
                candidate = int(candidates[idx])
                if candidate not in trial_order:
                    trial_order.append(candidate)
            added = False
            for candidate in trial_order:
                trial = np.asarray([*selected, candidate], dtype=int)
                # Partial caps are monotone; a violation cannot be repaired by additions.
                if np.unique(self.admin[trial], return_counts=True)[1].max(initial=0) > self.f.max_admin:
                    continue
                if np.unique(self.sigungu[trial], return_counts=True)[1].max(initial=0) > self.f.max_sigungu:
                    continue
                if self.f.one_cluster and np.unique(self.cluster[trial]).size != trial.size:
                    continue
                selected.append(candidate)
                priority[candidate] = -np.inf
                added = True
                break
            if not added:
                return None
        plan = np.asarray(selected, dtype=int)
        return plan if self._hard_feasible(plan) else None

    def search(
        self,
        seeds: Iterable[EvidenceSeed],
        *,
        objective: str,
        total_population_floor: float,
        min_sigungu_floor: float | None,
        rounds: int | None = None,
    ) -> LnsResult:
        started = time.perf_counter()
        seed_list = list(seeds)
        eligible = [
            seed
            for seed in seed_list
            if seed.metrics["total_population"] >= total_population_floor - 1e-6
            and (min_sigungu_floor is None or seed.metrics["min_sigungu_coverage"] >= min_sigungu_floor - 1e-10)
        ]
        if not eligible:
            raise RuntimeError("GPU LNS received no feasible seed")
        eligible.sort(key=lambda seed: seed.metrics[objective], reverse=True)
        elite_limit = max(2, int(self.cfg.get("elite_pool_size", 24)))
        elites = eligible[:elite_limit]
        best_before = float(elites[0].metrics[objective])
        rng = np.random.default_rng(int(self.cfg.get("random_seed", 42)))
        radius_schedule = [int(value) for value in self.cfg.get("radius_schedule", [1, 2, 3, 4, 5, 6])]
        rounds = int(rounds if rounds is not None else self.cfg.get("rounds", 80))
        ruin_sets = max(1, int(self.cfg.get("ruin_sets_per_round", 16)))
        proposals_per_ruin = max(1, int(self.cfg.get("proposals_per_ruin", 48)))
        pool_size = max(16, int(self.cfg.get("candidate_pool_size", 128)))
        generated = hard_feasible = metric_feasible = 0
        seen = {tuple(sorted(map(int, seed.selected_indices))) for seed in elites}

        for round_index in range(rounds):
            base_seed = elites[round_index % len(elites)] if round_index < len(elites) else elites[int(rng.integers(0, len(elites)))]
            selected = np.asarray(base_seed.selected_indices, dtype=int)
            unique = self._unique_contribution(selected, objective)
            remove_weights = 1.0 / (unique + max(1e-9, float(np.median(unique[unique > 0])) if np.any(unique > 0) else 1.0))
            radius = radius_schedule[round_index % len(radius_schedule)]
            radius = max(1, min(radius, self.f.visit_count - 1))
            proposals: list[np.ndarray] = []
            for _ in range(ruin_sets):
                removed_pos = rng.choice(
                    selected.size,
                    size=radius,
                    replace=False,
                    p=remove_weights / remove_weights.sum(),
                )
                base = np.delete(selected, removed_pos)
                priority_base = self._candidate_priority(
                    base,
                    objective=objective,
                    min_sigungu_floor=min_sigungu_floor,
                )
                for _ in range(proposals_per_ruin):
                    priority = priority_base.copy()
                    plan = self._recreate(base, radius, priority, rng, pool_size)
                    generated += 1
                    if plan is not None:
                        hard_feasible += 1
                        key = tuple(sorted(map(int, plan)))
                        if key not in seen:
                            proposals.append(plan)
                            seen.add(key)
            if not proposals:
                continue
            plan_matrix = np.vstack(proposals)
            total, high, min_ratio = self._evaluate_batch(plan_matrix)
            feasible = total >= total_population_floor - 1e-6
            if min_sigungu_floor is not None:
                feasible &= min_ratio >= min_sigungu_floor - 1e-10
            metric_feasible += int(feasible.sum())
            values = min_ratio if objective == "min_sigungu_coverage" else high
            order = np.argsort(np.where(feasible, values, -np.inf))[::-1]
            accepted = 0
            for idx in order[: max(4, elite_limit // 2)]:
                if not feasible[idx] or not np.isfinite(values[idx]):
                    continue
                seed = self._exact_seed(
                    f"{self.backend}_LNS::round{round_index}::r{radius}",
                    plan_matrix[idx],
                )
                if seed is None:
                    continue
                if seed.metrics["total_population"] < total_population_floor - 1e-6:
                    continue
                if min_sigungu_floor is not None and seed.metrics["min_sigungu_coverage"] < min_sigungu_floor - 1e-10:
                    continue
                elites.append(seed)
                accepted += 1
            if accepted:
                dedup: dict[tuple[int, ...], EvidenceSeed] = {}
                for seed in elites:
                    key = tuple(sorted(map(int, seed.selected_indices)))
                    old = dedup.get(key)
                    if old is None or seed.metrics[objective] > old.metrics[objective]:
                        dedup[key] = seed
                elites = sorted(dedup.values(), key=lambda seed: seed.metrics[objective], reverse=True)[:elite_limit]

        best_after = float(elites[0].metrics[objective])
        if self.backend == "CUPY_CUDA":
            try:
                self.xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        return LnsResult(
            backend=self.backend,
            proposals_generated=generated,
            proposals_hard_feasible=hard_feasible,
            proposals_metric_feasible=metric_feasible,
            elapsed_sec=float(time.perf_counter() - started),
            elites=elites,
            best_before=best_before,
            best_after=best_after,
        )
