from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mediroad.stage4_2_certification.formulation import CertificationFormulation


@dataclass
class ExpansionLnsResult:
    backend: str
    objective: str
    proposals_generated: int
    hard_feasible: int
    metric_feasible: int
    rounds: int
    elapsed_sec: float
    best_before: float
    best_after: float
    improvements: int
    exhaustive_single_swap_tested: int
    peak_batch_plans: int
    selected_indices: np.ndarray


class ExpansionGpuLns:
    """Vectorized multi-swap primal search for expanded candidate universes.

    This class is deliberately *not* part of the proof chain.  It only supplies
    feasible incumbents.  Every accepted winner is recomputed by the CPU
    formulation before it can become a MIP start.
    """

    def __init__(self, formulation: CertificationFormulation, cfg: dict[str, Any]) -> None:
        self.f = formulation
        self.cfg = cfg
        # top5/coarse are <1k candidates and ~5k compressed patterns; a dense
        # bool incidence matrix is only a few MiB and makes the 3080 Ti useful.
        self.B_np = formulation._candidate_pattern.toarray().astype(np.bool_, copy=False)
        self.pop_np = np.asarray(formulation.population, dtype=np.float64)
        self.need_np = np.asarray(formulation.need_weighted, dtype=np.float64)
        frame = formulation.candidates
        self.admin = frame["admin_code"].astype("category").cat.codes.to_numpy(np.int16)
        self.sig = frame["sigungu"].astype("category").cat.codes.to_numpy(np.int16)
        self.cluster = frame["cluster_id"].fillna(frame["venue_id"]).astype("category").cat.codes.to_numpy(np.int32)
        self.backend = "NUMPY_CPU"
        self.xp = np
        self.B = self.B_np
        self.pop = self.pop_np
        self.need = self.need_np
        if bool(cfg.get("enabled", True)):
            try:
                import cupy as cp  # type: ignore

                cp.cuda.Device(int(cfg.get("device", 0))).use()
                cp.zeros(1, dtype=cp.int8)
                fraction = float(cfg.get("memory_fraction", 0.85))
                free, total = cp.cuda.runtime.memGetInfo()
                try:
                    # Hard cap the allocator, but do not pre-allocate VRAM.
                    cp.get_default_memory_pool().set_limit(size=int(total * fraction))
                except Exception:
                    pass
                self.xp = cp
                self.B = cp.asarray(self.B_np)
                self.pop = cp.asarray(self.pop_np)
                self.need = cp.asarray(self.need_np)
                self.backend = "CUPY_CUDA"
            except Exception:
                self.backend = "NUMPY_CPU"

    def _hard_feasible_mask(self, plans: np.ndarray) -> np.ndarray:
        if plans.ndim != 2 or plans.shape[1] != self.f.visit_count:
            return np.zeros(len(plans), dtype=bool)
        # Candidate uniqueness.
        s = np.sort(plans, axis=1)
        mask = np.all(np.diff(s, axis=1) != 0, axis=1)
        if not np.any(mask):
            return mask
        # With only 20 visits, pairwise equality is faster than Python groupby
        # and is fully vectorized over thousands of proposals.
        ac = self.admin[plans]
        sc = self.sig[plans]
        mask &= (ac[:, :, None] == ac[:, None, :]).sum(axis=2).max(axis=1) <= self.f.max_admin
        mask &= (sc[:, :, None] == sc[:, None, :]).sum(axis=2).max(axis=1) <= self.f.max_sigungu
        if self.f.one_cluster:
            cc = np.sort(self.cluster[plans], axis=1)
            mask &= np.all(np.diff(cc, axis=1) != 0, axis=1)
        return mask

    def _batch_metrics(self, plans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xp = self.xp
        requested = int(self.cfg.get("batch_size", 4096))
        # CPU fallback is intentionally smaller; CUDA can comfortably handle
        # 4096 x 20 x ~5k boolean gathers on 12 GiB VRAM.
        chunk = requested if self.backend == "CUPY_CUDA" else min(256, requested)
        totals: list[np.ndarray] = []
        needs: list[np.ndarray] = []
        for start in range(0, len(plans), chunk):
            raw = plans[start : start + chunk]
            idx = xp.asarray(raw, dtype=xp.int32) if self.backend == "CUPY_CUDA" else raw
            covered = xp.any(self.B[idx], axis=1)
            total = covered @ self.pop
            need = covered @ self.need
            if self.backend == "CUPY_CUDA":
                totals.append(xp.asnumpy(total))
                needs.append(xp.asnumpy(need))
                del idx, covered, total, need
            else:
                totals.append(np.asarray(total))
                needs.append(np.asarray(need))
        if self.backend == "CUPY_CUDA":
            xp.cuda.Stream.null.synchronize()
        return np.concatenate(totals), np.concatenate(needs)

    def _priority(self, base: np.ndarray, weights_np: np.ndarray, weights_dev: Any) -> np.ndarray:
        covered_np = self.B_np[base].any(axis=0) if base.size else np.zeros(self.f.ny, dtype=bool)
        if self.backend == "CUPY_CUDA":
            xp = self.xp
            uncovered = xp.asarray(~covered_np)
            score = xp.asnumpy(self.B[:, uncovered] @ weights_dev[uncovered])
            del uncovered
        else:
            score = np.asarray(self.B_np[:, ~covered_np] @ weights_np[~covered_np], dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        # Stable deterministic tie break.
        score += 1e-12 * np.arange(self.f.nx, 0, -1)
        score[base] = -np.inf
        return score

    def _vector_proposals(
        self,
        seed: np.ndarray,
        radius: int,
        n: int,
        rng: np.random.Generator,
        priority: np.ndarray,
    ) -> np.ndarray:
        n = max(1, int(n))
        radius = max(1, min(int(radius), self.f.visit_count - 1))
        plans = np.tile(seed[None, :], (n, 1))
        # Pick distinct ruin positions per row by sorting 20 random keys.
        remove_pos = np.argpartition(rng.random((n, self.f.visit_count)), radius - 1, axis=1)[:, :radius]
        finite = np.flatnonzero(np.isfinite(priority))
        if finite.size == 0:
            return np.empty((0, self.f.visit_count), dtype=np.int32)
        order = finite[np.argsort(priority[finite])[::-1]]
        pool_size = min(int(self.cfg.get("candidate_pool_size", 256)), len(order))
        pool = order[:pool_size]
        add = rng.choice(pool, size=(n, radius), replace=True)
        # A controlled fraction explores the full universe to avoid trapping the
        # search in a score-based pool.
        full_p = float(self.cfg.get("random_addition_probability", 0.20))
        explore = rng.random((n, radius)) < full_p
        if np.any(explore):
            add[explore] = rng.integers(0, self.f.nx, size=int(explore.sum()))
        rows = np.arange(n)[:, None]
        plans[rows, remove_pos] = add
        mask = self._hard_feasible_mask(plans)
        return plans[mask].astype(np.int32, copy=False)

    def _exhaustive_single_swap(self, seed: np.ndarray) -> np.ndarray:
        blocks: list[np.ndarray] = []
        universe = np.arange(self.f.nx, dtype=np.int32)
        for pos in range(self.f.visit_count):
            plans = np.tile(seed[None, :], (self.f.nx, 1)).astype(np.int32, copy=False)
            plans[:, pos] = universe
            mask = self._hard_feasible_mask(plans)
            if np.any(mask):
                blocks.append(plans[mask])
        return np.vstack(blocks) if blocks else np.empty((0, self.f.visit_count), dtype=np.int32)

    def search(
        self,
        seed: np.ndarray,
        *,
        objective: str,
        total_population_floor: float,
        rounds: int | None = None,
    ) -> ExpansionLnsResult:
        seed = np.asarray(seed, dtype=int)
        valid, reason = self.f.validate_selection(seed)
        if not valid:
            raise RuntimeError(f"Expansion LNS seed invalid: {reason}")
        if objective not in {"total_population", "need_weighted"}:
            raise ValueError(objective)
        metrics = self.f.metrics(seed)
        best_value = float(
            metrics["unique_elderly_population"]
            if objective == "total_population"
            else metrics["need_weighted_population"]
        )
        best_before = best_value
        best = seed.copy()
        weights_np = self.pop_np if objective == "total_population" else self.need_np
        weights_dev = self.pop if objective == "total_population" else self.need
        rng = np.random.default_rng(int(self.cfg.get("random_seed", 42)))
        radii = [int(x) for x in self.cfg.get("radius_schedule", [1, 2, 3, 4, 5, 6, 8])]
        rounds = int(rounds if rounds is not None else self.cfg.get("rounds", 96))
        proposals_per_round = int(self.cfg.get("proposals_per_round", 4096))
        generated = hard = metric_ok = improvements = 0
        peak_batch = 0
        start = time.perf_counter()

        # Exhaust every feasible one-swap neighbour once; it is cheap and gives
        # a strong deterministic starting point before stochastic multi-swap LNS.
        one = self._exhaustive_single_swap(best)
        generated += self.f.nx * self.f.visit_count
        hard += len(one)
        exhaustive_count = len(one)
        if len(one):
            peak_batch = max(peak_batch, len(one))
            totals, needs = self._batch_metrics(one)
            feasible = totals >= float(total_population_floor) - 1e-6
            metric_ok += int(feasible.sum())
            vals = totals if objective == "total_population" else needs
            vals = np.where(feasible, vals, -np.inf)
            idx = int(np.argmax(vals))
            if np.isfinite(vals[idx]) and float(vals[idx]) > best_value:
                exact = self.f.metrics(one[idx])
                exact_total = float(exact["unique_elderly_population"])
                exact_value = float(
                    exact["unique_elderly_population"]
                    if objective == "total_population"
                    else exact["need_weighted_population"]
                )
                if exact_total >= total_population_floor - 1e-6 and exact_value > best_value:
                    best = one[idx].astype(int, copy=True)
                    best_value = exact_value
                    improvements += 1

        for r in range(rounds):
            radius = max(1, min(radii[r % len(radii)], self.f.visit_count - 1))
            priority = self._priority(best, weights_np, weights_dev)
            plans = self._vector_proposals(best, radius, proposals_per_round, rng, priority)
            generated += proposals_per_round
            hard += len(plans)
            if not len(plans):
                continue
            peak_batch = max(peak_batch, len(plans))
            # Drop duplicate unordered plans before the expensive GPU gather.
            canonical = np.sort(plans, axis=1)
            _, unique_idx = np.unique(canonical, axis=0, return_index=True)
            plans = plans[np.sort(unique_idx)]
            totals, needs = self._batch_metrics(plans)
            feasible = totals >= float(total_population_floor) - 1e-6
            metric_ok += int(feasible.sum())
            vals = totals if objective == "total_population" else needs
            vals = np.where(feasible, vals, -np.inf)
            idx = int(np.argmax(vals))
            if np.isfinite(vals[idx]) and float(vals[idx]) > best_value + 1e-10 * max(1.0, abs(best_value)):
                candidate = plans[idx]
                # CPU authoritative re-evaluation before accepting a GPU witness.
                valid, reason = self.f.validate_selection(candidate)
                if not valid:
                    continue
                exact = self.f.metrics(candidate)
                exact_total = float(exact["unique_elderly_population"])
                exact_value = float(
                    exact["unique_elderly_population"]
                    if objective == "total_population"
                    else exact["need_weighted_population"]
                )
                if exact_total >= total_population_floor - 1e-6 and exact_value > best_value:
                    best = candidate.astype(int, copy=True)
                    best_value = exact_value
                    improvements += 1

        if self.backend == "CUPY_CUDA":
            try:
                self.xp.cuda.Stream.null.synchronize()
                self.xp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        return ExpansionLnsResult(
            self.backend,
            objective,
            generated,
            hard,
            metric_ok,
            rounds,
            float(time.perf_counter() - start),
            best_before,
            best_value,
            improvements,
            exhaustive_count,
            peak_batch,
            best,
        )
