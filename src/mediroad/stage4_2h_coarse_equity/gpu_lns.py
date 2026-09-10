from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mediroad.stage4_2_certification.formulation import CertificationFormulation


@dataclass
class GpuSearchResult:
    backend: str
    objective: str
    proposals_generated: int
    hard_feasible: int
    metric_feasible: int
    elapsed_sec: float
    best_before: float
    best_after: float
    improvements: int
    selected_indices: np.ndarray
    proof_elite_indices: list[list[int]]


class CoarseEquityGpuLns:
    """Vectorized RTX primal search. It never produces a proof certificate."""

    def __init__(self, formulation: CertificationFormulation, cfg: dict[str, Any]) -> None:
        self.f = formulation
        self.cfg = cfg
        self.B_np = formulation._candidate_pattern.toarray().astype(np.bool_, copy=False)
        self.population_np = np.asarray(formulation.population, dtype=np.float64)
        self.high_np = np.asarray(formulation.high_need, dtype=np.float64)
        self.sig_weights_np = np.zeros((formulation.ns, formulation.ny), dtype=np.float64)
        self.sig_totals = np.empty(formulation.ns, dtype=np.float64)
        for s_idx, sigungu in enumerate(formulation.sigungu_values):
            idx = formulation.sigungu_pattern_indices[sigungu]
            self.sig_weights_np[s_idx, idx] = formulation.population[idx]
            self.sig_totals[s_idx] = formulation.sigungu_population_total[sigungu]
        frame = formulation.candidates
        self.admin = frame['admin_code'].astype('category').cat.codes.to_numpy(np.int16)
        self.sig = frame['sigungu'].astype('category').cat.codes.to_numpy(np.int16)
        self.cluster = frame['cluster_id'].fillna(frame['venue_id']).astype('category').cat.codes.to_numpy(np.int32)
        self.backend = 'NUMPY_CPU'
        self.xp = np
        self.B = self.B_np
        self.population = self.population_np
        self.high = self.high_np
        self.sig_weights = self.sig_weights_np
        if bool(cfg.get('enabled', True)):
            try:
                import cupy as cp  # type: ignore
                cp.cuda.Device(int(cfg.get('device', 0))).use()
                cp.zeros(1, dtype=cp.int8)
                total = int(cp.cuda.runtime.memGetInfo()[1])
                try:
                    cp.get_default_memory_pool().set_limit(size=int(total * float(cfg.get('memory_fraction', 0.85))))
                except Exception:
                    pass
                self.xp = cp
                self.B = cp.asarray(self.B_np)
                self.population = cp.asarray(self.population_np)
                self.high = cp.asarray(self.high_np)
                self.sig_weights = cp.asarray(self.sig_weights_np)
                self.backend = 'CUPY_CUDA'
            except Exception:
                self.backend = 'NUMPY_CPU'

    def _hard_mask(self, plans: np.ndarray) -> np.ndarray:
        if plans.ndim != 2 or plans.shape[1] != self.f.visit_count:
            return np.zeros(len(plans), dtype=bool)
        s = np.sort(plans, axis=1)
        mask = np.all(np.diff(s, axis=1) != 0, axis=1)
        if not np.any(mask):
            return mask
        ac = self.admin[plans]
        sc = self.sig[plans]
        mask &= (ac[:, :, None] == ac[:, None, :]).sum(axis=2).max(axis=1) <= self.f.max_admin
        mask &= (sc[:, :, None] == sc[:, None, :]).sum(axis=2).max(axis=1) <= self.f.max_sigungu
        if self.f.one_cluster:
            cc = np.sort(self.cluster[plans], axis=1)
            mask &= np.all(np.diff(cc, axis=1) != 0, axis=1)
        return mask

    def _batch_metrics(self, plans: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xp = self.xp
        requested = int(self.cfg.get('batch_size', 4096))
        chunk = requested if self.backend == 'CUPY_CUDA' else min(128, requested)
        totals: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        mins: list[np.ndarray] = []
        sig_totals_dev = xp.asarray(self.sig_totals)
        for start in range(0, len(plans), chunk):
            raw = plans[start:start + chunk]
            idx = xp.asarray(raw, dtype=xp.int32) if self.backend == 'CUPY_CUDA' else raw
            covered = xp.any(self.B[idx], axis=1)
            total = covered @ self.population
            high = covered @ self.high
            sig_cov = covered @ self.sig_weights.T
            min_ratio = xp.min(sig_cov / sig_totals_dev[None, :], axis=1)
            if self.backend == 'CUPY_CUDA':
                totals.append(xp.asnumpy(total)); highs.append(xp.asnumpy(high)); mins.append(xp.asnumpy(min_ratio))
                del idx, covered, total, high, sig_cov, min_ratio
            else:
                totals.append(np.asarray(total)); highs.append(np.asarray(high)); mins.append(np.asarray(min_ratio))
        if self.backend == 'CUPY_CUDA':
            xp.cuda.Stream.null.synchronize()
        return np.concatenate(totals), np.concatenate(highs), np.concatenate(mins)

    def _priority(self, base: np.ndarray, objective: str, min_floor: float | None) -> np.ndarray:
        covered = self.B_np[base].any(axis=0) if base.size else np.zeros(self.f.ny, dtype=bool)
        score = np.zeros(self.f.nx, dtype=np.float64)
        if objective == 'high_need_population':
            score += np.asarray(self.B_np[:, ~covered] @ self.high_np[~covered], dtype=np.float64)
        # Always reward the currently weakest/under-floor sigungu so high-need
        # search cannot cheaply destroy the lexicographic min-sigungu stage.
        sig_cov = self.sig_weights_np[:, covered].sum(axis=1)
        ratios = sig_cov / self.sig_totals
        target = float(min_floor) if min_floor is not None else float(ratios.max(initial=0.0))
        deficits = np.maximum(target - ratios, 0.0)
        if deficits.max(initial=0.0) <= 0:
            deficits = np.maximum(ratios.max(initial=0.0) - ratios, 0.0)
        if deficits.max(initial=0.0) > 0:
            d = deficits / deficits.max()
            marginal = self.B_np[:, ~covered] @ self.sig_weights_np[:, ~covered].T
            score += float(self.cfg.get('sigungu_deficit_boost', 8.0)) * (marginal @ d)
        # Small population term helps break flat min-ratio plateaus.
        score += float(self.cfg.get('population_tiebreak_weight', 1e-3)) * np.asarray(
            self.B_np[:, ~covered] @ self.population_np[~covered], dtype=np.float64
        )
        score += 1e-12 * np.arange(self.f.nx, 0, -1)
        score[base] = -np.inf
        return score

    def _proposals(self, seed: np.ndarray, radius: int, n: int, rng: np.random.Generator, priority: np.ndarray) -> np.ndarray:
        radius = max(1, min(int(radius), self.f.visit_count - 1))
        plans = np.tile(seed[None, :], (int(n), 1)).astype(np.int32, copy=False)
        remove_pos = np.argpartition(rng.random((len(plans), self.f.visit_count)), radius - 1, axis=1)[:, :radius]
        finite = np.flatnonzero(np.isfinite(priority))
        if finite.size == 0:
            return np.empty((0, self.f.visit_count), dtype=np.int32)
        order = finite[np.argsort(priority[finite])[::-1]]
        pool = order[: min(int(self.cfg.get('candidate_pool_size', 384)), len(order))]
        add = rng.choice(pool, size=(len(plans), radius), replace=True)
        explore = rng.random((len(plans), radius)) < float(self.cfg.get('random_addition_probability', 0.20))
        if np.any(explore):
            add[explore] = rng.integers(0, self.f.nx, size=int(explore.sum()))
        plans[np.arange(len(plans))[:, None], remove_pos] = add
        return plans[self._hard_mask(plans)]

    def _one_swap(self, seed: np.ndarray) -> np.ndarray:
        universe = np.arange(self.f.nx, dtype=np.int32)
        blocks = []
        for pos in range(self.f.visit_count):
            plans = np.tile(seed[None, :], (self.f.nx, 1)).astype(np.int32, copy=False)
            plans[:, pos] = universe
            mask = self._hard_mask(plans)
            if np.any(mask):
                blocks.append(plans[mask])
        return np.vstack(blocks) if blocks else np.empty((0, self.f.visit_count), dtype=np.int32)

    def search(self, seed: np.ndarray, *, objective: str, total_floor: float, min_floor: float | None = None, rounds: int | None = None) -> GpuSearchResult:
        if objective not in {'min_sigungu_coverage', 'high_need_population'}:
            raise ValueError(objective)
        seed = np.asarray(seed, dtype=int)
        valid, reason = self.f.validate_selection(seed)
        if not valid:
            raise RuntimeError(f'GPU seed invalid: {reason}')
        m = self.f.metrics(seed)
        if float(m['unique_elderly_population']) < total_floor - 1e-6:
            raise RuntimeError('GPU seed violates total-population floor')
        if min_floor is not None and float(m['min_sigungu_coverage_ratio']) < min_floor - 1e-10:
            raise RuntimeError('GPU seed violates min-sigungu floor')
        value = float(m['min_sigungu_coverage_ratio'] if objective == 'min_sigungu_coverage' else m['high_need_population'])
        before = value; best = seed.copy(); improvements = 0
        rng = np.random.default_rng(int(self.cfg.get('random_seed', 42042)))
        radii = [int(x) for x in self.cfg.get('radius_schedule', [1, 2, 3, 4, 5, 6, 8])]
        rounds = int(rounds if rounds is not None else self.cfg.get('rounds', 192))
        n = int(self.cfg.get('proposals_per_round', 4096))
        generated = hard = metric_ok = 0
        started = time.perf_counter()
        elite_limit = int(self.cfg.get('proof_elite_count', 32))
        elites_per_batch = int(self.cfg.get('proof_elites_per_batch', 2))
        elite_scores: dict[tuple[int, ...], float] = {tuple(sorted(int(i) for i in seed)): value}

        def remember_elites(plans: np.ndarray, values: np.ndarray) -> None:
            finite = np.flatnonzero(np.isfinite(values))
            if finite.size == 0 or elite_limit <= 0:
                return
            take = min(max(1, elites_per_batch), int(finite.size))
            local = finite[np.argpartition(values[finite], -take)[-take:]]
            for row in local:
                key = tuple(sorted(int(i) for i in plans[int(row)]))
                elite_scores[key] = max(float(values[int(row)]), elite_scores.get(key, -np.inf))
            if len(elite_scores) > elite_limit * 4:
                keep = sorted(elite_scores.items(), key=lambda item: (-item[1], item[0]))[:elite_limit * 2]
                elite_scores.clear(); elite_scores.update(keep)

        one = self._one_swap(best)
        generated += self.f.nx * self.f.visit_count; hard += len(one)
        if len(one):
            total, high, min_ratio = self._batch_metrics(one)
            feasible = total >= total_floor - 1e-6
            if min_floor is not None: feasible &= min_ratio >= min_floor - 1e-10
            metric_ok += int(feasible.sum())
            vals = min_ratio if objective == 'min_sigungu_coverage' else high
            vals = np.where(feasible, vals, -np.inf)
            remember_elites(one, vals)
            idx = int(np.argmax(vals))
            if np.isfinite(vals[idx]) and float(vals[idx]) > value:
                exact = self.f.metrics(one[idx])
                exact_value = float(exact['min_sigungu_coverage_ratio'] if objective == 'min_sigungu_coverage' else exact['high_need_population'])
                if exact_value > value:
                    best = one[idx].astype(int); value = exact_value; improvements += 1

        for r in range(rounds):
            priority = self._priority(best, objective, min_floor)
            plans = self._proposals(best, radii[r % len(radii)], n, rng, priority)
            generated += n; hard += len(plans)
            if not len(plans):
                continue
            total, high, min_ratio = self._batch_metrics(plans)
            feasible = total >= total_floor - 1e-6
            if min_floor is not None: feasible &= min_ratio >= min_floor - 1e-10
            metric_ok += int(feasible.sum())
            vals = min_ratio if objective == 'min_sigungu_coverage' else high
            vals = np.where(feasible, vals, -np.inf)
            remember_elites(plans, vals)
            idx = int(np.argmax(vals))
            if np.isfinite(vals[idx]) and float(vals[idx]) > value:
                candidate = plans[idx].astype(int)
                valid, _ = self.f.validate_selection(candidate)
                if valid:
                    exact = self.f.metrics(candidate)
                    exact_total = float(exact['unique_elderly_population'])
                    exact_min = float(exact['min_sigungu_coverage_ratio'])
                    exact_value = float(exact['min_sigungu_coverage_ratio'] if objective == 'min_sigungu_coverage' else exact['high_need_population'])
                    if exact_total >= total_floor - 1e-6 and (min_floor is None or exact_min >= min_floor - 1e-10) and exact_value > value:
                        best = candidate; value = exact_value; improvements += 1
        if self.backend == 'CUPY_CUDA':
            try: self.xp.get_default_memory_pool().free_all_blocks()
            except Exception: pass
        proof_elites = [list(key) for key, _ in sorted(elite_scores.items(), key=lambda item: (-item[1], item[0]))[:elite_limit]]
        return GpuSearchResult(
            backend=self.backend, objective=objective, proposals_generated=generated,
            hard_feasible=hard, metric_feasible=metric_ok,
            elapsed_sec=float(time.perf_counter() - started), best_before=before,
            best_after=value, improvements=improvements, selected_indices=best,
            proof_elite_indices=proof_elites,
        )
