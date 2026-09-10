from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .types import Floor, LinearMipModel, Sense


@dataclass
class _Layout:
    x: slice
    y: slice
    q: int | None
    count_lambda: slice | None
    nvars: int


class _Rows:
    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.names: list[str] = []

    def add(
        self,
        coeffs: Iterable[tuple[int, float]],
        lower: float,
        upper: float,
        name: str,
    ) -> None:
        row = len(self.lb)
        for col, value in coeffs:
            value = float(value)
            if value != 0.0:
                self.rows.append(row)
                self.cols.append(int(col))
                self.data.append(value)
        self.lb.append(float(lower))
        self.ub.append(float(upper))
        self.names.append(name)

    def matrix(self, nvars: int) -> sparse.csc_matrix:
        return sparse.coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.lb), nvars),
            dtype=np.float64,
        ).tocsc()


class CertificationFormulation:
    """Tighter Stage 4.2 formulation built for solver certification.

    The original model used one ``y >= x_i`` row for every pattern/candidate
    incidence. Those rows are unnecessary for the projection onto the venue
    variables because all coverage objectives and retained floors are
    non-negative. Stage4.2C keeps only ``y_p <= sum_i x_i``. For every feasible
    venue selection, setting all covered ``y`` variables to one remains feasible,
    so the venue plans and objective optima are preserved while the row count is
    reduced substantially.

    Variables are stage-specific. The minimum-coverage auxiliary exists only
    in its stage and is certified by a fixed-point threshold infeasibility oracle.
    The cost stage uses an integer-count convex-hull representation for visit
    allocation deviation, preventing fractional root-LP allocations from claiming
    a zero deviation that no integer 20-visit plan can attain.
    """

    BENEFIT_NAMES = {"total_population", "need_weighted", "high_need_population"}

    def __init__(
        self,
        candidates: pd.DataFrame,
        patterns: Any,
        base_config: dict[str, Any],
        cert_config: dict[str, Any],
        *,
        candidate_set: str,
    ) -> None:
        self.candidates = candidates.reset_index(drop=True).copy()
        self.patterns = patterns
        self.base_config = base_config
        self.cert_config = cert_config
        self.candidate_set = candidate_set
        self.venue_alias_map: dict[str, str] = {
            value: value for value in self.candidates["venue_id"].astype(str)
        }
        self.nx = int(len(self.candidates))
        self.ny = int(patterns.n_patterns)
        if self.nx != int(patterns.n_candidates):
            raise ValueError("Candidate and pattern candidate counts differ")
        self.visit_count = int(base_config.get("optimization", {}).get("visit_count", 20))
        self.max_admin = int(base_config.get("candidate_expansion", {}).get("max_visits_per_admin", 2))
        self.max_sigungu = int(base_config.get("candidate_expansion", {}).get("max_visits_per_sigungu", 4))
        self.one_cluster = bool(base_config.get("candidate_expansion", {}).get("one_per_coverage_cluster", True))
        self.min_sigungu_mode = str(
            cert_config.get("formulation", {}).get(
                "min_sigungu_mode", "continuous_auxiliary_with_threshold_oracle"
            )
        )
        if self.min_sigungu_mode != "continuous_auxiliary_with_threshold_oracle":
            raise ValueError(f"Unsupported min_sigungu_mode: {self.min_sigungu_mode}")
        self.min_sigungu_scale = int(
            cert_config.get("formulation", {}).get("min_sigungu_integer_scale", 1_000_000)
        )
        if self.min_sigungu_scale <= 0:
            raise ValueError("formulation.min_sigungu_integer_scale must be positive")
        self.sigungu_values = sorted(set(self.candidates["sigungu"].astype(str)))
        self.ns = len(self.sigungu_values)
        self.candidate_sigungu = self.candidates["sigungu"].astype(str).to_numpy()
        self.pattern_sigungu = np.asarray(patterns.sigungu, dtype=str)
        self.population = np.asarray(patterns.population, dtype=np.float64)
        self.need_weighted = np.asarray(patterns.need_weighted, dtype=np.float64)
        self.high_need = np.asarray(patterns.high_need_population, dtype=np.float64)
        for name, values in [
            ("population", self.population),
            ("need_weighted", self.need_weighted),
            ("high_need_population", self.high_need),
        ]:
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite pattern weights")
            if np.any(values < -1e-12):
                raise ValueError(
                    f"One-link formulation requires non-negative {name} weights; "
                    f"minimum={float(values.min())}"
                )
        self.pattern_coverers = [np.asarray(v, dtype=np.int32) for v in patterns.pattern_coverers]
        self.sigungu_pattern_indices = {
            s: np.flatnonzero(self.pattern_sigungu == s).astype(np.int32) for s in self.sigungu_values
        }
        self.sigungu_population_total = {
            s: float(self.population[idx].sum()) for s, idx in self.sigungu_pattern_indices.items()
        }
        self._candidate_pattern = self._build_candidate_pattern_matrix()
        self._candidate_cost = self._build_candidate_cost()

    def _build_candidate_pattern_matrix(self) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        for p, coverers in enumerate(self.pattern_coverers):
            rows.extend(int(i) for i in coverers)
            cols.extend([p] * len(coverers))
        data = np.ones(len(rows), dtype=np.int8)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.nx, self.ny))

    def _build_candidate_cost(self) -> np.ndarray:
        cfg = self.base_config.get("optimization", {})
        cost = np.zeros(self.nx, dtype=np.float64)
        travel = pd.to_numeric(
            self.candidates.get("travel_minutes", pd.Series(np.nan, index=self.candidates.index)),
            errors="coerce",
        )
        if not isinstance(travel, pd.Series):
            travel = pd.Series(np.nan, index=self.candidates.index)
        if travel.notna().any():
            fill = float(travel.median())
            cost += travel.fillna(fill).clip(lower=0).to_numpy(float) * float(cfg.get("travel_penalty_scale", 1.0))
        known = self.candidates.get("known_overlap", pd.Series("UNKNOWN", index=self.candidates.index))
        known_flag = (
            known.astype(str)
            .str.strip()
            .str.lower()
            .isin({"known", "yes", "true", "1", "confirmed", "high"})
        )
        cost += known_flag.to_numpy(float) * float(cfg.get("known_overlap_penalty", 1000.0))
        tie_raw = self.candidates.get("candidate_priority_tiebreak", pd.Series(0.0, index=self.candidates.index))
        tie = pd.to_numeric(tie_raw, errors="coerce")
        if not isinstance(tie, pd.Series):
            tie = pd.Series(0.0, index=self.candidates.index)
        rank = tie.fillna(0.0).rank(method="first", ascending=False).to_numpy(float)
        cost += rank * float(cfg.get("deterministic_tiebreak_scale", 1e-6))
        return cost

    def _layout(self, objective_name: str, floors: list[Floor], threshold: Floor | None) -> _Layout:
        need_q = objective_name == "min_sigungu_coverage"
        need_dev = objective_name == "cost" or any(f.objective == "cost" for f in floors)
        if threshold is not None and threshold.objective == "cost":
            need_dev = True
        x = slice(0, self.nx)
        y = slice(self.nx, self.nx + self.ny)
        cursor = self.nx + self.ny
        q: int | None = None
        if need_q:
            q = cursor
            cursor += 1
        count_lambda: slice | None = None
        if need_dev:
            width = self.max_sigungu + 1
            count_lambda = slice(cursor, cursor + self.ns * width)
            cursor += self.ns * width
        return _Layout(x=x, y=y, q=q, count_lambda=count_lambda, nvars=cursor)

    def _benefit_weights(self, name: str) -> np.ndarray:
        if name == "total_population":
            return self.population
        if name == "need_weighted":
            return self.need_weighted
        if name == "high_need_population":
            return self.high_need
        raise KeyError(name)

    def _cost_vector(self, layout: _Layout) -> np.ndarray:
        vec = np.zeros(layout.nvars, dtype=np.float64)
        vec[layout.x] = self._candidate_cost
        if layout.count_lambda is not None:
            penalty = float(
                self.base_config.get("optimization", {}).get("visit_location_deviation_penalty", 100.0)
            )
            target = self.visit_count / max(1, self.ns)
            width = self.max_sigungu + 1
            values = np.asarray(
                [abs(k - target) * penalty for _ in self.sigungu_values for k in range(width)],
                dtype=np.float64,
            )
            vec[layout.count_lambda] = values
        return vec

    def _objective_vector(self, layout: _Layout, objective_name: str) -> tuple[np.ndarray, float]:
        vec = np.zeros(layout.nvars, dtype=np.float64)
        scale = 1.0
        if objective_name in self.BENEFIT_NAMES:
            vec[layout.y] = self._benefit_weights(objective_name)
        elif objective_name == "min_sigungu_coverage":
            if layout.q is None:
                raise RuntimeError("q missing")
            vec[layout.q] = 1.0
        elif objective_name == "cost":
            vec = self._cost_vector(layout)
        elif objective_name == "feasibility":
            pass
        else:
            raise KeyError(objective_name)
        return vec, scale

    def _add_hard_rows(self, rows: _Rows, layout: _Layout) -> None:
        inf = math.inf
        rows.add(((i, 1.0) for i in range(self.nx)), self.visit_count, self.visit_count, "visit_count")
        admin = self.candidates["admin_code"].astype(str)
        for value, idx in admin.groupby(admin).groups.items():
            rows.add(((int(i), 1.0) for i in idx), -inf, self.max_admin, f"admin_cap::{value}")
        sig = self.candidates["sigungu"].astype(str)
        for value, idx in sig.groupby(sig).groups.items():
            rows.add(((int(i), 1.0) for i in idx), -inf, self.max_sigungu, f"sigungu_cap::{value}")
        if self.one_cluster:
            clusters = self.candidates["cluster_id"].fillna(self.candidates["venue_id"]).astype(str)
            for value, idx in clusters.groupby(clusters).groups.items():
                indices = list(map(int, idx))
                if len(indices) > 1:
                    rows.add(((i, 1.0) for i in indices), -inf, 1.0, f"cluster_cap::{value}")

        # One-link coverage formulation. Uncovered patterns are fixed to zero by bounds.
        for p, coverers in enumerate(self.pattern_coverers):
            if len(coverers) == 0:
                continue
            y = layout.y.start + p
            coeffs = [(y, 1.0), *((int(i), -1.0) for i in coverers)]
            rows.add(coeffs, -inf, 0.0, f"cover_link::{p}")

    def _add_q_rows(self, rows: _Rows, layout: _Layout) -> None:
        if layout.q is None:
            return
        for sigungu, idx in self.sigungu_pattern_indices.items():
            total = self.sigungu_population_total[sigungu]
            if total <= 0:
                continue
            coeffs: list[tuple[int, float]] = [(layout.q, total)]
            coeffs.extend((layout.y.start + int(p), -float(self.population[p])) for p in idx)
            rows.add(coeffs, -math.inf, 0.0, f"q_min_coverage::{sigungu}")

    def _add_deviation_rows(self, rows: _Rows, layout: _Layout) -> None:
        if layout.count_lambda is None:
            return
        width = self.max_sigungu + 1
        for s_idx, sigungu in enumerate(self.sigungu_values):
            start = layout.count_lambda.start + s_idx * width
            rows.add(
                ((start + k, 1.0) for k in range(width)),
                1.0,
                1.0,
                f"visit_count_lambda_sum::{sigungu}",
            )
            cidx = np.flatnonzero(self.candidate_sigungu == sigungu)
            coeffs: list[tuple[int, float]] = [(start + k, float(k)) for k in range(width)]
            coeffs.extend((int(i), -1.0) for i in cidx)
            rows.add(coeffs, 0.0, 0.0, f"visit_count_lambda_link::{sigungu}")

    def _add_floor(self, rows: _Rows, layout: _Layout, floor: Floor, suffix: str) -> None:
        if floor.objective in self.BENEFIT_NAMES:
            weights = self._benefit_weights(floor.objective)
            coeffs = ((layout.y.start + p, float(w)) for p, w in enumerate(weights) if w != 0.0)
            if floor.sense == "max":
                rows.add(coeffs, floor.value, math.inf, f"floor::{suffix}::{floor.objective}")
            else:
                rows.add(coeffs, -math.inf, floor.value, f"floor::{suffix}::{floor.objective}")
            return
        if floor.objective == "min_sigungu_coverage":
            # A ratio floor means every sigungu must achieve the ratio. This avoids
            # carrying q into later lexicographic stages.
            for sigungu, idx in self.sigungu_pattern_indices.items():
                total = self.sigungu_population_total[sigungu]
                coeffs = (
                    (layout.y.start + int(p), float(self.population[p]))
                    for p in idx
                    if self.population[p] != 0.0
                )
                target = floor.value * total
                if floor.sense == "max":
                    rows.add(coeffs, target, math.inf, f"floor::{suffix}::mincov::{sigungu}")
                else:
                    rows.add(coeffs, -math.inf, target, f"floor::{suffix}::mincov::{sigungu}")
            return
        if floor.objective == "cost":
            vec = self._cost_vector(layout)
            coeffs = ((i, float(v)) for i, v in enumerate(vec) if v != 0.0)
            if floor.sense == "min":
                rows.add(coeffs, -math.inf, floor.value, f"floor::{suffix}::cost")
            else:
                rows.add(coeffs, floor.value, math.inf, f"floor::{suffix}::cost")
            return
        raise KeyError(floor.objective)

    def build(
        self,
        *,
        objective_name: str,
        sense: Sense,
        floors: list[Floor] | None = None,
        threshold: Floor | None = None,
        name: str | None = None,
    ) -> LinearMipModel:
        floors = list(floors or [])
        layout = self._layout(objective_name, floors, threshold)
        rows = _Rows()
        self._add_hard_rows(rows, layout)
        self._add_q_rows(rows, layout)
        self._add_deviation_rows(rows, layout)
        for index, floor in enumerate(floors):
            self._add_floor(rows, layout, floor, f"retained_{index}")
        if threshold is not None:
            self._add_floor(rows, layout, threshold, "oracle_threshold")

        c, reported_scale = self._objective_vector(layout, objective_name)
        col_lower = np.zeros(layout.nvars, dtype=np.float64)
        col_upper = np.full(layout.nvars, math.inf, dtype=np.float64)
        col_upper[layout.x] = 1.0
        col_upper[layout.y] = 1.0
        for p, coverers in enumerate(self.pattern_coverers):
            if len(coverers) == 0:
                col_upper[layout.y.start + p] = 0.0
        integrality = np.zeros(layout.nvars, dtype=np.int8)
        integrality[layout.x] = 1
        variable_names = [f"x::{v}" for v in self.candidates["venue_id"].astype(str)]
        variable_names.extend(f"y::{p}" for p in range(self.ny))
        extra: dict[str, slice | int] = {}
        if layout.q is not None:
            # q is continuous. The authoritative 0.5% certificate is produced by
            # the threshold-infeasibility oracle, so there is no reason to add a
            # large-domain integer variable or scale the ratio by 1e6.
            col_upper[layout.q] = 1.0
            variable_names.append("q::min_sigungu_ratio")
            extra["q"] = layout.q
        if layout.count_lambda is not None:
            # Lambda variables give the convex hull of the integer visit-count
            # deviation function.  At an integer count the cost is exactly
            # |count - visit_count/ns|, while the root relaxation cannot exploit
            # the artificial zero at the non-integer equal-allocation target.
            width = self.max_sigungu + 1
            for sigungu in self.sigungu_values:
                variable_names.extend(f"count_lambda::{sigungu}::{k}" for k in range(width))
            col_upper[layout.count_lambda] = 1.0
            extra["count_lambda"] = layout.count_lambda

        model = LinearMipModel(
            name=name or f"{self.candidate_set}__{objective_name}",
            objective_name=objective_name,
            sense=sense,
            c=c,
            A=rows.matrix(layout.nvars),
            row_lower=np.asarray(rows.lb, dtype=np.float64),
            row_upper=np.asarray(rows.ub, dtype=np.float64),
            col_lower=col_lower,
            col_upper=col_upper,
            integrality=integrality,
            variable_names=variable_names,
            x_slice=layout.x,
            y_slice=layout.y,
            extra_slices=extra,
            reported_objective_scale=reported_scale,
            metadata={
                "candidate_set": self.candidate_set,
                "row_names": rows.names,
                "one_link_pattern_formulation": True,
                "integer_count_deviation_convex_hull": layout.count_lambda is not None,
                "candidate_count": self.nx,
                "pattern_count": self.ny,
                "coverage_incidence_count": int(self._candidate_pattern.nnz),
                "removed_redundant_or_rows": int(self._candidate_pattern.nnz),
                "floors": [f.__dict__ for f in floors],
                "threshold": threshold.__dict__ if threshold else None,
            },
        )
        model.validate()
        return model

    def covered_mask(self, selected_indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(selected_indices, dtype=int)
        if selected.size == 0:
            return np.zeros(self.ny, dtype=bool)
        counts = np.asarray(self._candidate_pattern[selected].sum(axis=0)).ravel()
        return counts > 0

    def metrics(self, selected_indices: np.ndarray) -> dict[str, Any]:
        selected = np.asarray(selected_indices, dtype=int)
        covered = self.covered_mask(selected)
        ratios: dict[str, float] = {}
        for sigungu, idx in self.sigungu_pattern_indices.items():
            total = self.sigungu_population_total[sigungu]
            cov = float(self.population[idx][covered[idx]].sum())
            ratios[sigungu] = cov / total if total > 0 else math.nan
        counts = pd.Series(self.candidate_sigungu[selected]).value_counts().reindex(self.sigungu_values, fill_value=0)
        return {
            "selected_venue_count": int(selected.size),
            "unique_elderly_population": float(self.population[covered].sum()),
            "need_weighted_population": float(self.need_weighted[covered].sum()),
            "high_need_population": float(self.high_need[covered].sum()),
            "min_sigungu_coverage_ratio": float(min(ratios.values())) if ratios else math.nan,
            "mean_sigungu_coverage_ratio": float(np.nanmean(list(ratios.values()))) if ratios else math.nan,
            "sigungu_coverage": ratios,
            "cost": float(self.cost_for_selection(selected)),
            "visit_location_sigungu_count": int((counts > 0).sum()),
        }

    def cost_for_selection(self, selected_indices: np.ndarray) -> float:
        selected = np.asarray(selected_indices, dtype=int)
        value = float(self._candidate_cost[selected].sum())
        counts = pd.Series(self.candidate_sigungu[selected]).value_counts().reindex(self.sigungu_values, fill_value=0)
        target = self.visit_count / max(1, self.ns)
        penalty = float(
            self.base_config.get("optimization", {}).get("visit_location_deviation_penalty", 100.0)
        )
        value += float(np.abs(counts.to_numpy(float) - target).sum()) * penalty
        return value

    def objective_for_selection(self, objective_name: str, selected_indices: np.ndarray) -> float:
        metrics = self.metrics(selected_indices)
        mapping = {
            "total_population": "unique_elderly_population",
            "need_weighted": "need_weighted_population",
            "high_need_population": "high_need_population",
            "min_sigungu_coverage": "min_sigungu_coverage_ratio",
            "cost": "cost",
        }
        return float(metrics[mapping[objective_name]])

    def complete_solution(self, model: LinearMipModel, selected_indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(selected_indices, dtype=int)
        values = np.zeros(model.n_col, dtype=np.float64)
        values[model.x_slice.start + selected] = 1.0
        covered = self.covered_mask(selected)
        values[model.y_slice] = covered.astype(float)
        if "q" in model.extra_slices:
            q = int(model.extra_slices["q"])
            ratio = float(self.metrics(selected)["min_sigungu_coverage_ratio"])
            # Move one representable float downward so the MIP start cannot
            # violate q <= exact covered ratio due to accumulation rounding.
            values[q] = max(0.0, math.nextafter(ratio, -math.inf))
        if "count_lambda" in model.extra_slices:
            count_lambda = model.extra_slices["count_lambda"]
            assert isinstance(count_lambda, slice)
            counts = (
                pd.Series(self.candidate_sigungu[selected])
                .value_counts()
                .reindex(self.sigungu_values, fill_value=0)
                .to_numpy(int)
            )
            width = self.max_sigungu + 1
            for s_idx, count in enumerate(counts):
                if count < 0 or count > self.max_sigungu:
                    raise ValueError(f"Visit count outside lambda domain: {count}")
                values[count_lambda.start + s_idx * width + int(count)] = 1.0
        return values

    def feasibility_violations(self, model: LinearMipModel, values: np.ndarray, tol: float = 1e-6) -> dict[str, float]:
        values = np.asarray(values, dtype=float)
        activity = np.asarray(model.A @ values).ravel()
        row_low = np.maximum(model.row_lower - activity, 0.0)
        row_high = np.maximum(activity - model.row_upper, 0.0)
        col_low = np.maximum(model.col_lower - values, 0.0)
        col_high = np.maximum(values - model.col_upper, 0.0)
        int_idx = np.flatnonzero(model.integrality != 0)
        int_err = np.abs(values[int_idx] - np.rint(values[int_idx])) if int_idx.size else np.zeros(0)
        return {
            "max_row_lower": float(row_low.max(initial=0.0)),
            "max_row_upper": float(row_high.max(initial=0.0)),
            "max_col_lower": float(col_low.max(initial=0.0)),
            "max_col_upper": float(col_high.max(initial=0.0)),
            "max_integrality": float(int_err.max(initial=0.0)),
            "feasible": float(
                max(
                    row_low.max(initial=0.0),
                    row_high.max(initial=0.0),
                    col_low.max(initial=0.0),
                    col_high.max(initial=0.0),
                    int_err.max(initial=0.0),
                )
                <= tol
            ),
        }

    def selected_from_solution(self, model: LinearMipModel, solution: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.asarray(solution[model.x_slice]) >= 0.5).astype(int)

    def validate_selection(self, selected_indices: np.ndarray) -> tuple[bool, str | None]:
        raw = np.asarray(selected_indices)
        if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
            return False, "selected indices must be a one-dimensional integer array"
        selected = raw.astype(np.int64, copy=False)
        if selected.size != self.visit_count:
            return False, f"expected {self.visit_count} selections; got {selected.size}"
        if np.any(selected < 0) or np.any(selected >= self.nx):
            return False, "selected index out of bounds"
        if len(set(map(int, selected))) != selected.size:
            return False, "duplicate selected index"
        frame = self.candidates.iloc[selected]
        if frame["admin_code"].astype(str).value_counts().max() > self.max_admin:
            return False, "admin cap"
        if frame["sigungu"].astype(str).value_counts().max() > self.max_sigungu:
            return False, "sigungu cap"
        if self.one_cluster:
            clusters = frame["cluster_id"].fillna(frame["venue_id"]).astype(str)
            if clusters.duplicated().any():
                return False, "coverage cluster cap"
        return True, None
