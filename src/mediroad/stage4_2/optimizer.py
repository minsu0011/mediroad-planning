from __future__ import annotations

import math
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .errors import ContractError, SolverCertificationError
from .candidates import candidate_cost_coefficients
from .types import GridPatternData, ScenarioResult, SolverStageTelemetry


STATUS_MAP = {
    0: "OPTIMAL",
    1: "LIMIT_OR_ITERATION",
    2: "INFEASIBLE",
    3: "UNBOUNDED",
    4: "OTHER_FAILURE",
}


class SpatialMILP:
    """Lossless grid-pattern MILP for MEDIROAD Stage 4.2.

    Candidate x variables are binary. Pattern y variables are continuous but are forced
    exactly to the OR of covering x variables by y <= sum(x) and y >= x_i constraints.
    This preserves unique coverage while reducing the number of population variables.
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        patterns: GridPatternData,
        config: dict[str, Any],
        *,
        candidate_set: str,
    ) -> None:
        self.candidates = candidates.reset_index(drop=True).copy()
        self.patterns = patterns
        self.config = config
        self.candidate_set = candidate_set
        if len(self.candidates) != patterns.n_candidates:
            raise ContractError("Candidate frame and pattern candidate count differ")
        self.nx = patterns.n_candidates
        self.ny = patterns.n_patterns
        self.sigungu_values = sorted(set(self.candidates["sigungu"].astype(str)))
        self.ns = len(self.sigungu_values)
        self.z_scale = int(self.config.get("solver", {}).get("min_sigungu_ratio_integer_scale", 1_000_000))
        if self.z_scale < 1:
            raise ContractError("solver.min_sigungu_ratio_integer_scale must be positive")
        self.x_slice = slice(0, self.nx)
        self.y_slice = slice(self.nx, self.nx + self.ny)
        self.z_index = self.nx + self.ny
        self.dev_slice = slice(self.z_index + 1, self.z_index + 1 + self.ns)
        self.nvars = self.z_index + 1 + self.ns
        self._objective_vectors = self._build_objective_vectors()
        self._base_constraint = self._build_base_constraints()
        self._bounds, self._integrality = self._build_bounds_integrality()

    def _build_bounds_integrality(self) -> tuple[Bounds, np.ndarray]:
        lb = np.zeros(self.nvars, dtype=float)
        ub = np.full(self.nvars, np.inf, dtype=float)
        ub[self.x_slice] = 1.0
        ub[self.y_slice] = 1.0
        ub[self.z_index] = float(self.z_scale)
        integrality = np.zeros(self.nvars, dtype=np.int8)
        integrality[self.x_slice] = 1
        # z represents the minimum coverage ratio in fixed-point integer units.
        # dev represents |n_sigungu * selected_count - visit_count|.  Both
        # quantities are discrete consequences of binary selections; declaring
        # them integer strengthens the relaxation without changing policy logic.
        integrality[self.z_index] = 1
        integrality[self.dev_slice] = 1
        return Bounds(lb, ub), integrality

    def _build_objective_vectors(self) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for name, values in [
            ("total_population", self.patterns.population),
            ("need_weighted", self.patterns.need_weighted),
            ("high_need_population", self.patterns.high_need_population),
        ]:
            vec = np.zeros(self.nvars, dtype=float)
            vec[self.y_slice] = np.asarray(values, dtype=float)
            vectors[name] = vec
        z = np.zeros(self.nvars, dtype=float)
        z[self.z_index] = 1.0 / float(self.z_scale)
        vectors["min_sigungu_coverage"] = z

        cost = np.zeros(self.nvars, dtype=float)
        cost[self.x_slice] = candidate_cost_coefficients(self.candidates, self.config)
        # Absolute visit-location deviations are represented by dev variables.
        cost[self.dev_slice] = float(
            self.config.get("optimization", {}).get("visit_location_deviation_penalty", 100.0)
        ) / max(1, self.ns)
        vectors["cost"] = cost
        return vectors

    def _build_base_constraints(self) -> LinearConstraint:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lb: list[float] = []
        ub: list[float] = []
        r = 0

        def add(coeffs: list[tuple[int, float]], lower: float, upper: float) -> None:
            nonlocal r
            for c, v in coeffs:
                if v:
                    rows.append(r)
                    cols.append(c)
                    vals.append(float(v))
            lb.append(float(lower))
            ub.append(float(upper))
            r += 1

        visit_count = int(self.config.get("optimization", {}).get("visit_count", 20))
        add([(i, 1.0) for i in range(self.nx)], visit_count, visit_count)

        max_admin = int(self.config.get("candidate_expansion", {}).get("max_visits_per_admin", 2))
        for _, group in self.candidates.groupby(self.candidates["admin_code"].astype(str)):
            add([(int(i), 1.0) for i in group.index], -np.inf, max_admin)

        max_sigungu = int(self.config.get("candidate_expansion", {}).get("max_visits_per_sigungu", 4))
        for _, group in self.candidates.groupby(self.candidates["sigungu"].astype(str)):
            add([(int(i), 1.0) for i in group.index], -np.inf, max_sigungu)

        if bool(self.config.get("candidate_expansion", {}).get("one_per_coverage_cluster", True)):
            clusters = self.candidates["cluster_id"].fillna(self.candidates["venue_id"]).astype(str)
            for _, idx in clusters.groupby(clusters).groups.items():
                idxs = list(map(int, idx))
                if len(idxs) > 1:
                    add([(i, 1.0) for i in idxs], -np.inf, 1.0)

        # Exact OR: y_p <= sum_i x_i and y_p >= x_i for every coverer.
        for p, coverers in enumerate(self.patterns.pattern_coverers):
            y = self.nx + p
            if len(coverers) == 0:
                add([(y, 1.0)], 0.0, 0.0)
                continue
            add([(y, 1.0), *[(int(i), -1.0) for i in coverers]], -np.inf, 0.0)
            for i in coverers:
                add([(int(i), 1.0), (y, -1.0)], -np.inf, 0.0)

        # z <= covered_population(sigungu) / total_population(sigungu)
        for sigungu in sorted(set(self.patterns.sigungu)):
            idx = np.flatnonzero(self.patterns.sigungu == sigungu)
            total = float(self.patterns.population[idx].sum())
            if total <= 0:
                continue
            coeffs = [(self.z_index, total / float(self.z_scale))]
            coeffs.extend((self.nx + int(p), -float(self.patterns.population[p])) for p in idx)
            add(coeffs, -np.inf, 0.0)

        # Absolute visit-location deviations from equal allocation.
        # Exact integer form of dev >= |count - visit_count/ns|.  Scaling by
        # ns removes the fractional target; the objective coefficient above
        # divides by ns, so the modeled policy cost is unchanged.
        for s_idx, sigungu in enumerate(self.sigungu_values):
            dev = self.dev_slice.start + s_idx
            cidx = np.flatnonzero(self.candidates["sigungu"].astype(str).to_numpy() == sigungu)
            # dev >= ns*count - visit_count
            add([*[(int(i), float(self.ns)) for i in cidx], (dev, -1.0)], -np.inf, visit_count)
            # dev >= visit_count - ns*count
            add([*[(int(i), -float(self.ns)) for i in cidx], (dev, -1.0)], -np.inf, -visit_count)

        matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(r, self.nvars)).tocsr()
        return LinearConstraint(matrix, np.asarray(lb), np.asarray(ub))

    def _constraint_with_floors(self, floors: dict[str, tuple[str, float]]) -> LinearConstraint:
        base = self._base_constraint
        matrices = [base.A]
        lbs = [np.asarray(base.lb, dtype=float)]
        ubs = [np.asarray(base.ub, dtype=float)]
        for name, (sense, value) in floors.items():
            vec = self._objective_vectors[name]
            matrices.append(sparse.csr_matrix(vec.reshape(1, -1)))
            if sense == "max":
                lbs.append(np.asarray([float(value)]))
                ubs.append(np.asarray([np.inf]))
            elif sense == "min":
                lbs.append(np.asarray([-np.inf]))
                ubs.append(np.asarray([float(value)]))
            else:
                raise ValueError(sense)
        return LinearConstraint(sparse.vstack(matrices, format="csr"), np.concatenate(lbs), np.concatenate(ubs))

    def _solve_stage(
        self,
        *,
        scenario: str,
        stage_index: int,
        objective_name: str,
        sense: str,
        floors: dict[str, tuple[str, float]],
        time_limit: float,
        relative_gap: float,
    ) -> tuple[Any, SolverStageTelemetry]:
        vec = self._objective_vectors[objective_name]
        c = -vec if sense == "max" else vec
        constraints = self._constraint_with_floors(floors)
        runtime = self.config.get("runtime", {})
        options: dict[str, Any] = {
            "time_limit": float(time_limit),
            "mip_rel_gap": float(relative_gap),
            "presolve": True,
            "disp": False,
        }
        # HiGHS owns a process-global scheduler.  Never inject a new default
        # thread count into callers that did not declare one; changing it after
        # another solve can yield ``HiGHS Status 0: Not Set``.  Official Stage
        # 4.2 workers always declare the frozen value 4 in a fresh process.
        if "highs_threads_per_worker" in runtime:
            options["threads"] = int(runtime["highs_threads_per_worker"])
        if "highs_random_seed" in runtime:
            options["random_seed"] = int(runtime["highs_random_seed"])
        started = time.perf_counter()
        result = milp(
            c,
            integrality=self._integrality,
            bounds=self._bounds,
            constraints=constraints,
            options=options,
        )
        wall = time.perf_counter() - started
        raw_status = int(result.status)
        status = STATUS_MAP.get(raw_status, f"STATUS_{result.status}")
        has_incumbent = getattr(result, "x", None) is not None and getattr(result, "fun", None) is not None
        objective_value = None
        best_bound = None
        if has_incumbent:
            objective_value = float(-result.fun if sense == "max" else result.fun)
        dual = getattr(result, "mip_dual_bound", None)
        if dual is not None and np.isfinite(dual):
            best_bound = float(-dual if sense == "max" else dual)
        absolute_gap = None
        gap = None
        solver_reported_gap = None
        raw_mip_gap = getattr(result, "mip_gap", None)
        if raw_mip_gap is not None and np.isfinite(raw_mip_gap):
            solver_reported_gap = max(0.0, float(raw_mip_gap))
        if objective_value is not None and best_bound is not None:
            directional = best_bound - objective_value if sense == "max" else objective_value - best_bound
            absolute_gap = max(0.0, float(directional))
            # Use the solver's own MIP-gap definition when it is available.  In
            # particular, objectives such as min_sigungu_coverage live in [0,1];
            # dividing their absolute gap by max(|incumbent|, 1) understates the
            # relative gap and can falsely certify a weak incumbent.
            derived_gap = absolute_gap / max(abs(float(objective_value)), 1e-12)
            gap = solver_reported_gap if solver_reported_gap is not None else derived_gap
        if raw_status == 0 and gap is not None and gap > 1e-9:
            status = "FEASIBLE_GAP_LIMIT"
        certified = bool(has_incumbent and gap is not None and gap <= relative_gap + 1e-12)
        telemetry = SolverStageTelemetry(
            scenario=scenario,
            candidate_set=self.candidate_set,
            stage_index=stage_index,
            objective_name=objective_name,
            sense=sense,
            solver_status=status,
            success=bool(has_incumbent),
            objective_value=objective_value,
            best_bound=best_bound,
            absolute_gap=absolute_gap,
            relative_gap=gap,
            solver_reported_mip_gap=solver_reported_gap,
            mip_node_count=int(getattr(result, "mip_node_count", 0)) if getattr(result, "mip_node_count", None) is not None else None,
            wall_time_sec=float(wall),
            message=str(result.message),
            time_limit_sec=float(time_limit),
            solver_engine="scipy_milp_highs",
            threads_requested=int(runtime.get("highs_threads_per_worker", 0)),
            random_seed=int(runtime.get("highs_random_seed", 42)),
            applied_objective_floors_json=json.dumps(floors, sort_keys=True, separators=(",", ":")),
            certified=certified,
        )
        if not has_incumbent:
            raise SolverCertificationError(
                f"No incumbent for {scenario}/{self.candidate_set}/{objective_name}: {status} {result.message}",
                telemetry=[telemetry],
            )
        return result, telemetry

    def _scenario_plan(self, scenario: str) -> list[tuple[str, str, float | None]]:
        retention = self.config.get("optimization", {}).get("objective_retention", {})
        if scenario == "efficiency":
            return [
                ("total_population", "max", float(retention.get("total_population", 0.99))),
                ("need_weighted", "max", float(retention.get("need_weighted", 0.99))),
                ("min_sigungu_coverage", "max", float(retention.get("min_sigungu_coverage", 1.0))),
                ("cost", "min", None),
            ]
        if scenario == "balanced":
            return [
                ("need_weighted", "max", float(retention.get("need_weighted", 0.99))),
                ("min_sigungu_coverage", "max", float(retention.get("min_sigungu_coverage", 1.0))),
                ("total_population", "max", float(retention.get("total_population", 0.99))),
                ("cost", "min", None),
            ]
        if scenario == "equity":
            return [
                ("min_sigungu_coverage", "max", float(retention.get("min_sigungu_coverage", 1.0))),
                ("high_need_population", "max", float(retention.get("high_need", 0.99))),
                ("need_weighted", "max", float(retention.get("need_weighted", 0.99))),
                ("cost", "min", None),
            ]
        raise ContractError(f"Unknown scenario {scenario}")

    def solve(
        self,
        scenario: str,
        *,
        efficiency_reference: float | None,
        greedy_floor: float | None,
        initial_incumbent_indices: np.ndarray | None = None,
        time_limit_per_stage: float,
        gap_threshold: float,
    ) -> ScenarioResult:
        floors: dict[str, tuple[str, float]] = {}
        if scenario == "efficiency" and greedy_floor is not None:
            floors["total_population"] = ("max", float(greedy_floor))
        if scenario == "balanced":
            if efficiency_reference is None:
                raise ContractError("Balanced scenario requires efficiency reference")
            fraction = float(self.config.get("optimization", {}).get("balanced_min_efficiency_fraction", 0.80))
            floors["total_population"] = ("max", efficiency_reference * fraction)
        if scenario == "equity":
            if efficiency_reference is None:
                raise ContractError("Equity scenario requires efficiency reference")
            fraction = float(self.config.get("optimization", {}).get("equity_min_efficiency_fraction", 0.60))
            floors["total_population"] = ("max", efficiency_reference * fraction)

        telemetry: list[SolverStageTelemetry] = []
        final_result = None
        incumbent_indices = (
            np.asarray(initial_incumbent_indices, dtype=int)
            if initial_incumbent_indices is not None
            else None
        )
        for stage_index, (name, sense, retention) in enumerate(self._scenario_plan(scenario), start=1):
            stage_floors = dict(floors)
            if incumbent_indices is not None and bool(
                self.config.get("solver", {}).get("known_incumbent_objective_cutoff", False)
            ):
                incumbent_value = self._selection_objective_value(incumbent_indices, name)
                tolerance = 1e-9 * max(1.0, abs(incumbent_value))
                if sense == "max":
                    stage_floors[name] = ("max", incumbent_value - tolerance)
                else:
                    stage_floors[name] = ("min", incumbent_value + tolerance)
            try:
                result, tel = self._solve_stage(
                    scenario=scenario,
                    stage_index=stage_index,
                    objective_name=name,
                    sense=sense,
                    floors=stage_floors,
                    time_limit=time_limit_per_stage,
                    relative_gap=gap_threshold,
                )
            except SolverCertificationError as exc:
                exc.telemetry = [*telemetry, *exc.telemetry]
                raise
            telemetry.append(tel)
            final_result = result
            incumbent_indices = np.flatnonzero(np.asarray(result.x[self.x_slice]) >= 0.5)
            if retention is not None and tel.objective_value is not None:
                if sense == "max":
                    retained = tel.objective_value * retention
                    if name == "total_population" and scenario == "efficiency" and greedy_floor is not None:
                        retained = max(retained, greedy_floor)
                    floors[name] = ("max", retained)
                else:
                    floors[name] = ("min", tel.objective_value / max(retention, 1e-12))

        assert final_result is not None and final_result.x is not None
        selected = np.flatnonzero(np.asarray(final_result.x[self.x_slice]) >= 0.5)
        visit_count = int(self.config.get("optimization", {}).get("visit_count", 20))
        if len(selected) != visit_count:
            raise SolverCertificationError(
                f"Expected {visit_count} selected venues; got {len(selected)}",
                telemetry=telemetry,
            )
        max_gap = max((t.relative_gap for t in telemetry if t.relative_gap is not None), default=None)
        all_certified = all(t.certified for t in telemetry)
        cert_class = "OPTIMAL" if all(t.solver_status == "OPTIMAL" for t in telemetry) else (
            "CERTIFIED_NEAR_OPTIMAL" if all_certified else "FEASIBLE_UNCERTIFIED"
        )
        metrics = self.recompute_metrics(selected)
        metrics.update(
            {
                "scenario": scenario,
                "candidate_set": self.candidate_set,
                "certification_class": cert_class,
                "certified": all_certified,
                "max_relative_gap": max_gap,
            }
        )
        chosen = self.candidates.iloc[selected].copy().reset_index(drop=True)
        chosen["candidate_set"] = self.candidate_set
        chosen.insert(0, "scenario", scenario)
        ordered = ["scenario", "candidate_set"] + [c for c in chosen.columns if c not in {"scenario", "candidate_set"}]
        chosen = chosen[ordered]
        return ScenarioResult(
            scenario=scenario,
            candidate_set=self.candidate_set,
            selected_indices=selected,
            selected_venue_ids=chosen["venue_id"].astype(str).tolist(),
            solver_status=telemetry[-1].solver_status,
            certification_class=cert_class,
            certified=all_certified,
            max_relative_gap=max_gap,
            telemetry=telemetry,
            metrics=metrics,
            selected_frame=chosen,
        )

    def _selection_objective_value(self, selected_indices: np.ndarray, objective_name: str) -> float:
        """Evaluate one objective exactly for a known feasible binary selection.

        The value is used only as a redundant incumbent cutoff.  It cannot
        change the optimum, but it prevents a later lexicographic stage from
        exploring regions already known to be worse when SciPy cannot pass a
        native MIP start to HiGHS.
        """

        selected = np.asarray(selected_indices, dtype=int)
        if len(selected) != int(self.config.get("optimization", {}).get("visit_count", 20)):
            raise ContractError("Known incumbent has the wrong visit count")
        selected_set = set(map(int, selected))
        covered = np.fromiter(
            (any(int(v) in selected_set for v in coverers) for coverers in self.patterns.pattern_coverers),
            dtype=bool,
            count=self.patterns.n_patterns,
        )
        if objective_name == "total_population":
            return float(self.patterns.population[covered].sum())
        if objective_name == "need_weighted":
            return float(self.patterns.need_weighted[covered].sum())
        if objective_name == "high_need_population":
            return float(self.patterns.high_need_population[covered].sum())
        if objective_name == "min_sigungu_coverage":
            ratios = []
            for sigungu in sorted(set(self.patterns.sigungu)):
                idx = self.patterns.sigungu == sigungu
                total = float(self.patterns.population[idx].sum())
                if total > 0:
                    ratios.append(float(self.patterns.population[idx & covered].sum()) / total)
            exact = min(ratios) if ratios else 0.0
            return math.floor((exact + 1e-12) * self.z_scale) / self.z_scale
        if objective_name == "cost":
            candidate_cost = float(candidate_cost_coefficients(self.candidates, self.config)[selected].sum())
            counts = self.candidates.iloc[selected]["sigungu"].astype(str).value_counts()
            visit_count = int(self.config.get("optimization", {}).get("visit_count", 20))
            target = visit_count / max(1, self.ns)
            deviation = sum(abs(float(counts.get(sigungu, 0)) - target) for sigungu in self.sigungu_values)
            penalty = float(self.config.get("optimization", {}).get("visit_location_deviation_penalty", 100.0))
            return candidate_cost + penalty * deviation
        raise ContractError(f"Unknown objective {objective_name!r}")

    def recompute_metrics(self, selected_indices: np.ndarray) -> dict[str, Any]:
        selected_set = set(map(int, selected_indices))
        covered = np.fromiter(
            (any(int(v) in selected_set for v in coverers) for coverers in self.patterns.pattern_coverers),
            dtype=bool,
            count=self.patterns.n_patterns,
        )
        pop = float(self.patterns.population[covered].sum())
        need = float(self.patterns.need_weighted[covered].sum())
        high = float(self.patterns.high_need_population[covered].sum())
        sigungu_ratios: dict[str, float] = {}
        for s in sorted(set(self.patterns.sigungu)):
            idx = self.patterns.sigungu == s
            total = float(self.patterns.population[idx].sum())
            cov = float(self.patterns.population[idx & covered].sum())
            sigungu_ratios[s] = cov / total if total > 0 else math.nan
        selected = self.candidates.iloc[selected_indices]
        counts = selected["sigungu"].astype(str).value_counts()
        visit_shares = counts / max(1, counts.sum())
        hhi = float((visit_shares**2).sum())
        gini = _gini(counts.reindex(self.sigungu_values, fill_value=0).to_numpy(float))
        return {
            "selected_venue_count": int(len(selected_indices)),
            "unique_elderly_population": pop,
            "need_weighted_population": need,
            "high_need_population": high,
            "min_sigungu_coverage_ratio": float(min(sigungu_ratios.values())) if sigungu_ratios else math.nan,
            "mean_sigungu_coverage_ratio": float(np.nanmean(list(sigungu_ratios.values()))) if sigungu_ratios else math.nan,
            "visit_location_sigungu_count": int(selected["sigungu"].astype(str).nunique()),
            "selected_admin_count": int(selected["admin_code"].astype(str).nunique()),
            "selected_cluster_count": int(selected["cluster_id"].astype(str).nunique()),
            "visit_location_hhi": hhi,
            "visit_location_gini": gini,
            "sigungu_coverage": sigungu_ratios,
        }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    return float((2 * np.sum((np.arange(1, n + 1)) * values) / (n * values.sum())) - (n + 1) / n)


def constrained_greedy(
    candidates: pd.DataFrame,
    coverage: sparse.csr_matrix,
    grid_population: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, float]:
    visit_count = int(config.get("optimization", {}).get("visit_count", 20))
    max_admin = int(config.get("candidate_expansion", {}).get("max_visits_per_admin", 2))
    max_sigungu = int(config.get("candidate_expansion", {}).get("max_visits_per_sigungu", 4))
    one_cluster = bool(config.get("candidate_expansion", {}).get("one_per_coverage_cluster", True))
    selected: list[int] = []
    covered = np.zeros(coverage.shape[1], dtype=bool)
    admin_counts: dict[str, int] = {}
    sig_counts: dict[str, int] = {}
    clusters: set[str] = set()
    for _ in range(visit_count):
        best = None
        best_key = None
        for i in range(len(candidates)):
            if i in selected:
                continue
            row = candidates.iloc[i]
            admin = str(row["admin_code"])
            sig = str(row["sigungu"])
            cluster = str(row.get("cluster_id", row["venue_id"]))
            if admin_counts.get(admin, 0) >= max_admin or sig_counts.get(sig, 0) >= max_sigungu:
                continue
            if one_cluster and cluster in clusters:
                continue
            cols = coverage.indices[coverage.indptr[i] : coverage.indptr[i + 1]]
            new_cols = cols[~covered[cols]]
            marginal = float(grid_population[new_cols].sum())
            need_tie = float(pd.to_numeric(pd.Series([row.get("need_weighted_exposure", 0.0)]), errors="coerce").fillna(0).iloc[0])
            key = (marginal, need_tie, str(row["venue_id"]))
            if best is None or key > best_key:
                best = i
                best_key = key
        if best is None:
            raise SolverCertificationError(f"Greedy could select only {len(selected)} of {visit_count} venues")
        selected.append(best)
        row = candidates.iloc[best]
        admin = str(row["admin_code"])
        sig = str(row["sigungu"])
        cluster = str(row.get("cluster_id", row["venue_id"]))
        admin_counts[admin] = admin_counts.get(admin, 0) + 1
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        clusters.add(cluster)
        cols = coverage.indices[coverage.indptr[best] : coverage.indptr[best + 1]]
        covered[cols] = True
    return np.asarray(selected, dtype=int), float(grid_population[covered].sum())


def telemetry_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for tel in result.telemetry:
            rows.append(asdict(tel))
    return pd.DataFrame(rows)
