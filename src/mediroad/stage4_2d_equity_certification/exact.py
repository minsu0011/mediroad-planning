from __future__ import annotations

import math
from dataclasses import asdict
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mediroad.stage4_2_certification.backends import HighsBackend, SolveOptions
from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.types import Floor, LinearMipModel

from .model_utils import append_cuts, apply_fixings, local_branching_cut
from .types import CutRecord, EvidenceSeed, FixingRecord, MetricCertificate, SolveEvidence


def _seal_empty_solver_artifact(path: Path, label: str) -> None:
    """Keep the positive-size inventory invariant for legitimate empty streams."""

    if path.is_file() and path.stat().st_size == 0:
        path.write_text(f"# {label}\n", encoding="utf-8")


def _gap(incumbent: float | None, bound: float | None) -> float | None:
    if incumbent is None or bound is None or not math.isfinite(float(bound)):
        return None
    return max(0.0, (float(bound) - float(incumbent)) / max(abs(float(incumbent)), 1e-12))


def _seed_for_model(
    formulation: CertificationFormulation,
    model: LinearMipModel,
    seeds: Iterable[EvidenceSeed],
    objective: str,
) -> tuple[EvidenceSeed | None, np.ndarray | None]:
    candidates: list[tuple[float, EvidenceSeed, np.ndarray]] = []
    for seed in seeds:
        valid, _ = formulation.validate_selection(seed.selected_indices)
        if not valid:
            continue
        full = formulation.complete_solution(model, seed.selected_indices)
        violations = formulation.feasibility_violations(model, full)
        if not bool(violations["feasible"]):
            continue
        value = formulation.objective_for_selection(objective, seed.selected_indices)
        candidates.append((float(value), seed, full))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _solve_options(
    config: dict[str, Any],
    *,
    objective: str,
    phase: str,
    time_limit_sec: float,
    threads: int,
    random_seed: int,
    log_path: Path,
    target: float | None,
) -> SolveOptions:
    solver = config.get("solver", {})
    phase_specific = dict(solver.get("phase_options", {}).get(phase, {}))
    phase_heuristic = phase_specific.pop("mip_heuristic_effort", None)
    phase_start_nodes = phase_specific.pop("mip_max_start_nodes", None)
    phase_options = dict(solver.get("extra_highs_options", {}))
    phase_options.update(phase_specific)
    replay = {}
    replay_table = solver.get("stage42c_oracle_replay", {})
    if phase == "oracle" and isinstance(replay_table, dict):
        candidate = replay_table.get(objective, {})
        if isinstance(candidate, dict) and bool(candidate.get("enabled", False)):
            replay = candidate
            phase_options.update(dict(replay.get("extra_highs_options", {})))
            random_seed = int(replay["random_seed"])
    protected_options = {
        "output_flag",
        "log_to_console",
        "log_file",
        "presolve",
        "parallel",
        "threads",
        "time_limit",
        "mip_rel_gap",
        "mip_abs_gap",
        "random_seed",
        "mip_heuristic_effort",
        "mip_max_start_nodes",
        "objective_target",
    }
    conflicts = sorted(protected_options.intersection(phase_options))
    if conflicts:
        raise ValueError(f"extra HiGHS options cannot override frozen runtime controls: {conflicts}")
    if target is not None and bool(replay.get("use_objective_target", solver.get("use_objective_target", True))):
        phase_options["objective_target"] = float(target)
    default_heuristic = (
        solver.get("direct_heuristic_effort", 0.18)
        if phase == "direct"
        else solver.get("oracle_heuristic_effort", 0.08)
    )
    heuristic_effort = replay.get(
        "heuristic_effort",
        phase_heuristic if phase_heuristic is not None else default_heuristic,
    )
    max_start_nodes = replay.get(
        "mip_max_start_nodes",
        phase_start_nodes
        if phase_start_nodes is not None
        else solver.get("mip_max_start_nodes", 20000),
    )
    return SolveOptions(
        time_limit_sec=float(time_limit_sec),
        relative_gap=float(config["gate"]["near_optimal_relative_gap_max"]) if phase == "direct" else 0.0,
        threads=max(1, int(threads)),
        random_seed=int(random_seed),
        parallel=int(threads) > 1,
        presolve=True,
        heuristic_effort=float(heuristic_effort),
        mip_max_start_nodes=int(max_start_nodes),
        log_to_console=bool(solver.get("log_to_console", True)),
        log_path=log_path,
        improving_solution_path=log_path.with_suffix(".improving.sol"),
        write_model_path=log_path.with_suffix(".mps") if bool(solver.get("write_mps", False)) else None,
        extra_options=phase_options,
    )


def _backend_evidence(
    formulation: CertificationFormulation,
    model: LinearMipModel,
    result: Any,
    *,
    seed: EvidenceSeed | None,
    objective: str,
    result_path: Path,
) -> SolveEvidence:
    best_selected: np.ndarray | None = None
    best_value: float | None = None
    source: str | None = None
    if seed is not None:
        best_selected = np.asarray(seed.selected_indices, dtype=int)
        best_value = float(formulation.objective_for_selection(objective, best_selected))
        source = seed.source
    if result.solution is not None:
        selected = formulation.selected_from_solution(model, result.solution)
        valid, _ = formulation.validate_selection(selected)
        if valid:
            full = formulation.complete_solution(model, selected)
            if bool(formulation.feasibility_violations(model, full)["feasible"]):
                value = float(formulation.objective_for_selection(objective, selected))
                if best_value is None or value > best_value + 1e-8 * max(1.0, abs(best_value)):
                    best_selected, best_value, source = selected, value, "HIGHS_INCUMBENT"
    metric_bound = None if model.objective_name == "feasibility" else result.best_bound
    return SolveEvidence(
        backend=f"HIGHS::{getattr(result, 'highs_version', 'unknown')}",
        status=str(result.status),
        has_incumbent=best_selected is not None,
        is_infeasible=bool(result.is_infeasible),
        is_optimal=bool(result.is_optimal),
        objective_value=best_value,
        best_bound=float(metric_bound) if metric_bound is not None else None,
        relative_gap=_gap(best_value, metric_bound),
        wall_time_sec=float(result.wall_time_sec),
        node_count=result.mip_node_count,
        selected_indices=best_selected,
        message=str(result.message),
        result_path=result_path,
        metadata={
            "incumbent_source": source,
            "solver_reported_gap": result.relative_gap,
            "solver_options": result.options,
            "model_rows": model.n_row,
            "model_cols": model.n_col,
            "model_nnz": int(model.A.nnz),
        },
    )


class EquityFrontStageCertifier:
    def __init__(
        self,
        formulation: CertificationFormulation,
        config: dict[str, Any],
        *,
        threads: int,
        output_dir: Path,
    ) -> None:
        self.f = formulation
        self.cfg = config
        self.threads = max(1, int(threads))
        self.output_dir = output_dir
        self.backend = HighsBackend()
        self.gap = float(config["gate"]["near_optimal_relative_gap_max"])

    def _time(self, metric: str, phase: str, default: float) -> float:
        table = self.cfg.get("solver", {}).get("time_limit_sec", {})
        metric_table = table.get(metric, {}) if isinstance(table, dict) else {}
        return float(metric_table.get(phase, table.get(phase, default)))

    def _certificate_threshold(self, metric: str, incumbent: float) -> float:
        if metric == "min_sigungu_coverage":
            scale = int(getattr(self.f, "min_sigungu_scale", 1_000_000))
            incumbent_units = max(0, int(math.floor(float(incumbent) * scale + 1e-9)))
            raw_units = Decimal(incumbent_units) * (Decimal(1) + Decimal(str(self.gap)))
            threshold_units = int(raw_units.to_integral_value(rounding=ROUND_FLOOR)) + 1
            return threshold_units / float(scale)
        return math.nextafter(float(incumbent) * (1.0 + self.gap), math.inf)

    def _solve(
        self,
        model: LinearMipModel,
        *,
        seeds: list[EvidenceSeed],
        objective: str,
        metric: str,
        phase: str,
        round_index: int,
        target: float | None = None,
        time_limit: float | None = None,
    ) -> SolveEvidence:
        seed, full = _seed_for_model(self.f, model, seeds, objective)
        log_path = self.output_dir / "logs" / f"{metric}__{phase}__r{round_index}.highs.log"
        options = _solve_options(
            self.cfg,
            objective=objective,
            phase=phase,
            time_limit_sec=time_limit or self._time(metric, phase, 1200),
            threads=self.threads,
            random_seed=int(self.cfg.get("solver", {}).get("random_seed", 42)) + round_index * 1009,
            log_path=log_path,
            target=target,
        )
        result = self.backend.solve(model, options, mip_start=full)
        _seal_empty_solver_artifact(log_path, "NO_HIGHS_LOG_OUTPUT")
        _seal_empty_solver_artifact(log_path.with_suffix(".improving.sol"), "NO_IMPROVING_SOLUTION")
        return _backend_evidence(
            self.f,
            model,
            result,
            seed=seed,
            objective=objective,
            result_path=log_path,
        )

    def _direct_certificate(
        self,
        evidence: SolveEvidence,
        *,
        metric: str,
        retained_floor: float,
    ) -> MetricCertificate | None:
        exact_optimal = bool(
            evidence.is_optimal
            and evidence.relative_gap is not None
            and evidence.relative_gap <= 1e-9
        )
        near_optimal = bool(
            evidence.relative_gap is not None
            and evidence.relative_gap <= self.gap + 1e-12
        )
        if exact_optimal or near_optimal:
            return MetricCertificate(
                metric=metric,  # type: ignore[arg-type]
                certified=True,
                certificate="OPTIMAL" if exact_optimal else "DIRECT_MIP_GAP",
                incumbent_value=float(evidence.objective_value),
                best_bound=evidence.best_bound,
                relative_gap=evidence.relative_gap,
                selected_indices=np.asarray(evidence.selected_indices, dtype=int),
                retained_floor=float(retained_floor),
                rounds=1,
                evidence=[asdict(evidence)],
            )
        return None

    def certify(
        self,
        *,
        metric: str,
        direct_model: LinearMipModel,
        oracle_builder: Any,
        seeds: list[EvidenceSeed],
        base_cuts: list[CutRecord],
        base_fixings: list[FixingRecord],
        retained_floor: float,
        cut_builder: Any,
    ) -> MetricCertificate:
        direct = append_cuts(apply_fixings(direct_model, base_fixings), base_cuts)
        evidence_log: list[dict[str, Any]] = []
        direct_evidence = self._solve(
            direct,
            seeds=seeds,
            objective=metric,
            metric=metric,
            phase="direct",
            round_index=0,
        )
        evidence_log.append(asdict(direct_evidence))
        certificate = self._direct_certificate(direct_evidence, metric=metric, retained_floor=retained_floor)
        if certificate is not None:
            return certificate
        if (
            not direct_evidence.has_incumbent
            or direct_evidence.selected_indices is None
            or direct_evidence.objective_value is None
        ):
            raise RuntimeError(
                f"{metric} direct solve returned no feasible incumbent after validated MIP starts; "
                "Stage4.2D refuses to continue with an invented plan."
            )

        incumbent_seed = EvidenceSeed(
            source="DIRECT_AUTHORITATIVE_INCUMBENT",
            selected_indices=np.asarray(direct_evidence.selected_indices, dtype=int),
            metrics={
                **self.f.metrics(np.asarray(direct_evidence.selected_indices, dtype=int)),
                metric: float(direct_evidence.objective_value),
            },
        )
        working_seeds = [incumbent_seed, *seeds]
        incumbent = float(direct_evidence.objective_value)
        selected = np.asarray(direct_evidence.selected_indices, dtype=int)
        best_bound = direct_evidence.best_bound
        exclusion_cuts: list[CutRecord] = []
        max_rounds = int(self.cfg.get("solver", {}).get("max_certificate_rounds", 4))

        for round_index in range(1, max_rounds + 1):
            target = self._certificate_threshold(metric, incumbent)
            target_cuts, target_fixings, strengthening_details = cut_builder(target, working_seeds)
            analytic = strengthening_details.get("analytic_impossibility_witnesses", [])
            if analytic:
                return MetricCertificate(
                    metric=metric,  # type: ignore[arg-type]
                    certified=True,
                    certificate="ANALYTIC_SUBMODULAR_CARDINALITY_BOUND",
                    incumbent_value=incumbent,
                    best_bound=target,
                    relative_gap=self.gap,
                    selected_indices=selected,
                    retained_floor=float(retained_floor),
                    rounds=round_index,
                    evidence=[*evidence_log, {"analytic_witnesses": analytic}],
                )

            # Prove or search the closest neighborhoods first. Infeasible local
            # balls become globally valid exclusion cuts for this and higher targets.
            radii = [int(v) for v in self.cfg.get("solver", {}).get("local_branching_radii", [1, 2, 3])]
            local_hit = False
            for seed_index, seed in enumerate(working_seeds[: int(self.cfg.get("solver", {}).get("local_seed_limit", 6))]):
                for radius in radii:
                    local_model = oracle_builder(target)
                    local_model = apply_fixings(local_model, [*base_fixings, *target_fixings])
                    local_model = append_cuts(
                        local_model,
                        [*base_cuts, *target_cuts, *exclusion_cuts, local_branching_cut(seed.selected_indices, self.f.visit_count, radius, mode="inside")],
                    )
                    local_evidence = self._solve(
                        local_model,
                        seeds=working_seeds,
                        objective=metric,
                        metric=metric,
                        phase="local",
                        round_index=round_index * 100 + seed_index * 10 + radius,
                        target=target,
                        time_limit=self._time(metric, "local", 180),
                    )
                    local_payload = asdict(local_evidence)
                    local_payload["target"] = target
                    local_payload["radius"] = radius
                    local_payload["seed_source"] = seed.source
                    evidence_log.append(local_payload)
                    if local_evidence.objective_value is not None and local_evidence.objective_value >= target - 1e-7 * max(1.0, abs(target)):
                        incumbent = float(local_evidence.objective_value)
                        selected = np.asarray(local_evidence.selected_indices, dtype=int)
                        new_seed = EvidenceSeed(
                            source=f"LOCAL_BRANCHING_TARGET_HIT::r{radius}",
                            selected_indices=selected,
                            metrics={metric: incumbent},
                        )
                        working_seeds.insert(0, new_seed)
                        local_hit = True
                        break
                    local_bound_proves = (
                        local_model.objective_name != "feasibility"
                        and
                        local_evidence.best_bound is not None
                        and local_evidence.best_bound < target - 1e-8 * max(1.0, abs(target))
                    )
                    if local_evidence.is_infeasible or local_bound_proves:
                        exclusion_cuts.append(local_branching_cut(seed.selected_indices, self.f.visit_count, radius, mode="exclude"))
                if local_hit:
                    break
            if local_hit:
                continue

            oracle = oracle_builder(target)
            oracle = apply_fixings(oracle, [*base_fixings, *target_fixings])
            global_strengthening = self.cfg.get("solver", {}).get("global_oracle_strengthening", {})
            use_global_strengthening = bool(
                global_strengthening.get(metric, True)
                if isinstance(global_strengthening, dict)
                else global_strengthening
            )
            oracle_cuts = [*exclusion_cuts]
            replay_table = self.cfg.get("solver", {}).get("stage42c_oracle_replay", {})
            replay = replay_table.get(metric, {}) if isinstance(replay_table, dict) else {}
            include_exclusions = bool(replay.get("include_local_exclusion_cuts", True))
            if not include_exclusions:
                oracle_cuts = []
            if use_global_strengthening:
                oracle_cuts = [*base_cuts, *target_cuts, *oracle_cuts]
            oracle = append_cuts(oracle, oracle_cuts)
            oracle_evidence = self._solve(
                oracle,
                seeds=working_seeds,
                objective=metric,
                metric=metric,
                phase="oracle",
                round_index=round_index,
                # A zero-objective feasibility model already contains the strict
                # target as a row.  Passing the metric target as a HiGHS objective
                # target changes search semantics and was not part of the verified
                # Stage4.2C oracle that proved this exact model infeasible.
                target=target if oracle.objective_name != "feasibility" else None,
            )
            payload = asdict(oracle_evidence)
            payload["target"] = target
            payload["strengthening"] = strengthening_details
            payload["global_oracle_strengthening_applied"] = use_global_strengthening
            payload["global_oracle_cut_count"] = len(oracle_cuts)
            payload["exclusion_cut_count"] = len(exclusion_cuts)
            payload["local_exclusion_cuts_applied_to_global_oracle"] = include_exclusions
            payload["stage42c_oracle_replay_applied"] = bool(replay.get("enabled", False))
            evidence_log.append(payload)
            bound_below_target = (
                oracle.objective_name != "feasibility"
                and
                oracle_evidence.best_bound is not None
                and oracle_evidence.best_bound < target - 1e-8 * max(1.0, abs(target))
            )
            if oracle_evidence.is_infeasible or bound_below_target:
                return MetricCertificate(
                    metric=metric,  # type: ignore[arg-type]
                    certified=True,
                    certificate=(
                        "THRESHOLD_INFEASIBILITY_ORACLE"
                        if oracle_evidence.is_infeasible
                        else "GLOBAL_BOUND_BELOW_THRESHOLD"
                    ),
                    incumbent_value=incumbent,
                    best_bound=target,
                    relative_gap=self.gap,
                    selected_indices=selected,
                    retained_floor=float(retained_floor),
                    rounds=round_index,
                    evidence=evidence_log,
                )
            if oracle_evidence.objective_value is not None and oracle_evidence.objective_value >= target - 1e-7 * max(1.0, abs(target)):
                incumbent = float(oracle_evidence.objective_value)
                selected = np.asarray(oracle_evidence.selected_indices, dtype=int)
                best_bound = oracle_evidence.best_bound
                working_seeds.insert(
                    0,
                    EvidenceSeed(
                        source="GLOBAL_THRESHOLD_TARGET_HIT",
                        selected_indices=selected,
                        metrics={metric: incumbent},
                    ),
                )
                continue
            if oracle_evidence.best_bound is not None:
                best_bound = oracle_evidence.best_bound
            break

        return MetricCertificate(
            metric=metric,  # type: ignore[arg-type]
            certified=False,
            certificate="FEASIBLE_UNCERTIFIED",
            incumbent_value=incumbent,
            best_bound=best_bound,
            relative_gap=_gap(incumbent, best_bound),
            selected_indices=selected,
            retained_floor=float(retained_floor),
            rounds=max_rounds,
            evidence=evidence_log,
        )
