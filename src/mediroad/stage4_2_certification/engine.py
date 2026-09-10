from __future__ import annotations

import math
from decimal import Decimal, ROUND_FLOOR
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .backends import HighsBackend, SolveOptions
from .formulation import CertificationFormulation
from .seeds import (
    SingleSwapImprover,
    choose_seed,
    discover_seed_selections,
    evaluate_seed_pool,
)
from .types import Floor, ScenarioCertificationResult, Seed, StageResult


class CertificationEngine:
    def __init__(
        self,
        *,
        project_root: Path,
        formulation: CertificationFormulation,
        base_config: dict[str, Any],
        cert_config: dict[str, Any],
        stage41: Any,
        greedy_indices: np.ndarray,
        backend: HighsBackend,
        highs_threads: int,
        output_dir: Path,
    ) -> None:
        self.project_root = project_root
        self.f = formulation
        self.base_config = base_config
        self.cfg = cert_config
        self.stage41 = stage41
        self.greedy_indices = np.asarray(greedy_indices, dtype=int)
        self.backend = backend
        self.highs_threads = int(highs_threads)
        self.output_dir = output_dir
        self.gap = float(cert_config["gate"]["near_optimal_relative_gap_max"])
        self.retention = base_config.get("optimization", {}).get("objective_retention", {})
        self.improver = SingleSwapImprover(
            formulation,
            allow_gpu=bool(cert_config.get("hardware", {}).get("gpu_seed_search", True)),
        )


    def _swap_round_limit(self) -> int:
        configured = int(self.cfg.get("seed_search", {}).get("max_rounds", 25))
        if self.improver.backend == "CUPY_CUDA":
            return configured
        cpu_cap = int(self.cfg.get("seed_search", {}).get("cpu_max_rounds", 4))
        return max(0, min(configured, cpu_cap))

    @staticmethod
    def _gap_from_bound(sense: str, incumbent: float, best_bound: float | None) -> float | None:
        if best_bound is None or not math.isfinite(float(best_bound)):
            return None
        denominator = max(abs(float(incumbent)), 1e-12)
        if sense == "max":
            return max(0.0, (float(best_bound) - float(incumbent)) / denominator)
        return max(0.0, (float(incumbent) - float(best_bound)) / denominator)

    def _local_improve_selection(
        self,
        selected: np.ndarray,
        *,
        objective_name: str,
        sense: str,
        floors: list[Floor],
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        before = self.f.objective_for_selection(objective_name, selected)
        rounds = self._swap_round_limit()
        details: dict[str, Any] = {
            "attempted": rounds > 0,
            "backend": self.improver.backend,
            "round_limit": rounds,
            "objective_before": before,
        }
        if rounds <= 0:
            details["objective_after"] = before
            details["rounds"] = 0
            return np.asarray(selected, dtype=int), float(before), details
        improved = self.improver.improve(
            selected,
            objective_name=objective_name,
            sense=sense,
            floors=floors,
            max_rounds=rounds,
        )
        details.update(
            {
                "rounds": improved.rounds,
                "objective_after": improved.objective_value,
            }
        )
        return (
            np.asarray(improved.selected_indices, dtype=int),
            float(improved.objective_value),
            details,
        )

    def _scenario_plan(self, scenario: str) -> list[tuple[str, str, float | None]]:
        if scenario == "efficiency":
            return [
                ("total_population", "max", float(self.retention.get("total_population", 0.99))),
                ("need_weighted", "max", float(self.retention.get("need_weighted", 0.99))),
                ("min_sigungu_coverage", "max", float(self.retention.get("min_sigungu_coverage", 1.0))),
                ("cost", "min", None),
            ]
        if scenario == "balanced":
            return [
                ("need_weighted", "max", float(self.retention.get("need_weighted", 0.99))),
                ("min_sigungu_coverage", "max", float(self.retention.get("min_sigungu_coverage", 1.0))),
                ("total_population", "max", float(self.retention.get("total_population", 0.99))),
                ("cost", "min", None),
            ]
        if scenario == "equity":
            return [
                ("min_sigungu_coverage", "max", float(self.retention.get("min_sigungu_coverage", 1.0))),
                ("high_need_population", "max", float(self.retention.get("high_need", 0.99))),
                ("need_weighted", "max", float(self.retention.get("need_weighted", 0.99))),
                ("cost", "min", None),
            ]
        raise ValueError(f"Unknown scenario: {scenario}")

    def _initial_floors(
        self,
        scenario: str,
        efficiency_reference: float | None,
        greedy_population: float,
    ) -> list[Floor]:
        floors: list[Floor] = []
        if scenario == "efficiency":
            floors.append(Floor("total_population", "max", float(greedy_population)))
        elif scenario == "balanced":
            if efficiency_reference is None:
                raise ValueError("Balanced scenario requires efficiency reference")
            fraction = float(
                self.base_config.get("optimization", {}).get("balanced_min_efficiency_fraction", 0.80)
            )
            floors.append(Floor("total_population", "max", efficiency_reference * fraction))
        elif scenario == "equity":
            if efficiency_reference is None:
                raise ValueError("Equity scenario requires efficiency reference")
            fraction = float(
                self.base_config.get("optimization", {}).get("equity_min_efficiency_fraction", 0.60)
            )
            floors.append(Floor("total_population", "max", efficiency_reference * fraction))
        return floors

    def _time_limit(self, scenario: str, objective_name: str) -> float:
        table = self.cfg.get("solver", {}).get("time_limit_sec", {})
        scenario_table = table.get(scenario, {}) if isinstance(table, dict) else {}
        if objective_name in scenario_table:
            return float(scenario_table[objective_name])
        common = table.get("common", {}) if isinstance(table, dict) else {}
        if objective_name in common:
            return float(common[objective_name])
        return float(self.cfg.get("solver", {}).get("default_time_limit_sec", 1200))

    def _oracle_time_limit(self, scenario: str, objective_name: str) -> float:
        oracle = self.cfg.get("oracle", {})
        per_objective = oracle.get("time_limit_sec", {})
        key = f"{scenario}.{objective_name}"
        return float(per_objective.get(key, per_objective.get(objective_name, oracle.get("default_time_limit_sec", 900))))

    def _solve_options(
        self,
        *,
        scenario: str,
        objective_name: str,
        stage_index: int,
        time_limit: float,
        oracle: bool,
        log_path: Path,
    ) -> SolveOptions:
        section = self.cfg.get("solver", {})
        seed_base = int(section.get("random_seed", 42))
        random_seed = seed_base + stage_index * 1009 + sum(ord(c) for c in scenario + objective_name)
        heuristic = float(
            section.get("oracle_heuristic_effort", 0.02)
            if oracle
            else section.get("mip_heuristic_effort", 0.12)
        )
        if oracle:
            overrides = section.get("oracle_heuristic_effort_by_objective", {})
            if isinstance(overrides, dict):
                key = f"{scenario}.{objective_name}"
                heuristic = float(overrides.get(key, overrides.get(objective_name, heuristic)))
        return SolveOptions(
            time_limit_sec=float(time_limit),
            relative_gap=0.0 if oracle else self.gap,
            threads=self.highs_threads,
            random_seed=random_seed,
            parallel=self.highs_threads > 1,
            presolve=True,
            heuristic_effort=heuristic,
            mip_max_start_nodes=int(section.get("mip_max_start_nodes", 5000)),
            log_to_console=bool(section.get("log_to_console", False)),
            log_path=log_path,
            improving_solution_path=log_path.with_suffix(".improving.sol"),
            write_model_path=(
                log_path.with_suffix(".mps")
                if bool(section.get("write_mps", False))
                else None
            ),
            extra_options=dict(section.get("extra_highs_options", {})),
        )

    def _is_directly_certified(self, result: Any) -> bool:
        return bool(result.is_optimal or (result.relative_gap is not None and result.relative_gap <= self.gap + 1e-12))

    @staticmethod
    def _better(sense: str, left: float, right: float) -> bool:
        return left > right if sense == "max" else left < right

    def _authoritative_incumbent(
        self,
        objective_name: str,
        sense: str,
        model: Any,
        seed: Seed | None,
        result: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, str | None]:
        candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []
        if seed is not None and seed.feasible and seed.full_values is not None:
            value = self.f.objective_for_selection(objective_name, seed.selected_indices)
            candidates.append((value, seed.selected_indices, seed.full_values, seed.source))
        if result.solution is not None:
            selected = self.f.selected_from_solution(model, result.solution)
            valid, reason = self.f.validate_selection(selected)
            if valid:
                full = self.f.complete_solution(model, selected)
                violations = self.f.feasibility_violations(model, full)
                if bool(violations["feasible"]):
                    value = self.f.objective_for_selection(objective_name, selected)
                    candidates.append((value, selected, full, "HIGHS_INCUMBENT"))
        if not candidates:
            return None, None, None, None
        candidates.sort(key=lambda row: row[0], reverse=sense == "max")
        value, selected, full, source = candidates[0]
        return np.asarray(selected, dtype=int), np.asarray(full, dtype=float), float(value), source

    def _oracle_threshold(self, objective_name: str, sense: str, incumbent: float) -> Floor:
        if objective_name == "min_sigungu_coverage" and sense == "max":
            scale = int(getattr(self.f, "min_sigungu_scale", 1_000_000))
            incumbent_units = max(0, int(np.floor(float(incumbent) * scale + 1e-9)))
            # The authoritative Stage4.2 objective is fixed at 1e-6 units.
            # Proving the next >0.5%-better integer unit infeasible certifies the
            # frozen contract while the direct model keeps q continuous.
            raw_units = Decimal(incumbent_units) * (Decimal(1) + Decimal(str(self.gap)))
            threshold_units = int(raw_units.to_integral_value(rounding=ROUND_FLOOR)) + 1
            return Floor(objective_name, "max", threshold_units / scale)
        if sense == "max":
            # Build the decimal target using an accurately rounded sum, then
            # move one representable float upward. This avoids accepting an
            # equality at the exact 0.5% boundary because of multiplication
            # rounding (for example, 100 * 1.005 -> 100.49999999999999).
            raw = math.fsum([incumbent, abs(incumbent) * self.gap])
            target = float(math.nextafter(raw, math.inf))
            return Floor(objective_name, "max", target)
        raw = math.fsum([incumbent, -abs(incumbent) * self.gap])
        target = float(math.nextafter(raw, -math.inf))
        return Floor(objective_name, "min", target)

    def _run_oracle(
        self,
        *,
        scenario: str,
        candidate_set: str,
        stage_index: int,
        objective_name: str,
        sense: str,
        floors: list[Floor],
        selected: np.ndarray,
        incumbent: float,
        stage_dir: Path,
    ) -> tuple[bool, str, np.ndarray, float, int, list[dict[str, Any]]]:
        oracle_cfg = self.cfg.get("oracle", {})
        if not bool(oracle_cfg.get("enabled", True)):
            return False, "ORACLE_DISABLED", selected, incumbent, 0, []
        max_rounds = int(oracle_cfg.get("max_rounds", 4))
        history: list[dict[str, Any]] = []
        current_selected = np.asarray(selected, dtype=int)
        current_value = float(incumbent)

        for oracle_round in range(1, max_rounds + 1):
            threshold = self._oracle_threshold(objective_name, sense, current_value)
            guided_objective = bool(oracle_cfg.get("guided_objective_enabled", False))
            oracle_objective_name = objective_name if guided_objective else "feasibility"
            oracle_sense = sense if guided_objective else "min"
            oracle_model = self.f.build(
                objective_name=oracle_objective_name,
                sense=oracle_sense,
                floors=floors,
                threshold=threshold,
                name=f"{candidate_set}__{scenario}__stage{stage_index}__oracle{oracle_round}",
            )
            log_path = stage_dir / f"oracle_{oracle_round}.highs.log"
            options = self._solve_options(
                scenario=scenario,
                objective_name=objective_name,
                stage_index=stage_index + oracle_round * 100,
                time_limit=self._oracle_time_limit(scenario, objective_name),
                oracle=True,
                log_path=log_path,
            )
            if guided_objective and bool(oracle_cfg.get("stop_at_objective_target", True)):
                # The strict threshold is already a model row.  Giving HiGHS the
                # same target makes a feasible threshold plan a terminal search
                # event while retaining a valid infeasibility proof when none
                # exists.  This changes search direction only, not the feasible
                # set or the frozen 0.5% certification contract.
                options.extra_options["objective_target"] = float(threshold.value)
            # The current incumbent deliberately violates the stricter threshold, so
            # passing it as a start would be counterproductive. HiGHS searches for any
            # better plan or proves none exists.
            result = self.backend.solve(oracle_model, options, mip_start=None)
            record = {
                "round": oracle_round,
                "threshold": asdict(threshold),
                "status": result.status,
                "has_incumbent": result.has_incumbent,
                "is_infeasible": result.is_infeasible,
                "wall_time_sec": result.wall_time_sec,
                "mip_node_count": result.mip_node_count,
                "message": result.message,
                "peak_rss_mb": result.options.get("peak_rss_mb"),
                "oracle_objective_name": oracle_objective_name,
                "oracle_objective_sense": oracle_sense,
                "objective_target": options.extra_options.get("objective_target"),
                "heuristic_effort": options.heuristic_effort,
            }
            history.append(record)
            if result.is_infeasible:
                return True, "INFEASIBILITY_ORACLE", current_selected, current_value, oracle_round, history
            if result.solution is None:
                return False, "ORACLE_INCONCLUSIVE", current_selected, current_value, oracle_round, history
            proposed = self.f.selected_from_solution(oracle_model, result.solution)
            valid, reason = self.f.validate_selection(proposed)
            if not valid:
                record["rejected_solution"] = reason
                return False, "ORACLE_INVALID_INCUMBENT", current_selected, current_value, oracle_round, history
            proposed_value = self.f.objective_for_selection(objective_name, proposed)
            oracle_swap: dict[str, Any] | None = None
            if bool(self.cfg.get("seed_search", {}).get("single_swap_enabled", True)):
                improved_indices, improved_value, oracle_swap = self._local_improve_selection(
                    proposed,
                    objective_name=objective_name,
                    sense=sense,
                    floors=floors,
                )
                improved_full = self.f.complete_solution(oracle_model, improved_indices)
                oracle_violations = self.f.feasibility_violations(oracle_model, improved_full)
                oracle_swap["feasibility"] = oracle_violations
                if bool(oracle_violations["feasible"]) and self._better(
                    sense, improved_value, proposed_value
                ):
                    proposed = improved_indices
                    proposed_value = improved_value
            record["post_oracle_local_search"] = oracle_swap
            if not self._better(sense, proposed_value, current_value):
                record["rejected_solution"] = (
                    f"threshold solution did not improve: proposed={proposed_value}, current={current_value}"
                )
                return False, "ORACLE_NUMERICAL_INCONCLUSIVE", current_selected, current_value, oracle_round, history
            current_selected = proposed
            current_value = float(proposed_value)
        return False, "ORACLE_ROUND_LIMIT", current_selected, current_value, max_rounds, history

    def _retained_floor(
        self,
        scenario: str,
        objective_name: str,
        sense: str,
        retention: float | None,
        objective_value: float,
        greedy_population: float,
    ) -> Floor | None:
        if retention is None:
            return None
        if sense == "max":
            value = objective_value * retention
            if objective_name == "min_sigungu_coverage":
                scale = int(getattr(self.f, "min_sigungu_scale", 1_000_000))
                incumbent_units = int(np.floor(float(objective_value) * scale + 1e-9))
                value = int(np.ceil(incumbent_units * retention - 1e-12)) / scale
            if scenario == "efficiency" and objective_name == "total_population":
                value = max(value, greedy_population)
            return Floor(objective_name, "max", float(value))
        return Floor(objective_name, "min", float(objective_value / max(retention, 1e-12)))

    def solve_scenario(
        self,
        scenario: str,
        *,
        efficiency_reference: float | None,
        greedy_population: float,
    ) -> ScenarioCertificationResult:
        candidate_set = self.f.candidate_set
        scenario_dir = self.output_dir / candidate_set / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        floors = self._initial_floors(scenario, efficiency_reference, greedy_population)
        stages: list[StageResult] = []
        previous_selected: np.ndarray | None = None
        final_selected: np.ndarray | None = None

        for stage_index, (objective_name, sense, retention) in enumerate(self._scenario_plan(scenario), start=1):
            stage_dir = scenario_dir / f"stage_{stage_index:02d}_{objective_name}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            model = self.f.build(
                objective_name=objective_name,
                sense=sense,  # type: ignore[arg-type]
                floors=floors,
                name=f"{candidate_set}__{scenario}__stage{stage_index}__{objective_name}",
            )
            raw_seeds = discover_seed_selections(
                self.project_root,
                self.f,
                self.stage41,
                scenario=scenario,
                candidate_set=candidate_set,
                greedy_indices=self.greedy_indices,
                previous_stage_indices=previous_selected,
            )
            seeds = evaluate_seed_pool(self.f, model, raw_seeds)
            seed = choose_seed(seeds, sense)

            local_details: dict[str, Any] = {"pre_solve": {"attempted": False}}
            if seed is not None and bool(self.cfg.get("seed_search", {}).get("single_swap_enabled", True)):
                improved_indices, improved_value, pre_details = self._local_improve_selection(
                    seed.selected_indices,
                    objective_name=objective_name,
                    sense=sense,
                    floors=floors,
                )
                improved_full = self.f.complete_solution(model, improved_indices)
                violations = self.f.feasibility_violations(model, improved_full)
                pre_details["feasibility"] = violations
                local_details["pre_solve"] = pre_details
                if bool(violations["feasible"]) and (
                    seed.objective_value is None
                    or self._better(sense, improved_value, float(seed.objective_value))
                ):
                    seed = Seed(
                        source=f"{seed.source}+{self.improver.backend}_SINGLE_SWAP",
                        selected_indices=improved_indices,
                        full_values=improved_full,
                        feasible=True,
                        objective_value=improved_value,
                        metrics=self.f.metrics(improved_indices),
                    )

            options = self._solve_options(
                scenario=scenario,
                objective_name=objective_name,
                stage_index=stage_index,
                time_limit=self._time_limit(scenario, objective_name),
                oracle=False,
                log_path=stage_dir / "direct.highs.log",
            )
            direct = self.backend.solve(model, options, mip_start=seed.full_values if seed else None)
            selected, full, incumbent, incumbent_source = self._authoritative_incumbent(
                objective_name, sense, model, seed, direct
            )
            if (
                selected is not None
                and full is not None
                and incumbent is not None
                and bool(self.cfg.get("seed_search", {}).get("single_swap_enabled", True))
            ):
                post_indices, post_value, post_details = self._local_improve_selection(
                    selected,
                    objective_name=objective_name,
                    sense=sense,
                    floors=floors,
                )
                post_full = self.f.complete_solution(model, post_indices)
                post_violations = self.f.feasibility_violations(model, post_full)
                post_details["feasibility"] = post_violations
                local_details["post_solve"] = post_details
                if bool(post_violations["feasible"]) and self._better(sense, post_value, incumbent):
                    selected = post_indices
                    full = post_full
                    incumbent = post_value
                    incumbent_source = f"{incumbent_source or 'INCUMBENT'}+{self.improver.backend}_POST_SWAP"
            if selected is None or full is None or incumbent is None:
                stages.append(
                    StageResult(
                        scenario=scenario,
                        candidate_set=candidate_set,
                        stage_index=stage_index,
                        objective_name=objective_name,
                        sense=sense,  # type: ignore[arg-type]
                        status=direct.status,
                        certificate="NO_INCUMBENT",
                        certified=False,
                        objective_value=None,
                        best_bound=direct.best_bound,
                        relative_gap=direct.relative_gap,
                        wall_time_sec=direct.wall_time_sec,
                        mip_node_count=direct.mip_node_count,
                        selected_indices=None,
                        solution=None,
                        retained_floor=None,
                        seed_source=seed.source if seed else None,
                        oracle_rounds=0,
                        model_rows=model.n_row,
                        model_cols=model.n_col,
                        model_nnz=int(model.A.nnz),
                        backend_message=direct.message,
                        details={
                            "seed_pool": [
                                {
                                    "source": s.source,
                                    "feasible": s.feasible,
                                    "objective_value": s.objective_value,
                                    "rejection_reason": s.rejection_reason,
                                }
                                for s in seeds
                            ],
                            "local_search": local_details,
                            "model": {
                                key: value
                                for key, value in model.metadata.items()
                                if key != "row_names"
                            },
                        },
                    )
                )
                break

            authoritative_direct_gap = self._gap_from_bound(sense, incumbent, direct.best_bound)
            certified = bool(
                direct.is_optimal
                or (
                    authoritative_direct_gap is not None
                    and authoritative_direct_gap <= self.gap + 1e-12
                )
            )
            certificate = "OPTIMAL" if direct.is_optimal else (
                "DIRECT_MIP_GAP" if certified else "DIRECT_UNCERTIFIED"
            )
            oracle_rounds = 0
            oracle_history: list[dict[str, Any]] = []
            if not certified:
                (
                    certified,
                    oracle_certificate,
                    selected,
                    incumbent,
                    oracle_rounds,
                    oracle_history,
                ) = self._run_oracle(
                    scenario=scenario,
                    candidate_set=candidate_set,
                    stage_index=stage_index,
                    objective_name=objective_name,
                    sense=sense,
                    floors=floors,
                    selected=selected,
                    incumbent=incumbent,
                    stage_dir=stage_dir,
                )
                certificate = oracle_certificate if certified else f"{certificate}+{oracle_certificate}"
                full = self.f.complete_solution(model, selected)

            retained = self._retained_floor(
                scenario,
                objective_name,
                sense,
                retention,
                incumbent,
                greedy_population,
            )
            if retained is not None:
                floors.append(retained)
            stage_gap = authoritative_direct_gap
            if certified and certificate == "INFEASIBILITY_ORACLE":
                stage_gap = self.gap
            stage = StageResult(
                scenario=scenario,
                candidate_set=candidate_set,
                stage_index=stage_index,
                objective_name=objective_name,
                sense=sense,  # type: ignore[arg-type]
                status=direct.status,
                certificate=certificate,
                certified=bool(certified),
                objective_value=float(incumbent),
                best_bound=direct.best_bound,
                relative_gap=stage_gap,
                wall_time_sec=float(direct.wall_time_sec + sum(x["wall_time_sec"] for x in oracle_history)),
                mip_node_count=direct.mip_node_count,
                selected_indices=np.asarray(selected, dtype=int),
                solution=np.asarray(full, dtype=float),
                retained_floor=retained,
                seed_source=incumbent_source,
                oracle_rounds=oracle_rounds,
                model_rows=model.n_row,
                model_cols=model.n_col,
                model_nnz=int(model.A.nnz),
                backend_message=direct.message,
                details={
                    "direct": {
                        "status": direct.status,
                        "objective_value": direct.objective_value,
                        "best_bound": direct.best_bound,
                        "relative_gap": direct.relative_gap,
                        "authoritative_gap_after_seed_improvement": authoritative_direct_gap,
                        "wall_time_sec": direct.wall_time_sec,
                        "mip_node_count": direct.mip_node_count,
                        "options": direct.options,
                    },
                    "oracle_history": oracle_history,
                    "seed_pool": [
                        {
                            "source": s.source,
                            "feasible": s.feasible,
                            "objective_value": s.objective_value,
                            "rejection_reason": s.rejection_reason,
                        }
                        for s in seeds
                    ],
                    "local_search": local_details,
                    "model": {
                        key: value
                        for key, value in model.metadata.items()
                        if key != "row_names"
                    },
                    "floors_before_stage": [asdict(f) for f in floors[:-1] if retained is not None]
                    if retained is not None
                    else [asdict(f) for f in floors],
                },
            )
            stages.append(stage)
            previous_selected = np.asarray(selected, dtype=int)
            final_selected = previous_selected
            if not certified and bool(self.cfg.get("gate", {}).get("stop_scenario_on_uncertified_stage", True)):
                break

        all_stages_present = len(stages) == len(self._scenario_plan(scenario))
        all_certified = all_stages_present and all(stage.certified for stage in stages)
        if final_selected is None:
            final_selected = self.greedy_indices.copy()
        metrics = self.f.metrics(final_selected)
        selected_frame = self.f.candidates.iloc[final_selected].copy().reset_index(drop=True)
        selected_frame["candidate_set"] = candidate_set
        selected_frame["scenario"] = scenario
        selected_frame = selected_frame[[
            "scenario",
            "candidate_set",
            *[column for column in selected_frame.columns if column not in {"scenario", "candidate_set"}],
        ]]
        max_gap = max(
            (stage.relative_gap for stage in stages if stage.relative_gap is not None),
            default=None,
        )
        certification_class = (
            "OPTIMAL"
            if all_certified and all(stage.certificate == "OPTIMAL" for stage in stages)
            else "CERTIFIED_NEAR_OPTIMAL"
            if all_certified
            else "FEASIBLE_UNCERTIFIED"
            if final_selected.size == self.f.visit_count
            else "INCONCLUSIVE"
        )
        metrics.update(
            {
                "scenario": scenario,
                "candidate_set": candidate_set,
                "certified": all_certified,
                "certification_class": certification_class,
                "max_relative_gap": max_gap,
                "gpu_seed_backend": self.improver.backend,
            }
        )
        return ScenarioCertificationResult(
            scenario=scenario,
            candidate_set=candidate_set,
            certified=all_certified,
            certification_class=certification_class,
            selected_indices=final_selected,
            selected_venue_ids=selected_frame["venue_id"].astype(str).tolist(),
            selected_frame=selected_frame,
            metrics=metrics,
            stages=stages,
            max_relative_gap=max_gap,
        )
