from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .types import BackendResult, LinearMipModel


@dataclass
class SolveOptions:
    time_limit_sec: float
    relative_gap: float
    threads: int = 1
    random_seed: int = 42
    parallel: bool = True
    presolve: bool = True
    heuristic_effort: float = 0.05
    mip_max_start_nodes: int = 5000
    log_to_console: bool = False
    log_path: Path | None = None
    improving_solution_path: Path | None = None
    write_model_path: Path | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return ()
    return tuple(int(v or 0) for v in match.groups())


class _MemoryMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.peak_rss_mb: float | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_MemoryMonitor":
        try:
            import psutil  # type: ignore

            process = psutil.Process()
        except Exception:
            return self

        def run() -> None:
            peak = 0
            while not self.stop_event.wait(0.2):
                try:
                    rss = process.memory_info().rss
                    for child in process.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                    peak = max(peak, rss)
                except Exception:
                    pass
            self.peak_rss_mb = peak / (1024.0 * 1024.0) if peak else None

        self._thread = threading.Thread(target=run, name="s42c-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class HighsBackend:
    """Direct HiGHS backend with MIP-start support.

    The preferred binding is the public ``highspy`` package.  Recent SciPy wheels
    also bundle the same pybind11 core under ``scipy.optimize._highspy``; that
    binding is used as a compatibility fallback so the certification runner does
    not fail merely because a separate highspy wheel is absent.  The fallback is
    deliberately isolated here because it is a private SciPy API.
    """

    minimum_version = (1, 8, 0)

    def __init__(self, *, require_minimum_version: bool = True) -> None:
        binding_source = "public_highspy"
        try:
            import highspy as highs_module  # type: ignore

            highs_class = highs_module.Highs
        except ImportError:
            try:
                from scipy.optimize._highspy import _core as highs_module  # type: ignore

                highs_class = highs_module._Highs
                binding_source = "scipy_bundled_private_highspy"
            except Exception as exc:
                raise RuntimeError(
                    "A direct HiGHS Python binding is required. Install highspy>=1.8, "
                    "or use a SciPy wheel that bundles scipy.optimize._highspy."
                ) from exc
        self.highspy = highs_module
        self.HighsClass = highs_class
        self.binding_source = binding_source
        self.version = self._get_version()
        if require_minimum_version and _version_tuple(self.version) < self.minimum_version:
            raise RuntimeError(
                f"Stage4.2C requires HiGHS>=1.8.0; found {self.version} via {binding_source}"
            )

    def _get_version(self) -> str:
        hs = self.highspy
        values = [
            getattr(hs, "HIGHS_VERSION_MAJOR", None),
            getattr(hs, "HIGHS_VERSION_MINOR", None),
            getattr(hs, "HIGHS_VERSION_PATCH", None),
        ]
        if all(v is not None for v in values):
            return ".".join(str(int(v)) for v in values)
        try:
            h = self.HighsClass()
            version = h.version()
            return str(version)
        except Exception:
            return str(getattr(hs, "__version__", "unknown"))

    def _pass_model(self, h: Any, model: LinearMipModel) -> None:
        hs = self.highspy
        A = model.A.tocsc(copy=False)
        lp = hs.HighsLp()
        lp.num_col_ = int(model.n_col)
        lp.num_row_ = int(model.n_row)
        lp.col_cost_ = np.asarray(model.c, dtype=np.float64)
        lp.col_lower_ = np.asarray(model.col_lower, dtype=np.float64)
        lp.col_upper_ = np.asarray(model.col_upper, dtype=np.float64)
        lp.row_lower_ = np.asarray(model.row_lower, dtype=np.float64)
        lp.row_upper_ = np.asarray(model.row_upper, dtype=np.float64)
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = int(model.n_col)
        lp.a_matrix_.num_row_ = int(model.n_row)
        lp.a_matrix_.start_ = np.asarray(A.indptr, dtype=np.int32)
        lp.a_matrix_.index_ = np.asarray(A.indices, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(A.data, dtype=np.float64)
        lp.sense_ = hs.ObjSense.kMaximize if model.sense == "max" else hs.ObjSense.kMinimize
        lp.integrality_ = [
            hs.HighsVarType.kInteger if int(value) else hs.HighsVarType.kContinuous
            for value in model.integrality
        ]
        status = h.passModel(lp)
        if "error" in str(status).lower():
            raise RuntimeError(f"HiGHS passModel failed: {status}")

    def _set_option(
        self,
        h: Any,
        name: str,
        value: Any,
        *,
        required: bool,
        skipped: dict[str, str],
    ) -> None:
        try:
            status = h.setOptionValue(name, value)
        except Exception as exc:
            if required:
                raise RuntimeError(f"Could not set required HiGHS option {name}={value!r}") from exc
            skipped[name] = f"exception:{type(exc).__name__}:{exc}"
            return
        if "error" in str(status).lower():
            if required:
                raise RuntimeError(f"HiGHS rejected required option {name}={value!r}: {status}")
            skipped[name] = str(status)

    def _set_start(self, h: Any, start: np.ndarray) -> str:
        hs = self.highspy
        values = np.asarray(start, dtype=np.float64)
        try:
            solution = hs.HighsSolution()
            solution.col_value = values
            if hasattr(solution, "value_valid"):
                solution.value_valid = True
            status = h.setSolution(solution)
            if "error" not in str(status).lower():
                return "FULL_HIGHS_SOLUTION"
        except Exception:
            pass
        # Newer highspy also exposes sparse setSolution(num_entries, indices, values).
        indices = np.flatnonzero(np.isfinite(values)).astype(np.int32)
        status = h.setSolution(int(indices.size), indices, values[indices])
        if "error" in str(status).lower():
            raise RuntimeError(f"HiGHS rejected MIP start: {status}")
        return "SPARSE_SET_SOLUTION"

    def solve(
        self,
        model: LinearMipModel,
        options: SolveOptions,
        *,
        mip_start: np.ndarray | None = None,
    ) -> BackendResult:
        hs = self.highspy
        model.validate()
        # HiGHS owns a process-global scheduler. SciPy's milp wrapper may have
        # initialized it with a different thread count earlier in the process;
        # reset before constructing a direct solver instance to avoid the
        # "scheduler already initialized with N threads" failure.
        try:
            self.HighsClass.resetGlobalScheduler(True)
        except Exception:
            pass
        h = self.HighsClass()
        skipped_options: dict[str, str] = {}
        self._set_option(h, "output_flag", True, required=True, skipped=skipped_options)
        self._set_option(h, "log_to_console", bool(options.log_to_console), required=True, skipped=skipped_options)
        if options.log_path is not None:
            options.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._set_option(h, "log_file", str(options.log_path), required=False, skipped=skipped_options)
        for name, value in [
            ("presolve", "on" if options.presolve else "off"),
            ("parallel", "on" if options.parallel and options.threads > 1 else "off"),
            ("threads", int(options.threads)),
            ("time_limit", float(options.time_limit_sec)),
            ("mip_rel_gap", float(options.relative_gap)),
            ("mip_abs_gap", 0.0),
            ("random_seed", int(options.random_seed)),
        ]:
            self._set_option(h, name, value, required=True, skipped=skipped_options)
        optional_options = {
            "mip_detect_symmetry": True,
            "mip_allow_restart": True,
            "mip_heuristic_effort": float(options.heuristic_effort),
            "mip_max_start_nodes": int(options.mip_max_start_nodes),
            "mip_min_logging_interval": 5.0,
        }
        # These controls were added after the HiGHS 1.8 line bundled by some
        # SciPy wheels. Avoid emitting unknown-option errors on older bindings.
        if _version_tuple(self.version) >= (1, 9, 0):
            optional_options.update(
                {
                    "mip_allow_cut_separation_at_nodes": True,
                    "mip_heuristic_run_feasibility_jump": True,
                    "mip_heuristic_run_rins": True,
                    "mip_heuristic_run_rens": True,
                    "mip_heuristic_run_root_reduced_cost": True,
                    "mip_heuristic_run_zi_round": True,
                    "mip_heuristic_run_shifting": True,
                }
            )
        for name, value in optional_options.items():
            self._set_option(h, name, value, required=False, skipped=skipped_options)
        if options.improving_solution_path is not None:
            options.improving_solution_path.parent.mkdir(parents=True, exist_ok=True)
            for name, value in [
                ("mip_improving_solution_save", True),
                ("mip_improving_solution_report_sparse", True),
                ("mip_improving_solution_file", str(options.improving_solution_path)),
            ]:
                self._set_option(h, name, value, required=False, skipped=skipped_options)
        for key, value in options.extra_options.items():
            self._set_option(h, str(key), value, required=False, skipped=skipped_options)

        self._pass_model(h, model)
        start_mode: str | None = None
        if mip_start is not None:
            start_mode = self._set_start(h, mip_start)
        if options.write_model_path is not None:
            options.write_model_path.parent.mkdir(parents=True, exist_ok=True)
            h.writeModel(str(options.write_model_path))

        started = time.perf_counter()
        with _MemoryMonitor() as monitor:
            run_status = h.run()
        wall = time.perf_counter() - started
        if "error" in str(run_status).lower():
            raise RuntimeError(f"HiGHS run failed: {run_status}")
        info = h.getInfo()
        model_status_enum = h.getModelStatus()
        status = str(h.modelStatusToString(model_status_enum)).upper().replace(" ", "_")
        is_infeasible = "INFEASIBLE" in status and "UNBOUNDED" not in status
        solver_reports_optimal = status == "OPTIMAL"

        primal_status_raw = getattr(info, "primal_solution_status", None)
        try:
            primal_status_text = str(h.solutionStatusToString(primal_status_raw)).upper()
        except Exception:
            primal_status_text = str(primal_status_raw).upper()
        primal_feasible = (
            "FEASIBLE" in primal_status_text and "INFEASIBLE" not in primal_status_text
        )

        objective_raw_candidate = getattr(info, "objective_function_value", None)
        objective_raw_candidate = (
            float(objective_raw_candidate)
            if objective_raw_candidate is not None
            and math.isfinite(float(objective_raw_candidate))
            else None
        )
        dual_bound_raw = getattr(info, "mip_dual_bound", None)
        dual_bound_raw = (
            float(dual_bound_raw)
            if dual_bound_raw is not None and math.isfinite(float(dual_bound_raw))
            else None
        )
        gap = getattr(info, "mip_gap", None)
        gap = float(gap) if gap is not None and math.isfinite(float(gap)) else None
        # HiGHS reports model status OPTIMAL when it stops at the configured MIP
        # gap tolerance.  Preserve that raw status, but reserve the exact-optimum
        # flag for a numerically closed bound so reports do not overstate a
        # 0.5%-certified result as an exact proof.
        is_optimal = bool(
            solver_reports_optimal and gap is not None and gap <= 1e-9
        )
        node_count = getattr(info, "mip_node_count", None)
        node_count = int(node_count) if node_count is not None else None
        solution_values: np.ndarray | None = None
        solution_value_valid = False
        try:
            highs_solution = h.getSolution()
            solution_value_valid = bool(getattr(highs_solution, "value_valid", False))
            values = np.asarray(list(highs_solution.col_value), dtype=np.float64)
            if (
                values.size == model.n_col
                and np.all(np.isfinite(values))
                and (solution_value_valid or primal_feasible or is_optimal)
            ):
                solution_values = values
        except Exception:
            solution_values = None
        has_incumbent = solution_values is not None and not is_infeasible
        objective_raw = objective_raw_candidate if has_incumbent else None
        message = (
            f"run_status={run_status}; model_status={status}; "
            f"primal_status={primal_status_text}; value_valid={solution_value_valid}; "
            f"start={start_mode or 'NONE'}"
        )
        details = {
            "threads": int(options.threads),
            "parallel": bool(options.parallel),
            "random_seed": int(options.random_seed),
            "heuristic_effort": float(options.heuristic_effort),
            "start_mode": start_mode,
            "peak_rss_mb": monitor.peak_rss_mb,
            "binding_source": self.binding_source,
            "skipped_options": skipped_options,
        }
        backend_result = BackendResult(
            model_name=model.name,
            status=status,
            has_incumbent=has_incumbent,
            is_infeasible=is_infeasible,
            is_optimal=is_optimal,
            objective_value_raw=objective_raw,
            objective_value=model.reported_objective(objective_raw),
            best_bound_raw=dual_bound_raw,
            best_bound=model.reported_objective(dual_bound_raw),
            relative_gap=gap,
            mip_node_count=node_count,
            wall_time_sec=float(wall),
            solution=solution_values,
            message=message,
            highs_version=self.version,
            options=details,
            log_path=options.log_path,
        )
        try:
            self.HighsClass.resetGlobalScheduler(True)
        except Exception:
            pass
        return backend_result


class ScipyBackend:
    """Small-problem fallback used by unit tests; it has no MIP-start support."""

    def solve(
        self,
        model: LinearMipModel,
        options: SolveOptions,
        *,
        mip_start: np.ndarray | None = None,
    ) -> BackendResult:
        model.validate()
        c = -model.c if model.sense == "max" else model.c
        started = time.perf_counter()
        result = milp(
            c,
            integrality=model.integrality,
            bounds=Bounds(model.col_lower, model.col_upper),
            constraints=LinearConstraint(model.A, model.row_lower, model.row_upper),
            options={
                "time_limit": float(options.time_limit_sec),
                "mip_rel_gap": float(options.relative_gap),
                "presolve": bool(options.presolve),
                "disp": False,
            },
        )
        wall = time.perf_counter() - started
        status_map = {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "ERROR"}
        status = status_map.get(int(result.status), f"STATUS_{result.status}")
        raw = getattr(result, "fun", None)
        raw = float(raw) if raw is not None and math.isfinite(float(raw)) else None
        if raw is not None and model.sense == "max":
            raw = -raw
        dual = getattr(result, "mip_dual_bound", None)
        dual = float(dual) if dual is not None and math.isfinite(float(dual)) else None
        if dual is not None and model.sense == "max":
            dual = -dual
        x = getattr(result, "x", None)
        solution = np.asarray(x, dtype=float) if x is not None else None
        gap = getattr(result, "mip_gap", None)
        gap = float(gap) if gap is not None and math.isfinite(float(gap)) else None
        return BackendResult(
            model_name=model.name,
            status=status,
            has_incumbent=solution is not None,
            is_infeasible=status == "INFEASIBLE",
            is_optimal=status == "OPTIMAL",
            objective_value_raw=raw,
            objective_value=model.reported_objective(raw),
            best_bound_raw=dual,
            best_bound=model.reported_objective(dual),
            relative_gap=gap,
            mip_node_count=int(getattr(result, "mip_node_count", 0) or 0),
            wall_time_sec=float(wall),
            solution=solution,
            message=str(result.message),
            highs_version="scipy-milp",
            options={"mip_start_ignored": mip_start is not None},
            log_path=options.log_path,
        )
