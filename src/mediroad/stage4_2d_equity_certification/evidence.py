from __future__ import annotations

import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mediroad.stage4_2_certification.formulation import CertificationFormulation
from mediroad.stage4_2_certification.seeds import discover_seed_selections

from .io_utils import sha256_file
from .types import EvidenceSeed, FrozenEquityContract


_EXPECTED_STAGE_PLAN: dict[str, list[tuple[str, str]]] = {
    "efficiency": [
        ("total_population", "max"),
        ("need_weighted", "max"),
        ("min_sigungu_coverage", "max"),
        ("cost", "min"),
    ],
    "equity": [
        ("min_sigungu_coverage", "max"),
        ("high_need_population", "max"),
        ("need_weighted", "max"),
        ("cost", "min"),
    ],
}


def _venue_column(frame: pd.DataFrame) -> str:
    for name in ("venue_id", "selected_venue_id", "candidate_venue_id"):
        if name in frame.columns:
            return name
    raise ValueError(f"Could not find venue ID column; columns={list(frame.columns)}")


def _read_plan_indices(path: Path, formulation: CertificationFormulation) -> np.ndarray | None:
    try:
        frame = pd.read_csv(path)
        column = _venue_column(frame)
    except Exception:
        return None
    lookup = {str(value): int(i) for i, value in enumerate(formulation.candidates["venue_id"].astype(str))}
    aliases = dict(formulation.venue_alias_map)
    selected: list[int] = []
    for raw in frame[column].astype(str):
        value = aliases.get(raw, raw)
        idx = lookup.get(value)
        if idx is None:
            return None
        selected.append(idx)
    indices = np.asarray(selected, dtype=int)
    valid, _ = formulation.validate_selection(indices)
    return indices if valid else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = _read_json(path)
    except Exception:
        return []
    return value if isinstance(value, list) else []


def _run_id(run_dir: Path) -> str:
    return run_dir.name


def _candidate_run_dirs(project_root: Path) -> list[Path]:
    patterns = [
        project_root / "outputs/model_v1/10_stage4_2_certification/runs/*/candidate_sets/top3",
        project_root / "outputs/model_v1/10_stage4_2_certification_diagnostic/runs/*/candidate_sets/top3",
        project_root / "outputs/model_v1/10_stage4_2_diagnostic/runs/*/candidate_sets/top3",
        project_root / "outputs/model_v1/10_stage4_2/runs/*/candidate_sets/top3",
    ]
    result: list[Path] = []
    for pattern in patterns:
        result.extend(Path(raw) for raw in glob.glob(str(pattern)))
    return sorted({path.resolve() for path in result}, key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)


def _scenario_certified(
    rows: list[dict[str, Any]],
    scenario: str,
    formulation: CertificationFormulation,
) -> bool:
    """Fail closed on the exact four-stage Top3 certification contract."""

    expected = _EXPECTED_STAGE_PLAN.get(scenario)
    if expected is None or len(rows) != len(expected):
        return False
    if any(type(row.get("stage_index")) is not int for row in rows):
        return False
    ordered = sorted(rows, key=lambda row: row["stage_index"])
    for stage_index, (row, (objective, sense)) in enumerate(zip(ordered, expected), start=1):
        if row.get("stage_index") != stage_index:
            return False
        if str(row.get("scenario", "")) != scenario:
            return False
        if str(row.get("candidate_set", "")) != "top3":
            return False
        if str(row.get("objective_name", "")) != objective or str(row.get("sense", "")) != sense:
            return False
        if row.get("certified") is not True:
            return False
        raw_numbers = [
            row.get("objective_value"),
            row.get("best_bound"),
            row.get("relative_gap"),
        ]
        if any(type(value) not in (int, float) for value in raw_numbers):
            return False
        incumbent, bound, gap = map(float, raw_numbers)
        if not all(math.isfinite(value) for value in (incumbent, bound, gap)):
            return False
        if gap < -1e-12 or gap > 0.005 + 1e-12:
            return False
        tolerance = 1e-8 * max(1.0, abs(incumbent), abs(bound))
        if sense == "max" and bound < incumbent - tolerance:
            return False
        if sense == "min" and bound > incumbent + tolerance:
            return False
        selected = _indices_from_stage_row(row, formulation)
        if selected is None:
            return False
        recomputed_objective = float(formulation.objective_for_selection(objective, selected))
        if abs(recomputed_objective - incumbent) > tolerance:
            return False
        denominator = max(abs(incumbent), 1e-12)
        recomputed_gap = max(
            0.0,
            (bound - incumbent) / denominator
            if sense == "max"
            else (incumbent - bound) / denominator,
        )
        if recomputed_gap > 0.005 + 1e-12:
            return False
        if abs(recomputed_gap - gap) > 1e-10 * max(1.0, abs(recomputed_gap), abs(gap)):
            return False
    return True


def _indices_from_stage_row(
    row: dict[str, Any],
    formulation: CertificationFormulation,
) -> np.ndarray | None:
    raw = row.get("selected_indices")
    if not isinstance(raw, list) or len(raw) != formulation.visit_count:
        return None
    if any(type(value) is not int or value < 0 or value >= formulation.nx for value in raw):
        return None
    values = np.asarray(raw, dtype=int)
    if values.size != formulation.visit_count:
        return None
    valid, _ = formulation.validate_selection(values)
    return values if valid else None


def _ordering_verified(
    final_stage_rows: list[dict[str, Any]],
    final_plan: np.ndarray | None,
    formulation: CertificationFormulation,
) -> bool:
    if final_plan is None or not final_stage_rows:
        return False
    ordered = sorted(final_stage_rows, key=lambda row: row.get("stage_index", -1))
    if [row.get("stage_index") for row in ordered] != [1, 2, 3, 4]:
        return False
    final_indices = _indices_from_stage_row(ordered[-1], formulation)
    if final_indices is None:
        return False
    return set(map(int, final_indices)) == set(map(int, final_plan))


def _collect_run_payload(run_candidate_dir: Path, formulation: CertificationFormulation) -> dict[str, Any] | None:
    efficiency_plan_path = run_candidate_dir / "plan__efficiency.csv"
    equity_plan_path = run_candidate_dir / "plan__equity.csv"
    efficiency_stages_path = run_candidate_dir / "stages__efficiency.json"
    equity_stages_path = run_candidate_dir / "stages__equity.json"
    efficiency_plan = _read_plan_indices(efficiency_plan_path, formulation) if efficiency_plan_path.exists() else None
    equity_plan = _read_plan_indices(equity_plan_path, formulation) if equity_plan_path.exists() else None
    efficiency_rows = _stage_rows(efficiency_stages_path)
    equity_rows = _stage_rows(equity_stages_path)
    if efficiency_plan is None and equity_plan is None:
        return None
    run_dir = run_candidate_dir.parents[1]
    payload: dict[str, Any] = {
        "run_id": _run_id(run_dir),
        "candidate_dir": run_candidate_dir,
        "efficiency_plan": efficiency_plan,
        "equity_plan": equity_plan,
        "efficiency_rows": efficiency_rows,
        "equity_rows": equity_rows,
        "efficiency_certified": _scenario_certified(efficiency_rows, "efficiency", formulation),
        "equity_certified": _scenario_certified(equity_rows, "equity", formulation),
        "efficiency_ordering_verified": _ordering_verified(
            efficiency_rows, efficiency_plan, formulation
        ),
        "ordering_verified": _ordering_verified(equity_rows, equity_plan, formulation),
    }
    if efficiency_plan is not None:
        payload["efficiency_metrics"] = formulation.metrics(efficiency_plan)
    if equity_plan is not None:
        payload["equity_metrics"] = formulation.metrics(equity_plan)
    if payload["ordering_verified"]:
        for row in equity_rows:
            objective = str(row.get("objective_name", ""))
            selected = _indices_from_stage_row(row, formulation)
            if selected is not None:
                payload[f"equity_stage_selection::{objective}"] = selected
                payload[f"equity_stage_value::{objective}"] = formulation.objective_for_selection(objective, selected)
    return payload


def _select_source_run(
    payloads: list[dict[str, Any]],
    preferred_run_id: str | None,
) -> dict[str, Any]:
    if preferred_run_id:
        for payload in payloads:
            if (
                payload["run_id"] == preferred_run_id
                and payload.get("efficiency_plan") is not None
                and bool(payload.get("efficiency_certified"))
                and bool(payload.get("efficiency_ordering_verified"))
            ):
                return payload
        raise RuntimeError(
            f"The frozen preferred Stage4.2C run {preferred_run_id!r} was not found with its exact "
            "solver-certified Top3 Efficiency contract. Stage4.2D will not silently select another run."
        )
    # Prefer a run with solver-certified Efficiency and a valid Equity stage trace.
    ranked = sorted(
        payloads,
        key=lambda p: (
            bool(p.get("efficiency_certified")),
            bool(p.get("ordering_verified")),
            p.get("efficiency_plan") is not None,
            p["candidate_dir"].stat().st_mtime,
        ),
        reverse=True,
    )
    for payload in ranked:
        if (
            payload.get("efficiency_plan") is not None
            and bool(payload.get("efficiency_certified"))
            and bool(payload.get("efficiency_ordering_verified"))
        ):
            return payload
    raise RuntimeError(
        "No prior Top3 Stage4.2C run with a valid solver-certified Efficiency plan was found. "
        "Stage4.2D refuses to invent the Equity efficiency reference."
    )


def retained_min_sigungu_floor(
    objective_value: float,
    retention: float,
    *,
    scale: int = 1_000_000,
) -> float:
    """Reproduce Stage4.2C's integerized min-sigungu retention floor exactly."""

    if scale <= 0:
        raise ValueError("min-sigungu scale must be positive")
    if not 0.0 < float(retention) <= 1.0:
        raise ValueError("min-sigungu retention must be in (0,1]")
    incumbent_units = int(math.floor(float(objective_value) * int(scale) + 1e-9))
    retained_units = int(math.ceil(incumbent_units * float(retention) - 1e-12))
    return retained_units / float(scale)


def _verify_inventory(project_root: Path, inventory_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(inventory_path, dtype={"relative_path": str, "sha256": str})
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(frame.columns) or frame.empty:
        raise RuntimeError(f"Invalid evidence inventory schema: {inventory_path}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate paths in evidence inventory: {inventory_path}")
    for row in frame.to_dict("records"):
        relative = Path(str(row["relative_path"]))
        candidates = (project_root / relative, inventory_path.parent / relative)
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise RuntimeError(f"Evidence artifact missing: {relative}")
        if not any(
            int(path.stat().st_size) == int(row["size_bytes"])
            and sha256_file(path) == str(row["sha256"]).lower()
            for path in existing
        ):
            raise RuntimeError(f"Evidence artifact hash/size mismatch: {relative}")
    return {
        "relative_path": inventory_path.resolve().relative_to(project_root.resolve()).as_posix(),
        "sha256": sha256_file(inventory_path),
        "row_count": int(len(frame)),
    }


def _verify_preferred_artifacts(
    project_root: Path,
    source: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = config.get("evidence", {}).get("preferred_artifact_sha256", {})
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("evidence.preferred_artifact_sha256 must pin the frozen source run")
    run_dir = Path(source["candidate_dir"]).parents[1]
    records: list[dict[str, Any]] = []
    for relative, raw_hash in sorted(expected.items()):
        path = run_dir / str(relative)
        expected_hash = str(raw_hash).lower()
        if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
            raise RuntimeError(f"Invalid preferred artifact SHA-256 for {relative}")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Frozen preferred evidence mismatch: {path}")
        records.append(
            {
                "relative_path": path.resolve().relative_to(project_root.resolve()).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": expected_hash,
            }
        )
    inventory_path = run_dir / "ARTIFACT_INVENTORY.csv"
    if inventory_path.is_file():
        _verify_inventory(project_root, inventory_path)
    return records


def _highs_improving_selections(
    path: Path,
    formulation: CertificationFormulation,
) -> list[tuple[float, np.ndarray]]:
    """Parse complete public-HiGHS improving-solution blocks and revalidate binary x."""

    blocks: list[tuple[float, dict[int, float]]] = []
    objective: float | None = None
    values: dict[int, float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        parts = raw_line.split()
        if len(parts) == 2 and parts[0] == "Objective":
            if objective is not None:
                blocks.append((objective, values))
            objective = float(parts[1])
            values = {}
        elif len(parts) == 3 and parts[0] == "NoName" and objective is not None:
            values[int(parts[2])] = float(parts[1])
    if objective is not None:
        blocks.append((objective, values))

    parsed: list[tuple[float, np.ndarray]] = []
    for raw_objective, block in blocks:
        fractional = [
            value
            for index, value in block.items()
            if 0 <= index < formulation.nx and 1e-7 < value < 1.0 - 1e-7
        ]
        if fractional:
            continue
        selected = np.asarray(
            sorted(
                index
                for index, value in block.items()
                if 0 <= index < formulation.nx and value >= 1.0 - 1e-7
            ),
            dtype=int,
        )
        valid, _ = formulation.validate_selection(selected)
        if valid:
            parsed.append((raw_objective, selected))
    return parsed


def _configured_improving_seed_pairs(
    project_root: Path,
    formulation: CertificationFormulation,
    config: dict[str, Any],
) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, Any]]]:
    configured = config.get("evidence", {}).get("improving_solution_artifacts", {})
    if not isinstance(configured, dict):
        raise RuntimeError("evidence.improving_solution_artifacts must be a mapping")
    pairs: list[tuple[str, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for raw_path, raw_hash in sorted(configured.items()):
        path = project_root / str(raw_path)
        expected_hash = str(raw_hash).lower()
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Pinned interrupted-solver evidence mismatch: {path}")
        parsed = _highs_improving_selections(path, formulation)
        if not parsed:
            raise RuntimeError(f"No complete valid binary incumbent in improving solution: {path}")
        records.append(
            {
                "relative_path": path.resolve().relative_to(project_root.resolve()).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": expected_hash,
                "complete_valid_blocks": len(parsed),
            }
        )
        for block_index, (raw_objective, selected) in enumerate(parsed, start=1):
            exact_high = float(formulation.objective_for_selection("high_need_population", selected))
            label = (
                f"PINNED_HIGHS_IMPROVING::{path.parent.parent.parent.parent.parent.name}::"
                f"block{block_index}::raw{raw_objective:.12g}::cpu{exact_high:.12g}"
            )
            pairs.append((label, selected))
    return pairs, records


def _previous_stage42d_seed_pairs(
    project_root: Path,
    formulation: CertificationFormulation,
) -> list[tuple[str, np.ndarray]]:
    """Recover feasible incumbents from earlier Stage4.2D attempts.

    Previous Stage4.2D runs never define the frozen Efficiency reference, but
    their valid plans and intermediate certificate selections are useful MIP
    starts. Both committed and failed runs are eligible; every selection is
    revalidated against the current frozen formulation before use.
    """

    roots = [
        project_root / "outputs/model_v1/10_stage4_2d_equity_certification/runs",
        project_root / "outputs/model_v1/10_stage4_2d_equity_certification_diagnostic/runs",
    ]
    pairs: list[tuple[str, np.ndarray]] = []
    for runs_root in roots:
        if not runs_root.exists():
            continue
        run_dirs = sorted(
            (path for path in runs_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            plan_path = run_dir / "plan__equity_front_stage_seed.csv"
            if plan_path.exists():
                selected = _read_plan_indices(plan_path, formulation)
                if selected is not None:
                    pairs.append((f"PREVIOUS_STAGE42D_PLAN::{run_dir.name}", selected))
            for metric, filename in (
                ("min_sigungu_coverage", "certificate_min_sigungu.json"),
                ("high_need_population", "certificate_high_need.json"),
            ):
                cert_path = run_dir / filename
                if not cert_path.exists():
                    continue
                try:
                    payload = _read_json(cert_path)
                except Exception:
                    continue
                selected = _indices_from_stage_row(payload, formulation)
                if selected is not None:
                    pairs.append((f"PREVIOUS_STAGE42D_CERT::{run_dir.name}::{metric}", selected))
    return pairs


def reconstruct_contract(
    project_root: Path,
    formulation: CertificationFormulation,
    stage41: Any,
    base_config: dict[str, Any],
    config: dict[str, Any],
    *,
    greedy_indices: np.ndarray | None,
) -> FrozenEquityContract:
    payloads = [
        value
        for path in _candidate_run_dirs(project_root)
        if (value := _collect_run_payload(path, formulation)) is not None
    ]
    preferred = config.get("evidence", {}).get("preferred_stage42c_run_id")
    source = _select_source_run(payloads, str(preferred) if preferred else None)
    preferred_records = _verify_preferred_artifacts(project_root, source, config)
    efficiency_plan = np.asarray(source["efficiency_plan"], dtype=int)
    efficiency_reference = float(formulation.metrics(efficiency_plan)["unique_elderly_population"])
    fraction = float(base_config.get("optimization", {}).get("equity_min_efficiency_fraction", 0.60))
    total_floor = efficiency_reference * fraction
    gap = float(config["gate"]["near_optimal_relative_gap_max"])

    improving_pairs, improving_records = _configured_improving_seed_pairs(project_root, formulation, config)
    seed_pairs: list[tuple[str, np.ndarray]] = [
        *improving_pairs,
        *_previous_stage42d_seed_pairs(project_root, formulation),
    ]
    for scenario in ("equity", "balanced", "efficiency"):
        seed_pairs.extend(
            discover_seed_selections(
                project_root,
                formulation,
                stage41,
                scenario=scenario,
                candidate_set="top3",
                greedy_indices=greedy_indices,
            )
        )
    # Prefer the exact intermediate Stage4.2C Equity incumbents over final plans.
    for objective in ("min_sigungu_coverage", "high_need_population", "need_weighted", "cost"):
        selected = source.get(f"equity_stage_selection::{objective}")
        if selected is not None:
            seed_pairs.insert(0, (f"SOURCE_RUN_STAGE::{source['run_id']}::{objective}", np.asarray(selected, dtype=int)))
    if source.get("equity_plan") is not None:
        seed_pairs.insert(0, (f"SOURCE_RUN_FINAL::{source['run_id']}", np.asarray(source["equity_plan"], dtype=int)))
    seed_pairs.insert(0, (f"SOURCE_RUN_EFFICIENCY::{source['run_id']}", efficiency_plan))

    seeds: list[EvidenceSeed] = []
    seen: set[tuple[int, ...]] = set()
    for label, selected in seed_pairs:
        selected = np.asarray(selected, dtype=int)
        valid, _ = formulation.validate_selection(selected)
        key = tuple(sorted(map(int, selected)))
        if not valid or key in seen:
            continue
        seen.add(key)
        raw_metrics = formulation.metrics(selected)
        metrics = {
            "total_population": float(raw_metrics["unique_elderly_population"]),
            "min_sigungu_coverage": float(raw_metrics["min_sigungu_coverage_ratio"]),
            "high_need_population": float(raw_metrics["high_need_population"]),
            "need_weighted": float(raw_metrics["need_weighted_population"]),
            "cost": float(raw_metrics["cost"]),
        }
        seeds.append(EvidenceSeed(source=label, selected_indices=selected, metrics=metrics))

    eligible = [seed for seed in seeds if seed.metrics["total_population"] >= total_floor - 1e-6]
    if not eligible:
        raise RuntimeError("No valid Equity seed satisfies the frozen 60% efficiency floor")
    best_min = max(eligible, key=lambda seed: seed.metrics["min_sigungu_coverage"])
    min_ref = float(best_min.metrics["min_sigungu_coverage"])
    min_retention = float(config["contract"].get("min_sigungu_retention", 1.0))
    min_floor = retained_min_sigungu_floor(
        min_ref,
        min_retention,
        scale=int(getattr(formulation, "min_sigungu_scale", 1_000_000)),
    )
    stage2_eligible = [seed for seed in eligible if seed.metrics["min_sigungu_coverage"] >= min_floor - 1e-10]
    best_high = max(stage2_eligible, key=lambda seed: seed.metrics["high_need_population"])
    high_ref = float(best_high.metrics["high_need_population"])
    threshold = math.nextafter(high_ref * (1.0 + gap), math.inf)

    # The formal oracle cannot let a newly-created historical diagnostic alter
    # its incumbent or cut anchors.  Freeze the exact selected sets and numeric
    # references in the official config; prior outputs remain heuristic input,
    # never an implicit policy-contract update.
    frozen = config.get("oracle_certification", {}).get("frozen_references", {})

    def selection_digest(selected: np.ndarray) -> str:
        payload = json.dumps(
            sorted(map(int, np.asarray(selected, dtype=int))),
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def exact_float(name: str, observed: float) -> None:
        expected = frozen.get(name)
        if type(expected) not in (int, float) or float(observed).hex() != float(expected).hex():
            raise RuntimeError(
                f"Frozen Stage4.2D reference drifted: {name}: "
                f"expected={expected!r}, observed={observed!r}"
            )

    exact_float("total_population_floor", total_floor)
    exact_float("min_sigungu_incumbent", min_ref)
    exact_float("min_sigungu_retained_floor", min_floor)
    exact_float("high_need_incumbent", high_ref)
    exact_float("high_need_strict_threshold", threshold)
    if selection_digest(best_min.selected_indices) != frozen.get("min_selection_sha256"):
        raise RuntimeError("Frozen min-sigungu incumbent selection changed")
    if selection_digest(best_high.selected_indices) != frozen.get("high_selection_sha256"):
        raise RuntimeError("Frozen high-need incumbent selection changed")
    seed_sets = sorted(
        sorted(map(int, np.asarray(seed.selected_indices, dtype=int))) for seed in seeds
    )
    seed_payload = json.dumps(seed_sets, separators=(",", ":")).encode("utf-8")
    seed_manifest_sha256 = hashlib.sha256(seed_payload).hexdigest()
    if type(frozen.get("anchor_seed_count")) is not int or len(seeds) != frozen["anchor_seed_count"]:
        raise RuntimeError("Frozen Stage4.2D anchor seed count changed")
    if seed_manifest_sha256 != frozen.get("anchor_seed_manifest_sha256"):
        raise RuntimeError("Frozen Stage4.2D anchor seed selection manifest changed")

    return FrozenEquityContract(
        source_run_id=str(source["run_id"]),
        candidate_set="top3",
        near_optimal_gap=gap,
        efficiency_reference=efficiency_reference,
        total_population_floor=total_floor,
        min_sigungu_reference=min_ref,
        min_sigungu_retained_floor=min_floor,
        high_need_reference=high_ref,
        known_high_need_threshold=threshold,
        seeds=seeds,
        evidence={
            "source_candidate_dir": str(source["candidate_dir"]),
            "source_efficiency_certified": bool(source.get("efficiency_certified")),
            "source_efficiency_ordering_verified": bool(
                source.get("efficiency_ordering_verified")
            ),
            "source_equity_ordering_verified": bool(source.get("ordering_verified")),
            "source_payload_count": len(payloads),
            "seed_count": len(seeds),
            "equity_min_efficiency_fraction": fraction,
            "best_min_seed": best_min.source,
            "best_high_seed": best_high.source,
            "min_selection_sha256": selection_digest(best_min.selected_indices),
            "high_selection_sha256": selection_digest(best_high.selected_indices),
            "anchor_seed_manifest_sha256": seed_manifest_sha256,
            "min_sigungu_integer_scale": int(getattr(formulation, "min_sigungu_scale", 1_000_000)),
            "preferred_artifacts": preferred_records,
            "improving_solution_artifacts": improving_records,
        },
    )
