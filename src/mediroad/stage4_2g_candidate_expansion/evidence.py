from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mediroad.stage4_2_certification.io_utils import sha256_file


FROZEN_STAGE42F_CURRENT_SHA256 = (
    "853670d7d7b44daedad939cae6c6e0942e8d2b896d2f408447b56cd40c8cb8b7"
)
FROZEN_STAGE42C_INVENTORY_SHA256 = (
    "c1198cd711730871538ff536614f724ea64ee2634ced21d1aa9d144b00a74362"
)
FROZEN_STAGE41_CURRENT_SHA256 = (
    "eb87049591e0acbaab581f57b1e63aba19dbe9179cf650dfd2b05933e5fad790"
)


def read_json(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(path)
    return value


def _verify_inventory(
    project_root: Path,
    inventory_path: Path,
    *,
    artifact_root: Path | None = None,
) -> None:
    rows = pd.read_csv(inventory_path, dtype={"relative_path": str, "sha256": str})
    required = {"relative_path", "size_bytes", "sha256"}
    if set(rows.columns) != required:
        raise RuntimeError(f"Unexpected inventory schema: {inventory_path}")
    base = (artifact_root or project_root).resolve()
    for row in rows.to_dict(orient="records"):
        artifact = (base / str(row["relative_path"])).resolve()
        try:
            artifact.relative_to(project_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Inventory path escapes project root: {artifact}") from exc
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row["size_bytes"])
            or sha256_file(artifact) != str(row["sha256"]).lower()
        ):
            raise RuntimeError(f"Inventory verification failed: {artifact}")


def load_top3_parent(project_root: Path) -> dict[str, Any]:
    ptr=project_root/"outputs/model_v1/10_stage4_2f_top3_aggregate/CURRENT_STAGE4_2F_TOP3_AGGREGATE_RUN.json"
    if not ptr.exists(): raise RuntimeError("Official Stage4.2F CURRENT is required")
    if sha256_file(ptr) != FROZEN_STAGE42F_CURRENT_SHA256:
        raise RuntimeError("Stage4.2F CURRENT SHA-256 is not the frozen official pointer")
    p=read_json(ptr)
    if (
        p.get("decision") != "PASS_STAGE4_2F_TOP3_REQUIRED_SCENARIOS_CERTIFIED"
        or p.get("scope") != "TOP3_REQUIRED_SCENARIOS_ONLY"
        or p.get("top3_required_scenarios_certified") is not True
        or p.get("candidate_expansion_certified") is not False
        or p.get("stage4_full_computational_complete") is not False
        or p.get("operational_final") is not False
        or p.get("stage5_started") is not False
        or p.get("stage5_release_allowed") is not False
    ):
        raise RuntimeError("Stage4.2F CURRENT is not the required PASS")
    run=project_root/str(p["run_relative_path"])
    if not (run / ".COMMITTED").is_file():
        raise RuntimeError("Stage4.2F official run is not committed")
    inventory_path = run / "ARTIFACT_INVENTORY.csv"
    gate_path = run / "04_quality_and_provenance/quality_gate.json"
    if sha256_file(inventory_path) != str(p.get("inventory_sha256", "")).lower():
        raise RuntimeError("Stage4.2F inventory is not pointer-bound")
    if sha256_file(gate_path) != str(p.get("quality_gate_sha256", "")).lower():
        raise RuntimeError("Stage4.2F quality gate is not pointer-bound")
    _verify_inventory(project_root, inventory_path)
    gate=read_json(gate_path)
    if (
        gate.get("decision") != "PASS_STAGE4_2F_TOP3_REQUIRED_SCENARIOS_CERTIFIED"
        or gate.get("passed") is not True
        or gate.get("promotable_scope") != "TOP3_REQUIRED_SCENARIOS_ONLY"
        or gate.get("top3_required_scenarios_certified") is not True
        or gate.get("candidate_expansion_certified") is not False
        or gate.get("stage4_full_computational_complete") is not False
        or gate.get("operational_final") is not False
        or gate.get("stage5_started") is not False
        or gate.get("stage5_release_allowed") is not False
    ):
        raise RuntimeError("Stage4.2F did not preserve the required Top3-only scope")
    agg=read_json(run/"03_top3_semantic_aggregate/scenario_certification.json")
    stages=agg.get("stages",[])
    efficiency=[x for x in stages if x.get("scenario")=="efficiency"]
    balanced=[x for x in stages if x.get("scenario")=="balanced"]
    if len(efficiency)!=4 or len(balanced)!=4: raise RuntimeError("Stage4.2F efficiency/balanced 4-stage evidence missing")
    source_run=str(balanced[-1]["source_run_id"])
    if any(str(x["source_run_id"]) != source_run for x in efficiency+balanced):
        raise RuntimeError("Stage4.2F E/B adoption does not share one pinned Stage4.2C source run")
    source_root=None
    for base in [
        project_root/"outputs/model_v1/10_stage4_2_certification_diagnostic/runs",
        project_root/"outputs/model_v1/10_stage4_2_certification/runs",
    ]:
        candidate=base/source_run
        if candidate.exists(): source_root=candidate; break
    if source_root is None: raise RuntimeError(f"Pinned Stage4.2C source run not found: {source_run}")
    source_inventory = source_root / "ARTIFACT_INVENTORY.csv"
    if sha256_file(source_inventory) != FROZEN_STAGE42C_INVENTORY_SHA256:
        raise RuntimeError("Preferred Stage4.2C inventory SHA-256 changed")
    _verify_inventory(project_root, source_inventory, artifact_root=source_root)
    efficiency_plan=pd.read_csv(source_root/"candidate_sets/top3/plan__efficiency.csv",low_memory=False)
    balanced_plan=pd.read_csv(source_root/"candidate_sets/top3/plan__balanced.csv",low_memory=False)
    efficiency_metrics=read_json(source_root/"candidate_sets/top3/metrics__efficiency.json")
    balanced_metrics=read_json(source_root/"candidate_sets/top3/metrics__balanced.json")
    return {
        "pointer":p,"run_root":run,"gate":gate,"source_run_id":source_run,"source_root":source_root,
        "efficiency_plan":efficiency_plan,"efficiency_metrics":efficiency_metrics,"efficiency_max_gap":max(float(x["relative_gap"]) for x in efficiency),
        "balanced_plan":balanced_plan,"balanced_metrics":balanced_metrics,"balanced_max_gap":max(float(x["relative_gap"]) for x in balanced),
    }


def load_frozen_candidate_decision(project_root: Path) -> dict[str, Any]:
    ptr_path = project_root / "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json"
    if sha256_file(ptr_path) != FROZEN_STAGE41_CURRENT_SHA256:
        raise RuntimeError("Stage4.1 CURRENT SHA-256 is not the frozen official pointer")
    ptr=read_json(ptr_path)
    if (
        ptr.get("run_id") != "stage4_1_20260819T203433Z_754bf1271229"
        or ptr.get("decision") != "PASS_STAGE4_READY_FOR_FIELD_VALIDATION"
        or ptr.get("candidate_reduction_certified") is not False
        or ptr.get("expanded_candidate_comparison_conclusive") is not False
        or ptr.get("stage4_operational_final") is not False
        or ptr.get("stage5_started") is not False
        or ptr.get("stage5_release_allowed") is not False
    ):
        raise RuntimeError("Stage4.1 CURRENT is not the frozen provisional field-validation PASS")
    metadata_path = project_root / str(ptr["metadata_relative_path"])
    run=metadata_path.parent
    inventory_path = project_root / str(ptr["inventory_relative_path"])
    for path_key, sha_key in (
        ("metadata_relative_path", "metadata_sha256"),
        ("inventory_relative_path", "inventory_sha256"),
        ("interface_relative_path", "interface_sha256"),
        ("report_relative_path", "report_sha256"),
    ):
        artifact = project_root / str(ptr[path_key])
        if not artifact.is_file() or sha256_file(artifact) != str(ptr[sha_key]).lower():
            raise RuntimeError(f"Stage4.1 pointer artifact binding failed: {path_key}")
    _verify_inventory(project_root, inventory_path, artifact_root=run)
    found=list(run.rglob("candidate_promotion_decision.json"))
    if len(found)!=1: raise RuntimeError(f"Expected one Stage4.1 candidate promotion decision, found {len(found)}")
    value=read_json(found[0])
    if value.get("threshold_fingerprint")!="b0a1a2c6dc6f1d1e7c3ae1141700fbb3c54854b03746d2e45d2d0ba33c3b1ad4":
        raise RuntimeError("Frozen Stage4.1 candidate threshold fingerprint changed")
    if value.get("baseline_set")!="top3" or value.get("mandatory_reference_sets")!=["top5","coarse_pareto"]:
        raise RuntimeError("Frozen Stage4.1 candidate ladder changed")
    return {"pointer":ptr,"run_root":run,"decision_path":found[0],"decision":value}
