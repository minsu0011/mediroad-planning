from __future__ import annotations

from pathlib import Path

from mediroad.stage4_2g_candidate_expansion.evidence import (
    load_frozen_candidate_decision,
    load_top3_parent,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_stage42f_parent_and_preferred_stage42c_inventory_load() -> None:
    parent = load_top3_parent(ROOT)
    assert parent["pointer"]["run_id"] == (
        "stage4_2f_top3_aggregate_20260821T112408Z_054f6cf6a90e"
    )
    assert parent["source_run_id"] == (
        "stage4_2c_certification_20260820T151657Z_56c5f1bad35c"
    )
    assert len(parent["efficiency_plan"]) == 20
    assert len(parent["balanced_plan"]) == 20


def test_frozen_stage41_candidate_decision_loads_from_pointer_bound_run() -> None:
    frozen = load_frozen_candidate_decision(ROOT)
    assert frozen["pointer"]["run_id"] == "stage4_1_20260819T203433Z_754bf1271229"
    assert frozen["decision"]["baseline_set"] == "top3"
    assert frozen["decision"]["mandatory_reference_sets"] == [
        "top5",
        "coarse_pareto",
    ]
