from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2d_equity_certification.evidence import (
    _candidate_run_dirs,
    _highs_improving_selections,
    _indices_from_stage_row,
    _ordering_verified,
    _previous_stage42d_seed_pairs,
    _scenario_certified,
    _select_source_run,
    retained_min_sigungu_floor,
)


def test_min_sigungu_retained_floor_matches_stage42c_integer_contract():
    assert retained_min_sigungu_floor(0.5252179790669655, 1.0) == 0.525217
    assert retained_min_sigungu_floor(0.5252179790669655, 0.99) == 0.519965


def test_candidate_discovery_includes_certification_diagnostic_root(tmp_path: Path):
    candidate_dir = (
        tmp_path
        / "outputs/model_v1/10_stage4_2_certification_diagnostic/runs/frozen/candidate_sets/top3"
    )
    candidate_dir.mkdir(parents=True)
    assert candidate_dir.resolve() in _candidate_run_dirs(tmp_path)


def test_explicit_preferred_run_never_silently_falls_back():
    payloads = [
        {
            "run_id": "different",
            "efficiency_plan": np.asarray([0, 3]),
            "efficiency_certified": True,
            "ordering_verified": True,
            "candidate_dir": Path.cwd(),
        }
    ]
    with pytest.raises(RuntimeError, match="preferred Stage4.2C run"):
        _select_source_run(payloads, "frozen")


def test_highs_improving_solution_uses_complete_binary_x_and_cpu_revalidation(tmp_path: Path):
    formulation = make_synthetic_formulation()
    path = tmp_path / "improving.sol"
    path.write_text(
        "\n".join(
            [
                "Objective 999999.0",
                "# Columns -2",
                "NoName 1 0",
                "NoName 1 3",
                "NoName 0.25 10",
                "Objective 888888.0",
                "# Columns -2",
                "NoName 1 1",
                "NoName 1 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = _highs_improving_selections(path, formulation)
    assert len(parsed) == 2
    assert set(parsed[0][1]) == {0, 3}
    assert formulation.validate_selection(parsed[0][1])[0]


def _certified_efficiency_rows(formulation):
    selected = np.asarray([0, 3])
    plan = [
        ("total_population", "max"),
        ("need_weighted", "max"),
        ("min_sigungu_coverage", "max"),
        ("cost", "min"),
    ]
    rows = []
    for index, (objective, sense) in enumerate(plan, start=1):
        value = float(formulation.objective_for_selection(objective, selected))
        rows.append(
            {
                "scenario": "efficiency",
                "candidate_set": "top3",
                "stage_index": index,
                "objective_name": objective,
                "sense": sense,
                "certified": True,
                "objective_value": value,
                "best_bound": value,
                "relative_gap": 0.0,
                "selected_indices": selected.tolist(),
            }
        )
    return rows


def test_source_certification_recomputes_objective_and_gap_from_selection():
    formulation = make_synthetic_formulation()
    rows = _certified_efficiency_rows(formulation)
    assert _scenario_certified(rows, "efficiency", formulation)

    forged_objective = [dict(row) for row in rows]
    forged_objective[0]["objective_value"] += 1.0
    forged_objective[0]["best_bound"] += 1.0
    assert not _scenario_certified(forged_objective, "efficiency", formulation)

    forged_gap = [dict(row) for row in rows]
    forged_gap[0]["best_bound"] += 10.0
    forged_gap[0]["relative_gap"] = 0.0
    assert not _scenario_certified(forged_gap, "efficiency", formulation)

    forged_bool = [dict(row) for row in rows]
    forged_bool[0]["certified"] = "true"
    assert not _scenario_certified(forged_bool, "efficiency", formulation)


@pytest.mark.parametrize(
    "selected",
    [
        [0.0, 3],
        ["0", 3],
        [-1, 3],
        [0, 999],
        [[0], [3]],
    ],
)
def test_stage_selection_indices_are_strict_bounded_integer_lists(selected):
    formulation = make_synthetic_formulation()
    assert _indices_from_stage_row({"selected_indices": selected}, formulation) is None


def test_ordering_check_uses_stage_index_four_not_raw_last_row():
    formulation = make_synthetic_formulation()
    rows = _certified_efficiency_rows(formulation)
    rows[0]["selected_indices"] = [1, 4]
    shuffled = [rows[1], rows[2], rows[3], rows[0]]
    assert not _ordering_verified(shuffled, np.asarray([1, 4]), formulation)


def test_negative_highs_column_index_is_never_an_x_alias(tmp_path: Path):
    formulation = make_synthetic_formulation()
    path = tmp_path / "negative-index.improving.sol"
    path.write_text(
        "Objective 1\n# Columns -2\nNoName 1 -1\nNoName 1 3\n",
        encoding="utf-8",
    )
    assert _highs_improving_selections(path, formulation) == []


@pytest.mark.parametrize("selected", [[-1, 3], [0.9, 3], ["0", 3], [[0], [3]]])
def test_previous_stage42d_certificate_indices_fail_closed(tmp_path: Path, selected):
    formulation = make_synthetic_formulation()
    run_dir = (
        tmp_path
        / "outputs/model_v1/10_stage4_2d_equity_certification_diagnostic/runs/prior"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "certificate_min_sigungu.json").write_text(
        json.dumps({"selected_indices": selected}) + "\n",
        encoding="utf-8",
    )
    assert _previous_stage42d_seed_pairs(tmp_path, formulation) == []
