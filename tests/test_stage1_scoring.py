from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediroad.scoring.ablation import (  # noqa: E402
    ablation_quality_gates,
    build_need_score,
    certify_weight_profiles,
    compare_score_runs,
    run_exhaustive_ablation,
)
from mediroad.scoring.transform import transformed_feature_matrix  # noqa: E402
from mediroad.stage1_pipeline import (  # noqa: E402
    _quality_gate_exit_code,
    _semantic_group_composites,
)


def _fixture() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(42)
    n = 40
    frame = pd.DataFrame(
        {
            "admin_dong_code": [f"{i:08d}" for i in range(n)],
            "admin_dong_name": [f"region_{i}" for i in range(n)],
            "policy_sigungu_name": [f"group_{i % 5}" for i in range(n)],
            "population_65plus": rng.integers(100, 5000, n),
            "population_75plus": rng.integers(50, 3000, n),
            "population_85plus": rng.integers(10, 1000, n),
            "is_rural_eup_myeon": rng.integers(0, 2, n),
        }
    )
    axes = ["demographic", "health", "medical", "transport", "equity"]
    features = []
    for i, axis in enumerate(axes):
        for j in range(3):
            name = f"f_{axis}_{j}"
            frame[name] = np.maximum(0, rng.normal(10 + i, 2 + j, n))
            features.append(
                {
                    "name": name,
                    "axis": axis,
                    "semantic_group": f"{axis}_{j}",
                    "ablation_cluster": f"{axis}_{'joint' if j < 2 else 'solo'}",
                    "direction": 1 if j == 0 else -1,
                    "transform": "identity",
                    "model_input": True,
                    "missing_policy": "error",
                }
            )
    profiles = {
        "balanced": {axis: 0.2 for axis in axes},
        "medical_priority": {
            "demographic": 0.15,
            "health": 0.15,
            "medical": 0.35,
            "transport": 0.20,
            "equity": 0.15,
        },
    }
    return frame, {
        "features": features,
        "scalers": ["rank_percentile", "robust_minmax"],
        "primary_scaler": "rank_percentile",
        "weight_profiles": profiles,
        "primary_weight_profile": "balanced",
    }


def test_score_is_bounded_and_complete() -> None:
    frame, config = _fixture()
    run = build_need_score(frame, config, scaler="rank_percentile", weight_profile="balanced")
    assert len(run.scores) == 40
    assert run.scores["need_score"].between(0, 100).all()
    assert run.scores["need_rank"].between(1, 40).all()


def test_ablation_covers_every_active_feature_cluster_and_axis() -> None:
    frame, config = _fixture()
    metrics, summary, region = run_exhaustive_ablation(frame, config)
    assert set(summary.loc[summary.scope.eq("feature"), "variant"]) == {
        feature["name"] for feature in config["features"]
    }
    assert summary.loc[summary.scope.eq("cluster"), "variant"].nunique() == 10
    assert summary.loc[summary.scope.eq("axis"), "variant"].nunique() == 5
    assert metrics["run_id"].is_unique
    assert region["run_id"].nunique() == len(metrics) + 4  # two baselines x two scalers

    # Influence is a share of the perturbation scope inside one fully specified
    # policy configuration.  Renormalize/neutral denominators must never mix.
    influence_keys = ["scope", "removal_mode", "scaler", "weight_profile"]
    for _, rows in metrics.groupby(influence_keys, sort=False):
        expected = 1.0 if rows["mean_abs_rank_shift"].sum() > 0 else 0.0
        assert np.isclose(rows["normalized_influence_within_config"].sum(), expected)
    for _, rows in summary.groupby(["scope", "removal_mode"], sort=False):
        expected = 1.0 if rows["mean_abs_rank_shift_across_configs"].sum() > 0 else 0.0
        assert np.isclose(rows["normalized_influence"].sum(), expected)

    # A broader ablation cluster must remove all of its members jointly, not
    # silently degrade to a duplicate of one-feature LOFO.
    joint_members = {"f_demographic_0", "f_demographic_1"}
    joint_metric = metrics.loc[
        metrics["scope"].eq("cluster")
        & metrics["variant"].eq("demographic_joint")
        & metrics["scaler"].eq("rank_percentile")
        & metrics["weight_profile"].eq("balanced")
    ]
    assert len(joint_metric) == 1
    joint_run_id = joint_metric.iloc[0]["run_id"]
    observed = region.loc[region["run_id"].eq(joint_run_id)].set_index("admin_dong_code")
    expected = build_need_score(
        frame,
        config,
        scaler="rank_percentile",
        weight_profile="balanced",
        removed_features=joint_members,
    ).scores.set_index("admin_dong_code")
    pd.testing.assert_series_equal(
        observed["need_score"].sort_index(),
        expected["need_score"].sort_index(),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        observed["need_rank"].sort_index(),
        expected["need_rank"].sort_index(),
    )

    gates = ablation_quality_gates(summary)
    assert not gates.empty
    assert {"completion", "stress_diagnostic"}.issubset(set(gates["gate_class"]))
    assert set(gates.status).issubset({"PASS", "FAIL"})


def _tied_score_fixture() -> tuple[pd.DataFrame, dict]:
    n = 40
    frame = pd.DataFrame(
        {
            "admin_dong_code": [f"{i:08d}" for i in range(n)],
            "admin_dong_name": [f"tie_region_{i}" for i in range(n)],
            "policy_sigungu_name": [f"tie_group_{i % 5}" for i in range(n)],
        }
    )
    binary = np.r_[np.ones(30), np.zeros(10)]
    features = []
    for axis in ("axis_a", "axis_b"):
        for suffix in ("one", "two"):
            name = f"{axis}_{suffix}"
            frame[name] = binary
            features.append(
                {
                    "name": name,
                    "axis": axis,
                    "semantic_group": name,
                    "ablation_cluster": f"{axis}_joint",
                    "direction": 1,
                    "transform": "identity",
                    "model_input": True,
                    "missing_policy": "error",
                }
            )
    config = {
        "features": features,
        "scalers": ["rank_percentile"],
        "primary_scaler": "rank_percentile",
        "weight_profiles": {"balanced": {"axis_a": 0.5, "axis_b": 0.5}},
        "primary_weight_profile": "balanced",
    }
    return frame, config


def test_top20_is_deterministic_under_ties_and_input_row_order() -> None:
    frame, config = _tied_score_fixture()
    baseline = build_need_score(
        frame,
        config,
        scaler="rank_percentile",
        weight_profile="balanced",
    ).scores
    reordered = baseline.iloc[::-1].reset_index(drop=True)

    comparison = compare_score_runs(baseline, reordered, frame)
    assert comparison["top20_jaccard"] == 1.0
    assert comparison["top20_changed"] == 0
    assert comparison["top20_bidirectional_recall_at30"] == 1.0

    _, _, region = run_exhaustive_ablation(frame, config)
    top20_counts = region.groupby("run_id", sort=False)["top20"].sum()
    assert top20_counts.eq(20).all(), top20_counts[top20_counts.ne(20)].to_dict()


def _certification_metrics() -> tuple[pd.DataFrame, list[str]]:
    scalers = ["s1", "s2", "s3", "s4"]
    profiles = ["balanced", "priority"]
    rows: list[dict[str, object]] = []
    feature_count = 12
    for profile in profiles:
        for scaler in scalers:
            for mode in ("renormalize", "neutral"):
                for index in range(feature_count):
                    if profile == "balanced":
                        influence = 1.0 / feature_count
                    elif index == 0:
                        influence = 0.18
                    else:
                        influence = 0.82 / (feature_count - 1)
                    rows.append(
                        {
                            "run_id": f"feature_{index}_{mode}_{scaler}_{profile}",
                            "scope": "feature",
                            "variant": f"feature_{index}",
                            "removal_mode": mode,
                            "scaler": scaler,
                            "weight_profile": profile,
                            "spearman": 0.99,
                            "kendall": 0.95,
                            "top20_changed": 1,
                            "top20_bidirectional_recall_at30": 1.0,
                            "rank_shift_p95": 5.0,
                            "top20_outward_displacement_p90": 2.0,
                            "top20_max_escape_beyond30": 0.0,
                            "top20_sigungu_tvd": 0.05,
                            "normalized_influence_within_config": influence,
                        }
                    )
            rows.append(
                {
                    "run_id": f"substitution_{scaler}_{profile}",
                    "scope": "substitution",
                    "variant": "active=>shadow",
                    "removal_mode": "renormalize",
                    "scaler": scaler,
                    "weight_profile": profile,
                    "spearman": 0.99,
                    "kendall": 0.95,
                    "top20_changed": 1,
                    "top20_bidirectional_recall_at30": 1.0,
                    "rank_shift_p95": 5.0,
                    "top20_outward_displacement_p90": 2.0,
                    "top20_max_escape_beyond30": 0.0,
                    "top20_sigungu_tvd": 0.05,
                    "normalized_influence_within_config": 1.0,
                }
            )
    return pd.DataFrame(rows), scalers


def _profile_status(certification: pd.DataFrame, profile: str) -> int:
    rows = certification.loc[certification["weight_profile"].eq(profile), "profile_certified"]
    assert len(rows) == 1
    return int(rows.iloc[0])


def test_profile_certification_requires_features_and_complete_four_scaler_grid() -> None:
    metrics, scalers = _certification_metrics()
    complete = certify_weight_profiles(
        metrics,
        default_profile="balanced",
        expected_scalers=scalers,
    )
    assert _profile_status(complete, "balanced") == 1
    assert _profile_status(complete, "priority") == 1
    assert complete["scaler_count"].eq(4).all()

    no_features = certify_weight_profiles(
        metrics.loc[metrics["scope"].ne("feature")],
        default_profile="balanced",
        expected_scalers=scalers,
    )
    assert no_features["profile_certified"].eq(0).all()

    missing_scaler = metrics.loc[
        ~(metrics["weight_profile"].eq("balanced") & metrics["scaler"].eq("s4"))
    ]
    incomplete = certify_weight_profiles(
        missing_scaler,
        default_profile="balanced",
        expected_scalers=scalers,
    )
    assert _profile_status(incomplete, "balanced") == 0
    assert _profile_status(incomplete, "priority") == 1


def test_profile_certification_uses_default_influence_cap_and_substitution_escape() -> None:
    metrics, scalers = _certification_metrics()

    influence_case = metrics.copy()
    default_feature = influence_case["weight_profile"].eq("balanced") & influence_case["scope"].eq(
        "feature"
    )
    dominant = default_feature & influence_case["variant"].eq("feature_0")
    other = default_feature & ~influence_case["variant"].eq("feature_0")
    influence_case.loc[dominant, "normalized_influence_within_config"] = 0.11
    influence_case.loc[other, "normalized_influence_within_config"] = 0.89 / 11.0
    influence_certification = certify_weight_profiles(
        influence_case,
        default_profile="balanced",
        expected_scalers=scalers,
    )
    assert _profile_status(influence_certification, "balanced") == 0
    assert _profile_status(influence_certification, "priority") == 1

    substitution_case = metrics.copy()
    priority_substitution = substitution_case["weight_profile"].eq("priority") & substitution_case[
        "scope"
    ].eq("substitution")
    substitution_case.loc[priority_substitution, "top20_max_escape_beyond30"] = 11.0
    substitution_certification = certify_weight_profiles(
        substitution_case,
        default_profile="balanced",
        expected_scalers=scalers,
    )
    assert _profile_status(substitution_certification, "balanced") == 1
    assert _profile_status(substitution_certification, "priority") == 0


def test_sigungu_repeated_scaling_is_invariant_to_unequal_admin_dong_counts() -> None:
    group_names = [f"sigungu_{index:02d}" for index in range(11)]
    group_values = {name: float(index + 1) for index, name in enumerate(group_names)}

    def repeated_frame(counts: list[int]) -> pd.DataFrame:
        groups = [name for name, count in zip(group_names, counts) for _ in range(count)]
        return pd.DataFrame(
            {
                "policy_sigungu_name": groups,
                "sigungu_context": [group_values[name] for name in groups],
            }
        )

    feature_specs = [
        {
            "name": "sigungu_context",
            "direction": 1,
            "transform": "identity",
            "granularity": "policy_sigungu_repeated_on_admin_dong",
            "missing_policy": "error",
        }
    ]
    balanced = repeated_frame([1] * 11)
    uneven = repeated_frame(list(range(1, 12)))

    balanced_scores = transformed_feature_matrix(
        balanced,
        feature_specs,
        scaler="rank_percentile",
    )
    uneven_scores = transformed_feature_matrix(
        uneven,
        feature_specs,
        scaler="rank_percentile",
    )
    balanced_by_group = balanced.assign(score=balanced_scores["sigungu_context"]).groupby(
        "policy_sigungu_name", sort=False
    )["score"].first()
    uneven_by_group = uneven.assign(score=uneven_scores["sigungu_context"]).groupby(
        "policy_sigungu_name", sort=False
    )["score"].first()

    pd.testing.assert_series_equal(balanced_by_group, uneven_by_group)
    assert uneven.assign(score=uneven_scores["sigungu_context"]).groupby(
        "policy_sigungu_name"
    )["score"].nunique().eq(1).all()


def test_semantic_composite_distinguishes_measurement_method_from_analysis_unit() -> None:
    need = pd.DataFrame(
        {
            "local_hospital_count": [10.0, 20.0, 30.0],
            "network_hospital_time": [30.0, 20.0, 10.0],
        }
    )
    manifest = pd.DataFrame(
        [
            {
                "source_column": "local_hospital_count",
                "granularity": "admin_dong",
                "label_ko": "병원 수",
                "label_en": "Hospital count",
                "proxy": False,
            },
            {
                "source_column": "network_hospital_time",
                "granularity": "admin_dong_centroid_to_hira_facility_network",
                "label_ko": "병원 접근시간",
                "label_en": "Hospital travel time",
                "proxy": True,
            },
        ]
    )
    config = {
        "features": [
            {"feature_id": name, "semantic_group": "hospital_availability"}
            for name in need.columns
        ]
    }

    composite, composite_manifest = _semantic_group_composites(need, manifest, config)
    assert composite.shape == (3, 1)
    assert composite_manifest.loc[0, "granularity"] == "admin_dong"
    assert "admin_dong_centroid_to_hira_facility_network" in composite_manifest.loc[
        0, "component_measurement_granularities"
    ]

    mixed_manifest = manifest.copy()
    mixed_manifest.loc[
        mixed_manifest["source_column"].eq("network_hospital_time"), "granularity"
    ] = "policy_sigungu_repeated_on_admin_dong"
    with pytest.raises(ValueError, match="independent analysis units"):
        _semantic_group_composites(need, mixed_manifest, config)


def test_stage1_forces_a_headless_matplotlib_backend() -> None:
    import matplotlib

    assert str(matplotlib.get_backend()).lower() == "agg"


def test_stage1_process_exit_code_is_fail_closed() -> None:
    assert _quality_gate_exit_code({"quality_gate": {"hard_gate_pass": True}}) == 0
    assert _quality_gate_exit_code({"quality_gate": {"hard_gate_pass": False}}) == 2
    with pytest.raises(ValueError, match="quality_gate.hard_gate_pass"):
        _quality_gate_exit_code({})
