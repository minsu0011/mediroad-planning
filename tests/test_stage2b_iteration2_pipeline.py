from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import mediroad.stage2b_iteration2_pipeline as pipeline


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_inventory(root: Path, path: Path, targets: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "relative_path": target.relative_to(root).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": _digest(target),
            }
            for target in targets
        ]
    ).to_csv(path, index=False)


def _package(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "package"
    stage2_payload = root / "outputs/model_v1/04_specialty/frozen.txt"
    hardening_root = root / "outputs/model_v1/05_stage2b_hardening"
    hardening_payload = hardening_root / "04_release/frozen.txt"
    for path, text in ((stage2_payload, "stage2"), (hardening_payload, "hardening")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    stage2_inventory = root / "outputs/model_v1/04_stage2/STAGE2_ARTIFACT_INVENTORY.csv"
    stage2_metadata = root / "outputs/model_v1/04_stage2/STAGE2_RUN_METADATA.json"
    stage2_metadata.parent.mkdir(parents=True, exist_ok=True)
    stage2_metadata.write_text('{"run_id":"stage2_test"}\n', encoding="utf-8")
    _write_inventory(root, stage2_inventory, [stage2_payload])
    hardening_metadata = hardening_root / "STAGE2B_HARDENING_RUN_METADATA.json"
    hardening_quality = hardening_root / "STAGE2B_HARDENING_QUALITY_GATE.csv"
    hardening_interface = hardening_root / "stage2_to_stage3_interface.parquet"
    hardening_metadata.write_text('{"run_id":"hardening_test"}\n', encoding="utf-8")
    hardening_quality.write_text("gate,passed\nfrozen,true\n", encoding="utf-8")
    hardening_interface.write_bytes(b"frozen-interface")
    _write_inventory(
        root,
        hardening_root / "STAGE2B_HARDENING_ARTIFACT_INVENTORY.csv",
        [hardening_payload],
    )

    bundle = {
        "bundles": [
            {
                "bundle_id": "bundle_a",
                "bundle_name_ko": "A",
                "included_services": {"s1": 0.5, "s2": 0.5},
            }
        ]
    }
    bundle_path = root / "configs/model_v1/service_bundles.yaml"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(yaml.safe_dump(bundle), encoding="utf-8")
    nhis_rows = []
    for year in (2022, 2023):
        for month in range(1, 13):
            for sigungu_index, sigungu in enumerate(("g1", "g2")):
                for service_index, service in enumerate(("s1", "s2")):
                    nhis_rows.append(
                        {
                            "year": year,
                            "month": month,
                            "policy_sigungu_name": sigungu,
                            "service_id": service,
                            "persons": 50 + sigungu_index,
                            "visits": 100 + 3 * month + 5 * service_index + sigungu_index,
                        }
                    )
    nhis_path = root / "inputs/nhis.csv"
    nhis_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nhis_rows).to_csv(nhis_path, index=False)
    calendar_path = root / "inputs/calendar.csv"
    pd.DataFrame(
        [
            {"year": year, "month": month, "calendar_days": 30, "working_days": 20}
            for year in (2022, 2023)
            for month in range(1, 13)
        ]
    ).to_csv(calendar_path, index=False)

    hardening_config = {
        "paths": {
            "nhis_normalized_panel": nhis_path.relative_to(root).as_posix(),
            "working_day_calendar": calendar_path.relative_to(root).as_posix(),
            "service_bundle_config": bundle_path.relative_to(root).as_posix(),
            "baseline_stage2_inventory": stage2_inventory.relative_to(root).as_posix(),
            "baseline_stage2_metadata": stage2_metadata.relative_to(root).as_posix(),
            "output_dir": hardening_root.relative_to(root).as_posix(),
        },
        "analysis_contract": {
            "observed_years": [2022, 2023],
            "expected_policy_sigungu": 2,
            "expected_services": 2,
        },
        "frozen_contract": {
            "nhis_normalized_panel_sha256": _digest(nhis_path),
            "working_day_calendar_sha256": _digest(calendar_path),
        },
    }
    hardening_path = root / "configs/model_v1/stage2b_hardening.yaml"
    hardening_path.write_text(yaml.safe_dump(hardening_config), encoding="utf-8")

    source_config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/model_v1/stage2b_iteration2.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_config["frozen_baseline"]["hardening_config"] = hardening_path.relative_to(root).as_posix()
    source_config["output_contract"]["raw_validation_dir"] = "external/raw"
    source_config["output_contract"]["normalized_validation_dir"] = "external/normalized"
    config_path = root / "configs/model_v1/stage2b_iteration2.yaml"
    config_path.write_text(yaml.safe_dump(source_config, sort_keys=False), encoding="utf-8")
    source_manifest = root / "external/SOURCE_MANIFEST.yaml"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        yaml.safe_dump(
            {"iteration_contract_sha256_current": _digest(config_path)},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root, config_path, stage2_payload


def _candidate_result() -> SimpleNamespace:
    candidate_ids = ["baseline__local__mean"] + [
        f"candidate_{index:02d}" for index in range(71)
    ]
    registry = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "eligible": [False] * 72,
            "stage2a_bundle_definition_change_count": [0] * 72,
            "stage2a_what_top1_retention": [1.0] * 72,
            "top1_agreement": [0.5] * 72,
            "primary_in_holdout_top2": [0.75] * 72,
            "selected_pair_holdout_top2_coverage": [0.5] * 72,
            "rank_spearman_median": [0.4] * 72,
            "normalized_regret_mean": [0.2] * 72,
            "lomo_top2_retention": [0.9] * 72,
            "loso_top2_retention": [0.9] * 72,
            "failure_reasons": ["synthetic_no_promotion"] * 72,
        }
    )
    selection = pd.DataFrame(
        {
            "candidate_id": ["nested"] * 4,
            "holdout_labels_visible_during_selection": [False] * 4,
        }
    )
    one = pd.DataFrame({"candidate_id": ["nested"], "value": [1.0]})
    comparison = pd.DataFrame(
        {
            "candidate_id": ["nested"],
            "top1_agreement": [0.5],
            "primary_in_holdout_top2": [0.75],
            "selected_pair_holdout_top2_coverage": [0.5],
            "rank_spearman_median": [0.4],
            "normalized_regret_mean": [0.2],
            "lomo_top2_retention": [0.9],
            "loso_top2_retention": [0.9],
            "failure_reasons": ["synthetic_no_promotion"],
        }
    )
    nested = SimpleNamespace(
        selection=selection,
        cross_year_detail=one,
        cross_year_summary=one,
        lomo_detail=one,
        lomo_summary=one,
        loso_detail=one,
        loso_summary=one,
        bootstrap_detail=one,
        bootstrap_summary=one,
        permutation_detail=one,
        permutation_summary=one,
        release_evidence=one,
        comparison=comparison,
        audit={"exploratory_only_due_no_third_year": True},
    )
    return SimpleNamespace(
        grid=SimpleNamespace(
            ledger=registry,
            bundle_ledger=pd.DataFrame(
                {"candidate_id": registry.candidate_id, "bundle_id": "bundle_a"}
            ),
            audit={"candidate_count": 72},
        ),
        nested=nested,
        promotion_decision="NO_PROMOTION_KEEP_PASS_ADVISORY_TEMPORAL",
    )


def _external(dataset_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        contract=SimpleNamespace(dataset_id=dataset_id),
        audit={
            "dataset_id": dataset_id,
            "row_count": 1,
            "sha256": dataset_id * 4,
            "promotion_allowed": False,
            "concat_allowed": False,
        },
        frame=pd.DataFrame(),
    )


def _patch_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    # Candidate-engine tests cover the immutable mapping seal.  This test
    # changes only filesystem paths so it can exercise orchestration in tmp.
    monkeypatch.setattr(
        pipeline,
        "validate_iteration2_config",
        lambda config: {"contract_mapping_sha256": config["contract_mapping_sha256"]},
    )
    monkeypatch.setattr(pipeline, "run_hardening_candidates", lambda *a, **k: _candidate_result())
    monkeypatch.setattr(
        pipeline, "read_provider_specialty_monthly_cp949", lambda *a, **k: _external("15141856")
    )
    monkeypatch.setattr(
        pipeline, "read_patient_allcare_monthly_cp949", lambda *a, **k: _external("15141213")
    )
    ledger = pd.DataFrame({"decision": ["NO_PROMOTION"]})
    provider = SimpleNamespace(
        bundle_season_profile=pd.DataFrame({"bundle_id": ["bundle_a"]}),
        detail=pd.DataFrame(
            {
                "bundle_id": ["bundle_a"],
                "season_rank_spearman": [0.5],
                "top1_agreement": [0.5],
                "train_primary_in_comparison_top2": [1.0],
            }
        ),
        summary=pd.DataFrame(
            {"bundle_id": ["bundle_a"], "mean_season_rank_spearman": [0.5]}
        ),
        ledger=ledger,
    )
    common = SimpleNamespace(
        common_profile=pd.DataFrame({"month": [1]}),
        shock_detail=pd.DataFrame({"month": [1]}),
        candidate=pd.DataFrame(
            {"month": [1], "candidate_multiplier_midpoint": [1.0]}
        ),
        ledger=ledger,
    )
    monkeypatch.setattr(pipeline, "diagnose_provider_bundle_phase", lambda *a, **k: provider)
    monkeypatch.setattr(pipeline, "build_allcare_common_shock_candidate", lambda *a, **k: common)


def test_iteration2_pipeline_is_run_scoped_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, frozen_payload = _package(tmp_path)
    frozen_sha = _digest(frozen_payload)
    _patch_apis(monkeypatch)

    result = pipeline.run_stage2b_iteration2(
        root,
        config,
        n_boot=2,
        n_permutations=3,
        jobs=1,
        run_id="synthetic_iteration2",
        tests_passed=1,
        tests_failed=0,
    )

    assert result.hard_pass
    assert result.decision == "NO_PROMOTION"
    assert _digest(frozen_payload) == frozen_sha
    assert not (result.run_dir.parent.parent / ".stage2b_iteration2.lock").exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["metadata_written_last"] is True
    assert metadata["decision"] == "NO_PROMOTION"
    assert metadata["stage3_started"] is False
    assert metadata["official_parent_mutation_count"] == 0
    registry = pd.read_csv(result.run_dir / "01_candidates/candidate_registry.csv")
    assert len(registry) == 72
    assert not registry["promotion_eligible"].astype(bool).any()
    gates = pd.read_csv(result.run_dir / "STAGE2B_ITERATION2_QUALITY_GATE.csv")
    assert gates["passed"].astype(bool).all()
    inventory = pd.read_csv(result.run_dir / "STAGE2B_ITERATION2_ARTIFACT_INVENTORY.csv")
    for row in inventory.itertuples(index=False):
        path = root / row.relative_path
        assert path.is_file()
        assert _digest(path) == row.sha256
    assert result.metadata_path.stat().st_mtime_ns >= max(
        (root / path).stat().st_mtime_ns for path in inventory.relative_path
    )
    assert (result.report_dir / "FINAL_REPORT.md").is_file()


def test_iteration2_pipeline_rejects_existing_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, _ = _package(tmp_path)
    _patch_apis(monkeypatch)
    output = root / "outputs/model_v1/06_stage2b_iteration2"
    output.mkdir(parents=True)
    (output / ".stage2b_iteration2.lock").write_text("busy", encoding="utf-8")
    with pytest.raises(RuntimeError, match="writer lock"):
        pipeline.run_stage2b_iteration2(root, config, run_id="locked")
