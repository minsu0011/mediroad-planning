from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2_certification.types import LinearMipModel
import mediroad.stage4_2d_equity_certification.runner as runner
from mediroad.stage4_2d_equity_certification.threshold_oracles import (
    ThresholdOracleModel,
    ThresholdOracleResult,
    build_min_highs_oracle,
    selection_digest,
)
from mediroad.stage4_2d_equity_certification.types import EvidenceSeed


def _tiny_model(name: str = "tiny") -> LinearMipModel:
    return LinearMipModel(
        name=name,
        objective_name="feasibility",
        sense="min",
        c=np.zeros(3, dtype=float),
        A=sparse.csc_matrix(
            (
                np.asarray([1.0, 2.0]),
                (np.asarray([0, 1]), np.asarray([0, 1])),
            ),
            shape=(2, 3),
        ),
        row_lower=np.asarray([0.0, -np.inf]),
        row_upper=np.asarray([1.0, 2.0]),
        col_lower=np.zeros(3, dtype=float),
        col_upper=np.ones(3, dtype=float),
        integrality=np.asarray([1, 1, 0], dtype=np.uint8),
        variable_names=["x0", "x1", "y0"],
        x_slice=slice(0, 2),
        y_slice=slice(2, 3),
    )


def _tiny_oracle() -> ThresholdOracleModel:
    exact = _tiny_model("exact")
    proof = _tiny_model("proof")
    selected = (0, 1)
    return ThresholdOracleModel(
        metric="min_sigungu_coverage",
        backend="HIGHS_IN_MEMORY",
        exact_model=exact,
        model=proof,
        threshold=0.55,
        incumbent_value=0.5,
        policy_relative_gap=0.005,
        retained_floor=10.0,
        total_population_floor=10.0,
        min_sigungu_floor=None,
        incumbent_selection=selected,
        selection_digest=selection_digest(selected),
        exact_model_digest="a" * 64,
        proof_model_digest="b" * 64,
        universe_digest="c" * 64,
        metadata={
            "proof_relaxation": {
                "exact_to_proof_transport_audit": {"sufficient": True}
            }
        },
    )


def test_pinned_selection_is_order_independent_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = selection_digest(np.asarray([4, 1], dtype=np.int64))
    seeds = [
        EvidenceSeed("first", np.asarray([1, 4], dtype=np.int64), {}),
        EvidenceSeed("same-selection", np.asarray([4, 1], dtype=np.int64), {}),
    ]

    np.testing.assert_array_equal(
        runner._pinned_selection(seeds, expected, label="test"),
        np.asarray([1, 4], dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="absent"):
        runner._pinned_selection(seeds, "0" * 64, label="test")

    # The ambiguity guard is deliberately tested independently from SHA-256's
    # collision resistance: if the digest selector ever admits two selections,
    # promotion must still stop.
    monkeypatch.setattr(runner, "selection_digest", lambda _selected: expected)
    with pytest.raises(RuntimeError, match="ambiguous"):
        runner._pinned_selection(
            seeds
            + [EvidenceSeed("different", np.asarray([0, 2], dtype=np.int64), {})],
            expected,
            label="test",
        )


def test_oracle_model_summary_omits_sparse_model_payload() -> None:
    summary = runner._oracle_model_summary(_tiny_oracle())

    assert summary["exact_model"] == {
        "rows": 2,
        "cols": 3,
        "nnz": 2,
        "sha256": "a" * 64,
    }
    assert summary["proof_model"] == {
        "rows": 2,
        "cols": 3,
        "nnz": 2,
        "sha256": "b" * 64,
    }
    payload = json.dumps(summary)
    for forbidden in ("row_lower", "col_upper", "variable_names", '"A"'):
        assert forbidden not in payload


def test_formal_report_states_solver_native_and_scope_ceiling(tmp_path: Path) -> None:
    report = tmp_path / "FINAL_REPORT.md"
    min_oracle = _tiny_oracle()
    high_oracle = _tiny_oracle()
    min_result = SimpleNamespace(
        solver_version="1.15.1",
        status="INFEASIBLE",
        incumbent_value=0.5,
        threshold=0.55,
        certified_threshold_infeasible=True,
    )
    high_result = SimpleNamespace(
        status="INFEASIBLE",
        incumbent_value=100.0,
        threshold=100.5,
        certified_threshold_infeasible=True,
    )
    runner._write_formal_oracle_report(
        report,
        run_id="stage4_2d_test",
        decision="PASS_STAGE4_2D_EQUITY_FRONT_STAGES_CERTIFIED",
        contract=SimpleNamespace(
            candidate_set="top3", near_optimal_gap=0.005, source_run_id="parent"
        ),
        hardware=SimpleNamespace(
            solver_threads=8,
            portfolio_workers=8,
            threads_per_portfolio_worker=1,
        ),
        min_oracle=min_oracle,
        min_result=min_result,
        high_oracle=high_oracle,
        high_result=high_result,
        high_mps=SimpleNamespace(sha256="d" * 64),
        final_metrics={"min_sigungu_coverage_ratio": 0.5},
    )

    text = report.read_text(encoding="utf-8")
    assert "Scope: frozen `top3` Equity front stages only" in text
    assert (
        "solver-native computational infeasibility certificates, not independently "
        "checkable formal proof logs"
    ) in text
    assert (
        "Candidate expansion, later Equity objectives, field validation, "
        "Operational Final, and Stage 5 remain uncertified or prohibited."
    ) in text
    assert "no GPU proof acceleration is claimed" in text


def test_formal_pipeline_uncertified_min_marks_failed_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formulation = make_synthetic_formulation()
    min_selected = np.asarray([0, 3], dtype=np.int64)
    high_selected = np.asarray([1, 3], dtype=np.int64)
    min_metrics = formulation.metrics(min_selected)
    min_oracle = build_min_highs_oracle(
        formulation,
        total_population_floor=10.0,
        threshold=0.5,
        incumbent_value=min_metrics["min_sigungu_coverage_ratio"],
        retained_floor=10.0,
        incumbent_selection=min_selected,
        expected_selection_digest=selection_digest(min_selected),
    )
    min_result = ThresholdOracleResult(
        metric="min_sigungu_coverage",
        backend="HIGHS_IN_MEMORY",
        status="TIME_LIMIT",
        certified_threshold_infeasible=False,
        counterexample_found=False,
        threshold=min_oracle.threshold,
        incumbent_value=min_oracle.incumbent_value,
        policy_relative_gap=min_oracle.policy_relative_gap,
        retained_floor=min_oracle.retained_floor,
        incumbent_selection=min_oracle.incumbent_selection,
        selection_digest=min_oracle.selection_digest,
        best_bound=None,
        relative_gap=None,
        wall_time_sec=1.0,
        node_count=1,
        solver_version="test-highs",
        exact_model_digest=min_oracle.exact_model_digest,
        proof_model_digest=min_oracle.proof_model_digest,
        universe_digest=min_oracle.universe_digest,
        cut_digest=None,
        mps_sha256=None,
        requested_options={"threads": 8},
        readback_options={"threads": 8},
        evidence={
            "model_transport": {
                "proof_model_digest": min_oracle.proof_model_digest
            }
        },
    )

    monkeypatch.setattr(runner, "build_min_highs_oracle", lambda *_a, **_k: min_oracle)
    monkeypatch.setattr(runner, "run_min_highs_oracle", lambda *_a, **_k: min_result)

    def high_must_not_run(*_args, **_kwargs):
        raise AssertionError("high oracle must be skipped after an uncertified min oracle")

    monkeypatch.setattr(runner, "build_high_scip_oracle", high_must_not_run)
    monkeypatch.setattr(runner, "run_scip_portfolio", high_must_not_run)
    monkeypatch.setattr(runner, "_pointer_snapshot", lambda _root: {})
    monkeypatch.setattr(runner, "_authoritative_input_snapshot", lambda _root: [])
    monkeypatch.setattr(
        runner, "_stage5_snapshot", lambda _root: {"records": [], "count": 0}
    )
    monkeypatch.setattr(runner, "_source_snapshot", lambda _root, _paths: [])
    monkeypatch.setattr(runner, "_verify_evidence_records", lambda _root, _records: True)
    monkeypatch.setattr(runner, "_verify_output_inventory", lambda _root, _path: None)

    min_digest = selection_digest(min_selected)
    high_digest = selection_digest(high_selected)
    config = {
        "oracle_certification": {
            "rebuild_model_in_official_run": True,
            "accept_diagnostic_artifacts": False,
            "frozen_references": {
                "min_selection_sha256": min_digest,
                "high_selection_sha256": high_digest,
                "total_population_floor": 10.0,
                "min_sigungu_strict_threshold": 0.5,
                "min_sigungu_incumbent": min_oracle.incumbent_value,
                "min_sigungu_retained_floor": 0.2,
                "high_need_strict_threshold": 30.0,
                "high_need_incumbent": 25.0,
                "anchor_seed_count": 2,
                "anchor_seed_manifest_sha256": "e" * 64,
            },
            "min_highs": {
                "expected_rows": min_oracle.exact_model.n_row,
                "expected_cols": min_oracle.exact_model.n_col,
                "expected_nnz": int(min_oracle.exact_model.A.nnz),
                "expected_exact_model_sha256": min_oracle.exact_model_digest,
                "expected_proof_model_sha256": min_oracle.proof_model_digest,
                "expected_universe_sha256": min_oracle.universe_digest,
                "highspy_version": "test-highs",
                "threads": 8,
            },
            "high_scip": {
                "expected_rows": 1,
                "expected_cols": 1,
                "expected_nnz": 1,
                "expected_exact_model_sha256": "1" * 64,
                "expected_proof_model_sha256": "2" * 64,
                "expected_universe_sha256": "3" * 64,
                "expected_cut_sha256": "4" * 64,
                "expected_mps_sha256": "5" * 64,
                "expected_cut_count": 1,
                "expected_unique_anchor_count": 1,
                "workers": 8,
                "threads_per_worker": 1,
                "pyscipopt_version": "6.2.1",
                "scip_version": [10, 0, 2],
            },
        }
    }
    contract = SimpleNamespace(
        seeds=[
            EvidenceSeed("min", min_selected, {}),
            EvidenceSeed("high", high_selected, {}),
        ],
        near_optimal_gap=0.005,
        candidate_set="synthetic",
        source_run_id="parent",
    )
    hardware = SimpleNamespace(
        solver_threads=8,
        portfolio_workers=8,
        threads_per_portfolio_worker=1,
        gpu_backend="none",
        to_dict=lambda: {
            "solver_threads": 8,
            "portfolio_workers": 8,
            "threads_per_portfolio_worker": 1,
            "gpu_backend": "none",
        },
    )

    root = tmp_path
    output_root = root / "outputs" / "stage4_2d"
    run_dir = output_root / "runs" / "test-run"
    report_dir = root / "reports" / "stage4_2d" / "test-run"
    run_dir.mkdir(parents=True)
    running = run_dir / ".RUNNING"
    failed = run_dir / ".FAILED"
    committed = run_dir / ".COMMITTED"
    running.write_text("running\n", encoding="utf-8")

    result = runner._run_formal_oracle_pipeline(
        root=root,
        config=config,
        run_mode="official",
        is_official=True,
        hardware=hardware,
        process_priority_status={"applied": True},
        fingerprint="f" * 64,
        fingerprint_payload={"test": True},
        run_id="test-run",
        output_root=output_root,
        run_dir=run_dir,
        report_dir=report_dir,
        bcfg_path=root / "base.yml",
        ccfg_path=root / "cert.yml",
        dcfg_path=root / "equity.yml",
        formulation=formulation,
        contract=contract,
        greedy_population=10.0,
        parent_before={},
        authoritative_inputs_before=[],
        stage5_before={"records": [], "count": 0},
        source_before=[],
        evidence_records=[],
        regression={"passed_count": 1},
        regression_passed=True,
        running=running,
        failed=failed,
        committed=committed,
    )

    assert result["passed"] is False
    assert result["decision"] == "FAIL_STAGE4_2D_EQUITY_FRONT_STAGES_UNCERTIFIED"
    assert not running.exists()
    assert failed.is_file()
    assert not committed.exists()
    assert (run_dir / "02_high_threshold_oracle" / "PREREQUISITE_SKIPPED.json").is_file()
    assert not (output_root / "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json").exists()
    latest_failed = json.loads(
        (output_root / "LATEST_FAILED_STAGE4_2D_EQUITY_RUN.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest_failed["run_id"] == "test-run"
    gate = json.loads(
        (run_dir / "04_quality_and_provenance" / "quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["passed"] is False
    assert gate["computational_certified"] is False
    assert gate["promotable_scope"] == "NONE"
