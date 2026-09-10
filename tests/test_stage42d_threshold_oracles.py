from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
import mediroad.stage4_2d_equity_certification.threshold_oracles as threshold_module
from mediroad.stage4_2d_equity_certification.threshold_oracles import (
    MpsArtifact,
    OFFICIAL_MIN_HIGHS_OPTIONS,
    build_high_scip_oracle,
    build_min_highs_oracle,
    certificate_from_threshold_oracle,
    run_min_highs_oracle,
    run_scip_portfolio,
    seed_set_digest,
    selection_digest,
    validate_cut_seed_set,
    validate_pinned_selection,
    validate_scip_portfolio_outcomes,
    write_mps,
)
from mediroad.stage4_2d_equity_certification.threshold_oracles import _require_exact_float
from mediroad.stage4_2d_equity_certification.types import EvidenceSeed


def _min_oracle():
    formulation = make_synthetic_formulation()
    selected = np.asarray([0, 3], dtype=np.int64)
    metrics = formulation.metrics(selected)
    oracle = build_min_highs_oracle(
        formulation,
        total_population_floor=10.0,
        threshold=0.5,
        incumbent_value=metrics["min_sigungu_coverage_ratio"],
        retained_floor=10.0,
        incumbent_selection=selected,
        expected_selection_digest=selection_digest(selected),
    )
    return formulation, oracle


def _high_oracle():
    formulation = make_synthetic_formulation()
    selected = np.asarray([0, 3], dtype=np.int64)
    metrics = formulation.metrics(selected)
    seeds = [
        EvidenceSeed("seed-a", np.asarray([0, 3], dtype=np.int64), {}),
        EvidenceSeed("seed-b", np.asarray([1, 3], dtype=np.int64), {}),
    ]
    oracle = build_high_scip_oracle(
        formulation,
        total_population_floor=10.0,
        min_sigungu_floor=0.2,
        threshold=30.0,
        incumbent_value=metrics["high_need_population"],
        retained_floor=0.2,
        incumbent_selection=selected,
        expected_selection_digest=selection_digest(selected),
        cut_seeds=seeds,
        expected_cut_seed_count=len(seeds),
        expected_cut_seed_digest=seed_set_digest(seeds),
        anchor_sizes=range(3),
        max_cuts=100,
    )
    return formulation, oracle, seeds


def test_selection_and_seed_digests_are_order_independent_but_multiplicity_sensitive():
    assert selection_digest(np.asarray([3, 0])) == selection_digest(np.asarray([0, 3]))
    with pytest.raises(ValueError, match="duplicate"):
        selection_digest(np.asarray([0, 0]))
    seeds_a = [
        EvidenceSeed("a", np.asarray([0, 3]), {}),
        EvidenceSeed("b", np.asarray([1, 3]), {}),
    ]
    seeds_b = list(reversed(seeds_a))
    assert seed_set_digest(seeds_a) == seed_set_digest(seeds_b)
    assert seed_set_digest(seeds_a) != seed_set_digest([seeds_a[0]])


def test_pinned_selection_fails_closed_on_raw_shape_count_range_and_sha():
    formulation = make_synthetic_formulation()
    selected = np.asarray([0, 3])
    metrics = formulation.metrics(selected)
    common = {
        "expected_digest": selection_digest(selected),
        "metric": "min_sigungu_coverage",
        "expected_value": metrics["min_sigungu_coverage_ratio"],
        "total_population_floor": 10.0,
        "threshold": 0.5,
    }
    for invalid in (
        np.asarray([[0, 3]]),
        np.asarray([0.0, 3.0]),
        np.asarray([0]),
        np.asarray([0, 99]),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            validate_pinned_selection(formulation, invalid, **common)
    with pytest.raises(RuntimeError, match="digest changed"):
        validate_pinned_selection(
            formulation,
            selected,
            **{**common, "expected_digest": "0" * 64},
        )


def test_pinned_high_metric_uses_exact_small_absolute_tolerance():
    formulation = make_synthetic_formulation()
    selected = np.asarray([0, 3])
    digest = selection_digest(selected)
    with pytest.raises(RuntimeError, match="value changed"):
        validate_pinned_selection(
            formulation,
            selected,
            expected_digest=digest,
            metric="high_need_population",
            expected_value=25.0 + 2e-8,
            total_population_floor=10.0,
            min_sigungu_floor=0.2,
            threshold=30.0,
        )


def test_frozen_contract_float_pin_rejects_a_single_ulp_change():
    _require_exact_float("gap", 0.005, 0.005)
    with pytest.raises(RuntimeError, match="Frozen gap changed"):
        _require_exact_float("gap", math.nextafter(0.005, math.inf), 0.005)


def test_min_builder_is_zero_objective_binary_x_continuous_y_and_row_only_relaxed():
    _, oracle = _min_oracle()
    assert np.all(oracle.exact_model.c == 0.0)
    assert np.all(oracle.exact_model.integrality[oracle.exact_model.x_slice] == 1)
    assert np.all(oracle.exact_model.integrality[oracle.exact_model.y_slice] == 0)
    np.testing.assert_array_equal(oracle.model.col_lower, oracle.exact_model.col_lower)
    np.testing.assert_array_equal(oracle.model.col_upper, oracle.exact_model.col_upper)
    assert np.all(oracle.model.row_lower <= oracle.exact_model.row_lower)
    assert np.all(oracle.model.row_upper >= oracle.exact_model.row_upper)
    manifest = oracle.metadata["proof_relaxation"]
    assert manifest["absolute_margin"] == 2e-6
    assert manifest["exact_column_bounds_restored"] is True


def test_high_builder_requires_pinned_seed_set_and_outward_all_binary_cuts():
    formulation, oracle, seeds = _high_oracle()
    assert np.all(oracle.model.integrality == 1)
    np.testing.assert_array_equal(oracle.model.col_lower, oracle.exact_model.col_lower)
    np.testing.assert_array_equal(oracle.model.col_upper, oracle.exact_model.col_upper)
    assert oracle.metadata["binary_coverage_theorem"][
        "projected_venue_feasible_region_preserved"
    ] is True
    assert oracle.metadata["cut_generation"]["outward_safe"] is True
    assert oracle.cut_seed_digest == seed_set_digest(seeds)
    with pytest.raises(RuntimeError, match="count changed"):
        validate_cut_seed_set(
            formulation,
            seeds,
            expected_count=3,
            expected_digest=seed_set_digest(seeds),
        )
    with pytest.raises(RuntimeError, match="digest changed"):
        validate_cut_seed_set(
            formulation,
            seeds,
            expected_count=2,
            expected_digest="f" * 64,
        )


class _FakeHighs:
    def __init__(self, *, status: str = "Infeasible", solution: np.ndarray | None = None):
        self.options = {}
        self.model = None
        self.status = status
        self.solution = solution

    def setOptionValue(self, name, value):
        self.options[name] = value
        return "OK"

    def getOptionValue(self, name):
        return "OK", self.options[name]

    def getModel(self):
        model = self.model
        A = model.A.tocsc(copy=False)
        lp = SimpleNamespace(
            num_col_=model.n_col,
            num_row_=model.n_row,
            col_cost_=model.c.tolist(),
            col_lower_=model.col_lower.tolist(),
            col_upper_=model.col_upper.tolist(),
            row_lower_=model.row_lower.tolist(),
            row_upper_=model.row_upper.tolist(),
            integrality_=model.integrality.tolist(),
            a_matrix_=SimpleNamespace(
                start_=A.indptr.tolist(),
                index_=A.indices.tolist(),
                value_=A.data.tolist(),
            ),
        )
        return SimpleNamespace(lp_=lp)

    def run(self):
        return "OK"

    def getInfo(self):
        return SimpleNamespace(mip_dual_bound=None, mip_node_count=17)

    def getSolution(self):
        values = [] if self.solution is None else self.solution.tolist()
        return SimpleNamespace(value_valid=self.solution is not None, col_value=values)

    def getModelStatus(self):
        return self.status

    def modelStatusToString(self, status):
        return status

    def version(self):
        return "1.15.1"

    def getRunTime(self):
        return 1.25


def _fake_pass_model(highs, model):
    highs.model = model


def test_min_highs_in_memory_requires_exact_option_readback_and_certifies(tmp_path: Path):
    formulation, oracle = _min_oracle()
    fake = _FakeHighs()
    result = run_min_highs_oracle(
        oracle,
        formulation,
        output_dir=tmp_path,
        highs_factory=lambda: fake,
        pass_model=_fake_pass_model,
    )
    assert result.certified_threshold_infeasible
    assert result.counterexample_found is False
    assert result.mps_sha256 is None
    assert result.requested_options["random_seed"] == 104752
    for name, value in OFFICIAL_MIN_HIGHS_OPTIONS.items():
        assert result.readback_options[name] == value
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.log_paths)
    certificate = certificate_from_threshold_oracle(result)
    assert certificate.certified
    assert certificate.certificate == "THRESHOLD_INFEASIBILITY_HIGHS"


def test_min_highs_fails_on_option_readback_or_status_contradiction(tmp_path: Path):
    formulation, oracle = _min_oracle()

    class BadReadback(_FakeHighs):
        def getOptionValue(self, name):
            status, value = super().getOptionValue(name)
            return status, value + 1 if name == "threads" else value

    with pytest.raises(RuntimeError, match="readback mismatch"):
        run_min_highs_oracle(
            oracle,
            formulation,
            output_dir=tmp_path / "bad-option",
            highs_factory=BadReadback,
            pass_model=_fake_pass_model,
        )
    full = formulation.complete_solution(oracle.model, np.asarray([1, 3]))
    with pytest.raises(RuntimeError, match="INFEASIBLE with a valid incumbent"):
        run_min_highs_oracle(
            oracle,
            formulation,
            output_dir=tmp_path / "contradiction",
            highs_factory=lambda: _FakeHighs(status="Infeasible", solution=full),
            pass_model=_fake_pass_model,
        )


def test_min_highs_validated_counterexample_is_not_a_certificate(tmp_path: Path):
    formulation, oracle = _min_oracle()
    full = formulation.complete_solution(oracle.model, np.asarray([1, 3]))
    result = run_min_highs_oracle(
        oracle,
        formulation,
        output_dir=tmp_path,
        highs_factory=lambda: _FakeHighs(status="Optimal", solution=full),
        pass_model=_fake_pass_model,
    )
    assert result.counterexample_found
    assert not result.certified_threshold_infeasible
    assert not certificate_from_threshold_oracle(result).certified


class _FakeMpsBackend:
    def __init__(self):
        self.store = {}
        store = self.store

        class FakeMpsHighs:
            @staticmethod
            def resetGlobalScheduler(_force):
                return None

            def __init__(self):
                self.model = None

            def writeModel(self, path):
                Path(path).write_bytes(b"synthetic proof mps\n")
                return "OK"

            def readModel(self, _path):
                self.model = store["model"]
                return "OK"

            def getModel(self):
                fake = _FakeHighs()
                fake.model = self.model
                raw = fake.getModel()
                raw.lp_.col_names_ = [f"c{index}" for index in range(self.model.n_col)]
                raw.lp_.row_names_ = [f"r{index}" for index in range(self.model.n_row)]
                return raw

        self.HighsClass = FakeMpsHighs

    def _pass_model(self, highs, model):
        self.store["model"] = model
        highs.model = model


def test_mps_writer_records_bytes_to_mib_and_sufficient_transport(tmp_path: Path):
    _, oracle, _ = _high_oracle()
    artifact = write_mps(
        oracle,
        tmp_path / "proof.mps",
        backend=_FakeMpsBackend(),
    )
    assert artifact.transport_sufficient
    assert artifact.size_mib == artifact.size_bytes / (2**20)
    assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()


def _mps_artifact(tmp_path: Path, oracle) -> MpsArtifact:
    path = tmp_path / "proof.mps"
    path.write_bytes(b"proof")
    size = path.stat().st_size
    return MpsArtifact(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=size,
        size_mib=size / (2**20),
        rows=oracle.model.n_row,
        cols=oracle.model.n_col,
        nnz=int(oracle.model.A.nnz),
        transport_sufficient=True,
        transport_audit={"sufficient": True},
    )


def _scip_outcome(
    tmp_path: Path,
    artifact: MpsArtifact,
    seed: int,
    *,
    status: str,
    selected: list[int] | None = None,
    primal: float | None = None,
    dual: float | None = None,
    gap: float | None = None,
    time_limit_sec: float = 900.0,
) -> dict:
    log = tmp_path / f"seed-{seed}.log"
    log.write_text("SCIP mock log\n", encoding="utf-8")
    params = {
        "limits/time": time_limit_sec,
        "limits/gap": 0.0,
        "limits/memory": 2800.0,
        "parallel/maxnthreads": 1,
        "lp/threads": 1,
        "randomization/randomseedshift": seed,
    }
    memory_bytes = 64 * 2**20
    return {
        "seed": seed,
        "status": status,
        "has_solution": selected is not None,
        "primal_bound": primal,
        "dual_bound": dual,
        "gap": gap,
        "nodes": 10,
        "lp_iterations": 20,
        "solving_time_sec": 1.0,
        "memory_used_bytes": memory_bytes,
        "memory_used_mib": memory_bytes / (2**20),
        "selected_indices": selected,
        "column_map_verified": True,
        "all_columns_binary": True,
        "requested_parameters": params,
        "readback_parameters": dict(params),
        "pyscipopt_version": "6.2.1",
        "scip_model_version": "10.0",
        "scip_version_components": {"major": 10, "minor": 0, "tech": 2},
        "log_path": str(log),
        "mps_sha256": artifact.sha256,
    }


def test_scip_portfolio_requires_complete_seeds_and_all_infeasible_proofs(
    tmp_path: Path,
):
    formulation, oracle, _ = _high_oracle()
    artifact = _mps_artifact(tmp_path, oracle)
    outcomes = [
        _scip_outcome(tmp_path, artifact, 11, status="infeasible"),
        _scip_outcome(tmp_path, artifact, 23, status="infeasible"),
    ]
    result = validate_scip_portfolio_outcomes(
        oracle,
        formulation,
        outcomes,
        expected_seeds=[11, 23],
        mps=artifact,
    )
    assert result.certified_threshold_infeasible
    assert certificate_from_threshold_oracle(result).certificate == (
        "THRESHOLD_INFEASIBILITY_SCIP_PORTFOLIO"
    )
    mixed = [outcomes[0], _scip_outcome(tmp_path, artifact, 23, status="timelimit")]
    mixed_result = validate_scip_portfolio_outcomes(
        oracle,
        formulation,
        mixed,
        expected_seeds=[11, 23],
        mps=artifact,
    )
    assert not mixed_result.certified_threshold_infeasible
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_scip_portfolio_outcomes(
            oracle,
            formulation,
            outcomes[:1],
            expected_seeds=[11, 23],
            mps=artifact,
        )


def test_scip_portfolio_validates_correct_metric_keys_and_fails_contradictions(tmp_path: Path):
    formulation, oracle, _ = _high_oracle()
    artifact = _mps_artifact(tmp_path, oracle)
    counterexample = _scip_outcome(
        tmp_path,
        artifact,
        11,
        status="optimal",
        selected=[1, 3],
        primal=45.0,
        dual=45.0,
    )
    timed = _scip_outcome(tmp_path, artifact, 23, status="timelimit")
    result = validate_scip_portfolio_outcomes(
        oracle,
        formulation,
        [counterexample, timed],
        expected_seeds=[11, 23],
        mps=artifact,
    )
    assert result.counterexample_found and not result.certified_threshold_infeasible
    metrics = result.outcomes[0]["recomputed_metrics"]
    assert "unique_elderly_population" in metrics
    assert "min_sigungu_coverage_ratio" in metrics

    infeasible = _scip_outcome(tmp_path, artifact, 23, status="infeasible")
    with pytest.raises(RuntimeError, match="contradiction"):
        validate_scip_portfolio_outcomes(
            oracle,
            formulation,
            [counterexample, infeasible],
            expected_seeds=[11, 23],
            mps=artifact,
        )


def test_scip_portfolio_fails_closed_on_invalid_incumbent_memory_and_readback(tmp_path: Path):
    formulation, oracle, _ = _high_oracle()
    artifact = _mps_artifact(tmp_path, oracle)
    invalid = _scip_outcome(
        tmp_path,
        artifact,
        11,
        status="optimal",
        selected=[0],
        primal=25.0,
        dual=25.0,
    )
    other = _scip_outcome(tmp_path, artifact, 23, status="timelimit")
    with pytest.raises(RuntimeError, match="invalid incumbent"):
        validate_scip_portfolio_outcomes(
            oracle,
            formulation,
            [invalid, other],
            expected_seeds=[11, 23],
            mps=artifact,
        )
    bad_memory = _scip_outcome(tmp_path, artifact, 11, status="infeasible")
    bad_memory["memory_used_mib"] += 1.0
    with pytest.raises(RuntimeError, match="bytes to MiB"):
        validate_scip_portfolio_outcomes(
            oracle,
            formulation,
            [bad_memory, other],
            expected_seeds=[11, 23],
            mps=artifact,
        )
    bad_readback = _scip_outcome(tmp_path, artifact, 11, status="infeasible")
    bad_readback["readback_parameters"]["limits/time"] = 299.0
    with pytest.raises(RuntimeError, match="readback"):
        validate_scip_portfolio_outcomes(
            oracle,
            formulation,
            [bad_readback, other],
            expected_seeds=[11, 23],
            mps=artifact,
        )


def test_scip_portfolio_normalizes_finite_infinity_sentinels(tmp_path: Path):
    formulation, oracle, _ = _high_oracle()
    artifact = _mps_artifact(tmp_path, oracle)
    outcomes = [
        _scip_outcome(
            tmp_path,
            artifact,
            seed,
            status="infeasible",
            primal=-1e20,
            dual=-1e20,
            gap=1e20,
        )
        for seed in (11, 23)
    ]
    result = validate_scip_portfolio_outcomes(
        oracle,
        formulation,
        outcomes,
        expected_seeds=[11, 23],
        mps=artifact,
    )
    assert result.certified_threshold_infeasible
    assert result.best_bound is None
    assert all(outcome["primal_bound"] is None for outcome in result.outcomes)
    assert all(outcome["dual_bound"] is None for outcome in result.outcomes)
    assert all(outcome["gap"] is None for outcome in result.outcomes)


def test_scip_portfolio_binds_worker_options_to_requested_contract(tmp_path: Path):
    formulation, oracle, _ = _high_oracle()
    artifact = _mps_artifact(tmp_path, oracle)
    outcomes = [
        _scip_outcome(tmp_path, artifact, seed, status="infeasible", time_limit_sec=300.0)
        for seed in (11, 23)
    ]
    requested = {
        "seeds": [11, 23],
        "workers": 2,
        "threads_per_worker": 1,
        "time_limit_sec_per_seed": 900.0,
        "memory_limit_mib_per_worker": 2800.0,
        "require_all_workers_infeasible": True,
    }
    with pytest.raises(RuntimeError, match="portfolio contract"):
        validate_scip_portfolio_outcomes(
            oracle,
            formulation,
            outcomes,
            expected_seeds=[11, 23],
            mps=artifact,
            requested_options=requested,
        )


def test_run_scip_portfolio_atomically_persists_each_worker_outcome(
    tmp_path: Path, monkeypatch
):
    formulation, oracle, _ = _high_oracle()
    artifact = _mps_artifact(tmp_path, oracle)
    output_dir = tmp_path / "portfolio"
    output_dir.mkdir()
    outcomes = {
        seed: _scip_outcome(output_dir, artifact, seed, status="infeasible")
        for seed in (11, 23)
    }

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class ImmediateExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, _function, payload):
            return ImmediateFuture(outcomes[int(payload["seed"])])

    monkeypatch.setattr(threshold_module, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(threshold_module, "as_completed", lambda futures: list(futures))
    result = run_scip_portfolio(
        oracle,
        formulation,
        artifact,
        output_dir=output_dir,
        seeds=[11, 23],
        workers=2,
    )
    assert result.certified_threshold_infeasible
    for seed in (11, 23):
        path = output_dir / f"worker_seed_{seed}.json"
        assert path.is_file() and path.stat().st_size > 0
        assert result.evidence["worker_result_artifacts"][str(seed)]["sha256"] == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
    assert (output_dir / "portfolio_progress.json").is_file()
