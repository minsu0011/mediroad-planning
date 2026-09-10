from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


FROZEN_GAP = 0.005
FROZEN_C_RUN_ID = "stage4_2c_certification_20260820T151657Z_56c5f1bad35c"
FROZEN_C_INVENTORY_SHA256 = (
    "c1198cd711730871538ff536614f724ea64ee2634ced21d1aa9d144b00a74362"
)
FROZEN_D_RUN_ID = "stage4_2d_equity_20260821T014952Z_cc9fd641ddf0"
FROZEN_D_POINTER_SHA256 = (
    "4bb004c172098534afecddf11a2cd45657a6cbec4f857ce5fcfb8f26b5e251b0"
)
PENDING_STAGE42E = "PENDING_AFTER_STAGE4_2E_OFFICIAL_SUCCESS"

EXPECTED_C_ARTIFACTS = {
    "ARTIFACT_INVENTORY.csv": FROZEN_C_INVENTORY_SHA256,
    "metadata.json": "d420d1d90439b22d473c4bc164dbf6198fd9f03331465db6c9cbc2e32adfc259",
    "candidate_sets/top3/stages__efficiency.json": (
        "4f19d3423288d6227a43a6ca5e9dfa2c6009107471e7e1bf4b0fbec875a51a0b"
    ),
    "candidate_sets/top3/stages__balanced.json": (
        "8e44a21594b143c517a799bce07d78780616454cc354c52c816a5c037b2ffafd"
    ),
    "candidate_sets/top3/plan__efficiency.csv": (
        "96d0f6a34976dfab9a580b02cf5e51a395d904275564950dbcc2d886fc6d267e"
    ),
    "candidate_sets/top3/plan__balanced.csv": (
        "da6d4747e4f0a08c13eb5038d2843e7bfc0b6a640968de9067773ec59b064cf8"
    ),
}

EXPECTED_C_SOURCE_PREIMAGE = {
    "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820/src/mediroad/stage4_2_certification/types.py": (
        "2c4616c29756c4bdd5870d13544f2b6a24a315bb818250143ed68aa49686895c"
    ),
    "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820/src/mediroad/stage4_2_certification/adapter.py": (
        "9ddda07ee3854bf2b9a12417a48b384a82561b486c3f4d4ccc61b54d6ce96ac5"
    ),
    "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820/src/mediroad/stage4_2_certification/formulation.py": (
        "b8c0a426b8ef163ceafb5e1f699a8fdee3fbd20f6849c0be64f5523ef6da1122"
    ),
    "MEDIROAD_STAGE4_2C_CERTIFICATION_BOTTLENECK_PATCH_20260820/configs/model_v1/stage4_2_certification_diagnostic.yaml": (
        "718aff8474f328f5174364cf241352af7434a0c46950e40914721d0e4c401830"
    ),
    "configs/model_v1/stage4_2.yaml": (
        "1d9ee501bbd15babf32e67d82d19a5ab8de5a7c4810e200ea1a55af3094782db"
    ),
}

EXPECTED_C_PARENT_POINTERS = {
    "outputs/model_v1/07_stage3/CURRENT_STAGE3_RUN.json": (
        "5a5fe659d038747f4d79e8dd5046c0da40bd6dd79359a8c622a175be08d62e70"
    ),
    "outputs/model_v1/09_stage4_finalization/CURRENT_STAGE4_FINALIZATION_RUN.json": (
        "eb87049591e0acbaab581f57b1e63aba19dbe9179cf650dfd2b05933e5fad790"
    ),
    "outputs/model_v1/10_stage4_2/CURRENT_STAGE4_2_RUN.json": (
        "ea2149ea1e921f1c5f2c087d958ef75b096020d120e991c154cf2ce906ea7297"
    ),
}

EXPECTED_D_ARTIFACTS = {
    "CURRENT_STAGE4_2D_EQUITY_CERTIFICATION_RUN.json": FROZEN_D_POINTER_SHA256,
    "ARTIFACT_INVENTORY.csv": (
        "a9a9e0ad28c53a254424b2e17c129c85430e84039b52f3919f1b484a88ab712b"
    ),
    "01_min_threshold_oracle/certificate.json": (
        "c00f4f3b870a665ec855c69679aabddf1b9e262bf68690f1b12ccbb8e5b0635b"
    ),
    "02_high_threshold_oracle/certificate.json": (
        "550980f4700d247990021170b32aff459775a8cdf1360fc0eefe04940cbfd8f1"
    ),
    "04_quality_and_provenance/quality_gate.json": (
        "1f81f5d21b5574e2bc8ebef8836d48c88ba66ba77d7c52816239382be3c9ef51"
    ),
}

EXPECTED_E_PARENT_KEYS = {
    "CURRENT_STAGE4_2E_EQUITY_TAIL_CERTIFICATION_RUN.json",
    "ARTIFACT_INVENTORY.csv",
    "01_need_weighted/certificate.json",
    "02_cost/certificate.json",
    "04_quality_and_provenance/quality_gate.json",
}

# These are exact, current-data compatibility seals.  They were computed by
# rebuilding each historical Efficiency/Balanced stage with its original
# binary64 floor list and then checking the historical selection in that model.
EXPECTED_ADOPTION_STAGES: dict[str, dict[str, Any]] = {
    "efficiency:1": {
        "objective": "total_population",
        "sense": "max",
        "model_sha256": "26d6283e2e25845fe3a96eb08b39d8485e2ddbd0c7d1811793609a945d894a99",
        "universe_sha256": "dcc91d8a844a78d7014480d7a7e2a9319c262b6659b12573fe80efe805477b2c",
        "selection_sha256": "056d1566a2147f439949b0a7bfeba57ecc059c7ac37f2245a217726db7a5db88",
        "rows": 5488,
        "cols": 5694,
        "nnz": 154693,
        "incumbent": 280544.1058422313,
        "best_bound": 280544.10584223183,
        "relative_gap": 1.867331864444207e-15,
    },
    "efficiency:2": {
        "objective": "need_weighted",
        "sense": "max",
        "model_sha256": "d1e7821dd39e09cef969c3cfda7b3c9bbae4fbab6bfca7d044ddf7f1d0244242",
        "universe_sha256": "582ee1494bd30e941e25776a0b9cd86f2e4cd0c82a1a1db14cd2a32a9b0542f6",
        "selection_sha256": "0999ac4560d08d2cef8ed638f12fc8330a9cb0c70b2980eff05ba9c3805b29cc",
        "rows": 5489,
        "cols": 5694,
        "nnz": 159935,
        "incumbent": 112064.41199338099,
        "best_bound": 112064.41199338096,
        "relative_gap": 0.0,
    },
    "efficiency:3": {
        "objective": "min_sigungu_coverage",
        "sense": "max",
        "model_sha256": "71307788a188e25c8cd17cf3515452255befe6f073b415b1c53671f8a8756d69",
        "universe_sha256": "016e8318aac687783d72eac3a446b571c507647f62e6354202dfd35b5591ff00",
        "selection_sha256": "7c30ec001615f58fa5c5f01ceb2d4a3f67821db6c58e9c0681415be5f17d5e89",
        "rows": 5501,
        "cols": 5695,
        "nnz": 170430,
        "incumbent": 0.3242960948937016,
        "best_bound": 0.32590083959654864,
        "relative_gap": 0.004948393545636299,
    },
    "efficiency:4": {
        "objective": "cost",
        "sense": "min",
        "model_sha256": "ad3c655feae0a38ea8f6d06678eec59ad9efd059d85875a70b07aa527d371597",
        "universe_sha256": "99ae9ccda5e77645b992c274ece2788611ed27872c1871472610111515a305bb",
        "selection_sha256": "da36ca1127b0be67ed16a7d0c1710156ede35fdde3eea37e633bb30aeb123574",
        "rows": 5523,
        "cols": 5749,
        "nnz": 170970,
        "incumbent": 981.8231838181819,
        "best_bound": 978.5529684321525,
        "relative_gap": 0.0033307579612368793,
    },
    "balanced:1": {
        "objective": "need_weighted",
        "sense": "max",
        "model_sha256": "988501816231e1bbb0be0b5ae6fb1b4ae4511018af987d2a17d9c7fafc6f4852",
        "universe_sha256": "dcc91d8a844a78d7014480d7a7e2a9319c262b6659b12573fe80efe805477b2c",
        "selection_sha256": "0999ac4560d08d2cef8ed638f12fc8330a9cb0c70b2980eff05ba9c3805b29cc",
        "rows": 5488,
        "cols": 5694,
        "nnz": 154693,
        "incumbent": 112064.41199338099,
        "best_bound": 112064.41199338099,
        "relative_gap": 0.0,
    },
    "balanced:2": {
        "objective": "min_sigungu_coverage",
        "sense": "max",
        "model_sha256": "c2600fa70225a2ec19758d28321a5382152db034ad7d07d779fd241f56144ba7",
        "universe_sha256": "669402dfc28d4342273d0b00e99f255d3c5f5cf7c6fe8c704917442c631cb39f",
        "selection_sha256": "20b4ef57575a113941f7c9cf97be96ad2de86033e837b61774789038dde7341f",
        "rows": 5500,
        "cols": 5695,
        "nnz": 165188,
        "incumbent": 0.33431024324673114,
        "best_bound": 0.33597989252702304,
        "relative_gap": 0.004994310865490424,
    },
    "balanced:3": {
        "objective": "total_population",
        "sense": "max",
        "model_sha256": "0f2bea8255fc95301434ba6d511ac9a9c71c53d41353abb396d5f9a1189c5726",
        "universe_sha256": "ff209a0b08f16eb7c75ac8001fea85b0e9acbf93825b4060a9a9b16628f240a8",
        "selection_sha256": "258e16aafe72ef5194efa2ca8d54ac9331d0a2a3bf5b4753c7be750551783b4e",
        "rows": 5500,
        "cols": 5694,
        "nnz": 165177,
        "incumbent": 277818.4929508749,
        "best_bound": 278896.8517221873,
        "relative_gap": 0.003881522644005847,
    },
    "balanced:4": {
        "objective": "cost",
        "sense": "min",
        "model_sha256": "143b25fe7c23b51a6584a6b9531b543479b44650688f7181825d55050ad8070a",
        "universe_sha256": "1262a5478c261e8f7dc79f4793dab9829f283134afb8cf10c30e414196400897",
        "selection_sha256": "b5fb44b966264383a88953ee1a50c90d582439d67572769889b110685c2cbad3",
        "rows": 5523,
        "cols": 5749,
        "nnz": 170970,
        "incumbent": 1018.1862831818181,
        "best_bound": 1013.3837303280469,
        "relative_gap": 0.004716772297072488,
    },
}


class PendingStage42E(RuntimeError):
    """The template is intentionally non-runnable until E is officially pinned."""


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same(value: Any, expected: float) -> bool:
    return bool(
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value).hex() == float(expected).hex()
    )


def _require_exact_sha_map(value: Any, expected: dict[str, str], label: str) -> None:
    if not isinstance(value, dict) or value != expected:
        raise ValueError(f"{label} must preserve its exact path/SHA trust anchors")


def validate_config(cfg: dict[str, Any]) -> None:
    required = {
        "version",
        "mode",
        "stage5_started",
        "paths",
        "contract",
        "preferred_stage42c",
        "stage42d_parent",
        "stage42e_parent",
        "adoption_stage_pins",
        "gate",
    }
    if set(cfg) != required:
        raise ValueError(f"Stage4.2F config keys must be exactly {sorted(required)}")
    if cfg.get("version") != "MEDIROAD_STAGE4_2F_TOP3_SEMANTIC_AGGREGATE_V1":
        raise ValueError("Stage4.2F config version changed")
    if cfg.get("mode") != "official" or cfg.get("stage5_started") is not False:
        raise ValueError("Stage4.2F is official-only and stage5_started must remain false")
    expected_paths = {
        "output_root": "outputs/model_v1/10_stage4_2f_top3_aggregate",
        "report_root": "reports/model_v1/stage4_2f_top3_aggregate",
        "lock": "outputs/model_v1/.stage4_2f_top3_aggregate_writer.lock",
    }
    if cfg.get("paths") != expected_paths:
        raise ValueError("Stage4.2F output/report/lock paths changed")

    contract = cfg["contract"]
    exact_contract = {
        "candidate_set": "top3",
        "candidate_count": 452,
        "pattern_count": 5242,
        "visit_count": 20,
        "required_scenarios": ["efficiency", "balanced", "equity"],
        "compatibility_adoption_scenarios": ["efficiency", "balanced"],
        "semantic_label": "CERTIFIED_NEAR_OPTIMAL",
        "scope": "TOP3_REQUIRED_SCENARIOS_ONLY",
        "candidate_expansion_enabled": False,
    }
    if not isinstance(contract, dict) or set(contract) != {
        *exact_contract,
        "relative_gap",
    }:
        raise ValueError("Stage4.2F contract changed")
    for key, expected in exact_contract.items():
        if contract.get(key) != expected or type(contract.get(key)) is not type(expected):
            raise ValueError(f"Stage4.2F contract.{key} changed")
    if not _same(contract.get("relative_gap"), FROZEN_GAP):
        raise ValueError("Stage4.2F relative gap changed")

    preferred = cfg["preferred_stage42c"]
    if not isinstance(preferred, dict) or set(preferred) != {
        "run_id",
        "artifact_sha256",
        "source_preimage_sha256",
        "historical_parent_pointer_sha256",
    }:
        raise ValueError("preferred_stage42c schema changed")
    if preferred.get("run_id") != FROZEN_C_RUN_ID:
        raise ValueError("Preferred Stage4.2C run changed")
    _require_exact_sha_map(
        preferred.get("artifact_sha256"), EXPECTED_C_ARTIFACTS, "preferred_stage42c"
    )
    _require_exact_sha_map(
        preferred.get("source_preimage_sha256"),
        EXPECTED_C_SOURCE_PREIMAGE,
        "preferred_stage42c.source_preimage_sha256",
    )
    _require_exact_sha_map(
        preferred.get("historical_parent_pointer_sha256"),
        EXPECTED_C_PARENT_POINTERS,
        "preferred_stage42c.historical_parent_pointer_sha256",
    )

    parent_d = cfg["stage42d_parent"]
    if not isinstance(parent_d, dict) or set(parent_d) != {"run_id", "artifact_sha256"}:
        raise ValueError("stage42d_parent schema changed")
    if parent_d.get("run_id") != FROZEN_D_RUN_ID:
        raise ValueError("Stage4.2D parent run changed")
    _require_exact_sha_map(
        parent_d.get("artifact_sha256"), EXPECTED_D_ARTIFACTS, "stage42d_parent"
    )

    parent_e = cfg["stage42e_parent"]
    if not isinstance(parent_e, dict) or set(parent_e) != {"run_id", "artifact_sha256"}:
        raise ValueError("stage42e_parent schema changed")
    if parent_e.get("run_id") == PENDING_STAGE42E:
        raise PendingStage42E(
            "Stage4.2F is fail-closed: replace every Stage4.2E PENDING pin only after an "
            "official PASS CURRENT exists"
        )
    if not isinstance(parent_e.get("run_id"), str) or not parent_e["run_id"].startswith(
        "stage4_2e_equity_tail_"
    ):
        raise ValueError("stage42e_parent.run_id is not an official Stage4.2E run id")
    e_artifacts = parent_e.get("artifact_sha256")
    if not isinstance(e_artifacts, dict) or set(e_artifacts) != EXPECTED_E_PARENT_KEYS:
        raise ValueError("stage42e_parent must pin the exact required artifact set")
    if any(not _sha256(value) for value in e_artifacts.values()):
        raise PendingStage42E("Every Stage4.2E artifact must be replaced by an exact SHA-256")

    pins = cfg["adoption_stage_pins"]
    if pins != EXPECTED_ADOPTION_STAGES:
        raise ValueError("The exact eight-stage model/universe/selection pins changed")
    for key, expected in EXPECTED_ADOPTION_STAGES.items():
        observed = pins[key]
        for field in ("incumbent", "best_bound", "relative_gap"):
            if not _same(observed.get(field), float(expected[field])):
                raise ValueError(f"adoption_stage_pins.{key}.{field} changed")
        for field in ("rows", "cols", "nnz"):
            if type(observed.get(field)) is not int:
                raise ValueError(f"adoption_stage_pins.{key}.{field} must be an integer")

    gate = cfg["gate"]
    expected_gate = {
        "require_current_recomputation": True,
        "require_exact_eight_stage_digests": True,
        "forbid_legacy_optimal_copy": True,
        "forbid_legacy_run_pass_copy": True,
        "write_current_only_on_full_pass": True,
        "candidate_expansion_certified": False,
        "stage4_full_computational_complete": False,
        "operational_final": False,
        "stage5_release_allowed": False,
    }
    if gate != expected_gate or any(type(gate.get(key)) is not bool for key in expected_gate):
        raise ValueError("Stage4.2F gate contract changed")
