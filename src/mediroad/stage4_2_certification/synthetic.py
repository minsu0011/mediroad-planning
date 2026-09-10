from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from mediroad.stage4_2_certification.formulation import CertificationFormulation


def make_synthetic_formulation() -> CertificationFormulation:
    candidates = pd.DataFrame(
        {
            "venue_id": [f"V{i}" for i in range(5)],
            "admin_code": ["A0", "A1", "A2", "A3", "A4"],
            "sigungu": ["S1", "S1", "S1", "S2", "S2"],
            "cluster_id": ["C0", "C1", "C2", "C3", "C4"],
            "known_overlap": ["NO", "NO", "NO", "NO", "NO"],
            "travel_minutes": [5.0, 6.0, 4.0, 8.0, 7.0],
            "candidate_priority_tiebreak": [5, 4, 3, 2, 1],
        }
    )
    coverers = [
        np.array([0, 1], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        np.array([3, 4], dtype=np.int32),
        np.array([0, 4], dtype=np.int32),
        np.array([], dtype=np.int32),
    ]
    patterns = SimpleNamespace(
        n_candidates=5,
        n_patterns=6,
        pattern_coverers=coverers,
        population=np.array([10.0, 20.0, 15.0, 25.0, 5.0, 7.0]),
        need_weighted=np.array([5.0, 18.0, 9.0, 20.0, 1.0, 3.0]),
        high_need_population=np.array([0.0, 20.0, 0.0, 25.0, 0.0, 0.0]),
        sigungu=np.array(["S1", "S1", "S2", "S2", "S1", "S2"]),
    )
    base = {
        "optimization": {
            "visit_count": 2,
            "known_overlap_penalty": 1000.0,
            "travel_penalty_scale": 1.0,
            "visit_location_deviation_penalty": 100.0,
            "deterministic_tiebreak_scale": 1e-6,
            "objective_retention": {
                "total_population": 0.99,
                "need_weighted": 0.99,
                "high_need": 0.99,
                "min_sigungu_coverage": 1.0,
            },
            "balanced_min_efficiency_fraction": 0.80,
            "equity_min_efficiency_fraction": 0.60,
        },
        "candidate_expansion": {
            "max_visits_per_admin": 2,
            "max_visits_per_sigungu": 2,
            "one_per_coverage_cluster": True,
        },
    }
    cert = {
        "formulation": {
            "one_link_pattern_formulation": True,
            "min_sigungu_integer_scale": 1_000_000,
        },
        "hardware": {"gpu_seed_search": False},
    }
    return CertificationFormulation(candidates, patterns, base, cert, candidate_set="synthetic")
