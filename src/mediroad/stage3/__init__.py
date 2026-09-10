"""Sparse spatial primitives for MEDIROAD MODEL V1 Stage 3.

The package intentionally contains no optimiser or scheduling logic.  It
prepares deterministic venue-by-grid coverage artifacts for the later Stage 4
optimiser without ever materialising the full dense venue-by-grid matrix.
"""

from .spatial import (
    CatchmentConfig,
    SparseCatchment,
    UnionCoverage,
    assert_hard_radius_monotonicity,
    build_catchment_family,
    build_sparse_catchment,
    candidate_pairs_within_distance,
    canonicalize_grid,
    canonicalize_venues,
    compute_pair_overlaps,
    compute_venue_exposures,
    connected_overlap_clusters,
    matrix_manifest_record,
    order_sha256,
    sparse_union,
    weighted_union_coverage,
)
from .decision import (
    AdminShortlistResult,
    SensitivityResult,
    SpearmanAuditResult,
    build_admin_shortlists,
    build_coverage_cluster_fallbacks,
    build_field_validation_queue,
    build_need_exposure_quadrants,
    build_stage3_to_stage4_interface,
    build_venue_feasibility_context,
    compute_pareto_tiers,
    compute_sensitivity_metrics,
    stage3_spearman_audit,
    validate_stage3_to_stage4_interface,
)

__all__ = [
    "CatchmentConfig",
    "SparseCatchment",
    "UnionCoverage",
    "assert_hard_radius_monotonicity",
    "build_catchment_family",
    "build_sparse_catchment",
    "candidate_pairs_within_distance",
    "canonicalize_grid",
    "canonicalize_venues",
    "compute_pair_overlaps",
    "compute_venue_exposures",
    "connected_overlap_clusters",
    "matrix_manifest_record",
    "order_sha256",
    "sparse_union",
    "weighted_union_coverage",
    "AdminShortlistResult",
    "SensitivityResult",
    "SpearmanAuditResult",
    "build_admin_shortlists",
    "build_coverage_cluster_fallbacks",
    "build_field_validation_queue",
    "build_need_exposure_quadrants",
    "build_stage3_to_stage4_interface",
    "build_venue_feasibility_context",
    "compute_pareto_tiers",
    "compute_sensitivity_metrics",
    "stage3_spearman_audit",
    "validate_stage3_to_stage4_interface",
]
