import numpy as np
import pandas as pd
from scipy import sparse

from mediroad.stage4_2_certification.types import LinearMipModel
from mediroad.stage4_2g_candidate_expansion.runner import (
    _candidate_only_proof_relaxation,
    _integerize_scip_coverage_projection,
)
from mediroad.stage4_2g_candidate_expansion.scip_portfolio import run_portfolio
from mediroad.stage4_2h_coarse_equity.solver import _map_plan

class F:
    candidates=pd.DataFrame({'venue_id':['A','B','C']})
    venue_alias_map={'OLD_A':'A'}
    def validate_selection(self,a): return (len(a)==len(set(a.tolist())),'ok')

def test_map_plan_uses_alias_and_preserves_20_style_semantics():
    p=pd.DataFrame({'venue_id':['OLD_A','C']}); out=_map_plan(F(),p); assert out.tolist()==[0,2]


def test_scip_coverage_integerization_is_projection_equivalent_strengthening():
    model = LinearMipModel(
        name='threshold', objective_name='feasibility', sense='min',
        c=np.zeros(4),
        A=sparse.csc_matrix(np.asarray([
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 3.0],
        ])),
        row_lower=np.asarray([1.0, -np.inf, 2.0]),
        row_upper=np.asarray([1.0, 0.0, np.inf]),
        col_lower=np.zeros(4), col_upper=np.asarray([1.0, 1.0, 1.0, 0.0]),
        integrality=np.asarray([1, 1, 0, 0], dtype=np.int8),
        variable_names=['x::a', 'x::b', 'y::0', 'y::1'],
        x_slice=slice(0, 2), y_slice=slice(2, 4),
        metadata={
            'one_link_pattern_formulation': True,
            'row_names': [
                'visit_count', 'cover_link::0', 'floor::oracle_threshold::mincov::S',
            ],
        },
    )
    strengthened = _integerize_scip_coverage_projection(
        model, [np.asarray([0], dtype=int), np.asarray([], dtype=int)]
    )
    assert strengthened.integrality.tolist() == [1, 1, 1, 1]
    assert strengthened.metadata['scip_projection_strengthening']['x_projection_equivalent'] is True
    assert strengthened.metadata['scip_projection_strengthening']['canonical_or_rows'] == 1
    assert strengthened.metadata['scip_projection_strengthening']['canonical_or_nnz'] == 2
    assert strengthened.A.shape == (4, 4)
    canonical_lift = np.asarray([1.0, 0.0, 1.0, 0.0])
    activity = strengthened.A @ canonical_lift
    assert np.all(activity >= strengthened.row_lower)
    assert np.all(activity <= strengthened.row_upper)


def test_candidate_only_proof_relaxation_keeps_x_rows_and_drops_y_rows():
    model = LinearMipModel(
        name='threshold', objective_name='feasibility', sense='min',
        c=np.zeros(4),
        A=sparse.csc_matrix(np.asarray([
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.5, 0.7, 0.0, 0.0],
        ])),
        row_lower=np.asarray([1.0, -np.inf, 2.0, 0.6]),
        row_upper=np.asarray([1.0, 0.0, np.inf, np.inf]),
        col_lower=np.zeros(4), col_upper=np.ones(4),
        integrality=np.asarray([1, 1, 0, 0], dtype=np.int8),
        variable_names=['x::a', 'x::b', 'y::0', 'y::1'],
        x_slice=slice(0, 2), y_slice=slice(2, 4),
        metadata={'row_names': ['visit_count', 'cover_link::0', 'floor::threshold', 'stage42d::cut']},
    )
    projected = _candidate_only_proof_relaxation(model)
    assert projected.A.shape == (2, 2)
    assert projected.metadata['row_names'] == ['visit_count', 'stage42d::cut']
    assert projected.integrality.tolist() == [1, 1]
    assert projected.y_slice == slice(2, 2)
    assert projected.metadata['stage42h_candidate_only_relaxation']['infeasible_is_authoritative'] is True


def test_scip_mps_witness_round_trips_candidate_indices(tmp_path):
    model = LinearMipModel(
        name='witness_round_trip', objective_name='feasibility', sense='min',
        c=np.zeros(2),
        A=sparse.csc_matrix(np.asarray([[1.0, 1.0]])),
        row_lower=np.asarray([1.0]), row_upper=np.asarray([1.0]),
        col_lower=np.zeros(2), col_upper=np.ones(2),
        integrality=np.ones(2, dtype=np.int8),
        variable_names=['x::original_a', 'x::original_b'],
        x_slice=slice(0, 2), y_slice=slice(2, 2),
        metadata={'row_names': ['visit_count']},
    )
    result = run_portfolio(
        model, tmp_path / 'portfolio', seeds=[7], workers=1,
        time_limit_sec=10.0, memory_limit_mib=256.0,
    )
    assert result['terminal']['kind'] == 'FEASIBLE_WITNESS'
    worker = result['terminal']['worker']
    assert len(worker['selected_indices']) == 1
    assert worker['selected_indices'][0] in {0, 1}
    assert worker['selected_variable_names'][0] in {'c0', 'c1'}
