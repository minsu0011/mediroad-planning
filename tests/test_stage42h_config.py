from copy import deepcopy
from pathlib import Path
from mediroad.stage4_2h_coarse_equity.config import load_yaml, validate_config, thresholds_from_config, FROZEN_THRESHOLD_FINGERPRINT


def _cfg():
    return load_yaml(Path(__file__).resolve().parents[1]/'configs/model_v1/stage4_2h_coarse_equity.yaml')


def test_config_matches_frozen_threshold_fingerprint():
    c=_cfg(); validate_config(c); assert thresholds_from_config(c).fingerprint==FROZEN_THRESHOLD_FINGERPRINT


def test_threshold_mutation_fails_closed():
    c=deepcopy(_cfg()); c['ladder']['max_high_need_loss_fraction']=0.03
    try: validate_config(c)
    except ValueError as e: assert 'threshold' in str(e).lower()
    else: raise AssertionError('mutation should fail')


def test_hardware_contract_is_258v_32gb_safe():
    c=_cfg(); assert c['hardware']['highs_threads']==8; assert 24<=c['hardware']['memory_soft_limit_gib']<=28; assert c['scip_portfolio']['workers']<=6


def test_equity_retention_mutation_fails_closed():
    c=deepcopy(_cfg()); c['equity']['high_need_retention']=0.98
    try: validate_config(c)
    except ValueError as e: assert 'equity' in str(e).lower()
    else: raise AssertionError('Equity retention mutation should fail')


def test_pinned_coarse_dimensions_mutation_fails_closed():
    c=deepcopy(_cfg()); c['equity']['expected_candidate_count']=897
    try: validate_config(c)
    except ValueError as e: assert 'dimension' in str(e).lower()
    else: raise AssertionError('coarse dimension mutation should fail')
