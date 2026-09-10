import pandas as pd
from mediroad.stage4_2g_candidate_expansion.compare import evaluate_top3_reduction
from mediroad.stage4_2g_candidate_expansion.config import CandidateThresholds

def _plan(prefix):
    return pd.DataFrame({'venue_id':[f'{prefix}{i}' for i in range(20)],'admin_code':[f'A{i}' for i in range(20)],'cluster_id':[f'C{i}' for i in range(20)]})

def test_retain_top3_when_frozen_thresholds_pass():
    base={'unique_elderly_population':100.0,'need_weighted_population':100.0,'high_need_population':100.0,'min_sigungu_coverage_ratio':0.40}
    alt={'unique_elderly_population':100.5,'need_weighted_population':100.5,'high_need_population':101.0,'min_sigungu_coverage_ratio':0.405}
    p=_plan('V')
    f,d=evaluate_top3_reduction(baseline_metrics=base,baseline_plan=p,baseline_max_gap=.0049,alternatives={'top5':(alt,p,.0048,True),'coarse_pareto':(alt,p,.0047,True)},thresholds=CandidateThresholds())
    assert d['decision']=='PASS_RETAIN_TOP3' and d['recommended_candidate_set']=='top3'
    assert f['comparison_passed'].all()

def test_promote_predeclared_top5_on_certified_harm():
    base={'unique_elderly_population':100.0,'need_weighted_population':100.0,'high_need_population':100.0,'min_sigungu_coverage_ratio':0.40}
    alt={'unique_elderly_population':103.0,'need_weighted_population':103.0,'high_need_population':104.0,'min_sigungu_coverage_ratio':0.43}
    p=_plan('V')
    f,d=evaluate_top3_reduction(baseline_metrics=base,baseline_plan=p,baseline_max_gap=.0049,alternatives={'top5':(alt,p,.0048,True),'coarse_pareto':(alt,p,.0047,True)},thresholds=CandidateThresholds())
    assert d['decision']=='PROMOTE_PREDECLARED_WIDER_CANDIDATE_SET' and d['recommended_candidate_set']=='top5'
