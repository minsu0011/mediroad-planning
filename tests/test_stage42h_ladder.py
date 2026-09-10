import pandas as pd
from mediroad.stage4_2h_coarse_equity.config import CandidateThresholds
from mediroad.stage4_2h_coarse_equity.ladder import CandidateView, compare_pair, evaluate_sequential_ladder


def _plan(admin_prefix='A', cluster_prefix='C'):
    return pd.DataFrame({
        'venue_id':[f'V{i}' for i in range(20)],
        'admin_code':[f'{admin_prefix}{i}' for i in range(20)],
        'cluster_id':[f'{cluster_prefix}{i}' for i in range(20)],
    })


def _view(name, need=100, unique=100, high=100, minc=.4, plan=None, gap=.0049):
    return CandidateView(name, {
        'unique_elderly_population':unique,
        'need_weighted_population':need,
        'high_need_population':high,
        'min_sigungu_coverage_ratio':minc,
    }, plan if plan is not None else _plan(), gap, True)


def test_actual_observed_asymmetry_promotes_through_top5_to_coarse():
    # Mirrors the real Stage4.2G shape: top3->top5 passes, but wider coarse
    # evidence violates high-need/Jaccard; top5 must then itself be tested.
    top3=_view('top3',need=110945,unique=277695,high=5315.21,minc=.33431)
    top5=_view('top5',need=110978.56,unique=277825.11,high=5075.73,minc=.33431)
    coarse_plan=_plan('B','D')
    # Preserve ~13/27 admin Jaccard and ~10/30 cluster overlap by editing prefixes.
    coarse_plan.loc[:12,'admin_code']=[f'A{i}' for i in range(13)]
    coarse_plan.loc[:9,'cluster_id']=[f'C{i}' for i in range(10)]
    coarse=_view('coarse_pareto',need=111201.88,unique=278394.77,high=5630.10,minc=.324296,plan=coarse_plan,gap=.005)
    comp,trans,decision=evaluate_sequential_ladder({'top3':top3,'top5':top5,'coarse_pareto':coarse},ladder=['top3','top5','coarse_pareto'],thresholds=CandidateThresholds())
    assert decision['recommended_candidate_set']=='coarse_pareto'
    assert list(trans['action'])==['PROMOTE_ONE_RUNG','PROMOTE_ONE_RUNG','FINAL_RUNG_REACHED']
    row=comp[(comp.candidate_set=='top5') & (comp.reference_candidate_set=='coarse_pareto')].iloc[0]
    assert not bool(row['comparison_passed'])
    assert not bool(row['high_need_loss_passed'])


def test_retain_top3_only_when_it_passes_all_wider_evidence():
    p=_plan(); top3=_view('top3',plan=p); top5=_view('top5',need=100.2,unique=100.2,high=100.5,minc=.401,plan=p); coarse=_view('coarse_pareto',need=100.3,unique=100.3,high=100.6,minc=.402,plan=p)
    _,trans,d=evaluate_sequential_ladder({'top3':top3,'top5':top5,'coarse_pareto':coarse},ladder=['top3','top5','coarse_pareto'],thresholds=CandidateThresholds())
    assert d['recommended_candidate_set']=='top3'
    assert trans.iloc[0]['action']=='RETAIN_CURRENT_RUNG'


def test_top5_can_be_final_when_top3_unsafe_but_top5_covers_coarse():
    p=_plan(); bad=_plan('B','D')
    top3=_view('top3',high=90,plan=bad)
    top5=_view('top5',high=100,plan=p)
    coarse=_view('coarse_pareto',high=101,need=100.5,unique=100.5,minc=.401,plan=p)
    _,trans,d=evaluate_sequential_ladder({'top3':top3,'top5':top5,'coarse_pareto':coarse},ladder=['top3','top5','coarse_pareto'],thresholds=CandidateThresholds())
    assert d['recommended_candidate_set']=='top5'
    assert list(trans['action'])==['PROMOTE_ONE_RUNG','RETAIN_CURRENT_RUNG']


def test_uncertified_wider_evidence_fails_closed():
    p=_plan(); views={'top3':_view('top3',plan=p),'top5':_view('top5',plan=p),'coarse_pareto':_view('coarse_pareto',plan=p)}
    views['coarse_pareto']=CandidateView('coarse_pareto',views['coarse_pareto'].metrics,p,.02,False)
    comp,trans,d=evaluate_sequential_ladder(views,ladder=['top3','top5','coarse_pareto'],thresholds=CandidateThresholds())
    assert not d['passed'] and d['recommended_candidate_set'] is None
    assert comp.empty and trans.empty
