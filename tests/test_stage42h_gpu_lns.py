import numpy as np
import pandas as pd
from scipy import sparse
from mediroad.stage4_2h_coarse_equity.gpu_lns import CoarseEquityGpuLns


class FakeF:
    visit_count=2; max_admin=2; max_sigungu=2; one_cluster=False; nx=4; ny=4; ns=2
    sigungu_values=['S1','S2']; sigungu_pattern_indices={'S1':np.array([0,1]),'S2':np.array([2,3])}; sigungu_population_total={'S1':2.0,'S2':2.0}
    _candidate_pattern=sparse.csr_matrix(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=bool))
    population=np.ones(4); high_need=np.array([0.,0.,1.,2.])
    candidates=pd.DataFrame({'admin_code':['A','A','B','B'],'sigungu':['S1','S1','S2','S2'],'cluster_id':['C1','C2','C3','C4'],'venue_id':['V1','V2','V3','V4']})
    def validate_selection(self,idx):
        idx=np.asarray(idx); return (len(idx)==2 and len(np.unique(idx))==2,'ok')
    def metrics(self,idx):
        covered=np.asarray(self._candidate_pattern[np.asarray(idx)].sum(axis=0)).ravel()>0
        s1=float(covered[:2].sum()/2); s2=float(covered[2:].sum()/2)
        return {'unique_elderly_population':float(covered.sum()),'high_need_population':float(self.high_need[covered].sum()),'min_sigungu_coverage_ratio':min(s1,s2),'need_weighted_population':float(covered.sum()),'cost':0.0}


def test_cpu_fallback_lns_preserves_hard_floors_and_improves_min():
    f=FakeF(); cfg={'enabled':False,'rounds':8,'proposals_per_round':64,'batch_size':64,'candidate_pool_size':4,'radius_schedule':[1]}
    lns=CoarseEquityGpuLns(f,cfg); r=lns.search(np.array([0,1]),objective='min_sigungu_coverage',total_floor=2.0,rounds=8)
    assert r.backend=='NUMPY_CPU'; assert r.best_after>=r.best_before
    m=f.metrics(r.selected_indices); assert m['unique_elderly_population']>=2.0


def test_high_need_search_respects_min_floor():
    f=FakeF(); cfg={'enabled':False,'rounds':8,'proposals_per_round':64,'batch_size':64,'candidate_pool_size':4,'radius_schedule':[1]}
    lns=CoarseEquityGpuLns(f,cfg); r=lns.search(np.array([0,2]),objective='high_need_population',total_floor=2.0,min_floor=.5,rounds=8)
    assert f.metrics(r.selected_indices)['min_sigungu_coverage_ratio']>=.5
