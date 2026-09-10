from mediroad.stage4_2_certification.synthetic import make_synthetic_formulation
from mediroad.stage4_2g_candidate_expansion.gpu_lns import ExpansionGpuLns
import numpy as np

def test_lns_cpu_fallback_preserves_feasibility():
    f=make_synthetic_formulation(); cfg={'enabled':False,'rounds':4,'proposals_per_round':8,'radius_schedule':[1,2],'candidate_pool_size':5,'random_seed':1}
    lns=ExpansionGpuLns(f,cfg); seed=np.array([0,3]); result=lns.search(seed,objective='total_population',total_population_floor=0.0)
    ok,_=f.validate_selection(result.selected_indices)
    assert ok and result.best_after>=result.best_before-1e-9
