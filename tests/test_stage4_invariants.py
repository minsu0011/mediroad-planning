from __future__ import annotations

import numpy as np

from mediroad.stage4.config import load_config
from mediroad.stage4.optimizer import prepare_weights
from mediroad.stage4.synthetic import make_synthetic_input


def test_grid_weights_are_nonnegative_and_high_need_subset():
    data = make_synthetic_input()
    weights = prepare_weights(data, load_config(None))
    assert np.all(weights.population >= 0)
    assert np.all(weights.need_weighted >= 0)
    assert np.all(weights.high_need_population >= 0)
    assert np.all(weights.high_need_population <= weights.population)


def test_cross_boundary_coverage_exists():
    data = make_synthetic_input()
    first = data.coverage.getrow(0).indices
    own_admin = data.grids.iloc[first[0]]["admin_code"]
    admins = set(data.grids.iloc[first]["admin_code"])
    assert own_admin in admins
    assert len(admins) > 1

