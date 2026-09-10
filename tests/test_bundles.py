import pandas as pd

from mediroad.stage4_2.bundles import assign_bundles


def test_bundle_assignment_one_each():
    selected = pd.DataFrame({"venue_id": ["V1", "V2", "V3"]})
    values = pd.DataFrame(
        [
            {"venue_id": v, "bundle_id": b, "bundle_value": value}
            for v, vals in {"V1": [3, 1], "V2": [2, 4], "V3": [5, 2]}.items()
            for b, value in zip(["B1", "B2"], vals)
        ]
    )
    out, info = assign_bundles(selected, values, minimum_per_bundle=1)
    assert len(out) == 3
    assert out["bundle_id"].nunique() == 2
    assert info["solver_status"] == "OPTIMAL"
