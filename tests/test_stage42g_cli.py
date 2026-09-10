from pathlib import Path
import numpy as np
from mediroad.stage4_2g_candidate_expansion.cli import _json_default

def test_numpy_json_conversion():
    assert _json_default(np.array([1,2]))==[1,2]
    assert _json_default(np.int64(3))==3
    assert _json_default(Path('x'))=='x'
