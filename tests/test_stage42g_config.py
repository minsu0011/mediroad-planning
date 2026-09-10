from mediroad.stage4_2g_candidate_expansion.config import CandidateThresholds,FROZEN_THRESHOLD_FINGERPRINT

def test_frozen_threshold_fingerprint():
    assert CandidateThresholds().fingerprint==FROZEN_THRESHOLD_FINGERPRINT
