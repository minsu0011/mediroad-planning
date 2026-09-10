from mediroad.stage4.config import load_config


def test_default_config_contract():
    config = load_config(None)
    assert config["optimization"]["visit_count"] == 20
    assert config["network_validation"]["enabled"] is True
    assert config["runtime"]["jobs"] == 8
    assert config["optimization"]["scenarios"] == ["efficiency", "balanced", "equity"]

