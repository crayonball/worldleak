from pathlib import Path

import pytest

from unlearn import worldleak as wl
from unlearn.common import load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_region_and_keys(path):
    cfg = load_config(path)
    region = wl.Region.from_config(cfg["region"])
    for key in ("repair", "erase", "optim", "audit", "labels", "final_labeler", "repair_labeler"):
        assert key in cfg
    assert set(region.leak) <= set(region.risky)
    for kind in ("open", "candidate"):
        assert set(cfg["labels"][kind]) <= set(region.classes)


def test_overrides_and_base():
    cfg = load_config(CONFIGS[0].parent / "wmdp_cyber.yaml", ["repair.eta=0.8", "erase.type=hinge"])
    assert cfg["benchmark"] == "wmdp-cyber" and cfg["repair"]["eta"] == 0.8 and cfg["erase"]["type"] == "hinge"
    assert cfg["repair"]["tau"] == 0.02
