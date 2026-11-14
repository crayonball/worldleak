import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_inputs", Path(__file__).resolve().parents[1] / "scripts" / "build_inputs.py")
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)


def test_tofu_candidates():
    qs = bi.tofu_candidates()
    assert len(qs) == 110 and len({q.qid for q in qs}) == 110
    for q in qs:
        assert len(q.candidates) == 4 and len(set(q.candidates)) == 4 and q.reference in q.candidates


@pytest.mark.parametrize("dom", ["bio", "cyber"])
def test_wmdp_paired(dom):
    f = bi.wmdp_paired()[dom]
    assert len(f["audit"]) == 256 and len({q.qid for q in f["audit"]}) == 256
    rep = {q.prompt for q in f["repair"]}
    assert not rep & {q.prompt for q in f["audit"]}  # disjoint templates
    for q in f["audit"] + f["repair"]:
        a, b = q.meta["reference_A"], q.meta["reference_B"]
        assert q.candidates[q.meta["A_cand"]] == a and q.candidates[q.meta["B_cand"]] == b and a != b
