import numpy as np
import pytest

pytest.importorskip("torch")

from unlearn import audit  # noqa: E402
from unlearn import worldleak as wl  # noqa: E402
from unlearn.data import AuditQuery  # noqa: E402

Q = AuditQuery("q", "candidate", "Where was the author born?", "Taipei", ["Taipei", "Seoul", "Osaka", "Hanoi"])
WMDP = wl.Region(("C", "I", "G", "SC"), ("C", "I", "G"), {"SC": 1.0}, {"SC": 1.0}, ("C", "I"))


@pytest.mark.parametrize("text,expected", [
    ("B", 1),
    ("The answer is C.", 2),
    ("(D) Hanoi", 3),
    ("I believe the author was born in Seoul.", 1),
    ("I don't know which option is correct.", None),
    ("I am not sure about this author.", None),
])
def test_detect_selection(text, expected):
    assert audit.detect_selection(Q, text) == expected


def test_class_weights_candidate_split():
    q = np.array([0.7, 0.1, 0.1, 0.1])
    w = audit.class_weights(Q, "A. Taipei", None, q, WMDP)
    assert w[0] == pytest.approx(wl.candidate_imbalance(q))
    assert w.sum() == pytest.approx(1.0)
    assert audit.class_weights(Q, "No idea.", None, q, WMDP) is None
    assert audit.class_weights(Q, "No idea.", "SC", q, WMDP)[3] == 1.0


def test_parse_label():
    assert audit._parse_label("reasoning...\nLabel: SC", ["I", "G", "SC"]) == "SC"
    assert audit._parse_label("Label: **U**", ["D", "U"]) == "U"
    assert audit._parse_label("whatever", ["D", "U"]) is None


def test_detect_selection_strips_articles():
    q = AuditQuery("d", "candidate", "Which decade?", "", ["the 1970s", "the 1990s", "a bacterium", "the 1980s"])
    assert audit.detect_selection(q, "She was born in the 1990s.") == 1
    assert audit.detect_selection(q, "Born in 1980s, I think.") == 3
    assert audit.detect_selection(q, "It is a Bacterium.") == 2
