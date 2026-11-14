import math

import numpy as np
import pytest

from unlearn import worldleak as wl

TOFU = wl.Region(
    classes=("D", "C", "I", "G", "SC", "H", "U"),
    risky=("D", "C", "I", "G"),
    lower={"SC": 110 / 510, "H": 0.1, "U": 0.5},
    upper={"SC": 110 / 510, "H": 0.5, "U": 0.9},
    leak=("D", "C", "I"),
)
WMDP = wl.Region(
    classes=("C", "I", "G", "SC"), risky=("C", "I", "G"),
    lower={"SC": 1.0}, upper={"SC": 1.0}, leak=("C", "I"),
)


def brute_B(p, region, grid=400):
    """Grid search over C for the admissible deviation (TOFU: SC fixed, H free)."""
    q = np.asarray(p)[region.d:]
    best = math.inf
    sc = region.lo[0]
    for h in np.linspace(region.lo[1], region.hi[1], grid):
        c = np.array([sc, h, 1 - sc - h])
        if not region.lo[2] - 1e-12 <= c[2] <= region.hi[2] + 1e-12:
            continue
        best = min(best, np.max(np.abs(np.log(q / c))))
    return best


def random_dists(rng, n, M):
    return rng.dirichlet(np.ones(M) * 0.7, size=n)


def test_region_validation():
    with pytest.raises(ValueError):
        wl.Region(("D", "H"), ("D",), {"H": 0.5}, {"H": 0.6})


def test_admissible_deviation_matches_grid():
    rng = np.random.default_rng(0)
    for p in random_dists(rng, 50, 7):
        assert wl.admissible_deviation(p, TOFU) == pytest.approx(brute_B(p, TOFU), abs=5e-3)


def test_in_region_has_zero_leak():
    c = np.array([0, 0, 0, 0, 110 / 510, 0.2, 1 - 110 / 510 - 0.2])
    assert wl.worldleak(c, TOFU, 0.05) == pytest.approx(0.0, abs=1e-9)


def test_wmdp_worldleak_closed_form():
    p = np.array([0.02, 0.01, 0.0, 0.97])
    assert wl.worldleak(p, WMDP, 0.02) == pytest.approx(max(0.03 / 0.02, -math.log(0.97)))


@pytest.mark.parametrize("eta,tau", [(2.5, 0.05), (0.8, 0.02), (0.1, 0.05)])
def test_certificate(eta, tau):
    sens = wl.sensitivity_bound(TOFU)
    assert sens == pytest.approx(np.max(TOFU.hi / TOFU.lo))
    eps, delta = wl.certificate(eta, tau, TOFU)
    assert eps == pytest.approx(2 * eta + math.log(sens))
    assert delta == pytest.approx(eta * tau)


@pytest.mark.parametrize("eta,tau", [(2.5, 0.05), (0.8, 0.02), (0.1, 0.05)])
def test_projection_feasible_and_optimal(eta, tau):
    rng = np.random.default_rng(1)
    for p in random_dists(rng, 10, 7):
        v = wl.project(p, TOFU, eta, tau)
        assert wl.worldleak(v, TOFU, tau) <= eta + 1e-5
        dist = np.linalg.norm(v - p)
        # No random feasible point (mixtures of v with points of C) may be closer.
        for _ in range(200):
            c = np.r_[np.zeros(4), 110 / 510, (h := rng.uniform(0.1, 1 - 110 / 510 - 0.5)), 1 - 110 / 510 - h]
            lam = rng.uniform()
            w = lam * v + (1 - lam) * c  # R(eta) is convex and contains C
            assert np.linalg.norm(w - p) >= dist - 1e-6


def test_projection_identity_inside_region():
    p = np.array([0.001, 0, 0, 0, 110 / 510, 0.2, 1 - 110 / 510 - 0.201])
    assert np.allclose(wl.project(p, TOFU, 2.5, 0.05), p, atol=1e-6)


def test_erasure_strength_ignores_degeneration():
    p = np.array([0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1])
    v = wl.project(p, TOFU, 2.5, 0.05)
    lam = wl.erasure_strength(p, v, TOFU)
    assert lam == pytest.approx(np.clip(p[:3] - v[:3], 0, None).sum())


def test_g_cand():
    assert wl.g_cand([0.25] * 4, 1.0) == 0.0
    assert wl.g_cand([1, 0, 0, 0], 0.5) == pytest.approx(0.5 * 0.75)


def test_kl_upper_bound_monotone():
    a = wl.kl_upper_bound(0.01, 64, 0.025)
    b = wl.kl_upper_bound(0.01, 256, 0.025)
    assert 0.01 < b < a < 1


def test_finite_sample_delta_shrinks_with_n():
    p = {"A": np.array([0.01, 0.0, 0.0, 0.99]), "B": np.array([0.02, 0.0, 0.0, 0.98])}
    pairs = [("A", "B"), ("B", "A")]
    small = wl.finite_sample_delta(p, {"A": 64, "B": 64}, pairs, 3, 0.05, 0.0)
    large = wl.finite_sample_delta(p, {"A": 1024, "B": 1024}, pairs, 3, 0.05, 0.0)
    assert large < small
