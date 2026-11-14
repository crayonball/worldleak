"""WorldLeak, admissible region, projection target, and certificates.

Notation follows Sections 3-5 and Appendix A of the paper. A behavior
distribution p is a length-M vector on the simplex whose first `d` entries
are the risky classes (leakage / degeneration) and whose remaining entries
are admissible classes constrained by [lower, upper] bounds.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class Region:
    """Admissible region C: zero risky mass, l_i <= c_i <= u_i, sum c_i = 1."""

    classes: tuple[str, ...]
    risky: tuple[str, ...]
    lower: dict[str, float]
    upper: dict[str, float]
    leak: tuple[str, ...] = field(default=())  # F: risky classes that reveal protected info

    def __post_init__(self):
        n_risky = len(self.risky)
        if tuple(self.classes[:n_risky]) != tuple(self.risky):
            raise ValueError("risky classes must come first in `classes`")
        lo, hi = self.lo, self.hi
        if np.any(lo <= 0) or np.any(lo > hi) or np.any(hi > 1):
            raise ValueError("need 0 < l_i <= u_i <= 1 for admissible classes")
        if lo.sum() > 1 + 1e-12 or hi.sum() < 1 - 1e-12:
            raise ValueError("admissible region is empty")
        if not set(self.leak) <= set(self.risky):
            raise ValueError("leak classes must be risky classes")

    @property
    def d(self) -> int:
        return len(self.risky)

    @property
    def M(self) -> int:
        return len(self.classes)

    @property
    def admissible(self) -> tuple[str, ...]:
        return self.classes[self.d:]

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.lower[c] for c in self.admissible], dtype=float)

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.upper[c] for c in self.admissible], dtype=float)

    @property
    def leak_idx(self) -> list[int]:
        return [self.classes.index(c) for c in self.leak]

    @classmethod
    def from_config(cls, cfg: dict) -> "Region":
        return cls(
            classes=tuple(cfg["classes"]),
            risky=tuple(cfg["risky"]),
            lower={k: float(v[0]) for k, v in cfg["bounds"].items()},
            upper={k: float(v[1]) for k, v in cfg["bounds"].items()},
            leak=tuple(cfg.get("leak", ())),
        )


def _as_dist(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any(p < -1e-9) or abs(p.sum() - 1) > 1e-6:
        raise ValueError(f"not a distribution: {p}")
    return np.clip(p, 0.0, None)


def risk(p, region: Region) -> float:
    """r(p): total mass on risky classes."""
    return float(_as_dist(p)[: region.d].sum())


def admissible_deviation(p, region: Region, tol: float = 1e-10) -> float:
    """B(p) = inf_{c in C} max_{i>d} |log(p_i / c_i)|.

    For fixed t, a witness c exists iff every interval
    [max(l_i, p_i e^-t), min(u_i, p_i e^t)] is nonempty and the interval
    sums bracket 1. Feasibility is monotone in t, so bisection is exact up to tol.
    """
    q = _as_dist(p)[region.d:]
    lo, hi = region.lo, region.hi
    if np.any(q <= 0):
        return math.inf

    def feasible(t: float) -> bool:
        a = np.maximum(lo, q * math.exp(-t))
        b = np.minimum(hi, q * math.exp(t))
        return bool(np.all(a <= b + 1e-15) and a.sum() <= 1 + 1e-12 and b.sum() >= 1 - 1e-12)

    if feasible(0.0):
        return 0.0
    left, right = 0.0, 1.0
    while not feasible(right):
        right *= 2
    while right - left > tol:
        mid = 0.5 * (left + right)
        left, right = (left, mid) if feasible(mid) else (mid, right)
    return right


def worldleak(p, region: Region, tau: float) -> float:
    """WorldLeak_tau(p; C) = max{ r(p)/tau, B(p) }."""
    return max(risk(p, region) / tau, admissible_deviation(p, region))


def sensitivity_bound(region: Region) -> float:
    """Lemma 3.1: Sens(C) <= max_{i>d} u_i / l_i."""
    return float(np.max(region.hi / region.lo))


def certificate(eta: float, tau: float, region: Region) -> tuple[float, float]:
    """Theorem 3.2: (eps, delta) = (2 eta + log Sens, eta * tau)."""
    return 2 * eta + math.log(sensitivity_bound(region)), eta * tau


def two_component_certificate(p_z, p_zp, region: Region) -> tuple[float, float]:
    """Proposition A.1 for a symmetric pair: eps = B_z + B_z' + log Sens, delta = max r."""
    eps = admissible_deviation(p_z, region) + admissible_deviation(p_zp, region)
    eps += math.log(sensitivity_bound(region))
    return eps, max(risk(p_z, region), risk(p_zp, region))


def _feasible_c(region: Region) -> np.ndarray:
    """A point of C: start at the lower bounds and fill up to the upper bounds."""
    c = region.lo.copy()
    slack = 1.0 - c.sum()
    for i, room in enumerate(region.hi - region.lo):
        add = min(room, slack)
        c[i] += add
        slack -= add
    return c


def project(p, region: Region, eta: float, tau: float) -> np.ndarray:
    """Euclidean projection of p onto R(eta) = {v : WorldLeak_tau(v; C) <= eta}.

    Solves the convex QP of Section 4.1 jointly over (v, c):
        min 1/2 ||v - p||^2  s.t.  v in simplex, sum_{i<=d} v_i <= eta * tau,
        e^-eta c_i <= v_i <= e^eta c_i (i > d),  c in C.
    """
    p = _as_dist(p)
    d, M = region.d, region.M
    k = M - d
    lo, hi = region.lo, region.hi
    ex, emx = math.exp(eta), math.exp(-eta)

    c0 = _feasible_c(region)
    x0 = np.concatenate([np.zeros(d), c0, c0])  # (v_risky, v_adm, c)

    cons = [
        {"type": "eq", "fun": lambda x: x[:M].sum() - 1.0, "jac": lambda x: np.r_[np.ones(M), np.zeros(k)]},
        {"type": "eq", "fun": lambda x: x[M:].sum() - 1.0, "jac": lambda x: np.r_[np.zeros(M), np.ones(k)]},
        {"type": "ineq", "fun": lambda x: eta * tau - x[:d].sum(),
         "jac": lambda x: np.r_[-np.ones(d), np.zeros(M - d + k)]},
        {"type": "ineq", "fun": lambda x: ex * x[M:] - x[d:M],
         "jac": lambda x: np.hstack([np.zeros((k, d)), -np.eye(k), ex * np.eye(k)])},
        {"type": "ineq", "fun": lambda x: x[d:M] - emx * x[M:],
         "jac": lambda x: np.hstack([np.zeros((k, d)), np.eye(k), -emx * np.eye(k)])},
    ]
    bounds = [(0.0, 1.0)] * M + list(zip(lo, hi))
    res = minimize(
        lambda x: 0.5 * np.sum((x[:M] - p) ** 2),
        x0,
        jac=lambda x: np.r_[x[:M] - p, np.zeros(k)],
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"projection failed: {res.message}")
    v = np.clip(res.x[:M], 0.0, None)
    return v / v.sum()


def erasure_strength(p, p_star, region: Region) -> float:
    """lambda_E = sum_{m in F} [p_m - p*_m]_+ over direct-leak classes F."""
    idx = region.leak_idx
    return float(np.clip(np.asarray(p)[idx] - np.asarray(p_star)[idx], 0.0, None).sum())


def candidate_imbalance(q) -> float:
    """1/2 sum_i |q_i - 1/L| for normalized candidate likelihoods q (per-response C weight)."""
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    return 0.5 * float(np.abs(q - 1.0 / len(q)).sum())


def g_cand(q, select_rate: float) -> float:
    """Remark 3.1 with p_i = s q_i: G_cand = s/2 sum_i |q_i - 1/L|."""
    return select_rate * candidate_imbalance(q)


def tv(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def sampling_tv_radius(M: int, n: int, alpha: float) -> float:
    """Proposition 3.1: TV(p_hat, p) <= M/2 sqrt(log(2M/alpha) / (2n)) w.p. >= 1 - alpha."""
    return 0.5 * M * math.sqrt(math.log(2 * M / alpha) / (2 * n))


def _set_extreme(p_hat: np.ndarray, h: float, d: int, S: list[int], maximize: bool) -> float:
    """max / min v(S) over P = {v in simplex : |v - p_hat|_inf <= h, |r(v) - r(p_hat)| <= h}."""
    M = len(p_hat)
    c = np.zeros(M)
    c[S] = -1.0 if maximize else 1.0
    r_hat = p_hat[:d].sum()
    risky_row = np.r_[np.ones(d), np.zeros(M - d)]
    res = linprog(
        c,
        A_ub=np.vstack([risky_row, -risky_row]),
        b_ub=[r_hat + h, -(r_hat - h)],
        A_eq=np.ones((1, M)),
        b_eq=[1.0],
        bounds=[(max(0.0, x - h), min(1.0, x + h)) for x in p_hat],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"confidence-set LP failed: {res.message}")
    return -res.fun if maximize else res.fun


def finite_sample_delta(p_hats: dict, ns: dict, pairs, d: int, alpha: float, eps: float) -> float:
    """Theorem A.1: delta_bar(eps) over ordered pairs (w, w') with confidence 1 - alpha."""
    W = len(p_hats)
    M = len(next(iter(p_hats.values())))
    h = {w: math.sqrt(math.log(2 * W * (M + 1) / alpha) / (2 * ns[w])) for w in p_hats}
    best = -math.inf
    for size in range(1, M + 1):
        for S in itertools.combinations(range(M), size):
            S = list(S)
            for w, wp in pairs:
                hi = _set_extreme(np.asarray(p_hats[w], float), h[w], d, S, True)
                lo = _set_extreme(np.asarray(p_hats[wp], float), h[wp], d, S, False)
                best = max(best, hi - math.exp(eps) * lo)
    return max(best, 0.0)


def bernoulli_kl(a: float, b: float) -> float:
    def term(x, y):
        return 0.0 if x == 0 else x * math.log(x / y)

    b = min(max(b, 1e-300), 1 - 1e-16)
    return term(a, b) + term(1 - a, 1 - b)


def kl_upper_bound(r_hat: float, n: int, alpha: float, tol: float = 1e-12) -> float:
    """Chernoff (KL form) upper confidence bound: max{r : kl(r_hat || r) <= log(1/alpha) / n}.

    The paper splits alpha=0.05 over two worlds, i.e. call with alpha/2 per world
    so that the threshold is log(2/alpha)/n.
    """
    bound = math.log(1 / alpha) / n
    left, right = r_hat, 1.0
    if bernoulli_kl(r_hat, right) <= bound:
        return 1.0
    while right - left > tol:
        mid = 0.5 * (left + right)
        left, right = (mid, right) if bernoulli_kl(r_hat, mid) <= bound else (left, mid)
    return left


def paired_delta_upper(r_a: float, r_b: float, n: int, alpha: float = 0.05) -> float:
    """Appendix A.4 paired-world bound at eps=0 when SC is the only admissible class."""
    return max(kl_upper_bound(r_a, n, alpha / 2), kl_upper_bound(r_b, n, alpha / 2))
