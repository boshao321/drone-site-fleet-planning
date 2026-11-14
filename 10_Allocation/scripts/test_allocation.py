"""Correctness tests for the two optimisers in `allocation`.

The allocation stage reports a policy comparison, so a rule that is merely good
would let a difference between policies be a difference between rounding
accidents. Both optimisers are therefore checked against exhaustive enumeration
on instances small enough to enumerate.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import allocation as A          # noqa: E402


def brute_force_minutes(e, M):
    """max sum_s min(15 x_s, e_s) over all integer x with sum x_s <= M."""
    S = len(e)
    best = -1.0
    for combo in itertools.product(range(M + 1), repeat=S):
        if sum(combo) > M:
            continue
        v = sum(min(A.SLOT_MINUTES * k, ei) for k, ei in zip(combo, e))
        best = max(best, v)
    return best


def test_allocate_is_exact(trials=400, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        S = int(rng.integers(2, 5))
        M = int(rng.integers(1, 9))
        e = rng.uniform(0, 70, size=S)
        x = A.allocate(e[None, :], M)[0]
        assert x.sum() <= M, "fleet constraint violated"
        got = float(np.minimum(A.SLOT_MINUTES * x, e).sum())
        want = brute_force_minutes(e, M)
        worst = max(worst, want - got)
        assert got >= want - 1e-9, f"suboptimal: {got} < {want}"
    print(f"allocate: exact on {trials} random instances (worst gap {worst:.2e})")


def _tiny_scenario(rng, T=2, S=3, n=6):
    """A scenario carrying only the fields the two optimisers read."""
    order_book = {}
    req_minutes = np.zeros((T, S))
    req_orders_hv = np.zeros((T, S))
    for t in range(T):
        for j in range(S):
            k = int(rng.integers(1, n))
            tau = rng.uniform(4, 20, size=k)
            price = rng.uniform(20, 120, size=k)
            arr = np.stack([tau, price], axis=1)
            arr = arr[np.argsort(-arr[:, 1])]
            order_book[(t, j)] = arr
            req_minutes[t, j] = tau.sum()
            req_orders_hv[t, j] = int((price >= A.HIGH_VALUE_THRESHOLD).sum())
    sc = A.Scenario(np.arange(S), np.zeros(1, dtype=int), req_minutes,
                    req_minutes.copy(), np.zeros((T, S)), req_orders_hv,
                    np.zeros((T, S)), order_book, np.arange(T), S)
    return sc


def brute_force_ceiling(sc, M):
    """Best served high-value count over all integer allocations, per slot."""
    T, S = sc.req_minutes.shape
    total = 0
    for t in range(T):
        caps = []
        for j in range(S):
            arr = sc.order_book[(t, j)]
            kmax = int(min(M, np.ceil(arr[:, 0].sum() / A.SLOT_MINUTES)))
            cum = np.cumsum(arr[:, 0])
            hv = np.concatenate(([0], np.cumsum(arr[:, 1] >= A.HIGH_VALUE_THRESHOLD)))
            f = [hv[int(np.searchsorted(cum, A.SLOT_MINUTES * k, side="right"))]
                 for k in range(kmax + 1)]
            caps.append(f)
        best = 0
        for combo in itertools.product(*[range(len(f)) for f in caps]):
            if sum(combo) > M:
                continue
            best = max(best, sum(caps[j][k] for j, k in enumerate(combo)))
        total += best
    return total


def test_ceiling_is_exact(trials=60, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        M = int(rng.integers(1, 7))
        sc = _tiny_scenario(rng)
        got, x = A.attainable_ceiling(sc, M)
        want = brute_force_ceiling(sc, M)
        assert got == want, f"ceiling {got} != brute force {want}"
        assert (x.sum(axis=1) <= M).all(), "fleet constraint violated"
        # the returned allocation must reproduce the reported value
        served = sum(A.evaluate(x[t:t + 1], _slice(sc, t))["served_high_value_orders"]
                     for t in range(x.shape[0]))
        assert abs(served - got) < 1e-9, f"allocation serves {served}, reported {got}"
    print(f"attainable_ceiling: exact on {trials} random instances")


def _slice(sc, t):
    book = {(0, j): v for (tt, j), v in sc.order_book.items() if tt == t}
    return A.Scenario(sc.site_idx, sc.grid_to_site, sc.req_minutes[t:t + 1],
                      sc.req_minutes_hv[t:t + 1], sc.req_orders[t:t + 1],
                      sc.req_orders_hv[t:t + 1], sc.req_value[t:t + 1],
                      book, sc.slots[t:t + 1], sc.n_sites)


def test_ceiling_dominates_foresight(trials=40, seed=2):
    """The ceiling can never be beaten by the minute-matching rule, which is the
    property the previous oracle lacked."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(trials):
        M = int(rng.integers(2, 10))
        sc = _tiny_scenario(rng, T=3, S=4, n=8)
        ceiling, _ = A.attainable_ceiling(sc, M)
        x = A.allocate(sc.req_minutes_hv, M)
        served = sum(A.evaluate(x[t:t + 1], _slice(sc, t))["served_high_value_orders"]
                     for t in range(x.shape[0]))
        assert served <= ceiling + 1e-9, "minute matching beat the ceiling"
        gaps.append(ceiling - served)
    print(f"ceiling >= minute matching on {trials} instances "
          f"(mean shortfall {np.mean(gaps):.2f} orders)")


def test_full_fleet_tiebreak(trials=400, seed=3):
    """allocate deploys the whole fleet, keeps the estimated optimum, matches
    sequential D'Hondt, and never serves less than the capped rule."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        T, S = 3, int(rng.integers(2, 7))
        M = int(rng.integers(1, 25))
        e = rng.uniform(0, 60, size=(T, S)) * (rng.random((T, S)) > 0.2)
        x = A.allocate(e, M)
        xc = A.allocate_capped(e, M)
        assert (x.sum(axis=1) == M).all(), "fleet not fully deployed"
        obj = np.minimum(A.SLOT_MINUTES * x, e).sum(axis=1)
        objc = np.minimum(A.SLOT_MINUTES * xc, e).sum(axis=1)
        assert np.allclose(obj, objc), "tie-break changed the estimated objective"
        assert (x >= xc).all(), "tie-break removed aircraft"
        for t in range(T):                       # sequential D'Hondt reference
            ref = xc[t].copy()
            ee = np.maximum(e[t], 1e-9)
            for _ in range(M - int(ref.sum())):
                ref[int(np.argmax(ee / (ref + 1)))] += 1
            assert (ref == x[t]).all(), "closed-form D'Hondt differs from sequential"
    rng = np.random.default_rng(seed + 1)
    for _ in range(60):                          # realised service never falls
        M = int(rng.integers(2, 12))
        sc = _tiny_scenario(rng, T=3, S=4, n=8)
        est = sc.req_minutes_hv * rng.lognormal(0, 0.6, size=sc.req_minutes_hv.shape)
        a = sum(A.evaluate(A.allocate(est, M)[t:t + 1], _slice(sc, t))["served_high_value_orders"]
                for t in range(3))
        b = sum(A.evaluate(A.allocate_capped(est, M)[t:t + 1], _slice(sc, t))["served_high_value_orders"]
                for t in range(3))
        assert a >= b - 1e-9, "full deployment served less than capped"
    print(f"allocate: full fleet, same estimated optimum, D'Hondt-exact, never below capped")


if __name__ == "__main__":
    test_allocate_is_exact()
    test_ceiling_is_exact()
    test_ceiling_dominates_foresight()
    test_full_fleet_tiebreak()
    print("all allocation tests passed")
