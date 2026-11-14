"""Shared helpers for the revision experiments (IEEE Access, round 1).

Everything here builds on 21_PlanDispatch/core_pd.py and 10_Allocation/scripts/
allocation.py without changing them, so the numbers of the submitted paper stay
reproducible from the original code.
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "21_PlanDispatch"))
import core_pd as P                      # noqa: E402

A, C = P.A, P.C
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
SLOT = P.SLOT
KS = [5, 10, 15, 20, 25, 30, 38, 45, 55, 70, 90, 120]
CAP = 10


# ------------------------------------------------------------------ policies
def policy_allocs(pl, M, cap=CAP, cal="val", ev="test", model="AMSGCN", seed=0,
                  which=("floor", "point", "method", "foresight", "ceiling", "unc_static")):
    """Allocation arrays [T, S] of each policy on the evaluation day."""
    out = {}
    if "floor" in which:
        out["floor"] = P.allocate_point(pl.static[ev], M, cap)
    if "foresight" in which:
        out["foresight"] = P.allocate_point(pl.truth[ev], M, cap)
    if "ceiling" in which:
        out["ceiling"] = P.ceiling_alloc(pl.sc[ev], M, cap)[1]
    if "unc_static" in which:
        smp = P.residual_sampler(pl.static[cal], pl.truth[cal], seed=seed)
        out["unc_static"] = P.allocate_expected(smp(pl.static[ev]), M, cap)
    if "point" in which or "method" in which:
        ec, ee = pl.forecast(model, cal), pl.forecast(model, ev)
        if "point" in which:
            out["point"] = P.allocate_point(ee, M, cap)
        if "method" in which:
            lam = pl.choose_lambda(ec, M, cal, cap)
            bc = (1 - lam) * pl.static[cal] + lam * ec
            be = (1 - lam) * pl.static[ev] + lam * ee
            out["method"] = P.allocate_expected(P.residual_sampler(bc, pl.truth[cal], seed=seed)(be), M, cap)
            out["_lambda"] = lam
    return out


def served_slots(pl, x, ev="test"):
    """High value orders served per slot under the paper's aggregate rule (A4)."""
    return P.served_per_slot(x, pl.sc[ev])


def boot_ci(diff, B=2000, block=4, seed=7, q=(2.5, 97.5)):
    """Paired block bootstrap over slots (blocks of `block` consecutive slots)."""
    rng = np.random.default_rng(seed)
    T = len(diff)
    blocks = [np.arange(i, min(i + block, T)) for i in range(0, T, block)]
    sums = np.array([diff[b].sum() for b in blocks])
    draws = rng.integers(0, len(blocks), size=(B, len(blocks)))
    bs = sums[draws].sum(axis=1)
    return np.percentile(bs, q), bs


# ------------------------------------------------------- aircraft level simulation
def hv_books(pl, ev="test"):
    """(slot, site) -> tour minutes of the high value orders, in value order."""
    thr = A.HIGH_VALUE_THRESHOLD
    return {k: arr[arr[:, 1] >= thr, 0] for k, arr in pl.sc[ev].order_book.items()}


def site_distance(st, pl):
    xy = st.coords[pl.site_idx]
    return A.haversine_km(xy[:, 0][:, None], xy[:, 1][:, None], xy[:, 0][None, :], xy[:, 1][None, :])


def simulate_aircraft(st, pl, x, ev="test", overrun=True, repo_speed=None, repo_handle=0.0,
                      books=None, detail=False):
    """Serve the day's high value orders with individually tracked aircraft.

    Each of the aircraft is at one site in a slot and flies one sortie at a time.
    Orders of a slot are taken in value order (A4) and given to the aircraft of
    their site that is free first. A sortie may start only while its slot is
    running. With `overrun=True` a sortie that starts in the slot may finish
    after it, and the aircraft stays busy into the next slot, so every minute of
    flying is charged to a real aircraft. With `overrun=False` a sortie must
    also finish inside the slot.

    Between slots the fleet is rearranged to the next commitment x(t+1). Idle
    aircraft move first. With `repo_speed` (km/h) a moved aircraft is available
    at its new site only after the flight between the two sites (plus
    `repo_handle` minutes), and the moves are matched to minimise distance.
    Without it repositioning is instantaneous, as in the paper's model.
    """
    x = np.asarray(x, dtype=int)
    T, S = x.shape
    books = hv_books(pl, ev) if books is None else books
    D = site_distance(st, pl)
    M = int(x.sum(axis=1).max())
    site = np.full(M, -1, dtype=int)             # -1: reserve, not committed in this slot
    free = np.zeros(M)                           # minutes from slot start; <= 0 means ready
    served = np.zeros(T)
    moves = np.zeros(T, dtype=int)
    move_km = np.zeros(T)
    busy_min = np.zeros(T)
    for t in range(T):
        if t > 0:
            free -= SLOT
        # ---- rearrange the fleet to x[t]
        cur = np.bincount(site[site >= 0], minlength=S) if (site >= 0).any() else np.zeros(S, dtype=int)
        need = x[t] - cur
        givers = []
        for j in np.where(need < 0)[0]:
            idx = np.where(site == j)[0]
            idx = idx[np.argsort(free[idx], kind="stable")][: -need[j]]     # the most available leave
            givers.extend(idx.tolist())
        res = np.where(site == -1)[0]
        takers = np.repeat(np.where(need > 0)[0], need[need > 0])
        n_take = len(takers)
        if n_take:
            # reserve aircraft fill deficits at no cost (they wait at the destination),
            # so real moves are only those from another site
            pool = givers + res.tolist()
            cost = np.zeros((len(pool), n_take))
            for r, a in enumerate(pool):
                if site[a] >= 0:
                    cost[r] = D[site[a], takers] + 1e-3 * max(free[a], 0.0)
                else:
                    cost[r] = 1e3            # use reserve only when no site aircraft is released
            rr, cc = linear_sum_assignment(cost)
            used = set()
            for r, c in zip(rr, cc):
                a = pool[r]
                j2 = takers[c]
                if site[a] >= 0:
                    d = D[site[a], j2]
                    moves[t] += 1
                    move_km[t] += d
                    if repo_speed:
                        free[a] = max(free[a], 0.0) + d / repo_speed * 60.0 + repo_handle
                site[a] = j2
                used.add(a)
            for a in givers:
                if a not in used:
                    site[a] = -1
        else:
            for a in givers:
                site[a] = -1
        # ---- serve
        for j in np.where(x[t] > 0)[0]:
            taus = books.get((t, j))
            if taus is None or len(taus) == 0:
                continue
            idx = np.where(site == j)[0]
            h = [(max(free[a], 0.0), a) for a in idx]
            heapq.heapify(h)
            for tau in taus:
                f, a = h[0]
                if f >= SLOT or (not overrun and f + tau > SLOT + 1e-9):
                    if overrun:
                        break
                    # without overrun a shorter order further down may still fit
                    cand = [(ff, aa) for ff, aa in h if ff + tau <= SLOT + 1e-9]
                    if not cand:
                        continue
                    f, a = min(cand)
                    h.remove((f, a)); heapq.heapify(h)
                    heapq.heappush(h, (f + tau, a))
                else:
                    heapq.heapreplace(h, (f + tau, a))
                free[a] = f + tau
                served[t] += 1
                busy_min[t] += tau
    if detail:
        return dict(served=served, moves=moves, move_km=move_km, busy_min=busy_min)
    return served


# ------------------------------------------------------- allocation under a move budget
def marginal_tables(pl, kind, M, cap=CAP, cal="val", ev="test", model="AMSGCN", seed=0):
    """g[t, s, k]: value the policy attaches to the (k+1)-th aircraft at site s in
    slot t, with the estimate e[t, s] used for the D'Hondt tie break."""
    kmax = cap if cap is not None else M
    k = np.arange(kmax)
    if kind in ("floor", "foresight", "point"):
        e = {"floor": pl.static[ev], "foresight": pl.truth[ev]}.get(kind)
        if e is None:
            e = pl.forecast(model, ev)
        g = np.clip(e[:, :, None] - SLOT * k[None, None, :], 0.0, SLOT)
        return g, e
    if kind == "method":
        ec, ee = pl.forecast(model, cal), pl.forecast(model, ev)
        lam = pl.choose_lambda(ec, M, cal, cap)
        bc = (1 - lam) * pl.static[cal] + lam * ec
        be = (1 - lam) * pl.static[ev] + lam * ee
        smp = P.residual_sampler(bc, pl.truth[cal], seed=seed)(be)          # [T, S, N]
        g = np.stack([np.clip(smp - SLOT * kk, 0.0, SLOT).mean(axis=2) for kk in k], axis=2)
        return g, smp.mean(axis=2)
    raise ValueError(kind)


def allocate_move_budget(g, e, x_free, budget):
    """Follow a policy when at most `budget` aircraft may change site per slot.

    Slot 0 takes the policy's own allocation. In each later slot the fleet
    starts from the previous allocation and exchanges are made one at a time,
    always moving the aircraft whose current seat is worth least to the seat
    worth most under the policy's own marginal values, until no exchange gains
    or the budget is used. For a separable concave objective this greedy
    exchange is optimal under the budget. Ties (seats worth zero or a full
    slot) are ordered by the unmet estimate, as in the paper's allocation.
    """
    T, S, kmax = g.shape
    x = np.zeros((T, S), dtype=int)
    x[0] = x_free[0]
    for t in range(1, T):
        cur = x[t - 1].copy()
        for _ in range(int(budget)):
            # value of the last aircraft at each site, and of one more
            have = cur > 0
            last = np.where(have, g[t, np.arange(S), np.maximum(cur - 1, 0)], np.inf)
            last_tb = np.where(have, e[t] - SLOT * (cur - 1), np.inf)
            room = cur < kmax
            nxt = np.where(room, g[t, np.arange(S), np.minimum(cur, kmax - 1)], -np.inf)
            nxt_tb = np.where(room, e[t] - SLOT * cur, -np.inf)
            i = np.lexsort((last_tb, last))[0]            # least valuable seat
            j = np.lexsort((-nxt_tb, -nxt))[0]            # most valuable empty seat
            if i == j:
                break
            if (nxt[j], nxt_tb[j]) <= (last[i], last_tb[i]):
                break
            cur[i] -= 1
            cur[j] += 1
        x[t] = cur
    return x


# ------------------------------------------------------- studies under other settings
DEFAULT_FC = P.FC_HV
_ORIG_BUILD = A.build_scenario
ENV = dict(v=60.0, ts=4.0, od=8.0, tour_cap=None)


def _build_scenario(orders, coords, site_idx, slots, radius_km=3.0, grid_index=None,
                    positioning=False, **kw):
    """`allocation.build_scenario` under the operating parameters in ENV, with an
    optional limit on the closed tour length (site to pickup to dropoff to site)."""
    sc = _ORIG_BUILD(orders, coords, site_idx, slots, radius_km=radius_km, v_kmh=ENV["v"],
                     t_service=ENV["ts"], max_range_km=ENV["od"], grid_index=grid_index,
                     positioning=positioning)
    cap = ENV["tour_cap"]
    if cap is None:
        return sc
    lim = cap / ENV["v"] * 60.0 + ENV["ts"] + 1e-9
    thr = A.HIGH_VALUE_THRESHOLD
    for a in (sc.req_minutes, sc.req_minutes_hv, sc.req_orders, sc.req_orders_hv, sc.req_value):
        a[:] = 0.0
    book = {}
    for (i, j), arr in sc.order_book.items():
        arr = arr[arr[:, 0] <= lim]
        if len(arr) == 0:
            continue
        book[(i, j)] = arr
        hv = arr[:, 1] >= thr
        sc.req_minutes[i, j] = arr[:, 0].sum(); sc.req_minutes_hv[i, j] = arr[hv, 0].sum()
        sc.req_orders[i, j] = len(arr); sc.req_orders_hv[i, j] = int(hv.sum())
        sc.req_value[i, j] = arr[:, 1].sum()
    sc.order_book = book
    return sc


A.build_scenario = _build_scenario


def rerank_sites(st, thr, days):
    """Rank the candidate cells by the high value order endpoints (pickups plus
    dropoffs) they carried on the given days, the ranking the paper uses."""
    o = st.orders[st.orders["dt_seq_id"].isin(list(days)) & (st.orders["package_price"] >= thr)]
    cnt = {}
    for la, lo in (("pickup_latitude", "pickup_longitude"), ("dropoff_latitude", "dropoff_longitude")):
        keys = zip(P.D.snap(o[la], 0.010), P.D.snap(o[lo], 0.010))
        for k, v in pd.Series(list(keys)).value_counts().items():
            cnt[(round(float(k[0]), 6), round(float(k[1]), 6))] = cnt.get((round(float(k[0]), 6), round(float(k[1]), 6)), 0) + int(v)
    hv = np.array([cnt.get((round(float(a), 6), round(float(b), 6)), 0) for a, b in zip(st.cand["lat"], st.cand["lng"])], dtype=float)
    base_rank = st.cand["high_value_network_rank"].to_numpy()
    order = np.lexsort((base_rank, -hv))                 # ties keep the submitted order
    st.cand_hv = hv
    st.rank_order = st.cand_node[order]
    return hv


def make_study(fc=None, train_days=5, thr=53.90, v=60.0, ts=4.0, rho=3.0, od=8.0, tour_cap=None,
               rank_days=None):
    """A Study under other parameters or another chronological fold. The settings
    are module globals of the original code, so only the study made last is valid."""
    ENV.update(v=float(v), ts=float(ts), od=float(od), tour_cap=tour_cap)
    P.FC_HV = Path(fc) if fc else DEFAULT_FC
    C.TRAIN_DAYS = int(train_days)
    A.HIGH_VALUE_THRESHOLD = float(thr)
    A.MAX_SORTIE_RANGE_KM = float(od)
    st = P.Study(positioning=True, radius_km=float(rho))
    if rank_days is not None:
        rerank_sites(st, float(thr), rank_days)
    return st


def equiv_aircraft(fleets, curve, service, M):
    """Aircraft the floor would need beyond M to reach `service` (linear interpolation
    on the floor's fleet curve, made monotone)."""
    phi = np.maximum.accumulate(np.asarray(curve, dtype=float))
    if service > phi.max():
        return np.inf
    i = int(np.argmax(phi >= service))
    need = fleets[i] if i == 0 else np.interp(service, phi[i - 1:i + 1], fleets[i - 1:i + 1])
    return float(need - M)


def floor_curve(pl, ev, fleets, cap=CAP):
    return np.array([served_slots(pl, P.allocate_point(pl.static[ev], int(m), cap), ev).sum() for m in fleets])
