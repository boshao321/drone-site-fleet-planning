"""Shared machinery for the planning-and-dispatch study.

Three decisions are studied together, on one evaluation day, one demand target
and one measure of service:

    planning   how many landing sites to build (K) and where
    planning   how many aircraft to buy (M)
    dispatch   how many aircraft stand at each site in each 15-minute slot,
               decided an hour ahead, before the slot's demand is known

Service is always counted as high-value air-servable orders served, divided by
the same orders over the whole district on the scored slots. That denominator
does not depend on where the pads are or how many there are, which is what lets
a pad, an aircraft and a dispatch rule be compared on one scale.

Cost model. A sortie is charged the full closed tour, site to pickup to dropoff
and back. With the pad placed at the pickup, as the earlier stage of the study
did, the leg from pad to demand is free, so pad density could only matter
through reach and the planning question would be answered by assumption. The
mean tour of a high-value sortie, which converts a forecast count into
aircraft-minutes, is re-estimated on the training days for every site set.

Dispatch rules compared (all exact optima of the program they solve):

    floor        time-of-day mean of the training days, whole fleet deployed
    unc_static   uncertainty-aware allocation centred on the time-of-day mean
                 (the proposed allocation rule with no forecast at all)
    point        a model's t+60 forecast, whole fleet deployed
    method       the proposed rule: forecast shrunk toward the time-of-day mean
                 with a weight chosen on the calibration day, a predictive
                 distribution from calibration-day residuals, and aircraft
                 assigned by expected marginal gain
    foresight    the realised requirement passed through the same program
    ceiling      the most high-value orders the fleet could serve on the
                 realised book (dynamic programme over the fleet)
"""
from __future__ import annotations

import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "09_Forecasting_Rebuilt"))
sys.path.insert(0, str(ROOT / "10_Allocation" / "scripts"))
from core import config as C            # noqa: E402
from core import data as D              # noqa: E402
import allocation as A                   # noqa: E402
from run_allocation import prepare_orders   # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
FC_HV = ROOT / "10_Allocation" / "outputs_hv" / "forecasts"
FC_ALL = ROOT / "10_Allocation" / "outputs" / "forecasts"
T_ST = ROOT / "07_NetworkBenefit_KStar_NoWeights" / "outputs" / "tables"
OFF = C.SEQ_LEN + C.PRED_LEN - 1
SLOT = A.SLOT_MINUTES
NS = 300                                    # predictive samples per site-slot
LAMBDAS = np.linspace(0.0, 1.0, 11)
CURVE_MMAX = 800                            # fleet axis of the ceiling curve
MODELS = ["AMSGCN", "AGCRN", "GraphWaveNet", "DCRNN", "STGCN", "LSTM", "Ridge",
          "SeasonalNaive", "HA"]


# ================================================================ allocation
def _dhondt_fill(x_t, e, r, cap):
    """Deploy r spare aircraft by the D'Hondt rule on estimate e, never putting
    more than `cap` aircraft at one site (cap None means no limit)."""
    S = len(e)
    e = np.maximum(np.asarray(e, dtype=float), 1e-9)
    j = np.arange(1, r + 1)
    q = e[:, None] / (x_t[:, None] + j[None, :])
    if cap is not None:
        q = np.where(x_t[:, None] + j[None, :] <= cap, q, -np.inf)
    order = np.lexsort((np.tile(j, S), np.repeat(np.arange(S), r), -q.ravel()))[:r]
    np.add.at(x_t, order // r, 1)


def allocate_point(estimate_minutes, M, cap=None):
    """`allocation.allocate` with an optional limit of `cap` aircraft per pad.

    With a limit the program  max sum_s min(15 x_s, e_s)  s.t. sum x_s <= M,
    0 <= x_s <= cap  is still separable and concave with box constraints, so the
    same descending sort on unmet estimate, truncated at cap, is exact. When the
    fleet exceeds what the pads can hold, the excess stays on the ground."""
    if cap is None:
        return A.allocate(estimate_minutes, M)
    T, S = estimate_minutes.shape
    x = np.zeros((T, S), dtype=int)
    M = int(min(M, S * cap))
    if M <= 0:
        return x
    for t in range(T):
        e = np.maximum(estimate_minutes[t], 0.0)
        need = np.minimum(np.ceil(e / SLOT).astype(int), cap)
        if need.sum() <= M:
            x[t] = need
        else:
            sites = np.repeat(np.arange(S), need)
            k_idx = np.concatenate([np.arange(c) for c in need])
            unmet = e[sites] - SLOT * k_idx
            order = np.lexsort((sites, -unmet))[:M]
            np.add.at(x[t], sites[order], 1)
        r = M - int(x[t].sum())
        if r > 0:
            _dhondt_fill(x[t], estimate_minutes[t], r, cap)
    return x


def ceiling_curve(sc, Mmax, cap=None):
    """Attainable ceiling for every fleet size 0..Mmax at once.

    The dynamic programme of `allocation.attainable_ceiling`, kept over the
    whole budget axis: dp[m] is the most high-value orders a slot can yield
    with at most m aircraft, so one pass gives the ceiling for every fleet.
    Returns an array [T, Mmax + 1] of high-value orders served per slot."""
    T = sc.req_minutes.shape[0]
    out = np.zeros((T, Mmax + 1))
    books = {}
    for (t, j), arr in sc.order_book.items():
        books.setdefault(t, []).append(arr)
    for t, entries in books.items():
        dp = np.zeros(Mmax + 1)
        for arr in entries:
            cum = np.cumsum(arr[:, 0])
            hv_cum = np.concatenate(([0.0], np.cumsum(arr[:, 1] >= A.HIGH_VALUE_THRESHOLD)))
            kmax = int(min(Mmax, np.ceil(cum[-1] / SLOT)))
            if cap is not None:
                kmax = min(kmax, cap)
            served = np.searchsorted(cum, SLOT * np.arange(kmax + 1), side="right")
            f = hv_cum[served]
            new = dp.copy()                       # k = 0
            for k in range(1, kmax + 1):
                np.maximum(new[k:], dp[:Mmax + 1 - k] + f[k], out=new[k:])
            dp = new
        out[t] = dp
    return out


def ceiling_alloc(sc, M, cap=None):
    """`allocation.attainable_ceiling` with an optional limit of `cap` aircraft
    per pad: the most high-value orders the fleet can serve on the realised
    book, and the allocation that attains it."""
    if cap is None:
        return A.attainable_ceiling(sc, M)
    M = int(M)
    T = sc.req_minutes.shape[0]
    x = np.zeros(sc.req_minutes.shape, dtype=int)
    books = {}
    for (t, j), arr in sc.order_book.items():
        books.setdefault(t, []).append((j, arr))
    total = 0.0
    for t in range(T):
        entries = books.get(t, [])
        if not entries or M <= 0:
            continue
        dp = np.zeros(M + 1)
        steps = []
        for j, arr in entries:
            cum = np.cumsum(arr[:, 0])
            hv_cum = np.concatenate(([0.0], np.cumsum(arr[:, 1] >= A.HIGH_VALUE_THRESHOLD)))
            kmax = int(min(M, cap, np.ceil(cum[-1] / SLOT)))
            served = np.searchsorted(cum, SLOT * np.arange(kmax + 1), side="right")
            f = hv_cum[served]
            new = dp.copy()
            arg = np.zeros(M + 1, dtype=int)
            for k in range(1, kmax + 1):
                cand = dp[:M + 1 - k] + f[k]
                better = cand > new[k:]
                new[k:][better] = cand[better]
                arg[k:][better] = k
            dp = new
            steps.append((j, arg))
        total += dp[M]
        m = M
        for j, arg in reversed(steps):
            k = int(arg[m])
            x[t, j] = k
            m -= k
    return float(total), x


def allocate_expected(samples, M, cap=None):
    """Exact optimum of the sample-average objective
        (1/N) sum_n sum_s min(15 x_s, R_s^n)   s.t.  sum_s x_s <= M,
    over predictive samples R [T, S, N] of the aircraft-minutes each site needs.

    The k-th aircraft at site s adds g_{s,k} = mean_n min(15, (R_s^n - 15(k-1))^+),
    which is non-increasing in k because each term is. The objective is
    therefore separable and concave in every x_s, and taking the M largest
    marginal gains over all sites at once is exact. Aircraft whose expected gain
    is zero are deployed by the same D'Hondt tie-break as `allocation.allocate`,
    on the sample mean, so the whole fleet is always in the air.
    """
    T, S, _ = samples.shape
    x = np.zeros((T, S), dtype=int)
    M = int(M) if cap is None else int(min(M, S * cap))
    for t in range(T):
        top_need = float(samples[t].max())
        kmax = int(min(M, np.ceil(top_need / SLOT))) if top_need > 0 else 0
        if cap is not None:
            kmax = min(kmax, cap)
        if kmax > 0:
            k = np.arange(kmax)
            g = np.clip(samples[t][:, None, :] - SLOT * k[None, :, None], 0.0, SLOT).mean(axis=2)
            # ties in expected gain (typically aircraft that are busy for the
            # whole slot in every sample) are broken by the unmet mean estimate
            # e_s - 15(k-1), the order the point allocation uses, so that with
            # no spread the rule returns exactly the point allocation
            unmet = samples[t].mean(axis=1)[:, None] - SLOT * k[None, :]
            flat = g.ravel()
            top = np.lexsort((-unmet.ravel(), -flat))[:M]
            top = top[flat[top] > 0]
            np.add.at(x[t], top // kmax, 1)
        r = M - int(x[t].sum())
        if r > 0:
            _dhondt_fill(x[t], samples[t].mean(axis=1), r, cap)
    return x


def residual_sampler(est_cal, truth_cal, nbins=6, seed=0):
    """Predictive samples around an estimate: the estimate plus an additive
    residual drawn from the calibration day, from the bin of calibration
    estimates of similar size. Residuals are taken in aircraft-minutes, the
    unit the program is solved in, and samples are truncated at zero."""
    rng = np.random.default_rng(seed)
    p, e = est_cal.ravel(), (truth_cal - est_cal).ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nbins + 1)))
    nb = max(len(edges) - 1, 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nb - 1)
    pools = [e[b == k] for k in range(nb)]

    def sample(point):
        bb = np.clip(np.searchsorted(edges, point, side="right") - 1, 0, nb - 1)
        out = np.repeat(point[..., None], NS, axis=-1).astype(float)
        for k, pool in enumerate(pools):
            m = bb == k
            if m.any() and len(pool):
                out[m] += rng.choice(pool, size=(int(m.sum()), NS))
        return np.maximum(out, 0.0)
    return sample


def served_per_slot(x, sc):
    """High-value orders served in each slot, value-first dispatch at each site."""
    out = np.zeros(x.shape[0])
    for (t, j), arr in sc.order_book.items():
        cap = x[t, j] * SLOT
        if cap <= 0:
            continue
        k = int(np.searchsorted(np.cumsum(arr[:, 0]), cap, side="right"))
        out[t] += (arr[:k, 1] >= A.HIGH_VALUE_THRESHOLD).sum()
    return out


# ================================================================ study data
class Study:
    """Everything that does not depend on the plan."""

    def __init__(self, positioning=True, radius_km=A.SERVICE_RADIUS_KM):
        self.positioning, self.radius = positioning, radius_km
        self.coords = np.load(FC_HV / "node_coords.npy")
        self.demand_hv = np.load(FC_HV / "demand_full.npy")      # [672, 120] HV orders per grid
        self.orders = prepare_orders()
        self.sp = D.build_spatial(D.load_orders(), 0.010, include_dropoff=True)
        self.orders["node"] = [self.sp.index.get(k, -1) for k in
                               zip(self.orders["p_lat_snap"], self.orders["p_lng_snap"])]
        self.train = np.arange(0, C.TRAIN_DAYS * C.SLOTS_PER_DAY)
        self.slots = {"val": np.load(FC_HV / "val_starts.npy") + OFF,
                      "test": np.load(FC_HV / "test_starts.npy") + OFF}
        o = self.orders
        base = o[(o["node"] >= 0) & (o["od_km"] <= A.MAX_SORTIE_RANGE_KM)
                 & (o["package_price"] >= A.HIGH_VALUE_THRESHOLD)]
        self.district = {d: int(base["slot"].isin(set(s.tolist())).sum())
                         for d, s in self.slots.items()}
        self.tod_train = {}
        for sl in range(C.SLOTS_PER_DAY):
            self.tod_train[sl] = self.train[self.train % C.SLOTS_PER_DAY == sl]
        # candidates, in the order the siting stage ranks them
        cand = pd.read_csv(T_ST / "candidate_network_indicators_no_weights.csv")
        key = {(round(float(a), 6), round(float(b), 6)): i for i, (a, b) in enumerate(self.coords)}
        self.cand = cand
        self.cand_node = np.array([key[(round(float(a), 6), round(float(b), 6))]
                                   for a, b in zip(cand["lat"], cand["lng"])])
        self.rank_order = self.cand_node[np.argsort(cand["high_value_network_rank"].to_numpy())]
        self.cand_hv = cand["high_value_orders"].to_numpy(dtype=float)
        la, lo = cand["lat"].to_numpy(), cand["lng"].to_numpy()
        self.cand_dist = A.haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        # forecasts: [windows, horizon, node] for the calibration and scored days
        self.pred = {}
        for tgt, fc in (("hv", FC_HV), ("all", FC_ALL)):
            for m in MODELS:
                for d, pre in (("val", "valpred_"), ("test", "pred_")):
                    f = fc / f"{pre}{m}.npy"
                    if f.exists():
                        self.pred[(tgt, m, d)] = np.load(f)
        # share of training-day air-servable orders that are high value, used to
        # turn a total-demand forecast into a high-value one
        tr = o[o["slot"].isin(set(self.train.tolist())) & (o["node"] >= 0)
               & (o["od_km"] <= A.MAX_SORTIE_RANGE_KM)]
        self.rho = float((tr["package_price"] >= A.HIGH_VALUE_THRESHOLD).mean())

    # ------------------------------------------------------------ siting rules
    def sites(self, rule, K):
        K = int(K)
        if rule == "proposed":
            return np.array(sorted(self.rank_order[:K].tolist()))
        if rule == "pmedian":
            return np.array(sorted(self.cand_node[list(_pmedian(self, K))].tolist()))
        raise ValueError(rule)

    @functools.lru_cache(maxsize=None)
    def plan(self, rule, K):
        return Plan(self, self.sites(rule, K), rule, K)


def _pmedian(study, K, restarts=20):
    """Teitz-Bart vertex substitution on demand-weighted distance (the siting
    baseline of Section 5), with a greedy seed plus random restarts."""
    p = ROOT / "13_SitingBaselines" / "scripts" / "run_siting_baselines.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("sb_mod", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sb_mod"] = m
    spec.loader.exec_module(m)
    chosen, _ = m.p_median(study.cand_dist, study.cand_hv, K, seeds=list(range(restarts)))
    return tuple(chosen)


# ================================================================ one plan
@dataclass
class Plan:
    study: Study
    site_idx: np.ndarray
    rule: str
    K: int
    sc: dict = field(default_factory=dict)

    def __post_init__(self):
        s = self.study
        self.g2s, _ = A.assign_grids_to_sites(s.coords, self.site_idx, s.radius)
        self.S = len(self.site_idx)
        for d, sl in s.slots.items():
            self.sc[d] = A.build_scenario(s.orders, s.coords, self.site_idx, sl,
                                          radius_km=s.radius, grid_index=s.sp.index,
                                          positioning=s.positioning)
        tr = A.build_scenario(s.orders, s.coords, self.site_idx, s.train,
                              radius_km=s.radius, grid_index=s.sp.index,
                              positioning=s.positioning)
        taus = [arr[arr[:, 1] >= A.HIGH_VALUE_THRESHOLD, 0] for arr in tr.order_book.values()]
        taus = np.concatenate([t for t in taus if len(t)])
        self.tau = float(taus.mean())
        self.reach = {d: float(self.sc[d].req_orders_hv.sum()) for d in s.slots}
        site_hv = np.zeros((s.demand_hv.shape[0], self.S))
        for n in range(s.demand_hv.shape[1]):
            if self.g2s[n] >= 0:
                site_hv[:, self.g2s[n]] += s.demand_hv[:, n]
        self.site_hv = site_hv
        tod_mean = np.stack([site_hv[s.tod_train[k]].mean(axis=0) for k in range(C.SLOTS_PER_DAY)])
        self.static = {d: tod_mean[sl % C.SLOTS_PER_DAY] * self.tau for d, sl in s.slots.items()}
        self.truth = {d: self.sc[d].req_minutes_hv for d in s.slots}
        self._served_cache = {}

    # ------------------------------------------------------------ estimates
    def forecast(self, model, day, h=3, target="hv"):
        p = self.study.pred[(target, model, day)][:, h, :]
        scale = 1.0 if target == "hv" else self.study.rho
        return A.nodes_to_sites(p, self.g2s, self.S) * self.tau * scale

    # ------------------------------------------------------------ scoring
    def served(self, x, day):
        return served_per_slot(x, self.sc[day])

    def share(self, served_slots, day):
        return 100.0 * float(np.sum(served_slots)) / self.study.district[day]

    def ceiling(self, M, day, cap=None):
        if cap is None:
            key = ("ceiling", M, day)
            if key not in self._served_cache:
                v, x = A.attainable_ceiling(self.sc[day], M)
                self._served_cache[key] = (v, self.served(x, day))
            return self._served_cache[key][1]
        return self.ceiling_curve(day, cap)[:, min(int(M), CURVE_MMAX)]

    def ceiling_curve(self, day, cap=None):
        key = ("curve", day, cap)
        if key not in self._served_cache:
            self._served_cache[key] = ceiling_curve(self.sc[day], CURVE_MMAX, cap)
        return self._served_cache[key]

    # ------------------------------------------------------------ policies
    def policy(self, name, M, cal="val", ev="test", est_cal=None, est_ev=None, seed=0,
               lam=None, cap=None):
        """Per-slot served high-value orders on the scored day, plus the chosen
        shrinkage weight when the policy has one. `cap` limits the aircraft a
        single pad can hold in a slot (None: no limit)."""
        if name == "floor":
            return self.served(allocate_point(self.static[ev], M, cap), ev), None
        if name == "foresight":
            return self.served(allocate_point(self.truth[ev], M, cap), ev), None
        if name == "ceiling":
            return self.ceiling(M, ev, cap), None
        if name == "unc_static":
            smp = residual_sampler(self.static[cal], self.truth[cal], seed=seed)
            return self.served(allocate_expected(smp(self.static[ev]), M, cap), ev), 0.0
        if name == "point":
            return self.served(allocate_point(est_ev, M, cap), ev), 1.0
        if name in ("method", "blend"):
            if lam is None:
                lam = self.choose_lambda(est_cal, M, cal, cap)
            bc = (1 - lam) * self.static[cal] + lam * est_cal
            be = (1 - lam) * self.static[ev] + lam * est_ev
            if name == "blend":
                return self.served(allocate_point(be, M, cap), ev), lam
            smp = residual_sampler(bc, self.truth[cal], seed=seed)
            return self.served(allocate_expected(smp(be), M, cap), ev), lam
        raise ValueError(name)

    def choose_lambda(self, est_cal, M, cal, cap=None):
        """Shrinkage weight toward the time-of-day mean, chosen by the service it
        delivers on the calibration day. Ties go to the smaller weight, so a
        forecast has to earn its influence."""
        vals = [self.served(allocate_point((1 - l) * self.static[cal] + l * est_cal, M, cap),
                            cal).sum()
                for l in LAMBDAS]
        return float(LAMBDAS[int(np.argmax(vals))])
