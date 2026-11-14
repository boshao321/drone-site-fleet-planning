"""Dispatch experiments at the reference plan (38 demand-driven sites, tour cost).

D1  every forecasting model, point forecast against the proposed rule, both
    folds (calibrate on the validation day and score the test day, then the
    reverse), paired block-bootstrap intervals over one-hour blocks
D2  lead time: the forecast horizon the dispatcher must commit at
D3  the accuracy a forecast must reach: synthetic forecasts with lognormal
    error of a known coefficient of variation, point against proposed rule
D4  target alignment: forecasts of total demand rescaled by the high-value
    share, point against proposed rule
D5  the five siting rules of Section 5 at K = 38, under every dispatch rule
"""
from __future__ import annotations

import importlib.util
import sys
import time

import numpy as np
import pandas as pd

import core_pd as P

A = P.A
FLEETS = [50, 100, 150, 200, 300, 400]
B = 2000
SEEDS = [0, 1, 2, 3, 4]
rng_boot = np.random.default_rng(7)
CAP = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--cap=")), None)


def boot_ci(diff, T):
    blocks = [np.arange(i, min(i + 4, T)) for i in range(0, T, 4)]
    bs = np.empty(B)
    for b in range(B):
        idx = np.concatenate([blocks[i] for i in rng_boot.integers(0, len(blocks), len(blocks))])
        bs[b] = diff[idx].sum()
    return np.percentile(bs, [2.5, 97.5])


def d1_models(st, pl):
    rows = []
    for cal, ev in (("val", "test"), ("test", "val")):
        T = len(st.slots[ev])
        tot = st.district[ev]
        for M in FLEETS:
            base = {n: pl.policy(n, M, cal, ev, cap=CAP)[0] for n in ("floor", "unc_static", "foresight", "ceiling")}
            floor, ceil = base["floor"].sum(), base["ceiling"].sum()
            gap = ceil - floor
            pct = lambda v: 100.0 * v.sum() / tot
            G = lambda v: 100.0 * (v.sum() - floor) / gap
            lo, hi = boot_ci(base["unc_static"] - base["floor"], T)
            common = dict(fold=f"{cal}->{ev}", M=M, floor=pct(base["floor"]),
                          ceiling=pct(base["ceiling"]), gap_pp=100 * gap / tot,
                          foresight_G=G(base["foresight"]), unc_static_G=G(base["unc_static"]),
                          unc_static_vs_floor_pp=100 * (base["unc_static"] - base["floor"]).sum() / tot,
                          unc_static_vs_floor_ci=f"[{100*lo/tot:.2f}, {100*hi/tot:.2f}]")
            for m in P.MODELS:
                ec, ee = pl.forecast(m, cal), pl.forecast(m, ev)
                point = pl.policy("point", M, cal, ev, ec, ee, cap=CAP)[0]
                blend, lam = pl.policy("blend", M, cal, ev, ec, ee, cap=CAP)
                meth = [pl.policy("method", M, cal, ev, ec, ee, seed=s, lam=lam, cap=CAP)[0] for s in SEEDS]
                mean_meth = np.mean(meth, axis=0)
                rec = dict(common, model=m, **{"lambda": lam},
                           point_G=G(point), blend_G=G(blend), method_G=G(mean_meth),
                           method_G_seed_sd=float(np.std([G(x) for x in meth], ddof=1)),
                           point=pct(point), method=pct(mean_meth))
                for nm, d in (("method_vs_point", mean_meth - point),
                              ("method_vs_floor", mean_meth - base["floor"]),
                              ("point_vs_floor", point - base["floor"])):
                    lo, hi = boot_ci(d, T)
                    rec[f"{nm}_pp"] = 100 * d.sum() / tot
                    rec[f"{nm}_lo"], rec[f"{nm}_hi"] = 100 * lo / tot, 100 * hi / tot
                rows.append(rec)
            print(f"  D1 {cal}->{ev} M={M} done", flush=True)
    return pd.DataFrame(rows)


def d2_lead(st, pl):
    rows = []
    for h in range(4):
        for M in (100, 200, 300):
            floor = pl.policy("floor", M, cap=CAP)[0].sum()
            ceil = pl.policy("ceiling", M, cap=CAP)[0].sum()
            for m in ("AMSGCN", "AGCRN", "GraphWaveNet", "DCRNN"):
                ec, ee = pl.forecast(m, "val", h), pl.forecast(m, "test", h)
                p = pl.policy("point", M, est_cal=ec, est_ev=ee, cap=CAP)[0].sum()
                q = pl.policy("method", M, est_cal=ec, est_ev=ee, cap=CAP)[0].sum()
                rows.append(dict(lead_min=15 * (h + 1), M=M, model=m,
                                 point_G=100 * (p - floor) / (ceil - floor),
                                 method_G=100 * (q - floor) / (ceil - floor)))
    return pd.DataFrame(rows)


def d3_accuracy(st, pl, reps=20):
    rng = np.random.default_rng(11)
    rows = []
    cvs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]
    for M in (50, 100, 200, 300):
        floor = pl.policy("floor", M, cap=CAP)[0].sum()
        ceil = pl.policy("ceiling", M, cap=CAP)[0].sum()
        G = lambda v: 100 * (v - floor) / (ceil - floor)
        for cv in cvs:
            pv, mv, lams = [], [], []
            for r in range(reps if cv > 0 else 1):
                if cv == 0:
                    ec, ee = pl.truth["val"].copy(), pl.truth["test"].copy()
                else:
                    sig = np.sqrt(np.log(1 + cv ** 2))
                    ec = pl.truth["val"] * rng.lognormal(-0.5 * sig ** 2, sig, pl.truth["val"].shape)
                    ee = pl.truth["test"] * rng.lognormal(-0.5 * sig ** 2, sig, pl.truth["test"].shape)
                pv.append(G(pl.policy("point", M, est_cal=ec, est_ev=ee, cap=CAP)[0].sum()))
                s, lam = pl.policy("method", M, est_cal=ec, est_ev=ee, seed=r, cap=CAP)
                mv.append(G(s.sum())); lams.append(lam)
            rows.append(dict(M=M, cv=cv, point_G=np.mean(pv), method_G=np.mean(mv),
                             point_sd=np.std(pv), method_sd=np.std(mv), lambda_mean=np.mean(lams)))
        print(f"  D3 M={M} done", flush=True)
    return pd.DataFrame(rows)


def d4_mismatch(st, pl):
    rows = []
    for M in (50, 100, 200, 300):
        floor = pl.policy("floor", M, cap=CAP)[0].sum()
        ceil = pl.policy("ceiling", M, cap=CAP)[0].sum()
        G = lambda v: 100 * (v - floor) / (ceil - floor)
        for m in P.MODELS:
            if ("all", m, "test") not in st.pred:
                continue
            for tgt in ("hv", "all"):
                ec, ee = pl.forecast(m, "val", target=tgt), pl.forecast(m, "test", target=tgt)
                p = pl.policy("point", M, est_cal=ec, est_ev=ee, cap=CAP)[0].sum()
                q = pl.policy("method", M, est_cal=ec, est_ev=ee, cap=CAP)[0].sum()
                rows.append(dict(M=M, model=m, target=tgt, point_G=G(p), method_G=G(q)))
    return pd.DataFrame(rows)


def d5_siting(st):
    p = P.ROOT / "15_SitingDownstream" / "scripts" / "run_siting_downstream.py"
    spec = importlib.util.spec_from_file_location("sd_mod", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sd_mod"] = m
    spec.loader.exec_module(m)
    sets = m.build_site_sets(st.coords)
    rows = []
    for name, idx in sets.items():
        pl = P.Plan(st, np.asarray(idx), name, 38)
        ec, ee = pl.forecast("AMSGCN", "val"), pl.forecast("AMSGCN", "test")
        for M in (50, 100, 200, 300):
            rec = dict(site_set=name, M=M, tau=pl.tau,
                       reach_pct=100 * pl.reach["test"] / st.district["test"])
            for n in ("floor", "unc_static", "ceiling"):
                rec[n] = pl.share(pl.policy(n, M, cap=CAP)[0], "test")
            rec["point"] = pl.share(pl.policy("point", M, est_cal=ec, est_ev=ee, cap=CAP)[0], "test")
            rec["method"] = pl.share(pl.policy("method", M, est_cal=ec, est_ev=ee, cap=CAP)[0], "test")
            rows.append(rec)
        print(f"  D5 {name} done", flush=True)
    return pd.DataFrame(rows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rule, K = args[0], int(args[1])
    which = args[2:] or ["d1", "d2", "d3", "d4", "d5"]
    tag = f"{rule}{K}" + (f"_c{CAP}" if CAP else "")
    global FLEETS
    if CAP:
        FLEETS = [m for m in FLEETS if m <= K * CAP]
    st = P.Study(positioning=True)
    pl = st.plan(rule, K)
    t0 = time.time()
    if "d1" in which:
        d1_models(st, pl).to_csv(P.OUT / f"d1_models_{tag}.csv", index=False)
        # floor curve on the validation day too, for the swapped fold
        pd.DataFrame([{"M": M, "floor": pl.share(pl.policy("floor", M, "test", "val", cap=CAP)[0], "val")}
                      for M in range(5, 805, 5)]).to_csv(P.OUT / f"floor_curve_val_{tag}.csv", index=False)
    if "d2" in which:
        d2_lead(st, pl).to_csv(P.OUT / f"d2_lead_{tag}.csv", index=False)
    if "d3" in which:
        d3_accuracy(st, pl).to_csv(P.OUT / f"d3_accuracy_{tag}.csv", index=False)
    if "d4" in which:
        d4_mismatch(st, pl).to_csv(P.OUT / f"d4_mismatch_{tag}.csv", index=False)
    if "d5" in which:
        d5_siting(st).to_csv(P.OUT / "d5_siting.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
