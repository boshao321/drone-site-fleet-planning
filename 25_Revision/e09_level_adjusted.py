"""E9 (R2.1): a historical baseline adjusted for the level of the scored day.

The floor of the paper is the time-of-day mean of the training days. The two
scored days are a weekend and run 17 to 18% above that mean. The adjusted
baseline scales the profile by the ratio of demand observed so far to the
profile over the same slots, using only slots up to t - L (the information
the dispatcher has when it commits):

    e_adj_s(t) = kappa(t) * e0_s(t),
    kappa(t)   = sum_{u in W(t)} Y(u) / sum_{u in W(t)} Ybar(h(u))

with Y the district count of high value orders inside the network. W(t) is the
last hour, the last three hours (the window the forecasting models see) or the
day so far. A site-level factor is also tried.

The proposed rule is then run twice: as in the paper (shrinking toward the
unadjusted mean) and on top of the adjusted mean.
"""
import sys, time
import numpy as np, pandas as pd
import revlib as R
P = R.P
LEAD = 4
t0 = time.time()
st = P.Study()

def adjusted(pl, day, window, site_level=False, clip=(0.5, 2.0)):
    sl = st.slots[day]
    hv = pl.site_hv                                             # [672, S] realised counts
    tod = np.stack([hv[st.tod_train[k]].mean(axis=0) for k in range(96)])
    out = np.zeros((len(sl), pl.S)); kap = np.ones(len(sl))
    for i, t in enumerate(sl):
        last = t - LEAD                                         # last observed slot
        if window == "day":
            first = (t // 96) * 96
        else:
            first = last - window + 1
        u = np.arange(max(first, 0), last + 1)
        if len(u) == 0:
            k = np.ones(pl.S) if site_level else 1.0
        elif site_level:
            num, den = hv[u].sum(axis=0), tod[u % 96].sum(axis=0)
            # shrink the site ratio toward the district ratio with one hour of pseudo demand
            kd = hv[u].sum() / max(tod[u % 96].sum(), 1e-9)
            k = (num + kd * 1.0) / (den + 1.0)
        else:
            den = tod[u % 96].sum()
            k = hv[u].sum() / den if den > 5 else 1.0
        k = np.clip(k, *clip)
        kap[i] = np.mean(k)
        out[i] = k * tod[t % 96] * pl.tau
    return out, kap

def run(pl, M, cal, ev, base_c, base_e, model="AMSGCN", seeds=range(5)):
    """service per slot of: point allocation on the base estimate, the proposed rule with no
    forecast around it, and the proposed rule shrinking the forecast toward it."""
    cap = R.CAP
    out = {"base": P.served_per_slot(P.allocate_point(base_e, M, cap), pl.sc[ev])}
    out["unc"] = np.mean([P.served_per_slot(P.allocate_expected(P.residual_sampler(base_c, pl.truth[cal], seed=s)(base_e), M, cap), pl.sc[ev]) for s in seeds], axis=0)
    ec, ee = pl.forecast(model, cal), pl.forecast(model, ev)
    vals = [P.served_per_slot(P.allocate_point((1 - l) * base_c + l * ec, M, cap), pl.sc[cal]).sum() for l in P.LAMBDAS]
    lam = float(P.LAMBDAS[int(np.argmax(vals))])
    bc, be = (1 - lam) * base_c + lam * ec, (1 - lam) * base_e + lam * ee
    out["method"] = np.mean([P.served_per_slot(P.allocate_expected(P.residual_sampler(bc, pl.truth[cal], seed=s)(be), M, cap), pl.sc[ev]) for s in seeds], axis=0)
    return out, lam

rows, kappas = [], []
for K in (45, 38):
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]
        variants = {"unadjusted": (pl.static[cal], pl.static[ev])}
        for name, (w, sl_) in {"last hour": (4, False), "last 3 hours": (12, False), "day so far": ("day", False),
                               "last 3 hours, by site": (12, True)}.items():
            (c, kc), (e, ke) = adjusted(pl, cal, w, sl_), adjusted(pl, ev, w, sl_)
            variants[name] = (c, e)
            if K == 45:
                kappas.append(dict(fold=f"{cal}->{ev}", variant=name, kappa_mean=ke.mean(), kappa_min=ke.min(), kappa_max=ke.max(),
                                   kappa_peak=ke[(st.slots[ev] % 96 >= 44) & (st.slots[ev] % 96 < 56)].mean()))
        for M in (100, 200, 300):
            if M > K * R.CAP: continue
            floor = P.served_per_slot(P.allocate_point(pl.static[ev], M, R.CAP), pl.sc[ev])
            ceil = pl.ceiling(M, ev, R.CAP)
            res = {}
            for name, (c, e) in variants.items():
                res[name], lam = run(pl, M, cal, ev, c, e)
                res[name]["lam"] = lam
            for name, r in res.items():
                rec = dict(K=K, fold=f"{cal}->{ev}", M=M, baseline=name, lam=r["lam"], floor_unadj=100 * floor.sum() / N,
                           ceiling=100 * ceil.sum() / N, baseline_service=100 * r["base"].sum() / N,
                           unc_service=100 * r["unc"].sum() / N, method_service=100 * r["method"].sum() / N)
                for lab, d in (("baseline_vs_floor", r["base"] - floor), ("method_vs_baseline", r["method"] - r["base"]),
                               ("unc_vs_baseline", r["unc"] - r["base"]),
                               ("paper_method_vs_baseline", res["unadjusted"]["method"] - r["base"])):
                    (lo, hi), _ = R.boot_ci(d)
                    rec[lab + "_pp"], rec[lab + "_lo"], rec[lab + "_hi"] = 100 * d.sum() / N, 100 * lo / N, 100 * hi / N
                rows.append(rec)
            print(f"K={K} {ev} M={M} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(R.OUT / "e09_level_adjusted.csv", index=False)
pd.DataFrame(kappas).to_csv(R.OUT / "e09_kappa.csv", index=False)
pd.set_option("display.width", 250)
d = pd.DataFrame(rows)
print(pd.DataFrame(kappas).round(3).to_string(index=False))
print(d[["K", "fold", "M", "baseline", "lam", "floor_unadj", "baseline_service", "unc_service", "method_service", "baseline_vs_floor_pp", "baseline_vs_floor_lo", "baseline_vs_floor_hi",
         "method_vs_baseline_pp", "method_vs_baseline_lo", "method_vs_baseline_hi", "paper_method_vs_baseline_pp", "paper_method_vs_baseline_lo", "paper_method_vs_baseline_hi"]].round(2).to_string(index=False))
