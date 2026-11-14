"""E11 (R3.1, R3.3): the 198 comparisons of the proposed rule with the floor.

198 = 9 forecasting models x 11 plan and fleet pairs (38 sites with 50 to 300
aircraft, 45 sites with 50 to 400) x 2 folds. Each comparison is a paired block
bootstrap over one-hour blocks of the scored day (24 blocks, 20,000
replicates). Reported per comparison: the difference, the unadjusted 95%
interval, a two-sided bootstrap p value, and the Holm and Bonferroni decisions
at a familywise level of 5%. The same is done with every allocation replayed at
aircraft level with repositioning flights.

Outputs: e11_comparisons.csv, e11_summary.csv
"""
import sys, time
import numpy as np, pandas as pd
from scipy import stats
import revlib as R
P = R.P
B = 100000
REPO_V, REPO_H = 60.0, 1.0
PLANS = [(38, [50, 100, 150, 200, 300]), (45, [50, 100, 150, 200, 300, 400])]
t0 = time.time()
st = R.make_study()
rows = []
for K, fleets in PLANS:
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev); fold = f"{cal}->{ev}"
        sim = lambda x: R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H)
        for M in fleets:
            xf = P.allocate_point(pl.static[ev], M, R.CAP)
            fl = {"aggregate": R.served_slots(pl, xf, ev), "aircraft_repo": sim(xf)}
            for m in P.MODELS:
                ec, ee = pl.forecast(m, cal), pl.forecast(m, ev)
                lam = pl.choose_lambda(ec, M, cal, R.CAP)
                bc = (1 - lam) * pl.static[cal] + lam * ec; be = (1 - lam) * pl.static[ev] + lam * ee
                xs = [P.allocate_expected(P.residual_sampler(bc, pl.truth[cal], seed=s)(be), M, R.CAP) for s in range(5)]
                xp = P.allocate_point(ee, M, R.CAP)
                for mode, fn in (("aggregate", lambda x: R.served_slots(pl, x, ev)), ("aircraft_repo", sim)):
                    meth = np.mean([fn(x) for x in xs], axis=0); pt = fn(xp)
                    for comp, d in (("method-floor", meth - fl[mode]), ("method-point", meth - pt), ("point-floor", pt - fl[mode])):
                        (lo, hi), bs = R.boot_ci(d, B=B)
                        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()); p = float(min(max(p, 1.0 / B), 1.0))
                        blk = np.array([d[i:i + 4].sum() for i in range(0, len(d), 4)])          # paired t test on the 24 hourly blocks
                        tstat = blk.mean() / (blk.std(ddof=1) / np.sqrt(len(blk))) if blk.std() > 0 else 0.0
                        pt_ = float(2 * stats.t.sf(abs(tstat), len(blk) - 1))
                        rows.append(dict(K=K, fold=fold, M=M, model=m, mode=mode, comparison=comp, lam=lam, diff=100 * d.sum() / N,
                                         lo=100 * lo / N, hi=100 * hi / N, p=p, p_t=pt_,
                                         lo_bonf=100 * np.percentile(bs, 100 * 0.025 / 198) / N, hi_bonf=100 * np.percentile(bs, 100 * (1 - 0.025 / 198)) / N,
                                         se=100 * bs.std(ddof=1) / N))
        print(f"K={K} {fold} done {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(R.OUT / "e11_comparisons.csv", index=False)
df = pd.DataFrame(rows)
out = []
for (mode, comp), g in df.groupby(["mode", "comparison"]):
    g = g.sort_values("p").reset_index(drop=True); n = len(g)
    holm_thr = 0.05 / (n - np.arange(n))
    rej = np.zeros(n, dtype=bool)
    for i in range(n):
        if g.p[i] <= holm_thr[i]: rej[i] = True
        else: break
    g["holm"] = rej; g["bonf"] = g.p <= 0.05 / n
    gt = g.sort_values("p_t").reset_index(drop=True); rt = np.zeros(n, dtype=bool)
    for i in range(n):
        if gt.p_t[i] <= holm_thr[i]: rt[i] = True
        else: break
    holm_t_above = int((rt & (gt["diff"] > 0)).sum()); holm_t_below = int((rt & (gt["diff"] < 0)).sum())
    out.append(dict(mode=mode, comparison=comp, n=n, positive=int((g["diff"] > 0).sum()), negative=int((g["diff"] < 0).sum()),
                    sig_above=int((g.lo > 0).sum()), sig_below=int((g.hi < 0).sum()),
                    holm_above=int((g.holm & (g["diff"] > 0)).sum()), holm_below=int((g.holm & (g["diff"] < 0)).sum()),
                    bonf_above=int((g.bonf & (g["diff"] > 0)).sum()), bonf_below=int((g.bonf & (g["diff"] < 0)).sum()),
                    holm_t_above=holm_t_above, holm_t_below=holm_t_below, bonf_ci_above=int((g.lo_bonf > 0).sum()), bonf_ci_below=int((g.hi_bonf < 0).sum()),
                    min_diff=g["diff"].min(), min_lo=g.lo.min(), min_lo_bonf=g.lo_bonf.min(), median_diff=g["diff"].median(), max_diff=g["diff"].max()))
pd.DataFrame(out).to_csv(R.OUT / "e11_summary.csv", index=False)
print(pd.DataFrame(out).to_string())
print("ALL_DONE")
