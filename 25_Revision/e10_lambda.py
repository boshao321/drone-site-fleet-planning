"""E10 (R2.6, R3.2): the shrinkage weight lambda.

  curve    service of the proposed rule on the scored day for every lambda on
           the grid, next to the service the selection step sees on the
           calibration day
  fixed    lambda fixed in advance (0.3, 0.5, 0.7) instead of selected
  split    lambda selected on one half of the calibration day and residuals
           drawn from the other half (alternating one-hour blocks), both ways

Outputs: e10_curve.csv, e10_variants.csv
"""
import sys, time
import numpy as np, pandas as pd
import revlib as R
P = R.P
SEEDS = range(5)
t0 = time.time()
st = R.make_study()

def method(pl, M, cal, ev, lam, ec, ee, cal_mask=None, seed=0):
    bc = (1 - lam) * pl.static[cal] + lam * ec
    be = (1 - lam) * pl.static[ev] + lam * ee
    tc = pl.truth[cal]
    if cal_mask is not None:
        bc, tc = bc[cal_mask], tc[cal_mask]
    return P.allocate_expected(P.residual_sampler(bc, tc, seed=seed)(be), M, R.CAP)

def choose(pl, M, cal, ec, mask=None):
    vals = []
    for l in P.LAMBDAS:
        s = R.served_slots(pl, P.allocate_point((1 - l) * pl.static[cal] + l * ec, M, R.CAP), cal)
        vals.append(s.sum() if mask is None else s[mask].sum())
    return float(P.LAMBDAS[int(np.argmax(vals))]), np.array(vals)

rows_c, rows_v = [], []
for K in (45, 38):
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N, Nc = st.district[ev], st.district[cal]; fold = f"{cal}->{ev}"
        Tc = len(st.slots[cal]); hour = np.arange(Tc) // 4
        even, odd = hour % 2 == 0, hour % 2 == 1
        for model in (["AMSGCN", "LSTM", "Ridge"] if K == 45 else ["AMSGCN"]):
            ec, ee = pl.forecast(model, cal), pl.forecast(model, ev)
            for M in (100, 200, 300):
                floor = R.served_slots(pl, P.allocate_point(pl.static[ev], M, R.CAP), ev)
                point = R.served_slots(pl, P.allocate_point(ee, M, R.CAP), ev)
                lam_sel, cal_vals = choose(pl, M, cal, ec)
                sv = {}
                for i, l in enumerate(P.LAMBDAS):
                    s = np.mean([R.served_slots(pl, method(pl, M, cal, ev, l, ec, ee, seed=sd), ev) for sd in SEEDS], axis=0)
                    sv[round(float(l), 1)] = s
                    blend_ev = R.served_slots(pl, P.allocate_point((1 - l) * pl.static[ev] + l * ee, M, R.CAP), ev)
                    rows_c.append(dict(K=K, fold=fold, model=model, M=M, lam=round(float(l), 1), selected=bool(abs(l - lam_sel) < 1e-9),
                                       cal_blend=100 * cal_vals[i] / Nc, test_blend=100 * blend_ev.sum() / N,
                                       test_method=100 * s.sum() / N, floor=100 * floor.sum() / N, point=100 * point.sum() / N))
                def add(name, lam, s):
                    d = s - floor; (lo, hi), _ = R.boot_ci(d)
                    d2 = s - point; (lo2, hi2), _ = R.boot_ci(d2)
                    rows_v.append(dict(K=K, fold=fold, model=model, M=M, variant=name, lam=lam, service=100 * s.sum() / N,
                                       vs_floor=100 * d.sum() / N, lo=100 * lo / N, hi=100 * hi / N,
                                       vs_point=100 * d2.sum() / N, lo_p=100 * lo2 / N, hi_p=100 * hi2 / N))
                add("selected (paper)", lam_sel, sv[round(lam_sel, 1)])
                for l in (0.3, 0.5, 0.7, 1.0):
                    add(f"fixed {l}", l, sv[l])
                best = max(sv, key=lambda l: sv[l].sum()); worst = min(sv, key=lambda l: sv[l].sum())
                add("best on grid (hindsight)", best, sv[best]); add("worst on grid", worst, sv[worst])
                for nm, a, b in (("split: lambda even hours, residuals odd", even, odd), ("split: lambda odd hours, residuals even", odd, even)):
                    lam_s, _ = choose(pl, M, cal, ec, a)
                    s = np.mean([R.served_slots(pl, method(pl, M, cal, ev, lam_s, ec, ee, cal_mask=b, seed=sd), ev) for sd in SEEDS], axis=0)
                    add(nm, lam_s, s)
            print(f"K={K} {fold} {model} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows_c).to_csv(R.OUT / "e10_curve.csv", index=False); pd.DataFrame(rows_v).to_csv(R.OUT / "e10_variants.csv", index=False)
print("ALL_DONE")
