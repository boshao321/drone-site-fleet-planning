"""E13 (R1.6, R1.7, R2.1, R2.2, R3.1): folds that only look backwards.

In every fold each step uses only days before the scored day: the high value
threshold, the site ranking, the historical mean, the forecasting models, the
shrinkage weight and the residuals.

  W       scored day 5 (weekday).  models on days 1-3, early stopping and calibration on day 4
  S1      scored day 6 (Saturday). models on days 1-4, early stopping and calibration on day 5
  MAIN    scored day 7 (Sunday).   models on days 1-5, early stopping and calibration on day 6 (fold 1 of the paper)
  STRICT  scored day 7 (Sunday).   models on days 1-4, early stopping on day 5, calibration on day 6

Outputs: e13_grid.csv (service by number of sites), e13_ref.csv (reference plans with intervals)
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import revlib as R
P = R.P
HERE = Path(__file__).resolve().parent
REPO_V, REPO_H = 60.0, 1.0
FLEETS = np.arange(5, 805, 5)
FOLDS = {"W": dict(fc=HERE / "folds/W/forecasts"), "S1": dict(fc=HERE / "folds/S1/forecasts"),
         "MAIN": dict(fc=None), "STRICT": dict(fc=HERE / "folds/STRICT/forecasts")}
only = sys.argv[1:] or list(FOLDS)
gp, rp = R.OUT / "e13_grid.csv", R.OUT / "e13_ref.csv"
grid = [r for r in (pd.read_csv(gp).to_dict("records") if gp.exists() else []) if r["fold"] not in only]
ref = [r for r in (pd.read_csv(rp).to_dict("records") if rp.exists() else []) if r["fold"] not in only]
t0 = time.time()
for tag in only:
    if tag == "MAIN":
        meta = dict(cal_day=6, test_day=7, hv_threshold=53.90, train_days=5, es_day=6)
    else:
        meta = json.load(open(FOLDS[tag]["fc"] / "meta.json"))
    st = R.make_study(fc=FOLDS[tag]["fc"], train_days=meta["cal_day"] - 1, thr=meta["hv_threshold"], rank_days=range(1, meta["test_day"]))
    models = sorted({m for (tgt, m, d) in st.pred if tgt == "hv"}, key=P.MODELS.index)
    cal, ev = "val", "test"; N = st.district[ev]
    info = dict(fold=tag, test_day=meta["test_day"], cal_day=meta["cal_day"], thr=meta["hv_threshold"], N=N)
    print(tag, info, models, flush=True)
    for K in R.KS:
        pl = st.plan("proposed", K)
        for M in (100, 200, 300):
            al = R.policy_allocs(pl, M, R.CAP, cal, ev)
            for p in ("floor", "unc_static", "point", "method", "foresight", "ceiling"):
                grid.append(dict(info, K=K, M=M, policy=p, service=100 * R.served_slots(pl, al[p], ev).sum() / N, lam=al["_lambda"], reach=100 * pl.reach[ev] / N))
    for K in (45, 38):
        pl = st.plan("proposed", K); books = R.hv_books(pl, ev)
        sim = lambda x: R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H)
        curve = 100 * R.floor_curve(pl, ev, FLEETS) / N
        # level of the scored day against the historical mean (high value orders inside the network)
        level = float(pl.truth[ev].sum() / pl.static[ev].sum()); level_cal = float(pl.truth[cal].sum() / pl.static[cal].sum())
        for M in (100, 200, 300):
            if M > K * R.CAP: continue
            xf = P.allocate_point(pl.static[ev], M, R.CAP); xo = P.allocate_point(pl.truth[ev], M, R.CAP)
            fl = {"aggregate": R.served_slots(pl, xf, ev), "aircraft_repo": sim(xf)}
            fo = {"aggregate": R.served_slots(pl, xo, ev), "aircraft_repo": sim(xo)}
            ce = R.served_slots(pl, P.ceiling_alloc(pl.sc[ev], M, R.CAP)[1], ev)
            for m in models:
                ec, ee = pl.forecast(m, cal), pl.forecast(m, ev)
                lam = pl.choose_lambda(ec, M, cal, R.CAP)
                bc = (1 - lam) * pl.static[cal] + lam * ec; be = (1 - lam) * pl.static[ev] + lam * ee
                xs = [P.allocate_expected(P.residual_sampler(bc, pl.truth[cal], seed=s)(be), M, R.CAP) for s in range(5)]
                xp = P.allocate_point(ee, M, R.CAP)
                for mode, fn in (("aggregate", lambda x: R.served_slots(pl, x, ev)), ("aircraft_repo", sim)):
                    if mode == "aircraft_repo" and (m != "AMSGCN" or K != 45): continue
                    meth = np.mean([fn(x) for x in xs], axis=0); pt = fn(xp)
                    rec = dict(info, K=K, M=M, model=m, mode=mode, lam=lam, level_test=level, level_cal=level_cal,
                               floor=100 * fl[mode].sum() / N, point=100 * pt.sum() / N, method=100 * meth.sum() / N,
                               perfect=100 * fo[mode].sum() / N, ceiling=100 * ce.sum() / N if mode == "aggregate" else np.nan)
                    for nm, d in (("method-floor", meth - fl[mode]), ("method-point", meth - pt), ("point-floor", pt - fl[mode]), ("perfect-floor", fo[mode] - fl[mode])):
                        (lo, hi), _ = R.boot_ci(d)
                        rec[nm] = 100 * d.sum() / N; rec[nm + "_lo"] = 100 * lo / N; rec[nm + "_hi"] = 100 * hi / N
                    if mode == "aggregate":
                        rec["worth_point"] = R.equiv_aircraft(FLEETS, curve, rec["point"], M)
                        rec["worth_method"] = R.equiv_aircraft(FLEETS, curve, rec["method"], M)
                        rec["worth_perfect"] = R.equiv_aircraft(FLEETS, curve, rec["perfect"], M)
                    ref.append(rec)
    pd.DataFrame(grid).to_csv(gp, index=False); pd.DataFrame(ref).to_csv(rp, index=False)
    print(f"{tag} done {time.time()-t0:.0f}s", flush=True)
print("ALL_DONE")
