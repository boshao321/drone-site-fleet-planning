"""E6 (R1.10): the residual model behind the proposed rule.

  bins       1, 3, 6 (paper) and 10 bins of the calibration estimate
  samples    50, 100, 300 (paper) and 1000 predictive samples per site and slot
  history    residuals from all, half or a quarter of the calibration day
             (random one-hour blocks, five subsets each)
  strata     residual pools formed by estimate size (paper), by time of day, by
             site volume, by size and time jointly, or one pool; and a normal
             model with the bias and spread of each size bin
  coverage   share of realised site requirements inside the central 50, 80 and
             90% predictive intervals on the scored day, by site volume, time
             of day and size of the estimate

Outputs: e06_service.csv, e06_coverage.csv
"""
import sys, time
import numpy as np, pandas as pd
import revlib as R
P = R.P
t0 = time.time()
st = R.make_study()
PERIODS = [(0, 7, "night 00-07"), (7, 11, "morning 07-11"), (11, 14, "lunch 11-14"), (14, 17, "afternoon 14-17"), (17, 21, "dinner 17-21"), (21, 24, "late 21-24")]

def period_of(slots):
    h = (slots % 96) / 4.0
    out = np.zeros(len(slots), dtype=int)
    for i, (a, b, _) in enumerate(PERIODS):
        out[(h >= a) & (h < b)] = i
    return out

def make_sampler(bc, tc, strata="size", nbins=6, ns=300, seed=0, per_c=None, per_e=None, site_grp=None, mask=None):
    """Predictive samples around an estimate from calibration-day residuals. `strata`
    decides which calibration residuals count as similar."""
    rng = np.random.default_rng(seed)
    T, S = bc.shape
    keep = np.ones(T, dtype=bool) if mask is None else mask
    def size_bin(x, edges):
        return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    nb_size = nbins if strata in ("size", "normal") else (3 if strata == "size_time" else 1)
    edges = np.unique(np.quantile(bc[keep].ravel(), np.linspace(0, 1, nb_size + 1)))
    if len(edges) < 2: edges = np.array([0.0, 1.0])
    def key(est, per, is_cal):
        b = size_bin(est, edges) if strata in ("size", "normal", "size_time") else np.zeros(est.shape, dtype=int)
        if strata == "time":
            b = np.repeat(per[:, None], S, axis=1)
        elif strata == "size_time":
            coarse = np.array([0, 1, 2, 1, 2, 0])[per]             # off-peak, shoulder, peak
            b = b * 3 + np.repeat(coarse[:, None], S, axis=1)
        elif strata == "site":
            b = np.repeat(site_grp[None, :], est.shape[0], axis=0)
        return b
    kc = key(bc, per_c, True)[keep]; res = (tc - bc)[keep]
    pools = {int(k): res[kc == k] for k in np.unique(kc)}
    allres = res.ravel()
    def sample(be):
        ke = key(be, per_e, False)
        out = np.repeat(be[..., None], ns, axis=-1).astype(float)
        for k in np.unique(ke):
            m = ke == k
            pool = pools.get(int(k), allres)
            if len(pool) == 0: pool = allres
            if strata == "normal":
                out[m] += rng.normal(pool.mean(), pool.std() + 1e-9, size=(int(m.sum()), ns))
            else:
                out[m] += rng.choice(pool, size=(int(m.sum()), ns))
        return np.maximum(out, 0.0)
    return sample

rows, cov = [], []
for K in (45, 38):
    pl = st.plan("proposed", K)
    vol = pl.site_hv[st.train].mean(axis=0); q = np.quantile(vol, [1 / 3, 2 / 3]); site_grp = np.searchsorted(q, vol)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; fold = f"{cal}->{ev}"
        per_c, per_e = period_of(st.slots[cal]), period_of(st.slots[ev])
        ec, ee = pl.forecast("AMSGCN", cal), pl.forecast("AMSGCN", ev)
        Tc = len(st.slots[cal]); hour = np.arange(Tc) // 4
        for M in (100, 200, 300):
            if M > K * R.CAP: continue
            lam = pl.choose_lambda(ec, M, cal, R.CAP)
            bc = (1 - lam) * pl.static[cal] + lam * ec; be = (1 - lam) * pl.static[ev] + lam * ee
            floor = R.served_slots(pl, P.allocate_point(pl.static[ev], M, R.CAP), ev)
            blend = R.served_slots(pl, P.allocate_point(be, M, R.CAP), ev)
            def run(tag, group, seeds=range(5), **kw):
                sv = [R.served_slots(pl, P.allocate_expected(make_sampler(bc, pl.truth[cal], seed=s, per_c=per_c, per_e=per_e, site_grp=site_grp, **kw)(be), M, R.CAP), ev) for s in seeds]
                return tag, group, sv
            res = [run("paper: size, 6 bins, 300 samples", "reference")]
            res += [run(f"{b} bins", "bins", nbins=b) for b in (1, 3, 10)]
            res += [run(f"{n} samples", "samples", ns=n) for n in (50, 100, 1000)]
            res += [run(nm, "strata", strata=s_) for nm, s_ in (("by time of day", "time"), ("by site volume", "site"), ("by size and time", "size_time"), ("normal, bias and spread by size bin", "normal"))]
            for frac in (0.5, 0.25):
                sv = []
                for r in range(5):
                    hrs = np.random.default_rng(100 + r).choice(24, size=int(round(24 * frac)), replace=False)
                    sv += run("", "", seeds=range(2), mask=np.isin(hour, hrs))[2]
                res.append((f"{int(frac*100)}% of the calibration day", "history", sv))
            ref = np.mean(res[0][2], axis=0)
            rows.append(dict(K=K, fold=fold, M=M, lam=lam, group="reference", variant="no sampling (point allocation on the shrunk estimate)", service=100 * blend.sum() / N,
                             seed_sd=0.0, vs_paper=100 * (blend - ref).sum() / N, vs_floor=100 * (blend - floor).sum() / N))
            for tag, group, sv in res:
                m = np.mean(sv, axis=0); d = m - ref; (lo, hi), _ = R.boot_ci(d)
                df_ = m - floor; (lof, hif), _ = R.boot_ci(df_)
                rows.append(dict(K=K, fold=fold, M=M, lam=lam, group=group, variant=tag, service=100 * m.sum() / N,
                                 seed_sd=100 * np.std([x.sum() for x in sv], ddof=1) / N, vs_paper=100 * d.sum() / N, lo=100 * lo / N, hi=100 * hi / N,
                                 vs_floor=100 * df_.sum() / N, lo_f=100 * lof / N, hi_f=100 * hif / N))
            if M == 300:
                # coverage of the predictive intervals on the scored day
                for strata in ("size", "time", "site", "size_time", "normal"):
                    smp = make_sampler(bc, pl.truth[cal], strata=strata, ns=2000, seed=0, per_c=per_c, per_e=per_e, site_grp=site_grp)(be)
                    y = pl.truth[ev]
                    szq = np.quantile(be.ravel(), [1 / 3, 2 / 3]); szg = np.searchsorted(szq, be)
                    groups = {"all": np.ones_like(y, dtype=bool)}
                    for gi, gn in enumerate(("low volume sites", "mid volume sites", "high volume sites")): groups[gn] = np.repeat((site_grp == gi)[None, :], y.shape[0], axis=0)
                    for pi, (_, _, pn) in enumerate(PERIODS): groups[pn] = np.repeat((per_e == pi)[:, None], y.shape[1], axis=1)
                    for gi, gn in enumerate(("small estimates", "medium estimates", "large estimates")): groups[gn] = szg == gi
                    for level in (50, 80, 90):
                        lo_q = np.percentile(smp, (100 - level) / 2, axis=2); hi_q = np.percentile(smp, 100 - (100 - level) / 2, axis=2)
                        inside = (y >= lo_q - 1e-9) & (y <= hi_q + 1e-9)
                        for gn, gm in groups.items():
                            cov.append(dict(K=K, fold=fold, strata=strata, level=level, group=gn, n=int(gm.sum()), coverage=100 * inside[gm].mean(),
                                            width=float((hi_q - lo_q)[gm].mean()), mean_requirement=float(y[gm].mean()),
                                            bias=float((smp.mean(axis=2) - y)[gm].mean())))
            print(f"K={K} {fold} M={M} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(R.OUT / "e06_service.csv", index=False); pd.DataFrame(cov).to_csv(R.OUT / "e06_coverage.csv", index=False)
print("ALL_DONE")
