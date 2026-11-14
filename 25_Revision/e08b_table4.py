"""E8b: 95% intervals for the worth in aircraft of every dispatch entry of the lever table,
read exactly as in the paper (floor curve in steps of five aircraft, linear interpolation,
one sampling seed for the proposed rule). Paired block bootstrap over one-hour blocks, B = 5000.
Output: e08b_table4.csv
"""
import numpy as np, pandas as pd
import revlib as R

P = R.P
B, BLOCK = 5000, 4
st = R.make_study()

def worth(fleets, phi, sv, M):
    phi = np.maximum.accumulate(phi)
    if sv > phi[-1]:
        return np.inf
    if sv <= phi[0]:
        return float(fleets[0] - M)
    i = int(np.argmax(phi >= sv))
    return float(np.interp(sv, phi[i - 1:i + 1], fleets[i - 1:i + 1]) - M)

rows = []
for K in (45, 38):
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; fold = f"{cal}->{ev}"; T = len(st.slots[ev])
        blocks = [np.arange(i, min(i + BLOCK, T)) for i in range(0, T, BLOCK)]
        draws = np.random.default_rng(7).integers(0, len(blocks), size=(B, len(blocks)))
        bsum = lambda v: np.array([v[..., b].sum(axis=-1) for b in blocks])
        fleets = np.arange(5, min(K * R.CAP, 800) + 1, 5)
        F = np.stack([R.served_slots(pl, P.allocate_point(pl.static[ev], int(m), R.CAP), ev) for m in fleets]); Fb = bsum(F)
        tot = 100 * F.sum(axis=1) / N
        for M in (100, 200, 300):
            if M > K * R.CAP: continue
            al = R.policy_allocs(pl, M, R.CAP, cal, ev, seed=0)
            lev = {p: R.served_slots(pl, al[p], ev) for p in ("unc_static", "point", "method", "foresight", "ceiling")}
            fl = R.served_slots(pl, al["floor"], ev)
            est = {p: worth(fleets, tot, 100 * v.sum() / N, M) for p, v in lev.items()}
            bw = {p: np.empty(B) for p in lev}; Lb = {p: bsum(v) for p, v in lev.items()}
            for b in range(B):
                curve = 100 * Fb[draws[b]].sum(axis=0) / N
                for p in lev:
                    bw[p][b] = worth(fleets, curve, 100 * Lb[p][draws[b]].sum() / N, M)
            for p in lev:
                d = lev[p] - fl; (lo_s, hi_s), _ = R.boot_ci(d)
                lo, hi = np.percentile(bw[p], [2.5, 97.5])
                rows.append(dict(K=K, fold=fold, M=M, policy=p, service=100 * lev[p].sum() / N, gain=100 * d.sum() / N, gain_lo=100 * lo_s / N, gain_hi=100 * hi_s / N,
                                 worth=est[p], lo=lo, hi=hi, share_inf=float(np.isinf(bw[p]).mean())))
            for a, b_ in (("method", "point"), ("method", "unc_static")):
                ok = np.isfinite(bw[a]) & np.isfinite(bw[b_]); d = (bw[a] - bw[b_])[ok]
                okr = ok & (bw[b_] > 0); ratio = bw[a][okr] / bw[b_][okr]
                rows.append(dict(K=K, fold=fold, M=M, policy=f"{a}-{b_} (aircraft)", worth=est[a] - est[b_], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), share_inf=float((d <= 0).mean())))
                rows.append(dict(K=K, fold=fold, M=M, policy=f"{a}/{b_} (ratio)", worth=est[a] / est[b_] if est[b_] > 0 else np.nan,
                                 lo=np.percentile(ratio, 2.5), hi=np.percentile(ratio, 97.5), share_inf=float(1 - okr.mean())))
pd.DataFrame(rows).to_csv(R.OUT / "e08b_table4.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(2).to_string(index=False))
