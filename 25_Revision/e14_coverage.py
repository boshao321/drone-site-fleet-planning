"""E14 (R3.5): service split into coverage and fulfilment.

  coverage     share of the district's eligible high value orders whose pickup is
               within the service radius of a site (service with unlimited aircraft)
  fulfilment   share of the covered orders that the fleet serves
  service      coverage x fulfilment

Output: e14_coverage.csv
"""
import time
import numpy as np, pandas as pd
import revlib as R
P = R.P
t0 = time.time()
st = R.make_study()
rows = []
for K in R.KS:
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; cov = 100 * pl.reach[ev] / N
        for M in (100, 200, 300, 400):
            al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["floor", "method", "foresight", "ceiling"])
            for p in ("floor", "method", "foresight", "ceiling"):
                sv = 100 * R.served_slots(pl, al[p], ev).sum() / N
                rows.append(dict(K=K, fold=f"{cal}->{ev}", M=M, usable=min(M, K * R.CAP), policy=p, coverage=cov, service=sv,
                                 fulfilment=100 * sv / cov, shortfall=cov - sv))
    print(K, round(time.time() - t0), flush=True)
pd.DataFrame(rows).to_csv(R.OUT / "e14_coverage.csv", index=False)
print("ALL_DONE")
