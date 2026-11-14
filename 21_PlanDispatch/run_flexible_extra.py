"""Zoned against pooled execution for the base reference plan (45 demand-ranked
pads, at most ten aircraft per pad) at the fleets the currency table uses."""
import pandas as pd
import core_pd as P
from run_flexible import Executor

CAP = 10
st = P.Study(positioning=True)
rows = []
for rule, K in (("proposed", 45), ("proposed", 38)):
    pl = st.plan(rule, K)
    ex = Executor(st, pl, "test")
    for M in (150, 200, 250, 300):
        x = P.allocate_point(pl.static["test"], M, CAP)
        rows.append(dict(rule=rule, K=K, M=M,
                         floor_zoned=100 * ex.serve(x, False) / st.district["test"],
                         floor_pooled=100 * ex.serve(x, True) / st.district["test"]))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(P.OUT / "flexible_c10_ref.csv", index=False)
