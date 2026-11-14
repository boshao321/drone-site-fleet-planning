"""Every number the revised manuscript cites, computed from the outputs in one
place and written to outputs/paper_facts.json. The manuscript's verification
script checks the text against this file."""
import json
import numpy as np
import pandas as pd
import core_pd as P

O = P.OUT
F = {}
g = pd.read_csv(O / "cap_grid.csv", dtype={"cap": str})
gs = pd.read_csv(O / "cap_grid_swap.csv", dtype={"cap": str})

def best(d, p, tol=0.05):
    top = d[p].max(); k = int(d[d[p] >= top - tol].K.min())
    return k, float(d[d.K == k][p].iloc[0])

for tag, G in (("test", g), ("val", gs)):
    for cap in ("10", "20", "none"):
        for rule in ("proposed", "pmedian"):
            for M in (100, 200, 300):
                d = G[(G.cap == cap) & (G.rule == rule) & (G.M == M)]
                if cap != "none":
                    d = d[d.K >= M / float(cap)]
                F[f"{tag}_c{cap}_{rule}_M{M}_best"] = {p: best(d, p) for p in ("floor", "point", "method", "foresight", "ceiling")}
                F[f"{tag}_c{cap}_{rule}_M{M}_K120"] = {p: float(d[d.K == 120][p].iloc[0]) for p in ("floor", "method", "point")}
                for K in (15, 25, 38, 45, 55, 90):
                    r = d[d.K == K]
                    if len(r):
                        F[f"{tag}_c{cap}_{rule}_M{M}_K{K}"] = {p: float(r[p].iloc[0]) for p in ("floor", "point", "method", "unc_static", "foresight", "ceiling")}
    F[f"{tag}_c10_max_ceiling"] = float(G[G.cap == "10"].ceiling.max())
    F[f"{tag}_c5_max_ceiling"] = float(G[G.cap == "5"].ceiling.max())
    F[f"{tag}_c20_max_ceiling"] = float(G[G.cap == "20"].ceiling.max())

# pooling
f = pd.read_csv(O / "flexible_c10.csv")
fr = pd.read_csv(O / "flexible_c10_ref.csv")
F["pool_c10_proposed_M300"] = f[(f.rule == "proposed") & (f.M == 300)].set_index("K")[["floor_zoned", "floor_pooled", "method_zoned", "method_pooled"]].round(3).to_dict("index")
F["pool_c10_ref"] = fr.round(3).to_dict("records")
fu = pd.read_csv(O / "flexible.csv")
F["pool_nocap_pmedian_M300"] = fu[(fu.rule == "pmedian") & (fu.M == 300)].set_index("K")[["floor_zoned", "floor_pooled"]].round(3).to_dict("index")

# mechanism
m = pd.read_csv(O / "base_mechanism.csv")
F["mech"] = m.round(4).to_dict("records")

# currency
F["currency_base"] = pd.read_csv(O / "base_currency.csv").round(3).to_dict("records")
F["currency_nocap"] = pd.read_csv(O / "currency.csv").round(3).to_dict("records")

# dispatch
for t in ("proposed45_c10", "proposed38_c10", "proposed38", "pmedian15"):
    d = pd.read_csv(O / f"d1_models_{t}.csv")
    F[f"d1_{t}"] = d[["fold", "M", "model", "floor", "ceiling", "gap_pp", "lambda", "point_vs_floor_pp", "point_vs_floor_lo", "point_vs_floor_hi",
                     "method_vs_floor_pp", "method_vs_floor_lo", "method_vs_floor_hi", "method_vs_point_pp", "method_vs_point_lo", "method_vs_point_hi",
                     "unc_static_vs_floor_pp", "method_G", "point_G", "foresight_G", "unc_static_G", "method_G_seed_sd"]].round(4).to_dict("records")
for t in ("proposed45_c10",):
    for k in ("d2_lead", "d3_accuracy", "d4_mismatch"):
        F[f"{k}_{t}"] = pd.read_csv(O / f"{k}_{t}.csv").round(4).to_dict("records")

# designs
F["cost_optimal"] = pd.read_csv(O / "cap_cost_optimal.csv", dtype={"cap": str}).round(3).to_dict("records")
F["savings"] = pd.read_csv(O / "cap_savings.csv", dtype={"cap": str}).round(3).to_dict("records")
F["savings_swap"] = pd.read_csv(O / "cap_savings_swap.csv", dtype={"cap": str}).round(3).to_dict("records")
F["best_plans_cap"] = pd.read_csv(O / "cap_best_plans.csv", dtype={"cap": str}).to_dict("records")
F["best_plans_cap_swap"] = pd.read_csv(O / "cap_best_plans_swap.csv", dtype={"cap": str}).to_dict("records")
F["iso_c10_80_proposed"] = pd.read_csv(O / "cap_iso.csv", dtype={"cap": str}).query("cap=='10' and target==80 and rule=='proposed'").round(2).to_dict("records")

# study-level facts and site-level forecast error on the reference plan
st = P.Study(positioning=True)
F["district"] = dict(st.district)
F["rho"] = st.rho
pl = st.plan("proposed", 45)
F["tau45"] = pl.tau
F["reach45_test"] = 100 * pl.reach["test"] / st.district["test"]
err = {}
for model in P.MODELS:
    for day in ("val", "test"):
        e = pl.forecast(model, day); t = pl.truth[day]; nz = t > 0
        err[f"{model}_{day}"] = float(((e - t)[nz] / t[nz]).std(ddof=1))
s = pl.static["test"]; t = pl.truth["test"]; nz = t > 0
err["static_test"] = float(((s - t)[nz] / t[nz]).std(ddof=1))
F["site_rel_err_sd_45"] = err
F["n_slots"] = {d: int(len(v)) for d, v in st.slots.items()}
F["n_candidates"] = int(len(st.cand))
json.dump(F, open(O / "paper_facts.json", "w"), indent=1, default=float)
print("tau45", round(pl.tau, 3), "reach45", round(F["reach45_test"], 2), "district", F["district"], "slots", F["n_slots"])
print({k: round(v, 3) for k, v in err.items()})
