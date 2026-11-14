"""Key numbers of the base scenario (ten aircraft per pad), printed for the
write-up and checked against the CSVs the figures and tables are drawn from."""
import json
import numpy as np
import pandas as pd

O = "outputs/"
out = {}
for suf in ("", "_swap"):
    g = pd.read_csv(O + f"cap_grid{suf}.csv", dtype={"cap": str})
    g = g[(g.cap == "10") & (g.rule == "proposed")]
    for M in (100, 200, 300):
        d = g[(g.M == M) & (g.K >= M / 10)]
        rec = {}
        for p in ("floor", "point", "method", "foresight", "ceiling"):
            top = d[p].max()
            k = int(d[d[p] >= top - 0.05].K.min())
            rec[p] = (k, round(float(d[d.K == k][p].iloc[0]), 2))
        rec["floor_K120"] = round(float(d[d.K == 120].floor.iloc[0]), 2)
        rec["method_K120"] = round(float(d[d.K == 120].method.iloc[0]), 2)
        out[f"best_K{suf}_M{M}"] = rec
m = pd.read_csv(O + "base_mechanism.csv"); m = m[m.rule == "proposed"].set_index("K")
out["mech"] = {k: dict(sd=round(m.loc[k, "static_rel_err_sd"], 2),
                       floor_unmet300=round(m.loc[k, "floor_unmet_pct_M300"], 1),
                       ceil_unmet300=round(m.loc[k, "ceiling_unmet_pct_M300"], 1),
                       mean_req=round(m.loc[k, "mean_req_min"], 1))
               for k in (15, 45, 120)}
f = pd.read_csv(O + "flexible_c10.csv"); f = f[(f.rule == "proposed") & (f.M == 300)]
out["pool300"] = dict(zoned45=round(float(f[f.K == 45].floor_zoned.iloc[0]), 2),
                      pooled45=round(float(f[f.K == 45].floor_pooled.iloc[0]), 2),
                      pooled_min_K38plus=round(float(f[f.K >= 38].floor_pooled.min()), 2),
                      pooled_max=round(float(f[f.K >= 38].floor_pooled.max()), 2))
c = pd.read_csv(O + "base_currency.csv")
out["currency"] = {f"{r.M} {r.lever}": (round(r.service_pp, 2), None if not np.isfinite(r.aircraft) else round(r.aircraft, 1))
                   for r in c.itertuples()}
cells = 0; neg = 0
for tag in ("proposed38_c10", "proposed45_c10"):
    d = pd.read_csv(O + f"d1_models_{tag}.csv")
    cells += len(d); neg += int((d.method_vs_floor_hi < 0).sum())
    for M in (200, 300):
        x = d[(d.fold == "val->test") & (d.M == M)]
        y = d[d.M == M]
        out[f"d1_{tag}_M{M}"] = dict(method=(round(x.method_vs_floor_pp.min(), 2), round(x.method_vs_floor_pp.max(), 2)),
                                     point=(round(x.point_vs_floor_pp.min(), 2), round(x.point_vs_floor_pp.max(), 2)),
                                     point_sig_neg=int((x.point_vs_floor_hi < 0).sum()),
                                     method_beats_point_both_folds=int((y.method_vs_point_lo > 0).sum()),
                                     gap=round(float(x.gap_pp.iloc[0]), 2))
out["d1_cells"] = cells; out["d1_method_sig_below_floor"] = neg
d3 = pd.read_csv(O + "d3_accuracy_proposed45_c10.csv"); d3 = d3[d3.M == 300].sort_values("cv")
p = d3.point_G.to_numpy(); cv = d3.cv.to_numpy()
i = np.where((p[:-1] > 0) & (p[1:] <= 0))[0][0]
out["breakeven_cv_point_45_300"] = round(float(cv[i] + (cv[i + 1] - cv[i]) * p[i] / (p[i] - p[i + 1])), 2)
out["method_G_at_cv1"] = round(float(d3[d3.cv == 1.0].method_G.iloc[0]), 1)
gap300 = out["d1_proposed45_c10_M300"]["gap"]
d4 = pd.read_csv(O + "d4_mismatch_proposed45_c10.csv"); d4 = d4[(d4.M == 300) & (d4.target == "all")]
out["mismatch300_pp"] = dict(point=(round(d4.point_G.min() * gap300 / 100, 2), round(d4.point_G.max() * gap300 / 100, 2)),
                             method=(round(d4.method_G.min() * gap300 / 100, 2), round(d4.method_G.max() * gap300 / 100, 2)))
d2 = pd.read_csv(O + "d2_lead_proposed45_c10.csv"); d2 = d2[d2.M == 300].groupby("lead_min")[["point_G", "method_G"]].mean() * gap300 / 100
out["lead300_pp"] = d2.round(2).to_dict()
b = pd.read_csv(O + "cap_best_plans.csv", dtype={"cap": str})
out["staircase_cap10"] = b[(b.cap == "10") & (b.policy == "floor")].set_index("M").K.to_dict()
out["staircase_cap10_swap"] = pd.read_csv(O + "cap_best_plans_swap.csv", dtype={"cap": str}).query("cap=='10' and policy=='floor'").set_index("M").K.to_dict()
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(O + "base_key_numbers.json", "w"), indent=1, default=str)
