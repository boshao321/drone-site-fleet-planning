# Joint Landing Site and Fleet Capacity Planning for Urban Drone Delivery

Code, parameter configurations, evaluation scripts and result tables for

> B. Shao, X. Li, S. Dong, T. Chen, Z. He and C. Ye, "Joint Landing Site and Fleet Capacity Planning for Urban Drone Delivery under Demand Uncertainty," submitted to *IEEE Access*.

The study compares landing site networks, fleet sizes and dispatch policies for drone delivery of high value orders in Nanshan District, Shenzhen, using 592,667 delivery orders over seven days. Every table and figure of the paper is drawn from the CSV and JSON files in the `outputs` folders of this repository.

## Data

The order records are a desensitized operational sample provided under a confidentiality agreement and are **not** included. The code expects a CSV with the columns

```
dt_seq_id, order_seq_id, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude,
customer_pay_time, fetch_time, arrived_time, package_price, biz_line
```

at `02_Demand_Prediction/data/raw/` or at the path given by the environment variable `DRONES_RAW_CSV`. `dt_seq_id` is the day index (1 to 7; days 1 to 5 are weekdays, 6 and 7 the weekend), times are `H:MM` within the day, and `package_price` is the order value in CNY. The candidate ranking table `07_NetworkBenefit_KStar_NoWeights/outputs/tables/candidate_network_indicators_no_weights.csv` is derived from the same records and is not distributed either.

## Layout

| Folder | Content |
|---|---|
| `09_Forecasting_Rebuilt` | Demand forecasting: historical average, seasonal naive, ridge, LSTM, STGCN, DCRNN, Graph WaveNet, AGCRN and AMSGCN (`core/models`), shared training protocol (`core/engine.py`, `core/config.py`) |
| `10_Allocation` | Scenario construction (`scripts/allocation.py`): catchments, closed tours, aircraft-minute requirements and order books |
| `13_SitingBaselines` | p-median site selection |
| `21_PlanDispatch` | Planning and dispatch study of the paper (`core_pd.py`): historical floor, point forecast, proposed rule, perfect forecast, zoned ceiling, equivalent aircraft, site limits and prices, pooling |
| `25_Revision` | Validation and sensitivity analyses (Section VII and Appendix B), see below |

## Validation and sensitivity analyses (`25_Revision`)

| Script | Paper | Output |
|---|---|---|
| `e01_aircraft_level.py` | VII-A, Table 8 | replay with individually tracked aircraft, repositioning flights, move limits, static deployment |
| `e17_causal.py`, `causal.py` | VII-A | replay in which an order can be served only after its payment time |
| `e03_sensitivity.py` | VII-B, Table 9 | speed, service time, radius, pickup to dropoff limit, closed tour limit, combined settings, value thresholds |
| `train_fold.py`, `e13_rolling.py`, `e09_level_adjusted.py` | VII-C, Table 10 | chronological folds, level adjusted historical baselines |
| `e11_multiplicity.py`, `e08_equivalent.py` | VII-D, Table 6 | Holm and Bonferroni corrections, bootstrap intervals for equivalent aircraft |
| `e10_lambda.py`, `e06_residual.py`, `e05_objective.py` | VII-E, Table 11 | shrinkage weight, residual model, orders objective |
| `e12_pooling.py`, `e14_coverage.py`, `e16_static.py` | VI-A, VI-B, Tables 4 and 5 | pooling frictions and upper bound, coverage and fulfilment |
| `tune_sweep.py`, `conv_check.py` | Appendix B, Table 15 | hidden dimension sweep and convergence check of the neural models |

Shared helpers are in `revlib.py`. All random steps use fixed seeds (42 to 44 for training, 0 to 4 for predictive sampling, 7 for the bootstrap).

## Operating parameters

Cruise speed 60 km/h, service time 4 minutes per sortie, service radius 3 km, pickup to dropoff limit 8 km, at most 10 aircraft per site, 15 minute slots, commitments 60 minutes ahead, high value threshold at the 75th percentile of order value (53.90 CNY), 300 predictive samples, shrinkage weight chosen on the calibration day from {0, 0.1, ..., 1}.

## Requirements

Python 3.10 or later; see `requirements.txt`. Training uses PyTorch on CPU.

```
pip install -r requirements.txt
cd 21_PlanDispatch && python run_grid.py          # main planning grid
cd ../25_Revision && python e01_aircraft_level.py  # tracked aircraft replay
```

## License

MIT, see `LICENSE`.
