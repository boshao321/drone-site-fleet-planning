"""Configuration for the rebuilt forecasting pipeline.

Differences from the original 02_Demand_Prediction setup, each traceable to an
audit finding:
  - the temporal resolution is fixed at 15 minutes, so 15/30/45/60 are genuine
    forecast horizons rather than four different aggregation widths;
  - the spatial node set is built once from all business lines and reused, so
    the node index means the same thing everywhere;
  - every model name maps to its own implementation;
  - seeds are re-applied before each run, so runs differ by seed only.
"""
from __future__ import annotations

import os
import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "09_Forecasting_Rebuilt" / "outputs"
# the order records are confidential; set DRONES_RAW_CSV or place the file at the default path
RAW_CSV = Path(os.environ.get("DRONES_RAW_CSV", PROJECT_ROOT / "02_Demand_Prediction" / "data" / "raw" / "深圳市南山区运单抽样_已脱敏.csv"))
CACHE_DIR = PROJECT_ROOT / "09_Forecasting_Rebuilt" / "cache"

# ---------------------------------------------------------------- task setup
RESOLUTION_MIN = 15
SLOTS_PER_DAY = 1440 // RESOLUTION_MIN          # 96
N_DAYS = 7
TOTAL_SLOTS = SLOTS_PER_DAY * N_DAYS            # 672

SEQ_LEN = 12                                    # past 3 hours
PRED_LEN = 4                                    # t+15, t+30, t+45, t+60
HORIZON_LABELS = ["t+15", "t+30", "t+45", "t+60"]

# Chronological split on whole days, so each split spans complete daily cycles.
# A 7:2:1 cut of the slot index leaves the test set with no night hours at all,
# which makes validation-based model selection score a different time of day
# than the test set measures.
TRAIN_DAYS, VAL_DAYS, TEST_DAYS = 5, 1, 1

BIZ_LINES = ["waimai", "shangou", "pinhaofan", "health"]
BIZ_LABELS = {"waimai": "外卖", "shangou": "闪购", "pinhaofan": "拼好饭", "health": "健康"}

GRID_STEPS = [0.010, 0.001]                     # ~1.1 km and ~110 m
MAIN_GRID_STEP = 0.010

# distance-based static adjacency (Li et al. 2018 convention)
ADJ_SIGMA_KM = 1.5
ADJ_THRESHOLD = 0.1

# ---------------------------------------------------------------- training
SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE = 20
GRAD_CLIP = 5.0
HIDDEN = 32                                     # small: 455 training windows
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETERMINISTIC_MODELS = ["HA", "SeasonalNaive", "Ridge"]
NEURAL_MODELS = ["LSTM", "STGCN", "DCRNN", "GraphWaveNet", "AGCRN", "AMSGCN"]
ALL_MODELS = DETERMINISTIC_MODELS + NEURAL_MODELS

ABLATION_VARIANTS = {
    "AMSGCN (Full)": {},
    "w/o Multi-scale Temporal": {"use_multiscale": False},
    "w/o Dynamic Flow Graph": {"use_dynamic_graph": False},
    "w/o Periodic Embedding": {"use_periodic": False},
    "w/o Node Heterogeneity": {"use_heterogeneity": False},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
