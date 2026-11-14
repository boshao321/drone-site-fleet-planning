"""Training and evaluation loop shared by every neural model."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .metrics import per_horizon


def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=shuffle, drop_last=False)


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def collect(model, loader, scaler, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        p = model(batch)
        preds.append(p.cpu().numpy())
        labels.append(batch["y"].cpu().numpy())
    p = np.concatenate(preds)[..., 0]            # [S, P, N]
    y = np.concatenate(labels)[..., 0]
    return scaler.inverse(p), scaler.inverse(y)


def train_model(model, bundle, seed, device=C.DEVICE, max_epochs=C.MAX_EPOCHS,
                patience=C.PATIENCE, verbose=False):
    C.seed_everything(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(3, patience // 3))
    crit = nn.L1Loss()

    tr = make_loader(bundle.train, True)
    va = make_loader(bundle.val, False)
    te = make_loader(bundle.test, False)

    best = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in tr:
            batch = _to_device(batch, device)
            opt.zero_grad()
            loss = crit(model(batch), batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        epochs_run = epoch

        p, y = collect(model, va, bundle.scaler, device)
        val_mae = float(np.abs(p - y).mean())
        sched.step(val_mae)

        if val_mae < best - 1e-6:
            best = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 10 == 0:
            print(f"      epoch {epoch:3d}  val MAE {val_mae:.4f}  best {best:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    p, y = collect(model, te, bundle.scaler, device)
    metrics = per_horizon(p, y)
    metrics["val_MAE"] = best
    metrics["epochs"] = epochs_run
    metrics["train_sec"] = time.time() - t0
    metrics["params"] = sum(q.numel() for q in model.parameters() if q.requires_grad)
    return metrics, p, y
