"""Training loop: Adam + One-Cycle LR + gradient clipping + weight decay.

These four choices come straight from the project's research paper, which
credits them for the model's convergence behaviour. Mixed precision (--amp) is
an addition for CUDA machines; it roughly halves epoch time and is a no-op on
CPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it


@torch.no_grad()
def evaluate(model, valid_dl):
    """Mean validation loss and accuracy."""
    model.eval()
    outputs = [model.validation_step(batch) for batch in valid_dl]
    if not outputs:
        return {"val_loss": float("nan"), "val_acc": float("nan")}
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group["lr"]
    return None


def fit_one_cycle(epochs, model, train_dl, valid_dl, *,
                  max_lr=0.01, weight_decay=1e-4, grad_clip=0.1,
                  opt_func=torch.optim.Adam, use_amp=False, device=None,
                  on_epoch_end=None):
    """Train with the One-Cycle learning rate policy.

    Returns a list of per-epoch dicts: train_loss, val_loss, val_acc, lr, secs.
    """
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))

    device_type = (device.type if device is not None else "cpu")
    amp_on = bool(use_amp) and device_type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=amp_on)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        lrs = []

        bar = tqdm(train_dl, desc="epoch %d/%d" % (epoch + 1, epochs),
                   unit="batch", disable=None)
        for batch in bar:
            with torch.amp.autocast(device_type, enabled=amp_on):
                loss = model.training_step(batch)

            scaler.scale(loss).backward()

            if grad_clip:
                # Unscale before clipping, or the threshold is meaningless.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss.detach())
            lrs.append(get_lr(optimizer))
            sched.step()

            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss="%.4f" % loss.item())

        result = evaluate(model, valid_dl)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lr"] = lrs[-1] if lrs else max_lr
        result["max_lr_seen"] = max(lrs) if lrs else max_lr
        result["secs"] = time.time() - t0
        result["epoch"] = epoch + 1

        print("  epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1f min)"
              % (epoch + 1, epochs, result["train_loss"], result["val_loss"],
                 result["val_acc"], result["secs"] / 60))

        history.append(result)
        if on_epoch_end is not None:
            on_epoch_end(epoch, result, model)

    return history


@torch.no_grad()
def predict_logits(model, dl):
    """Collect raw logits and labels over a loader. Used by evaluate.py."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in tqdm(dl, desc="inference", unit="batch", disable=None):
        all_logits.append(model(images).float().cpu())
        all_labels.append(labels.cpu())
    if not all_logits:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_logits), torch.cat(all_labels)
