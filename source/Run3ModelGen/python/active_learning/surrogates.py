'''The neural surrogates: MC-dropout transformer and feed-forward network, training loop,
MC-dropout prediction. Ported from pmssm-al-benchmark (``pmssm/models/transformer.py``,
``pmssm/models/feedforward.py``, ``pmssm/training.py``, ``pmssm/uncertainty.py``; MIT).

Inputs are the FREE parameters, z-scored with the training split's mean and standard
deviation (``+1e-8``); the benchmark fed all 19 parameters with the 10 fixed ones z-scored
to a constant, which carries no information. Targets are ``t = log(Y / value)`` (or ``Y``)
mapped by the head (identity for regression, the 0/1 verdict for classification).

Training (per iteration): AdamW, cosine annealing to 1e-6 over ``epochs``, gradient clip
1.0, minibatch ``batch_size``, early stopping on the validation loss with ``patience``
epochs and the best state restored; a warm start loads the previous iteration's weights
(optimiser and schedule are rebuilt). The epoch log goes to ``iteration_NNN/training.log``.
'''
from __future__ import annotations

import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Run3ModelGen.active_learning.heads import AcquisitionHead


class PMSSMTransformerTabular(nn.Module):
    '''Per-feature token embeddings, pre-norm transformer encoder, attention pooling,
    MLP head: the benchmark's production surrogate (803k parameters at 19 inputs).'''

    def __init__(self, n_params=19, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, dropout=0.0):
        super().__init__()
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model)) for _ in range(n_params)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))
        self.regressor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.LayerNorm(dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward // 2), nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1))

    def forward(self, x):
        embedded = [emb(x[:, i:i + 1].unsqueeze(-1)) for i, emb in enumerate(self.feature_embeddings)]
        h = torch.cat(embedded, dim=1)              # (batch, n_params, d_model)
        h = self.encoder(h)
        w = self.attention_pool(h)                  # (batch, n_params, 1)
        h = (h * w).sum(dim=1)
        return self.regressor(h)                    # (batch, 1)


class PMSSMFeedForward(nn.Module):
    '''Shared scalar embedding, flatten, ``num_layers`` x [Linear, ReLU, Dropout].'''

    def __init__(self, n_params=19, d_model=64, num_layers=4, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.input_embed = nn.Linear(1, d_model)
        layers, in_features = [], n_params * d_model
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, dim_feedforward))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_features = dim_feedforward
        self.fc_layers = nn.Sequential(*layers)
        self.regressor = nn.Linear(dim_feedforward, 1)

    def forward(self, x):
        h = self.input_embed(x.unsqueeze(-1)).flatten(start_dim=1)
        return self.regressor(self.fc_layers(h))


def build_model(model: str, n_inputs: int, cfg: dict) -> nn.Module:
    if model == "transformer":
        return PMSSMTransformerTabular(n_params=n_inputs, d_model=cfg["d_model"], nhead=cfg["nhead"],
                                       num_layers=cfg["num_layers"],
                                       dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"])
    if model == "dnn":
        return PMSSMFeedForward(n_params=n_inputs, d_model=cfg["d_model"], num_layers=cfg["num_layers"],
                                dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"])
    raise ValueError(f"not a neural surrogate: {model!r}")


class Standardiser:
    '''z-score of the inputs with the TRAINING split's statistics.'''

    def __init__(self, X_train: np.ndarray):
        X_train = np.asarray(X_train, dtype=np.float32)
        self.mean = torch.as_tensor(X_train.mean(axis=0))
        self.std = torch.as_tensor(X_train.std(axis=0, ddof=1) if len(X_train) > 1
                                   else np.ones(X_train.shape[1], np.float32)) + 1e-8

    def __call__(self, X) -> torch.Tensor:
        return (torch.as_tensor(np.asarray(X, dtype=np.float32)) - self.mean) / self.std

    def state(self) -> dict:
        return {"mean": self.mean.clone(), "std": self.std.clone()}


def training_logger(log_path: str | None):
    '''A logger that writes every epoch line to ``log_path`` (and nothing to the
    console). Call :func:`close_logger` when done so the handler is released.'''
    logger = logging.getLogger(f"r3mg.active_learning.training.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    if log_path:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    return logger


def close_logger(logger) -> None:
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def train_neural(model: nn.Module, head: AcquisitionHead, X_train, t_train, X_val, t_val,
                 training: dict, device, log_path: str | None = None,
                 warm_state: dict | None = None, progress=None) -> tuple:
    '''Train ``model`` in place; returns ``(standardiser, metrics)``.

    ``X_*`` are physical free-parameter arrays, ``t_*`` transformed targets (the head maps
    them to its own targets). ``progress(epoch, train_loss, val_loss)`` is called every 100
    epochs for the main log.'''
    logger = training_logger(log_path)
    try:
        std = Standardiser(X_train)
        x_tr = std(X_train)
        x_va = std(X_val)
        y_tr = head.make_targets(torch.as_tensor(np.asarray(t_train, np.float32)).view(-1, 1))
        y_va = head.make_targets(torch.as_tensor(np.asarray(t_val, np.float32)).view(-1, 1))
        if head.name != "regression":
            logger.info(f"head {head.name}: positive fraction train {float(y_tr.mean()):.4f}, "
                        f"val {float(y_va.mean()):.4f}")
        logger.info(f"training set {len(x_tr)}, validation set {len(x_va)}, device {device}")

        if warm_state is not None:
            model.load_state_dict(warm_state)
            logger.info("warm start: previous iteration's weights loaded")
        model.to(device)

        bs = int(training["batch_size"])
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=bs, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                      weight_decay=training["weight_decay"])
        epochs = int(training["epochs"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = head.criterion()
        loss_name = {"MSELoss": "MSE", "BCEWithLogitsLoss": "BCE"}.get(type(criterion).__name__,
                                                                       type(criterion).__name__)
        patience = int(training["patience"])

        train_losses, val_losses = [], []
        best_val, best_state, counter, best_epoch = math.inf, None, 0, -1
        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total += loss.item() * x.size(0)
            train_loss = total / max(1, len(train_loader.dataset))
            model.eval()
            total = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    total += criterion(model(x), y).item() * x.size(0)
            val_loss = total / max(1, len(val_loader.dataset))
            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"epoch {epoch:05d} | train {loss_name} {train_loss:.6f} | "
                        f"val {loss_name} {val_loss:.6f} | lr {lr:.3e}")
            if progress and (epoch % 100 == 0):
                progress(epoch, train_loss, val_loss)
            if val_loss < best_val:
                best_val, best_epoch, counter = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        seconds = time.time() - t0

        # metrics at the restored best state
        model.eval()
        with torch.no_grad():
            out_tr = torch.cat([model(x.to(device)).cpu() for x, _ in train_loader]).view(-1) if len(x_tr) else torch.zeros(0)
            out_va = torch.cat([model(x.to(device)).cpu() for x, _ in val_loader]).view(-1) if len(x_va) else torch.zeros(0)
        metrics = {
            "epochs_run": len(train_losses), "best_epoch": best_epoch,
            "best_train_loss": float(train_losses[best_epoch]) if best_epoch >= 0 else math.nan,
            "best_val_loss": float(best_val) if best_epoch >= 0 else math.nan,
            "loss_name": loss_name, "train_losses": train_losses, "val_losses": val_losses,
            "n_train": int(len(x_tr)), "n_val": int(len(x_va)), "seconds": seconds,
        }
        metrics.update(verdict_metrics(head, out_tr, y_tr.view(-1), out_va, y_va.view(-1)))
        logger.info(f"done: best epoch {best_epoch}, best val {loss_name} {best_val:.6f}, "
                    f"{seconds:.0f} s")
        return std, metrics
    finally:
        close_logger(logger)


def verdict_metrics(head, out_tr, y_tr, out_va, y_va) -> dict:
    '''Accuracy of the sign verdict (every head), plus R2/MSE in transformed space for
    the regression head.'''
    def acc(out, y):
        if len(out) == 0:
            return math.nan
        return float(((out > 0) == (y > (0.0 if head.name != "regression" else head.threshold))).float().mean())

    m = {"verdict_accuracy_train": acc(out_tr, y_tr), "verdict_accuracy_val": acc(out_va, y_va)}
    if head.value_metrics and len(out_va):
        resid = (out_va - y_va)
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y_va - y_va.mean()) ** 2).sum())
        m["val_mse_t"] = ss_res / len(out_va)
        m["val_r2_t"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return m


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x_norm: torch.Tensor, mc_samples: int, device,
                       batch_size: int = 8192) -> torch.Tensor:
    '''``(T, N)`` stochastic forward passes with dropout active (``model.train()``;
    there is no batch norm in either architecture).'''
    model.to(device)
    model.train()
    x_norm = x_norm.to(device)
    out = []
    for _ in range(int(mc_samples)):
        preds = torch.cat([model(x_norm[i:i + batch_size]) for i in range(0, len(x_norm), batch_size)])
        out.append(preds.view(-1).cpu())
    return torch.stack(out, dim=0)
