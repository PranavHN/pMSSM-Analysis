'''One interface over the four surrogates, so the loop and the acquisition rules never
branch on the model family.

``NeuralSurrogate`` (transformer, dnn): z-scored inputs, MC dropout for uncertainty.
``GPSurrogate`` (exact_gp, deep_gp): min-max inputs over the config's free box, the
posterior for uncertainty. Both expose

* ``fit(X_train, t_train, X_val, t_val, warm, log_path)`` -> metrics dict,
* ``evaluate(X)`` -> :class:`Evaluation` (predictive mean/variance in transformed space
  and the head summary with the acquisition ``score``),
* ``pool_posterior(evaluation, pool_indices)`` -> ``(mean, cov)`` for the entropy batch,
* ``checkpoint()`` -> what ``iteration_NNN/model.pt`` stores and a warm start reloads.
'''
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from Run3ModelGen.active_learning import gp as gpmod
from Run3ModelGen.active_learning.config import AlConfig
from Run3ModelGen.active_learning.heads import head_for
from Run3ModelGen.active_learning import surrogates as nn_mod


@dataclass
class Evaluation:
    mean: torch.Tensor            # (N,) predictive mean in transformed space (latent for the GPs)
    var: torch.Tensor             # (N,) predictive variance
    summary: dict                 # the head summary (score, entropy, p_mean, ...)
    samples: torch.Tensor | None  # (T, N) MC-dropout passes (neural surrogates only)
    x_norm: torch.Tensor          # normalised inputs the model saw (for GP pool blocks)


class NeuralSurrogate:
    kind = "neural"

    def __init__(self, cfg: AlConfig, n_inputs: int, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.head = head_for(cfg.head, cfg.model, cfg.threshold)
        self.arch = cfg.transformer if cfg.model == "transformer" else cfg.dnn
        self.n_inputs = n_inputs
        self.model = nn_mod.build_model(cfg.model, n_inputs, self.arch)
        self.std = None

    def fit(self, X_train, t_train, X_val, t_val, warm=None, log_path=None, progress=None) -> dict:
        self.model = nn_mod.build_model(self.cfg.model, self.n_inputs, self.arch)   # cold start by default
        warm_state = warm.get("model_state_dict") if warm else None
        self.std, metrics = nn_mod.train_neural(
            self.model, self.head, X_train, t_train, X_val, t_val, self.cfg.training,
            self.device, log_path=log_path, warm_state=warm_state, progress=progress)
        return metrics

    def checkpoint(self) -> dict:
        return {"model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "standardiser": self.std.state() if self.std else None, "model": self.cfg.model}

    @torch.no_grad()
    def evaluate(self, X) -> Evaluation:
        x_norm = self.std(X)
        samples = nn_mod.mc_dropout_predict(self.model, x_norm, self.cfg.acquisition["mc_samples"],
                                            self.device)
        summary = self.head.summarise(samples)
        return Evaluation(mean=samples.mean(0), var=samples.var(0), summary=summary,
                          samples=samples, x_norm=x_norm)

    def pool_posterior(self, ev: Evaluation, pool: torch.Tensor) -> tuple:
        '''Sample covariance of the MC passes over the pool (rank <= T - 1), regularised
        by 1e-4 on the diagonal as in the benchmark.'''
        preds = ev.samples[:, pool]
        centered = preds - preds.mean(dim=0)
        T = preds.shape[0]
        cov = (centered.T @ centered) / (T - 1)
        cov += 1e-4 * torch.eye(cov.shape[0], dtype=cov.dtype)
        return ev.mean[pool].clone(), cov


class GPSurrogate:
    kind = "gp"

    def __init__(self, cfg: AlConfig, n_inputs: int, device, lo, hi, seed: int):
        self.cfg = cfg
        self.device = torch.device(device)
        self.head = head_for(cfg.head, cfg.model, cfg.threshold)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.is_deep = cfg.model == "deep_gp"
        self.seed = int(seed)
        self.model = None

    def _norm(self, X) -> torch.Tensor:
        return gpmod.normalise(X, self.lo, self.hi)

    def fit(self, X_train, t_train, X_val, t_val, warm=None, log_path=None, progress=None) -> dict:
        logger = nn_mod.training_logger(log_path)
        try:
            x_tr, x_va = self._norm(X_train), self._norm(X_val)
            y_tr = self.head.make_targets(torch.as_tensor(np.asarray(t_train, np.float32)).view(-1))
            y_va = self.head.make_targets(torch.as_tensor(np.asarray(t_val, np.float32)).view(-1))
            logger.info(f"training set {len(x_tr)}, validation set {len(x_va)}, device {self.device}, "
                        f"model {self.cfg.model}, head {self.head.name}")
            self.model = gpmod.create_gp_model(self.cfg.model, self.cfg.gp, x_tr, y_tr, x_va, y_va,
                                               self.cfg.head, self.device, self.seed, self.cfg.threshold)
            if warm:
                gpmod.load_warm_start(self.model, warm, logger)
            tr_losses, va_losses, seconds = gpmod.train_gp(
                self.model, self.cfg.model, self.cfg.gp, int(self.cfg.training["epochs"]),
                int(self.cfg.training["patience"]), logger)
            if progress:
                for e in range(0, len(tr_losses), 100):
                    progress(e, tr_losses[e], va_losses[e])
            best = int(np.argmin(va_losses)) if va_losses else -1
            out_tr = gpmod.latent_mean(self.model, x_tr, self.is_deep, self.cfg.gp["jitter"], self.cfg.gp["num_samples"])
            out_va = gpmod.latent_mean(self.model, x_va, self.is_deep, self.cfg.gp["jitter"], self.cfg.gp["num_samples"])
            metrics = {
                "epochs_run": len(tr_losses), "best_epoch": best,
                "best_train_loss": float(tr_losses[best]) if best >= 0 else math.nan,
                "best_val_loss": float(va_losses[best]) if best >= 0 else math.nan,
                "loss_name": "-MLL", "train_losses": tr_losses, "val_losses": va_losses,
                "n_train": int(len(x_tr)), "n_val": int(len(x_va)), "seconds": seconds,
            }
            metrics.update(nn_mod.verdict_metrics(self.head, out_tr, y_tr.cpu(), out_va, y_va.cpu()))
            logger.info(f"done: best iter {best}, best val -mll {metrics['best_val_loss']:.4f}, {seconds:.0f} s")
            return metrics
        finally:
            nn_mod.close_logger(logger)

    def checkpoint(self) -> dict:
        return {"model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "likelihood_state_dict": {k: v.detach().cpu() for k, v in self.model.likelihood.state_dict().items()},
                "model": self.cfg.model}

    @torch.no_grad()
    def evaluate(self, X) -> Evaluation:
        x_norm = self._norm(X)
        jitter, ns = self.cfg.gp["jitter"], self.cfg.gp["num_samples"]
        if self.cfg.head == "classification":
            lat_mean, lat_var = gpmod.latent_moments(self.model, x_norm, self.is_deep, jitter, ns)
            summary = self.head.summarise_gaussian(lat_mean, lat_var)
            return Evaluation(mean=summary["mean"], var=summary["var"], summary=summary,
                              samples=None, x_norm=x_norm)
        mean, var = gpmod.predictive_moments(self.model, x_norm, self.is_deep, jitter, ns)
        summary = self.head.summarise_gaussian(mean, var)
        return Evaluation(mean=mean, var=var, summary=summary, samples=None, x_norm=x_norm)

    def pool_posterior(self, ev: Evaluation, pool: torch.Tensor) -> tuple:
        mean, cov = gpmod.posterior_block(self.model, ev.x_norm[pool], self.is_deep,
                                          self.cfg.gp["jitter"], self.cfg.gp["num_samples"])
        return mean.cpu(), cov.cpu()


def make_surrogate(cfg: AlConfig, n_inputs: int, device, lo, hi, seed: int):
    if cfg.is_neural:
        return NeuralSurrogate(cfg, n_inputs, device)
    return GPSurrogate(cfg, n_inputs, device, lo, hi, seed)
