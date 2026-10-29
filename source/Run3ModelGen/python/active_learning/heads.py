'''Acquisition heads: what the surrogate's scalar output means, and how samples of it
become an acquisition score.

Ported from pmssm-al-benchmark ``pmssm/heads.py`` (MIT). Every head keeps one scalar
output and the decision point "output > 0" (a logit is positive exactly when p > 0.5):

* ``regression`` — MSE on the transformed target ``t``; acquisition ranks by variance.
* ``classification`` — BCE on the verdict ``1[t > threshold]``; the raw output is a logit
  (logistic link) for the neural surrogates and a probit latent for the GPs, whose
  Bernoulli likelihood is ``Phi(f)``; the score is the predictive entropy ``H[mean p]``.
* ``lsq_classification`` — the +-1 verdict regressed under a Gaussian likelihood, read
  through a probit link: the only way an EXACT GP can classify without giving up
  conjugate inference (Rasmussen & Williams ch. 6). Used internally for
  ``model: exact_gp`` + ``head: classification``; not a user-facing head.

For Gaussian latents (the GPs) the link expectation is a 32-node Gauss-Hermite
quadrature rather than a Monte-Carlo average, because the acquisition ranks a million
candidates and MC noise on a score is indistinguishable from signal in a top-k rule. A
leading sample-path dimension (deep GP) is a mixture: probabilities are averaged after
the quadrature, never before.
'''
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AcquisitionHead:
    name = "abstract"
    value_metrics = False          # True when the output estimates t itself (MSE/R2 meaningful)

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def make_targets(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def criterion(self) -> nn.Module:
        raise NotImplementedError

    def summarise(self, raw_samples: torch.Tensor) -> dict:
        '''(T, N) MC samples -> dict with at least ``mean`` and ``score`` (N,).'''
        raise NotImplementedError

    def summarise_gaussian(self, mean: torch.Tensor, var: torch.Tensor) -> dict:
        '''(N,) or (S, N) Gaussian latent moments -> the same dict.'''
        raise NotImplementedError


class RegressionHead(AcquisitionHead):
    name = "regression"
    value_metrics = True

    def make_targets(self, t):
        return t

    def criterion(self):
        return nn.MSELoss()

    def summarise(self, raw_samples):
        mean = raw_samples.mean(dim=0)
        var = raw_samples.var(dim=0)
        return {"mean": mean, "var": var, "score": var}

    def summarise_gaussian(self, mean, var):
        if mean.dim() == 2:          # mixture over sample paths: total variance
            m = mean.mean(0)
            v = (var + mean.pow(2)).mean(0) - m.pow(2)
            return {"mean": m, "var": v, "score": v}
        return {"mean": mean, "var": var, "score": var}


class VerdictClassificationHead(AcquisitionHead):
    name = "classification"
    value_metrics = False

    def __init__(self, threshold: float = 0.0, link: str = "logistic"):
        super().__init__(threshold=threshold)
        if link not in ("logistic", "probit"):
            raise ValueError(f"unknown link {link!r}; expected 'logistic' or 'probit'")
        self.link = link

    def _link_fn(self):
        return torch.sigmoid if self.link == "logistic" else standard_normal_cdf

    def make_targets(self, t):
        return (t > self.threshold).float()

    def criterion(self):
        return nn.BCEWithLogitsLoss()

    def summarise(self, raw_samples):
        p = self._link_fn()(raw_samples.double())
        p_bar = p.mean(dim=0)
        total = bernoulli_entropy(p_bar)
        aleatoric = bernoulli_entropy(p).mean(dim=0)
        return {
            "mean": raw_samples.mean(dim=0),
            "p_mean": p_bar.float(),
            "var": p.var(dim=0).float(),
            "entropy": total.float(),
            "mutual_information": (total - aleatoric).float(),
            "score": total.float(),
        }

    def summarise_gaussian(self, mean, var):
        return summarise_link_gaussian(mean, var, self._link_fn())


class LeastSquaresVerdictHead(AcquisitionHead):
    name = "lsq_classification"
    value_metrics = False

    def make_targets(self, t):
        return torch.where(t > self.threshold, 1.0, -1.0).to(t.dtype)

    def criterion(self):
        return nn.MSELoss()

    def summarise(self, raw_samples):
        return summarise_link_gaussian(raw_samples.mean(dim=0), raw_samples.var(dim=0),
                                       standard_normal_cdf)

    def summarise_gaussian(self, mean, var):
        return summarise_link_gaussian(mean, var, standard_normal_cdf)


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-12, 1.0 - 1e-12)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def standard_normal_cdf(f: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(f / np.sqrt(2.0)))


def _gauss_hermite(n: int = 32):
    x, w = np.polynomial.hermite.hermgauss(n)
    return (torch.as_tensor(x, dtype=torch.float64),
            torch.as_tensor(w / np.sqrt(np.pi), dtype=torch.float64))


def summarise_link_gaussian(mean, var, link, n_quad: int = 32) -> dict:
    '''E over f ~ N(mean, var) of the link and of the Bernoulli entropy, per sample path
    (leading dimension), then averaged over paths.'''
    mu = mean.double()
    sd = var.double().clamp_min(0).sqrt()
    squeezed = mu.dim() == 1
    if squeezed:
        mu, sd = mu.unsqueeze(0), sd.unsqueeze(0)
    x, w = _gauss_hermite(n_quad)
    x, w = x.to(mu.device), w.to(mu.device)
    f = mu.unsqueeze(-1) + np.sqrt(2.0) * sd.unsqueeze(-1) * x          # (S, N, Q)
    p = link(f)
    p_path = (p * w).sum(-1)
    h_path = (bernoulli_entropy(p) * w).sum(-1)
    p_bar = p_path.mean(0)
    aleatoric = h_path.mean(0)
    total = bernoulli_entropy(p_bar)
    mean_latent = mu.mean(0)
    var_latent = (sd.pow(2) + mu.pow(2)).mean(0) - mean_latent.pow(2)
    return {
        "mean": mean_latent.float(),
        "p_mean": p_bar.float(),
        "var": var_latent.float(),
        "entropy": total.float(),
        "mutual_information": (total - aleatoric).clamp_min(0).float(),
        "score": total.float(),
    }


HEADS = {
    RegressionHead.name: RegressionHead,
    VerdictClassificationHead.name: VerdictClassificationHead,
    LeastSquaresVerdictHead.name: LeastSquaresVerdictHead,
}


def get_head(name: str, threshold: float = 0.0, **kwargs) -> AcquisitionHead:
    try:
        cls = HEADS[name]
    except KeyError:
        raise ValueError(f"unknown head {name!r}; known: {sorted(HEADS)}") from None
    return cls(threshold=threshold, **kwargs)


def head_for(config_head: str, model: str, threshold: float) -> AcquisitionHead:
    '''The head object for a (config head, surrogate model) pair.

    ``classification`` with a neural surrogate is the logistic BCE head; with the deep GP
    the probit head over its Bernoulli likelihood; with the exact GP the least-squares
    verdict head (exact inference cannot carry a Bernoulli likelihood).'''
    if config_head == "regression":
        return RegressionHead(threshold=threshold)
    if model == "exact_gp":
        return LeastSquaresVerdictHead(threshold=threshold)
    if model == "deep_gp":
        return VerdictClassificationHead(threshold=threshold, link="probit")
    return VerdictClassificationHead(threshold=threshold, link="logistic")
