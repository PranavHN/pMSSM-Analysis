'''The three acquisition rules, as the benchmark runs them (``pmssm/selection.py`` and the
GP driver's ``select_entropy_batch``; MIT):

* ``cls_entropy`` (classification head): top ``n_select`` by the predictive entropy
  ``H[mean p]`` over ALL candidates — no tolerance cut, no proximity weighting (the score
  is already anchored at the decision boundary).
* ``tol_only_random`` (regression head): keep the candidates whose predicted mean lies
  strictly within ``threshold +- tolerance`` (all of them if none survive, or if the
  tolerance is 0), then a uniform draw of ``n_select`` among the survivors; the
  uncertainty is never consulted.
* ``entropy_batch`` (regression head): the tolerance cut, then proximity-weighted
  variance ``var * exp(-(mean - threshold)^2 / proximity)`` ranks a focused pool of
  ``entropy_pool_size`` points, whose posterior (sample covariance of the MC passes for
  the neural surrogates, the GP's predictive covariance for the GPs) feeds the smoothed
  batch-entropy selector with Gibbs temperature ``entropy_beta`` and blur ``entropy_blur``.

Every rule returns ``Selection(indices, pred, score)`` with ``pred`` the predictive mean
of the selected candidates in transformed space and ``score`` the quantity the rule ranked
by (variance-weighted entropy per point for entropy_batch; NaN for tol_only_random,
which ranks nothing).
'''
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from Run3ModelGen.active_learning.gp import EntropySelectionStrategy


@dataclass
class Selection:
    indices: np.ndarray    # into the candidate pool, unique, in selection order
    pred: np.ndarray       # predictive mean (transformed space) of the selected
    score: np.ndarray      # acquisition score of the selected


def tolerance_survivors(mean: torch.Tensor, threshold: float, tolerance: float, logger=None) -> torch.Tensor:
    N = len(mean)
    if tolerance > 0:
        mask = (mean > threshold - tolerance) & (mean < threshold + tolerance)
        surviving = torch.where(mask)[0]
        if logger:
            logger.info(f"tolerance cut (+/-{tolerance:g} around {threshold:g}): {len(surviving)}/{N} survive")
        if len(surviving) == 0:
            if logger:
                logger.warning("no candidate survived the tolerance cut; using all candidates")
            surviving = torch.arange(N)
        return surviving
    return torch.arange(N)


def _dedupe(indices) -> np.ndarray:
    '''Keep the first occurrence of every index, preserving order (the Gibbs sampler of
    the entropy batch can draw a pool point twice; the benchmark labelled such
    duplicates and dropped them afterwards, we skip the simulator call).'''
    seen, out = set(), []
    for i in np.asarray(indices).tolist():
        if i not in seen:
            seen.add(i)
            out.append(i)
    return np.asarray(out, dtype=int)


def select_cls_entropy(ev, n_select: int, logger=None) -> Selection:
    score = ev.summary["score"].to(torch.float64).view(-1)
    take = min(int(n_select), len(score))
    top = torch.topk(score, take).indices
    if logger:
        logger.info(f"cls_entropy: {take} candidates by predictive entropy "
                    f"(range {float(score[top].min()):.4g} to {float(score[top].max()):.4g})")
    idx = top.numpy()
    return Selection(indices=idx, pred=ev.mean[top].numpy(), score=score[top].numpy())


def select_tol_only_random(ev, n_select: int, threshold: float, tolerance: float, seed: int,
                           logger=None) -> Selection:
    surviving = tolerance_survivors(ev.mean.view(-1), threshold, tolerance, logger)
    take = min(int(n_select), len(surviving))
    g = torch.Generator().manual_seed(int(seed))
    chosen = surviving[torch.randperm(len(surviving), generator=g)[:take]]
    if logger:
        logger.info(f"tol_only_random: {take} survivors drawn uniformly (seed {int(seed)}); "
                    "no uncertainty was consulted")
    return Selection(indices=chosen.numpy(), pred=ev.mean[chosen].numpy(),
                     score=np.full(take, np.nan))


def select_entropy_batch(surrogate, ev, n_select: int, acq: dict, threshold: float, device,
                         logger=None) -> Selection:
    mean, var = ev.mean.view(-1), ev.var.view(-1)
    N = len(mean)
    n_select = min(int(n_select), N)
    n_pool = min(int(acq["entropy_pool_size"]), N)
    if n_select > n_pool:
        if logger:
            logger.warning(f"entropy_pool_size ({n_pool}) < num_models ({n_select}); pool raised to {n_select}")
        n_pool = n_select

    surviving = tolerance_survivors(mean, threshold, acq["tolerance"], logger)
    surv_mean, surv_var = mean[surviving], var[surviving]
    if acq["proximity"] > 0:
        proximity = torch.exp(-((surv_mean - threshold) ** 2) / acq["proximity"])
        weighted = proximity * surv_var
    else:
        weighted = surv_var
    k = min(n_pool, len(surviving))
    pool = surviving[torch.argsort(weighted, descending=True)[:k]]
    if logger:
        logger.info(f"entropy_batch: focused pool of {len(pool)} candidates "
                    f"(from {len(surviving)} survivors of {N})")

    pool_mean, pool_cov = surrogate.pool_posterior(ev, pool)
    strategy = EntropySelectionStrategy(blur=acq["entropy_blur"], beta=acq["entropy_beta"])
    score_fn = strategy.smoothed_batch_entropy(blur=acq["entropy_blur"], device=device)
    choice_fn = lambda score, indices: strategy.gibbs_sample(score, acq["entropy_beta"], device)
    dev_mean = (pool_mean - threshold).to(device)
    dev_cov = pool_cov.to(device)
    selected_in_pool = strategy.iterative_batch_selector(score_fn, choice_fn, dev_mean, dev_cov,
                                                         n_select, device, logger=logger)
    selected_in_pool = _dedupe(selected_in_pool)
    # per-point smoothed entropy of each pick (the GP driver's `per_point_entropy`)
    per_point = np.array([
        float(score_fn(dev_mean[i].view(1, 1), dev_cov[i, i].view(1, 1, 1)).item())
        for i in selected_in_pool.tolist()])
    chosen = pool[torch.as_tensor(selected_in_pool, dtype=torch.long)]
    if logger and len(selected_in_pool) < n_select:
        logger.info(f"entropy_batch: {n_select - len(selected_in_pool)} duplicate draws skipped")
    return Selection(indices=chosen.numpy(), pred=mean[chosen].numpy(), score=per_point)


def select(strategy: str, surrogate, ev, n_select: int, acq: dict, threshold: float,
           seed: int, device, logger=None) -> Selection:
    if strategy == "cls_entropy":
        return select_cls_entropy(ev, n_select, logger)
    if strategy == "tol_only_random":
        return select_tol_only_random(ev, n_select, threshold, acq["tolerance"], seed, logger)
    if strategy == "entropy_batch":
        return select_entropy_batch(surrogate, ev, n_select, acq, threshold, device, logger)
    raise ValueError(f"unknown strategy {strategy!r}")
