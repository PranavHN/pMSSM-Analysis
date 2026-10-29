'''Gaussian-process surrogates and the entropy-batch selector, vendored from al_pmssmwithgp.

Provenance: https://github.com/jwuerzinger/al_pmssmwithgp, commit
a7f53fe19d66f8202f9b4e62cad97af5ec69b318, files
``model/gp_pipeline/models/linearMean.py`` (LinearMean),
``model/gp_pipeline/models/exact_gp.py`` (ExactGP),
``model/gp_pipeline/models/deep_gp.py`` (DeepGPHiddenLayer, DeepGP) and
``model/gp_pipeline/utils/selection.py`` (EntropySelectionStrategy), by Dominik Voss,
Irina Espejo Morales and Jonas Wuerzinger. The helper functions after the classes are
ports of pmssm-al-benchmark ``pmssm/training.py`` (create_gp_model, warm start),
``pmssm/uncertainty.py`` (latent/predictive moments) and the GP driver's
``select_entropy_batch``.

Edits to the vendored classes, all deliberate:
  * the deep-kernel (DKL) branch, the spectral-mixture / additive kernels and the
    ``select_new_points`` pipeline hook are dropped (not exposed by the config);
  * the constructors no longer re-seed the GLOBAL torch/numpy RNGs (the originals called
    ``torch.manual_seed(42)`` on every construction): ``seed`` feeds KMeans and a local
    generator for the output layer's inducing points; the loop owns the global RNG;
  * the device is the training tensors' device, not "cuda if available";
  * ``do_train_loop`` takes a logger (the epoch log goes to iteration_NNN/training.log).
'''
from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

_module_logger = logging.getLogger(__name__)


# ── vendored: LinearMean ──────────────────────────────────────────────────────────

class LinearMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(0.1 * torch.ones(*batch_shape, input_size, 1)),
        )
        if bias:
            self.register_parameter(
                name="bias", parameter=torch.nn.Parameter(torch.ones(*batch_shape, 1))
            )
        else:
            self.bias = None

    def forward(self, x):
        res = x.to(self.weights.device).matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res


def _base_kernel(kernel: str, ard_num_dims, m_nu: float = 1.5, batch_shape=torch.Size()):
    if kernel == "RBF":
        return RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
    if kernel == "Matern":
        return MaternKernel(nu=m_nu, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
    if kernel == "RQK":
        return RQKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
    raise ValueError(f"unknown kernel {kernel!r}; expected RBF, Matern or RQK")


# ── vendored: ExactGP ────────────────────────────────────────────────────────────

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, x_valid, y_valid, n_dim, lengthscale, use_ard,
                 noise, kernel, m_nu=1.5, thr=0, seed=42):
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.thr = thr
        self.seed = seed
        self.device = x_train.device
        likelihood = GaussianLikelihood()
        super().__init__(x_train, y_train, likelihood)
        self.likelihood = likelihood

        self.mean_module = LinearMean(input_size=n_dim)
        self.covar_module = ScaleKernel(_base_kernel(kernel, n_dim if use_ard else None, m_nu))
        if hasattr(self.covar_module.base_kernel, "lengthscale"):
            self.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.likelihood.noise = noise
        self.to(self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def do_train_loop(self, lr=0.005, iters=1000, jitter=1e-5, patience=None, logger=None):
        '''Full-batch exact marginal likelihood; validation loss is -mll on the
        validation set; the best state is restored.'''
        logger = logger or _module_logger
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        best_loss, best_model, patience_counter = 1e10, None, 0
        losses_train, losses_valid = [], []
        n_train = self.x_train.shape[0]
        with gpytorch.settings.max_cholesky_size(max(n_train + 1, 5000)), \
             gpytorch.settings.cg_tolerance(1e-2), \
             gpytorch.settings.max_cg_iterations(500), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
            for i in range(iters):
                self.train()
                self.likelihood.train()
                optimizer.zero_grad()
                output = self(self.x_train)
                loss = -mll(output, self.y_train.view(-1))
                loss.backward()
                optimizer.step()
                losses_train.append(loss.item())

                self.eval()
                self.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    output_valid = self(self.x_valid)
                    loss_valid = -mll(output_valid, self.y_valid.view(-1))
                    losses_valid.append(loss_valid.item())

                if loss_valid.item() < best_loss:
                    best_loss = loss_valid.item()
                    best_model = {k: v.clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        logger.info(f"early stopping at iter {i + 1}")
                        break
                if i % 100 == 0:
                    logger.info(f"iter {i + 1}/{iters} - loss (train) {loss.item():.3f} - "
                                f"loss (val) {loss_valid.item():.3f}")
                    if getattr(self.covar_module.base_kernel, "lengthscale", None) is not None:
                        logger.info(f"lengthscales: {self.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")
        if best_model is not None:
            self.load_state_dict(best_model)
        return self, losses_train, losses_valid


# ── vendored: DeepGP ─────────────────────────────────────────────────────────────

class DeepGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    '''A single deep hidden layer with variational inference approximation with inducing points'''
    def __init__(self, input_dims, output_dims, inducing_points, mean_type='constant', kernel='RBF', m_nu=1.5):
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[0], batch_shape=batch_shape)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        self.covar_module = ScaleKernel(_base_kernel(kernel, input_dims, m_nu, batch_shape),
                                        batch_shape=batch_shape)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class DeepGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, x_train, y_train, x_valid, y_valid, n_dim, lengthscale, noise, num_hidden_dims=10,
                 num_inducing_max=256, thr=0, kernel='RBF', m_nu=1.5, num_samples=8, seed=42,
                 likelihood="gaussian"):
        super().__init__()
        from sklearn.cluster import KMeans

        self.seed = seed
        self.x_train, self.x_valid, self.y_train, self.y_valid = x_train, x_valid, y_train, y_valid
        self.thr = thr
        self.num_middle_dims = 0
        self.num_samples = num_samples
        self.num_inducing_max = num_inducing_max
        self.device = x_train.device

        # Inducing points: KMeans centres of the training inputs, their number growing
        # with the training set up to num_inducing_max.
        num_inducing = max(1, min(int(0.5 * len(x_train)), num_inducing_max))
        _module_logger.info(f"Number of Inducing Points: {num_inducing}")
        kmeans = KMeans(n_clusters=num_inducing, random_state=int(seed), n_init=10).fit(x_train.cpu().numpy())
        inducing_points_hidden = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        self.hidden_layer = DeepGPHiddenLayer(
            input_dims=n_dim, output_dims=num_hidden_dims,
            inducing_points=inducing_points_hidden, mean_type='linear')

        g = torch.Generator().manual_seed(int(seed))
        inducing_points_output = torch.randn(num_inducing, num_hidden_dims, generator=g).to(self.device)
        self.output_layer = DeepGPHiddenLayer(
            input_dims=num_hidden_dims, output_dims=None,
            inducing_points=inducing_points_output, mean_type='constant', kernel=kernel, m_nu=m_nu)

        self.likelihood_kind = str(likelihood).lower()
        if self.likelihood_kind == "bernoulli":
            self.likelihood = BernoulliLikelihood()
        elif self.likelihood_kind == "gaussian":
            self.likelihood = GaussianLikelihood()
            self.likelihood.noise = noise
        else:
            raise ValueError(f"unknown likelihood {likelihood!r}; expected 'gaussian' or 'bernoulli'")

        for layer in (self.hidden_layer, self.output_layer):
            layer.covar_module.base_kernel.initialize(lengthscale=lengthscale)
        self.to(self.device)

    def forward(self, inputs):
        return self.output_layer(self.hidden_layer(inputs))

    def do_train_loop(self, lr=0.005, iters=1000, batch_size=512, jitter=1e-4, patience=None, logger=None):
        '''Minibatch ELBO over a fresh permutation per epoch; the validation loss is a
        size-weighted mean of -mll over minibatches (equal to the unbatched value).'''
        logger = logger or _module_logger
        _n_train = self.x_train.shape[0]
        _n_batches = max(1, (_n_train + batch_size - 1) // batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, self.x_train.shape[-2]))
        best_loss, best_model, patience_counter = 1e10, None, 0
        losses_train, losses_valid = [], []

        with gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.fast_pred_var(False):
            for i in range(iters):
                self.train()
                self.likelihood.train()
                _perm = torch.randperm(_n_train, device=self.x_train.device)
                _epoch_loss = torch.zeros((), device=self.device)
                for _b in range(_n_batches):
                    _idx = _perm[_b * batch_size:(_b + 1) * batch_size]
                    x_batch = self.x_train[_idx].to(self.device)
                    y_batch = self.y_train[_idx].to(self.device)
                    optimizer.zero_grad()
                    with gpytorch.settings.num_likelihood_samples(self.num_samples):
                        output = self(x_batch)
                        loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    _epoch_loss = _epoch_loss + loss.detach()
                epoch_avg = float(_epoch_loss) / _n_batches
                losses_train.append(epoch_avg)

                self.eval()
                self.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.num_samples):
                    _acc, _seen = 0.0, 0
                    for _j in range(0, self.x_valid.shape[0], batch_size):
                        _xb = self.x_valid[_j:_j + batch_size].to(self.device)
                        _yb = self.y_valid[_j:_j + batch_size].to(self.device)
                        _out = self.output_layer(self.hidden_layer(_xb))
                        _n = _xb.shape[0]
                        _acc += float(-mll(_out, _yb)) * _n
                        _seen += _n
                    loss_valid_value = _acc / max(_seen, 1)
                losses_valid.append(loss_valid_value)

                if loss_valid_value < best_loss:
                    best_loss = loss_valid_value
                    best_model = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        logger.info(f"early stopping at iter {i}")
                        break
                if i % 100 == 0:
                    logger.info(f"iter {i}/{iters} - loss (train) {epoch_avg:.3f} - loss (val) {loss_valid_value:.3f}")
        if best_model is not None:
            self.load_state_dict(best_model)
        return self, losses_train, losses_valid


# ── vendored: the entropy-batch strategy ─────────────────────────────────────────

class EntropySelectionStrategy:
    '''Batch selection by the smoothed batch entropy of the posterior at the threshold
    (Irina Espejo Morales): the first point is the argmax of the single-point score;
    each further point is Gibbs-sampled (temperature ``beta``) from the score of the
    batch extended by every candidate. The mean is pre-shifted by the threshold and
    blurred by ``blur * sign(mean)`` so points sitting exactly on the boundary do not
    produce a singular ``diag(1 / mean)``.'''
    def __init__(self, blur=0.15, beta=50, tolerance_sampling=1.0, proximity_sampling=0.1):
        self.blur = blur
        self.beta = beta
        self.tolerance_sampling = tolerance_sampling
        self.proximity_sampling = proximity_sampling

    def gibbs_sample(self, score, beta, device):
        probs = torch.exp(beta * (score - torch.max(score))).to(device)
        probs /= torch.sum(probs)
        cums = torch.cumsum(probs, dim=0)
        rand = torch.rand(size=(1,)).to(device)[0]
        return torch.sum(cums < rand)

    def approximate_batch_entropy(self, mean, cov, device):
        mean = mean.float()
        cov = cov.float()
        n = mean.shape[-1]
        d = torch.diag_embed(1. / mean).to(device)
        x = d @ cov @ d
        I = torch.eye(n)[None, :, :].to(device)
        return (torch.logdet(x + I) - torch.logdet(x + 2 * I) + n * torch.log(torch.tensor(2.0))) / torch.log(torch.tensor(2.0))

    def smoothed_batch_entropy(self, blur, device):
        return lambda mean, cov: self.approximate_batch_entropy(mean + blur * torch.sign(mean).to(device), cov, device)

    def iterative_batch_selector(self, score_function, choice_function, gp_mean, gp_covar, N, device, logger=None):
        logger = logger or _module_logger
        score = score_function(gp_mean[:, None], torch.diag(gp_covar)[:, None, None]).to(device)
        first_index = torch.argmax(score).to(device)
        indices = [int(first_index)]
        num_pts = len(gp_mean)
        logger.info(f"Iterative batch selector: selecting {N} points from {num_pts} candidates")
        gp_diag = torch.diag(gp_covar)
        chunk_size = max(256, 50_000 // max(N, 1))
        for iteration in range(N - 1):
            if iteration % 10 == 0:
                logger.info(f"Selection progress: {iteration+1}/{N-1} points ({100*(iteration+1)/(N-1):.1f}%)")
            center_cov_cpu = gp_covar[indices, :][:, indices]
            center_mean_cpu = gp_mean[indices]
            score_chunks = []
            for start in range(0, num_pts, chunk_size):
                end = min(start + chunk_size, num_pts)
                cs = end - start
                cc = center_cov_cpu.unsqueeze(0).expand(cs, -1, -1).to(device)
                sc = gp_covar[start:end, None, indices].to(device)
                bc = gp_covar[start:end, indices, None].to(device)
                ec = gp_diag[start:end, None, None].to(device)
                cov_batch = torch.cat([torch.cat([cc, sc], dim=1), torch.cat([bc, ec], dim=1)], dim=2)
                cm = center_mean_cpu.unsqueeze(0).expand(cs, -1).to(device)
                nm = gp_mean[start:end, None].to(device)
                mean_batch = torch.cat([cm, nm], dim=1)
                score_chunks.append(score_function(mean_batch, cov_batch).cpu())
                del cc, sc, bc, ec, cov_batch, cm, nm, mean_batch
            score = torch.cat(score_chunks, dim=0).to(device)
            next_index = choice_function(score, indices)
            indices.append(int(next_index))
        logger.info(f"Iterative batch selector complete: selected {len(indices)} points")
        return indices


# ── ports of the benchmark's GP helpers ───────────────────────────────────────────

def normalise(X, lo, hi) -> torch.Tensor:
    '''Min-max to [0, 1] with the free box of the config (float32).'''
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    lo = torch.as_tensor(np.asarray(lo, dtype=np.float32))
    hi = torch.as_tensor(np.asarray(hi, dtype=np.float32))
    return (X - lo) / (hi - lo)


def create_gp_model(model: str, gp: dict, x_train, y_train, x_val, y_val, head_name: str,
                    device, seed: int, threshold: float = 0.0):
    '''``exact_gp``: Gaussian likelihood (the +-1 least-squares verdict when classifying);
    ``deep_gp``: Bernoulli likelihood when classifying, Gaussian otherwise.'''
    device = torch.device(device)
    x_train, x_val = x_train.to(device), x_val.to(device)
    y_train, y_val = y_train.view(-1).to(device), y_val.view(-1).to(device)
    n_dim = x_train.shape[1]
    if model == "exact_gp":
        return ExactGP(x_train, y_train, x_val, y_val, n_dim,
                       lengthscale=gp["lengthscale"], use_ard=gp["use_ard"], noise=gp["noise"],
                       kernel=gp["kernel"], thr=threshold, seed=seed)
    if model == "deep_gp":
        return DeepGP(x_train, y_train, x_val, y_val, n_dim,
                      lengthscale=gp["lengthscale"], noise=gp["noise"],
                      num_hidden_dims=gp["num_hidden_dims"], num_inducing_max=gp["num_inducing_max"],
                      kernel=gp["kernel"], num_samples=gp["num_samples"], seed=seed, thr=threshold,
                      likelihood="bernoulli" if head_name == "classification" else "gaussian")
    raise ValueError(f"not a GP surrogate: {model!r}")


def load_warm_start(model, ckpt: dict, logger) -> None:
    '''Load the previous iteration's state, keeping only shape-compatible tensors (the
    inducing points grow with the training set), likelihood included.'''
    state = model.state_dict()
    compatible = {k: v for k, v in ckpt["model_state_dict"].items()
                  if k in state and v.shape == state[k].shape}
    skipped = [k for k in ckpt["model_state_dict"] if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    lik = ckpt.get("likelihood_state_dict") or {}
    lik_state = model.likelihood.state_dict()
    model.likelihood.load_state_dict({k: v for k, v in lik.items()
                                      if k in lik_state and v.shape == lik_state[k].shape}, strict=False)
    logger.info(f"warm start: {len(compatible)} tensors loaded, {len(skipped)} skipped (shape changed)")


def train_gp(model, model_type: str, gp: dict, epochs: int, patience: int, logger) -> tuple:
    '''Returns ``(train_losses, val_losses, seconds)``; the model is left at its best state.'''
    t0 = time.time()
    if model_type == "exact_gp":
        _, tr, va = model.do_train_loop(lr=gp["learning_rate"], iters=epochs, jitter=gp["jitter"],
                                        patience=patience, logger=logger)
    else:
        _, tr, va = model.do_train_loop(lr=gp["learning_rate"], iters=epochs, batch_size=gp["batch_size"],
                                        jitter=gp["jitter"], patience=patience, logger=logger)
    return tr, va, time.time() - t0


def _settings(is_deep: bool, jitter: float, num_samples: int):
    return (gpytorch.settings.fast_pred_var(not is_deep),
            gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter),
            gpytorch.settings.num_likelihood_samples(num_samples))


@torch.no_grad()
def latent_moments(model, x_norm: torch.Tensor, is_deep: bool, jitter: float, num_samples: int) -> tuple:
    '''Latent posterior ``(mean, var)``: ``(N,)`` for the exact GP, ``(S, N)`` sample paths
    for the deep GP (a mixture; the head averages after the link).'''
    device = next(model.parameters()).device
    model.eval()
    model.likelihood.eval()
    batch = 10000 if is_deep else 5000
    means, variances = [], []
    for i in range(0, len(x_norm), batch):
        xb = x_norm[i:i + batch].to(device)
        fpv, jit, nls = _settings(is_deep, jitter, num_samples)
        with fpv, jit, nls:
            latent = model(xb)
            m, v = latent.mean.detach(), latent.variance.detach()
        means.append(m.reshape(-1, m.shape[-1]) if is_deep else m.reshape(-1))
        variances.append(v.reshape(-1, v.shape[-1]) if is_deep else v.reshape(-1))
    cat_dim = -1 if is_deep else 0
    return torch.cat(means, dim=cat_dim).cpu(), torch.cat(variances, dim=cat_dim).cpu()


@torch.no_grad()
def predictive_moments(model, x_norm: torch.Tensor, is_deep: bool, jitter: float, num_samples: int) -> tuple:
    '''``likelihood(model(x))`` moments ``(mean, var)``, ``(N,)``; deep GP paths averaged.
    This is what the GP driver's entropy batch and tolerance cut consume (Gaussian
    likelihood: the latent mean, variance plus noise).'''
    device = next(model.parameters()).device
    model.eval()
    model.likelihood.eval()
    batch = 10000 if is_deep else 5000
    means, variances = [], []
    for i in range(0, len(x_norm), batch):
        xb = x_norm[i:i + batch].to(device)
        with torch.no_grad(), gpytorch.settings.eval_cg_tolerance(1e-4), \
             gpytorch.settings.max_cg_iterations(300), gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.fast_pred_samples(True), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples if is_deep else 1):
            preds = model.likelihood(model(xb))
            m, v = preds.mean.detach(), preds.variance.detach()
        if is_deep:
            means.append(m.mean(dim=0).reshape(-1))
            variances.append(v.mean(dim=0).reshape(-1))
        else:
            means.append(m.reshape(-1))
            variances.append(v.reshape(-1))
    return torch.cat(means).cpu(), torch.cat(variances).cpu()


@torch.no_grad()
def posterior_block(model, x_pool: torch.Tensor, is_deep: bool, jitter: float, num_samples: int) -> tuple:
    '''Full predictive ``(mean, covariance)`` on a focused pool (deep GP: mean over paths).'''
    device = next(model.parameters()).device
    model.eval()
    model.likelihood.eval()
    with gpytorch.settings.eval_cg_tolerance(1e-4), gpytorch.settings.max_cg_iterations(300), \
         gpytorch.settings.fast_pred_var(False), gpytorch.settings.fast_pred_samples(True), \
         gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         gpytorch.settings.num_likelihood_samples(num_samples if is_deep else 1):
        preds = model.likelihood(model(x_pool.to(device)))
        if is_deep:
            mean = preds.mean.detach().mean(dim=0).reshape(-1)
            covar = preds.covariance_matrix.detach().mean(dim=0)
        else:
            mean = preds.mean.detach().reshape(-1)
            covar = preds.covariance_matrix.detach()
    return mean, covar


@torch.no_grad()
def latent_mean(model, x_norm: torch.Tensor, is_deep: bool, jitter: float, num_samples: int,
                batch: int = 1024) -> torch.Tensor:
    '''Mean latent (paths averaged), for the verdict/value metrics; the decision point is 0.'''
    device = next(model.parameters()).device
    model.eval()
    out = []
    with gpytorch.settings.fast_pred_var(not is_deep), \
         gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         gpytorch.settings.num_likelihood_samples(num_samples):
        for i in range(0, len(x_norm), batch):
            m = model(x_norm[i:i + batch].to(device)).mean.detach()
            out.append((m.mean(dim=0) if is_deep else m).reshape(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0)
