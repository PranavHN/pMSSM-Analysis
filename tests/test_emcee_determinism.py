'''The emcee chain's reproducibility contract.

An emcee run at a fixed --seed is bit-reproducible: two runs produce identical chains,
log-probs and ntuple branches, and the result does not depend on mcmc.nworkers. That is
worth pinning because it was originally true only by accident -- emcee's EnsembleSampler
seeds its proposal RNG from `backend.random_state` if the backend has one and from the
process-wide `np.random.get_state()` otherwise, and ModelGenerator.__init__ happens to
call np.random.seed(seed) in the parent before the sampler is built. seed_sampler_rng()
replaced that coincidence with an explicit assignment; these tests fail if an emcee
upgrade (pinned only to <4) changes the mechanism, or if someone reorders the driver.

Deliberately no physics here: a toy log-prob exercises the same proposal machinery in
milliseconds, so the guarantee is checked on every run rather than in a slow nightly.
'''
import numpy as np
import pytest

import emcee

from Run3ModelGen.emcee_sampler import build_moves, init_walkers, seed_sampler_rng


NWALKERS, NDIM, STEPS = 12, 3, 8


def _toy_log_prob(theta):
    '''Unit Gaussian. Deterministic in theta, so any run-to-run difference in the
    resulting chain comes from the sampler's RNG and nothing else.'''
    return -0.5 * float(np.dot(theta, theta))


def _run(seed, steps=STEPS, moves=None, p0=None):
    # Scramble the process-wide RNG first, so inheriting it (emcee's fallback) can NOT
    # make two runs agree. Without this every sampler built in one pytest process reads
    # the same global state and the reproducibility assertions pass vacuously -- while
    # two real runs, in two processes with entropy-seeded global state, would diverge.
    # This models the production case: one fresh process per chain.
    np.random.seed(None)
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, _toy_log_prob, moves=moves)
    seed_sampler_rng(sampler, seed)
    if p0 is None:
        p0 = np.random.default_rng(0).standard_normal((NWALKERS, NDIM))
    sampler.run_mcmc(p0, steps, progress=False)
    return sampler.get_chain(), sampler.get_log_prob()


# ── the contract ────────────────────────────────────────────────────────────────

def test_same_seed_gives_an_identical_chain():
    c1, l1 = _run(7)
    c2, l2 = _run(7)
    assert np.array_equal(c1, c2), "same seed must reproduce the chain bitwise"
    assert np.array_equal(l1, l2)


def test_different_seeds_give_a_different_chain():
    '''Guards the opposite failure: a seeding bug that pins every run to one stream
    would make the test above pass while destroying ensemble independence, which is
    what the cross-ensemble R-hat depends on.'''
    c1, _ = _run(7)
    c2, _ = _run(8)
    assert not np.array_equal(c1, c2), "different seeds must explore differently"


def test_seeding_actually_took_effect():
    '''emcee's random_state setter is `try: ... except: pass` -- it fails SILENTLY.
    seed_sampler_rng must therefore verify, not just assign.'''
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, _toy_log_prob)
    seed_sampler_rng(sampler, 123)
    want = np.random.RandomState(123).get_state()
    got = sampler.random_state
    assert got[0] == want[0]
    assert np.array_equal(got[1], want[1])
    assert tuple(got[2:]) == tuple(want[2:])


def test_seed_sampler_rng_raises_if_the_state_will_not_take():
    '''If a future emcee stops accepting a legacy MT19937 state tuple, the run must
    fail loudly rather than silently produce an unreproducible chain.'''
    class _Deaf:
        '''Mimics emcee's swallow-everything setter.'''
        random_state = property(lambda self: np.random.RandomState(999).get_state(),
                                lambda self, state: None)

    with pytest.raises(RuntimeError, match="could not pin the ensemble proposal RNG"):
        seed_sampler_rng(_Deaf(), 5)


def test_the_chain_does_not_depend_on_the_move_mix_being_reseeded():
    '''A multi-move ensemble draws its move choice from the same RNG
    (`self._random.choice(self._moves, ...)`), so the mix must be reproducible too --
    this is the path a DEMove/DESnookerMove production config takes.'''
    moves = build_moves([{'type': 'DEMove', 'kwargs': {}, 'weight': 0.8},
                         {'type': 'DESnookerMove', 'kwargs': {}, 'weight': 0.2}])
    c1, _ = _run(11, moves=moves)
    c2, _ = _run(11, moves=moves)
    assert np.array_equal(c1, c2)


# ── resume must extend the chain, not retrace it ────────────────────────────────

def _legs(tmp_path, name, reseed_second_leg):
    '''Run 5+5 iterations as two samplers over one backend. Returns the full chain.'''
    p0 = np.random.default_rng(0).standard_normal((NWALKERS, NDIM))
    backend = emcee.backends.HDFBackend(str(tmp_path / name))
    backend.reset(NWALKERS, NDIM)

    np.random.seed(None)
    first = emcee.EnsembleSampler(NWALKERS, NDIM, _toy_log_prob, backend=backend)
    seed_sampler_rng(first, 42)                     # fresh chain -> seed it
    first.run_mcmc(p0, 5, progress=False)

    # A second process picking the chain up: fresh interpreter, scrambled global RNG,
    # same backend file. initial_state=None continues from the backend's last sample.
    np.random.seed(None)
    second = emcee.EnsembleSampler(NWALKERS, NDIM, _toy_log_prob, backend=backend)
    if reseed_second_leg:
        seed_sampler_rng(second, 42)                # the bug this guards against
    second.run_mcmc(None, 5, progress=False)
    return backend.get_chain()


def test_a_resumed_chain_equals_an_uninterrupted_one(tmp_path):
    '''emcee saves its RNG state to the backend every iteration and restores it on
    resume, which is why run_emcee_scan seeds ONLY when is_fresh. 5+5 must equal 10.'''
    p0 = np.random.default_rng(0).standard_normal((NWALKERS, NDIM))
    whole = emcee.backends.HDFBackend(str(tmp_path / "whole.h5"))
    whole.reset(NWALKERS, NDIM)
    np.random.seed(None)
    s = emcee.EnsembleSampler(NWALKERS, NDIM, _toy_log_prob, backend=whole)
    seed_sampler_rng(s, 42)
    s.run_mcmc(p0, 10, progress=False)

    assert np.array_equal(whole.get_chain(), _legs(tmp_path, "split.h5", False)), \
        "a resumed leg must continue the chain the uninterrupted run would have made"


def test_a_resume_is_governed_by_the_backend_not_by_our_seed(tmp_path):
    '''`sample()` assigns `self.random_state = state.random_state`, and a resumed State
    carries the RNG state saved in the backend — so a continuation is driven by the
    backend whether or not anyone re-seeds, and re-seeding it is a no-op rather than a
    corruption. run_emcee_scan still gates seeding on is_fresh: expressing the intent
    beats depending on that override staying a no-op.

    The same assignment is why seeding a FRESH chain works at all: State(p0) carries
    random_state=None, which emcee's setter rejects silently, leaving our state in
    place. test_same_seed_gives_an_identical_chain covers that end of it.'''
    assert np.array_equal(_legs(tmp_path, "plain.h5", False),
                          _legs(tmp_path, "reseeded.h5", True))


# ── walker initialisation ───────────────────────────────────────────────────────

def test_init_walkers_refuses_an_implicit_rng():
    '''The unseeded default_rng() fallback was the other silent route to an
    unreproducible run.'''
    with pytest.raises(ValueError, match="requires an explicit `rng`"):
        init_walkers({'a': [0, 1]}, ['a'], 4, pool=None, rng=None)


def test_init_walkers_is_reproducible_for_a_given_rng():
    class _SerialPool:
        '''pool.map stand-in: the real Pool preserves order, which is why the chain is
        invariant to mcmc.nworkers.'''
        @staticmethod
        def map(fn, xs):
            return [fn(x) for x in xs]

    params = {'a': [0.0, 1.0], 'b': [-1.0, 1.0]}
    kw = dict(parameters=params, free_names=['a', 'b'], nwalkers=6, pool=_SerialPool())
    # a finite toy log-prob for every point: patch the module's log_prob via the pool
    import Run3ModelGen.emcee_sampler as es
    orig = es.log_prob
    es.log_prob = lambda theta: 0.0
    try:
        p1 = init_walkers(**kw, rng=np.random.default_rng(4))
        p2 = init_walkers(**kw, rng=np.random.default_rng(4))
        p3 = init_walkers(**kw, rng=np.random.default_rng(5))
    finally:
        es.log_prob = orig
    assert np.array_equal(p1, p2), "same rng seed must give the same walker ball"
    assert not np.array_equal(p1, p3)
    assert p1.shape == (6, 2)
