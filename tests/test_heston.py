"""Tests for heston.py: Heston parameters and paths, Eq. (42).

Fast tests (python -m pytest -q) check bookkeeping and one step worked out by hand.
Slow tests (python -m pytest -q -m slow) check averages over many paths against exact formulas.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from sim.heston import HestonParams, cir_exact_step, log_price_step, simulate_heston

# One set of sensible values. Each test copies this and changes only what it needs.
VALID = dict(s0=100.0, v0=0.04, mu=0.0, kappa=1.0, theta=0.04, xi=2.0, rho=-0.7)
ALL_NAMES = list(VALID)  # ["s0", "v0", "mu", "kappa", "theta", "xi", "rho"]

DT = 1 / 365  # one day, in years
N_STEPS = 30  # 30 days
HORIZON = N_STEPS * DT  # total time in years


def make_params(**changes):
    """HestonParams with the VALID values, except for any given here, e.g. make_params(mu=0.2)."""
    return HestonParams(**{**VALID, **changes})


def simulate(params=None, n_paths=1_000, n_steps=N_STEPS, seed=0):
    """Run simulate_heston with a fresh generator made from `seed`."""
    params = params if params is not None else make_params()
    return simulate_heston(params, n_paths, n_steps, DT, np.random.default_rng(seed))


# =============================================================== HestonParams


def test_valid_parameters_are_stored():
    params = HestonParams(**VALID)
    for name, value in VALID.items():
        assert getattr(params, name) == value  # getattr(params, "kappa") is params.kappa


@pytest.mark.parametrize(
    "name, value",
    [
        ("s0", 0.0),  # a price of exactly zero is not allowed
        ("s0", -1.0),
        ("v0", -0.01),  # variance can be zero, but never negative
        ("kappa", 0.0),
        ("kappa", -1.0),
        ("theta", 0.0),
        ("xi", 0.0),
        ("rho", -1.5),
        ("rho", 1.01),
    ],
)
def test_out_of_range_value_is_rejected(name, value):
    # match= is searched in the error message. "^" means "at the start", so this checks the
    # message begins with the parameter's name: the check for *this* parameter fired.
    # (Without "^", "mu" would also match the word "must" in any other message.)
    with pytest.raises(ValueError, match=f"^{name} "):
        make_params(**{name: value})


# Two stacked parametrize decorators run every combination: 7 names x 3 values = 21 tests.
@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_value_is_rejected(name, value):
    with pytest.raises(ValueError, match=f"^{name} "):
        make_params(**{name: value})


@pytest.mark.parametrize(
    "name, value",
    [
        ("rho", 1.0),  # the edges of [-1, 1] are allowed
        ("rho", -1.0),
        ("v0", 0.0),  # zero variance is allowed
        ("mu", -0.1),  # the drift can have either sign
    ],
)
def test_boundary_values_are_allowed(name, value):
    # No assert needed: if the constructor raises, the test fails.
    make_params(**{name: value})


def test_arguments_must_be_given_by_name():
    # Positional arguments could silently swap kappa and theta, so they are refused.
    with pytest.raises(TypeError):
        HestonParams(100.0, 0.04, 0.0, 1.0, 0.04, 2.0, -0.7)


def test_parameters_cannot_be_changed_after_creation():
    params = make_params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.kappa = -1.0


# =============================================================== cir_exact_step


def test_cir_step_keeps_shape_and_is_never_negative():
    v = np.array([0.0, 1e-12, 0.04, 0.5])  # includes v = 0 exactly
    v_next = cir_exact_step(v, DT, make_params(), np.random.default_rng(0))
    assert v_next.shape == v.shape  # one new value per path
    assert np.all(v_next >= 0)
    assert np.all(np.isfinite(v_next))


# =============================================================== log_price_step


def test_log_price_step_matches_hand_calculation():
    # One step with numbers chosen to be easy to follow on paper.
    params = make_params(mu=0.1, kappa=2.0, theta=0.04, xi=0.5, rho=-0.5)
    dt = 0.01
    log_s = np.array([math.log(100.0)])
    v, v_next, z = np.array([0.04]), np.array([0.05]), np.array([1.0])

    # Total variance over the step (trapezoid rule): 0.5 * (0.04 + 0.05) * 0.01
    int_var = 0.00045
    # Variance shock recovered from the variance equation:
    #   M = (v_next - v - kappa*theta*dt + kappa*int_var) / xi = (0.01 - 0.0008 + 0.0009) / 0.5
    M = 0.0202
    expected_increment = (
        0.1 * dt  # mu*dt              =  0.001
        - 0.5 * int_var  # -0.5*I          = -0.000225
        + (-0.5) * M  # rho*M              = -0.0101
        + math.sqrt(1 - 0.25) * math.sqrt(int_var) * 1.0  # sqrt(1-rho^2)*sqrt(I)*z
    )  # total ~ 0.0090462

    log_s_next = log_price_step(log_s, v, v_next, z, dt, params)
    assert log_s_next.shape == (1,)
    assert log_s_next[0] == pytest.approx(math.log(100.0) + expected_increment, abs=1e-12)


# =============================================================== simulate_heston: fast


def test_output_shape_and_type():
    X = simulate(n_paths=7, n_steps=5)
    assert isinstance(X, torch.Tensor)
    assert X.dtype == torch.float32
    assert X.shape == (7, 6, 2)  # [N, T+1, 2]: 5 steps give 6 dates


def test_day_zero_is_the_starting_point():
    # Non-default values, so the test cannot pass just because the defaults happen to match.
    X = simulate(make_params(s0=95.0, v0=0.09), n_paths=11)
    # float32 cannot store 0.09 exactly, so compare with a small tolerance, not ==.
    assert X[:, 0, 0].tolist() == pytest.approx([math.log(95.0)] * 11)  # channel 0 = log S
    assert X[:, 0, 1].tolist() == pytest.approx([0.09] * 11)  # channel 1 = v


def test_paths_are_finite_and_variance_never_negative():
    # With xi = 2 the variance often gets very close to zero, so this is a real test.
    X = simulate(n_paths=5_000)
    assert torch.isfinite(X).all()
    assert (X[:, :, 1] >= 0).all()


def test_same_seed_gives_identical_paths():
    assert torch.equal(simulate(seed=123), simulate(seed=123))


def test_different_seeds_give_different_paths():
    assert not torch.equal(simulate(seed=1), simulate(seed=2))


@pytest.mark.parametrize(
    "n_paths, n_steps, dt, bad_name",
    [
        (0, 30, DT, "n_paths"),
        (10, 0, DT, "n_steps"),
        (10, 30, 0.0, "dt"),
        (10, 30, -DT, "dt"),
    ],
)
def test_bad_sizes_are_rejected(n_paths, n_steps, dt, bad_name):
    with pytest.raises(ValueError, match=f"^{bad_name} "):
        simulate_heston(make_params(), n_paths, n_steps, dt, np.random.default_rng(0))


# =============================================================== simulate_heston: slow


def mean_is_close(sample, target, n_se=4.0):
    """True if the sample's average is within n_se standard errors of target.

    Standard error = (sample std) / sqrt(number of samples): how much the average of this
    many random numbers typically wobbles. A correct simulator lands within 4 standard errors
    except about once in 16,000 seeds.
    """
    standard_error = sample.std(ddof=1) / math.sqrt(sample.size)
    return abs(sample.mean() - target) <= n_se * standard_error


@pytest.mark.slow
def test_final_variance_has_the_exact_cir_mean_and_variance():
    # Start away from the long-run level, so the variance has to move towards theta.
    # (Starting at v0 = theta, the mean stays at theta and a kappa bug could hide.)
    kappa, theta, xi, v0 = 1.0, 0.04, 2.0, 0.09
    X = simulate(make_params(v0=v0, kappa=kappa, theta=theta, xi=xi), n_paths=400_000, seed=7)
    v_T = X[:, -1, 1].double().numpy()  # float64 for accurate averages
    e = math.exp(-kappa * HORIZON)

    mean_exact = theta + (v0 - theta) * e
    var_exact = v0 * xi**2 * e * (1 - e) / kappa + theta * xi**2 * (1 - e) ** 2 / (2 * kappa)

    assert mean_is_close(v_T, mean_exact)
    # (v_T - exact mean)^2 is itself a sample whose average should be the exact variance.
    assert mean_is_close((v_T - mean_exact) ** 2, var_exact)


@pytest.mark.slow
@pytest.mark.parametrize("mu", [0.0, 0.2])
def test_average_final_price_grows_at_rate_mu(mu):
    # dS = mu*S*dt + ...  means the average price grows like S0 * exp(mu * time).
    # With mu = 0 the price is a "martingale": on average it stays where it started.
    X = simulate(make_params(mu=mu), n_paths=400_000, seed=11)
    s_T = np.exp(X[:, -1, 0].double().numpy())  # X stores log S, so undo the log
    assert mean_is_close(s_T, 100.0 * math.exp(mu * HORIZON))


@pytest.mark.slow
def test_tiny_vol_of_vol_gives_correlation_rho():
    # With a tiny xi the variance barely moves from theta, so daily log-price changes have
    # standard deviation sqrt(theta*dt) and their correlation with variance changes is rho.
    # This is the only test that notices if the sign of rho is flipped.
    rho, theta = -0.7, 0.04
    X = simulate(make_params(v0=theta, theta=theta, xi=1e-3, rho=rho), n_paths=20_000, seed=3)
    X = X.double().numpy()
    daily_log_s_change = np.diff(X[:, :, 0], axis=1).ravel()  # all paths, all days
    daily_v_change = np.diff(X[:, :, 1], axis=1).ravel()

    assert daily_log_s_change.std() == pytest.approx(math.sqrt(theta * DT), rel=0.01)
    assert np.corrcoef(daily_log_s_change, daily_v_change)[0, 1] == pytest.approx(rho, abs=0.01)