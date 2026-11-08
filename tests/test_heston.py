"""Tests for HestonParams in sim/heston.py (the parameters of the Heston model, Eq. (42))."""

import dataclasses
import math

import pytest

from sim.heston import HestonParams

# One set of sensible values. Each test copies this and changes one entry, so that any
# error raised can only come from the entry under test.
VALID = dict(s0=100.0, v0=0.04, mu=0.0, kappa=1.0, theta=0.04, xi=2.0, rho=-0.7)

ALL_NAMES = list(VALID)  # ["s0", "v0", "mu", "kappa", "theta", "xi", "rho"]


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
        HestonParams(**{**VALID, name: value})


# Two stacked parametrize decorators run every combination: 7 names x 3 values = 21 tests.
@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_value_is_rejected(name, value):
    with pytest.raises(ValueError, match=f"^{name} "):
        HestonParams(**{**VALID, name: value})


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
    HestonParams(**{**VALID, name: value})


def test_arguments_must_be_given_by_name():
    # Positional arguments could silently swap kappa and theta, so they are refused.
    with pytest.raises(TypeError):
        HestonParams(100.0, 0.04, 0.0, 1.0, 0.04, 2.0, -0.7)


def test_parameters_cannot_be_changed_after_creation():
    # Otherwise someone could set kappa = -1 later and skip the checks.
    params = HestonParams(**VALID)
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.kappa = -1.0