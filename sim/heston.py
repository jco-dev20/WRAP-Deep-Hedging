"""
Generate simulated stock price paths using Heston model. This generated data will be used to train the dynamic hedging model.
"""
import math
import numpy as np
import torch
from dataclasses import dataclass

#seed = 42
#rng = np.random.default_rng(seed)

@dataclass(frozen=True, kw_only=True)
class HestonParams:
    """Parameters of the Heston model, Eq. (42). Time is measured in years.
 
    The paper does not print its values; the ones we use are set in the config
    (see docs/DECISIONS.md, D-003).
 
    Attributes:
        s0: stock price at time 0. Must be > 0.
        v0: variance at time 0. Must be >= 0. (Volatility is sqrt(v0): v0 = 0.04 means 20%.)
        mu: drift, the average growth rate of the price per year. Any sign.
        kappa: mean-reversion speed, how strongly v is pulled back towards theta. Must be > 0.
        theta: long-run variance, the level v is pulled back towards. Must be > 0.
        xi: vol-of-vol, how randomly the variance itself moves. Must be > 0.
        rho: correlation between the price shock and the variance shock. In [-1, 1].
    """
 
    s0: float
    v0: float
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float
 
    def __post_init__(self) -> None:
        # NaN ("not a number") fails every comparison, so a check like `kappa <= 0` would
        # quietly let it through. Rule out NaN and infinity for every parameter first.
        for name in ("s0", "v0", "mu", "kappa", "theta", "xi", "rho"):
            value = getattr(self, name)  # getattr(self, "kappa") is self.kappa
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value}")
 
        # Each message starts with the parameter's name; the tests rely on this.
        if self.s0 <= 0:
            raise ValueError(f"s0 must be > 0, got {self.s0}")
        if self.v0 < 0:
            raise ValueError(f"v0 must be >= 0, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be > 0, got {self.xi}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")


def cir_exact_step(v: np.ndarray, dt: float, params: HestonParams, rng):
    exp_scale = np.exp(-1.0 * params.kappa * dt)
    c = (params.xi)**2 * (1 - exp_scale) / (4 * params.kappa)
    d = (4.0 * params.kappa * params.theta) / (params.xi)**2
    f_lambda = v * exp_scale / c
    Y = rng.noncentral_chisquare(d, f_lambda)

    return c * Y

def log_price_step(log_s: np.ndarray, v: float, v_next: float, z: float, dt: float, params: HestonParams):
    int_var = 0.5 * (v + v_next) * dt
    M = (v_next - v - params.kappa * params.theta * dt + params.kappa * int_var) / params.xi
    log_s_next = log_s + params.mu * dt - 0.5 * int_var + params.rho * M + np.sqrt(1 - params.rho**2) * np.sqrt(int_var) * z
    return log_s_next

def simulate_heston(params: HestonParams, n_paths: int, n_steps: int, dt: float, rng):

    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    X = np.zeros((n_paths, n_steps + 1, 2))

    # Day 0: every path starts from the same price and variance.
    log_s = np.full(n_paths, math.log(params.s0))  # shape [N]: N copies of log(s0)
    v = np.full(n_paths, params.v0)                 # shape [N]: N copies of v0
    X[:, 0, 0] = log_s                              # ":" means "every path"
    X[:, 0, 1] = v

    for t in range(n_steps):  # t = 0, 1, ..., n_steps - 1
        # Keep this draw order (variance first, then z) so a seed always gives the same paths.
        v_next = cir_exact_step(v, dt, params, rng)
        z = rng.standard_normal(n_paths)
        log_s = log_price_step(log_s, v, v_next, z, dt, params)  # needs the OLD v and v_next
        v = v_next  # only now move on to tomorrow
        X[:, t + 1, 0] = log_s
        X[:, t + 1, 1] = v

    # Simulate in float64 for accuracy; hand the network a float32 torch tensor.
    return torch.from_numpy(X).to(torch.float32)
