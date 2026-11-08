"""
Generate simulated stock price paths using Heston model. This generated data will be used to train the dynamic hedging model.
"""
import math
import numpy as np
from dataclasses import dataclass


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





