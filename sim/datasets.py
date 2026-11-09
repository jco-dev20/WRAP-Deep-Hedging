"""
This contains the functionality to generate the datasets which will used to train the neural networks for hedging. 
"""

import torch
import numpy as np
from dataclasses import dataclass,replace
from heston import simulate_heston,HestonParams
from config import ExperimentConfig

@dataclass(frozen=True, kw_only=True)
class PathSet:
    """A group of simulated paths, plus the Heston parameters each path was simulated with.

    Every field has the same first dimension N (one entry per path), so entry n of every
    field belongs to path n.

    Attributes:
        X: simulated paths, float32 [N, T+1, 2]; channel 0 = log S, channel 1 = v.
        mu: drift of each path, float32 [N].
        kappa: mean-reversion speed of each path, float32 [N].
        theta: long-run variance of each path, float32 [N].
        block: block each path came from (1 to n_blocks), int64 [N].
    """

    X: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor
    block: torch.Tensor

    def __post_init__(self) -> None:
        # X must be [N, T+1, 2]: three dimensions, the last one holding (log S, v).
        if self.X.ndim != 3 or self.X.shape[2] != 2:
            raise ValueError(f"X must have shape [N, T+1, 2], got {tuple(self.X.shape)}")

        # Each per-path field must be 1-D with one entry per path. Dimension 0 is N.
        n = self.X.shape[0]
        for name in ("mu", "kappa", "theta", "block"):
            shape = tuple(getattr(self, name).shape)
            if shape != (n,):
                raise ValueError(f"{name} must have shape ({n},) to match X, got {shape}")

        # Types: numbers are float32, block labels are whole numbers (int64).
        for name in ("X", "mu", "kappa", "theta"):
            dtype = getattr(self, name).dtype
            if dtype != torch.float32:
                raise ValueError(f"{name} must be float32, got {dtype}")
        if self.block.dtype != torch.int64:
            raise ValueError(f"block must be int64, got {self.block.dtype}")

    @property
    def n_paths(self) -> int:
        """Number of paths N."""
        return self.X.shape[0]


def simulate_block(params, n_paths, grid, rng, block) -> PathSet:
    X = simulate_heston(params, n_paths, grid.n_steps, grid.dt, rng)
    mu = torch.full((n_paths,), params.mu)
    kappa = torch.full((n_paths,),params.kappa)
    theta = torch.full((n_paths,),params.theta)
    block_labels = torch.full((n_paths,),block)

    return PathSet(X=X, mu=mu, kappa=kappa,theta=theta,block=block_labels)

def concat(pathsets: list[PathSet]) -> PathSet:
    """Join several PathSets end to end into one, keeping their order.

    Order matters: make_drifting_dataset passes the blocks oldest first, so the joined set
    stays ordered oldest -> newest, which the drift-aware weights rely on.

    Args:
        pathsets: PathSets that all have the same number of dates T+1.

    Returns:
        One PathSet holding every path, with N = total number of paths:
        X [N, T+1, 2]; mu, kappa, theta, block [N].
    """
    if len(pathsets) == 0:
        raise ValueError("concat needs at least one PathSet")

    # torch.cat joins tensors along an existing dimension. dim=0 is the path dimension,
    # so the paths of the second PathSet go after those of the first, and so on.
    # [p.X for p in pathsets] is a list of the X tensor from each PathSet, in order.
    return PathSet(
        X=torch.cat([p.X for p in pathsets], dim=0),
        mu=torch.cat([p.mu for p in pathsets], dim=0),
        kappa=torch.cat([p.kappa for p in pathsets], dim=0),
        theta=torch.cat([p.theta for p in pathsets], dim=0),
        block=torch.cat([p.block for p in pathsets], dim=0),
    )

def make_drifting_dataset(cfg :ExperimentConfig,rng):
    """Simulate B blocks of paths whose drift and long-run variance rise block by block.

    Section 5.1 / Appendix D.6.1. Block b = 1..B uses
        mu_b    = mu    + mu_increase    * (b - 1) / (B - 1)
        theta_b = theta + theta_increase * (b - 1) / (B - 1),   v0_b = theta_b
    so the market's rules shift steadily from the first block to the last. Blocks are
    simulated oldest first, all from the same generator.

    Args:
        cfg: experiment settings; uses cfg.heston (baseline parameters), cfg.grid and cfg.data.
        rng: seeded NumPy generator, created once in make_dataset.

    Returns:
        {"train": blocks 1..B-3 joined, "val": block B-2, "near": block B-1, "far": block B}.
        With B = 8: train has 20,000 paths, the others 4,000 each.
    """
    base = cfg.heston
    B = cfg.data.n_blocks
    blocks = []
    for b in range(1, B + 1):  # b = 1, 2, ..., B (range stops before its end value)
        frac = (b - 1) / (B - 1)  # 0 for the first block, 1 for the last
        mu_b = base.mu + cfg.data.mu_increase * frac
        theta_b = base.theta + cfg.data.theta_increase * frac
        # A copy of the baseline parameters with mu, theta and v0 changed (checks run again).
        block_params = replace(base, mu=mu_b, theta=theta_b, v0=theta_b)
        blocks.append(simulate_block(block_params, cfg.data.paths_per_block, cfg.grid, rng, b))

    # The list starts at 0, so blocks[0] is block 1 and blocks[B - 1] is block B.
    return {
        "train": concat(blocks[: B - 3]),
        "val": blocks[B - 3],
        "near": blocks[B - 2],
        "far": blocks[B - 1],
    }

def make_stationary_dataset(cfg: ExperimentConfig, rng):
    """Simulate three equal-sized splits with the same, unchanging Heston parameters.

    Stationary Heston experiment (Table 3): independent paths under fixed parameters,
    split equally into train / validation / test. The block numbers 1, 2, 3 are only labels.

    Args:
        cfg: experiment settings; uses cfg.heston, cfg.grid and cfg.data.paths_per_split.
        rng: seeded NumPy generator, created once in make_dataset.

    Returns:
        {"train", "val", "test"}: one PathSet each, with cfg.data.paths_per_split paths.
    """
    blocks = []
    B = cfg.data.n_blocks
    for b, name in enumerate(["train", "val", "test"], start=1):
        blocks.append(simulate_block(cfg.heston,cfg.data.paths_per_block, cfg.grid, rng, b))

    return blocks


def make_dataset(cfg: ExperimentConfig):
    rng = np.random.default_rng(cfg.data_seed)
    if cfg.data.kind == "drifting":
        return make_drifting_dataset(cfg,rng)

    return make_stationary_dataset(cfg,rng)



