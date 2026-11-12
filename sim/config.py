"""Experiment settings, read from a YAML file into dataclasses.

All settings live in a YAML file such as config.yaml. load_config reads it and checks it;
save_config writes a copy next to a run's results, so every result can be traced back to
the settings that produced it (CLAUDE.md rules 3, 5, 6).
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from sim.heston import HestonParams

DATA_KINDS = ("stationary", "drifting")


@dataclass(frozen=True, kw_only=True)
class GridConfig:
    """The time grid: n_steps steps, each of length dt = 1 / days_per_year years.

    Attributes:
        n_steps: number of steps T; each path has T + 1 dates. Must be > 0.
        days_per_year: steps per year; 365 means one step per calendar day. Must be > 0.
    """

    n_steps: int
    days_per_year: float

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.days_per_year <= 0:
            raise ValueError(f"days_per_year must be > 0, got {self.days_per_year}")

    # @property lets you write grid.dt like a field, but the value is computed each time.
    # It is not a field, so it is not saved to the YAML file (YAML cannot store "1/365").
    @property
    def dt(self) -> float:
        """Length of one step, in years."""
        return 1.0 / self.days_per_year


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    """How to build the dataset (Section 5.1, Appendix D.6.1).

    For kind = "drifting", block b = 1..n_blocks uses
        mu_b    = mu    + mu_increase    * (b - 1) / (n_blocks - 1)
        theta_b = theta + theta_increase * (b - 1) / (n_blocks - 1),     v0_b = theta_b
    where mu and theta are the baseline values from the heston section.

    Attributes:
        kind: "stationary" or "drifting".
        n_blocks: number of blocks B. Must be >= 2, because the formula divides by B - 1.
        paths_per_block: number of paths simulated in each block. Must be > 0.
        mu_increase: total rise in mu from the first block to the last (paper: 0.20).
        theta_increase: total rise in theta from the first block to the last (paper: 0.12).
    """

    kind: str
    n_blocks: int
    paths_per_block: int
    mu_increase: float
    theta_increase: float

    def __post_init__(self) -> None:
        if self.kind not in DATA_KINDS:
            raise ValueError(f"kind must be one of {DATA_KINDS}, got {self.kind!r}")
        if self.n_blocks < 2:
            raise ValueError(f"n_blocks must be >= 2, got {self.n_blocks}")
        if self.paths_per_block <= 0:
            raise ValueError(f"paths_per_block must be > 0, got {self.paths_per_block}")


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """All settings for one experiment. Mirrors the sections of the YAML file.

    Attributes:
        name: experiment name; results will go to results/<name>/.
        data_seed: seed for simulating the paths. Must be >= 0.
        model_seed: seed for the network (used later). Kept separate from data_seed so the
            same paths can be reused with differently initialised networks. Must be >= 0.
        heston: baseline Heston parameters (DECISIONS D-003).
        grid: the time grid.
        data: how to build the dataset.
    """

    name: str
    data_seed: int
    model_seed: int
    heston: HestonParams
    grid: GridConfig
    data: DataConfig

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        # np.random.default_rng refuses negative seeds, so catch them here, early.
        if self.data_seed < 0:
            raise ValueError(f"data_seed must be >= 0, got {self.data_seed}")
        if self.model_seed < 0:
            raise ValueError(f"model_seed must be >= 0, got {self.model_seed}")


def _build(cls, section: dict):
    """Build the dataclass `cls` from one section of the YAML file.

    PyYAML reads a number like 1e-3 (no decimal point) as the text "1e-3", so every field
    declared as float is passed through float() first. fields(cls) lists the dataclass's
    fields; field.type is the type written after the colon, e.g. `xi: float`.
    """
    values = dict(section)  # a copy, so the caller's dictionary is not changed
    for field in fields(cls):
        if field.type is float and field.name in values:
            # (A missing field is left out here; cls(**values) then reports it by name.)
            values[field.name] = float(values[field.name])
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file into an ExperimentConfig.

    Each section is built with **, so a misspelt or missing key at any level raises a
    TypeError that names it, and every dataclass runs its own checks as it is built.
    """
    with open(path) as f:
        # safe_load only builds plain data (dicts, lists, numbers, text).
        # Never use yaml.load: it can run code hidden in a file.
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} should contain 'name: value' settings, got {raw!r}")

    raw["heston"] = _build(HestonParams, raw["heston"])
    raw["grid"] = _build(GridConfig, raw["grid"])
    raw["data"] = _build(DataConfig, raw["data"])
    return ExperimentConfig(**raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write config to a YAML file that load_config can read back.

    asdict turns the dataclasses (including the ones inside) into nested dictionaries,
    because yaml cannot write a dataclass directly. sort_keys=False keeps the fields in the
    same order as the dataclasses, so the file reads like the original.
    """
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)