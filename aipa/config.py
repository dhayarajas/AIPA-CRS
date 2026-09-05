"""Configuration and reproducibility helpers."""
from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    values: dict[str, Any] = field(default_factory=dict)
    run_mode: str = "quick"

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:  # pragma: no cover - programming error
            raise AttributeError(name) from exc

    def path(self, key: str) -> Path:
        p = Path(self.values[key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return {"run_mode": self.run_mode, **self.values}


def load_config(run_mode: str | None = None, path: Path | None = None) -> Config:
    path = path or PROJECT_ROOT / "configs" / "default.yaml"
    raw = yaml.safe_load(path.read_text())
    run_mode = run_mode or os.environ.get("AIPA_RUN_MODE", "quick")
    if run_mode not in raw:
        raise ValueError(f"Unknown RUN_MODE {run_mode!r}; expected one of {list(raw)}")
    values = dict(raw["common"])
    values.update(raw[run_mode])
    if values.get("device", "auto") == "auto":
        values["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return Config(values=values, run_mode=run_mode)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_report() -> dict[str, str]:
    import importlib.metadata as md

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cpu_count": str(os.cpu_count()),
    }
    for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "pyyaml"]:
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = "missing"
    return info
