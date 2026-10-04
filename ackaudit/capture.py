"""Reduced from ackaudit/capture.py: only `_resolve`, which the experiment scripts use to build models.

The original also contains small synthetic models and the knapsack-auditing tooling; neither is used by the paper.
"""
from __future__ import annotations

from .hf_models import ModelSpec, hf_models


def _resolve(name: str, scale: int = 1) -> ModelSpec:
    zoo = hf_models(scale=scale)
    if name in zoo:
        return zoo[name]
    raise KeyError(f"unknown model {name!r} (this subset provides: {sorted(zoo)})")
