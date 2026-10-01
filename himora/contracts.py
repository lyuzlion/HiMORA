"""Minimal online interfaces; stages are zero-based and budgets use raw units."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DecisionContext:
    stage_idx: int
    remaining_budget: float
    daily_budget: float
    baseline: np.ndarray
    production_weights: np.ndarray
    history: List[Dict[str, Any]]

    def __post_init__(self):
        self.baseline = np.asarray(self.baseline, dtype=float)
        self.production_weights = np.asarray(self.production_weights, dtype=float)
        if (self.baseline.ndim != 1 or not len(self.baseline)
                or self.production_weights.shape != self.baseline.shape):
            raise ValueError("baseline and production_weights must be same-length nonempty vectors")
        for weights in (self.baseline, self.production_weights):
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("Budget weights must be finite and nonnegative")
        if not isinstance(self.stage_idx, (int, np.integer)) or not 0 <= self.stage_idx < self.num_stages:
            raise ValueError("stage_idx must identify an unfinished stage")
        if (not np.isfinite(self.daily_budget) or self.daily_budget <= 0
                or not np.isfinite(self.remaining_budget)
                or not 0 <= self.remaining_budget <= self.daily_budget):
            raise ValueError("Require 0 <= remaining_budget <= daily_budget and daily_budget > 0")
        if [record["stage_idx"] for record in self.history] != list(range(self.stage_idx)):
            raise ValueError("history must contain every completed stage in order")
        expected = self.history[-1]["remaining_budget"] if self.history else self.daily_budget
        if not np.isclose(expected, self.remaining_budget, rtol=1e-6, atol=1e-6):
            raise ValueError("remaining_budget must match the latest real feedback")

    @property
    def num_stages(self):
        return len(self.baseline)


@dataclass
class Decision:
    budget: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[np.ndarray] = None


STAGE_FIELDS = (
    "episode", "player_index", "stage_idx", "daily_budget", "assigned_budget",
    "spend", "value", "remaining_budget", "auction_volume", "market_price",
)
