"""The deployed logging law and its induced, context-dependent action support.

The production proposal is computed from production weights, never from a
candidate SABS plan. All stage indices are zero-based and budgets use raw units.
"""
from dataclasses import dataclass

import numpy as np


def production_proposal(stage_idx, remaining_budget, production_weights):
    """Proportionally allocate actual/predicted remaining money to this stage."""
    weights = np.asarray(production_weights, dtype=float)
    remaining = float(remaining_budget)
    if weights.ndim != 1 or not 0 <= stage_idx < len(weights):
        raise ValueError("production_weights must be a vector containing stage_idx")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("production_weights must be finite and nonnegative")
    if not np.isfinite(remaining) or remaining < 0:
        raise ValueError("remaining_budget must be finite and nonnegative")
    tail = weights[stage_idx:]
    denominator = float(tail.sum())
    share = float(tail[0] / denominator) if denominator > 0 else 1.0 / len(tail)
    return remaining * share


def support_bounds(stage_idx, remaining_budget, production_weights, rho=1.0):
    """Closed support of clip(b0 * (1 + delta), 0, remaining), delta uniform."""
    return ProductionSupport(rho).bounds(stage_idx, remaining_budget, production_weights)


@dataclass(frozen=True)
class ProductionSupport:
    """Uniform multiplicative perturbations with clipping and explicit atoms.

    ``rho`` is the perturbation radius. A radius of zero gives a deterministic
    policy, represented by probability mass rather than an artificial density.
    """

    rho: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.rho) or not 0 <= self.rho <= 1:
            raise ValueError("rho must lie in [0, 1]")

    def proposal(self, stage_idx, remaining_budget, production_weights):
        return production_proposal(stage_idx, remaining_budget, production_weights)

    def bounds(self, stage_idx, remaining_budget, production_weights):
        b0 = self.proposal(stage_idx, remaining_budget, production_weights)
        return (float(np.clip(b0 * (1 - self.rho), 0, remaining_budget)),
                float(np.clip(b0 * (1 + self.rho), 0, remaining_budget)))

    def contains(self, budget, stage_idx, remaining_budget, production_weights, atol=1e-7):
        lower, upper = self.bounds(stage_idx, remaining_budget, production_weights)
        return bool(np.isfinite(budget) and lower - atol <= budget <= upper + atol)

    def perturb(self, stage_idx, remaining_budget, production_weights, delta):
        """Apply one already-drawn perturbation and describe its mixed law.

        ``action_density`` is the continuous density in inverse currency units.
        ``action_mass`` is the discrete probability at the executed action. The
        final stage has an upper atom of 1/2 when rho > 0 and money remains.
        These quantities must not be confused with the perturbation density.
        """
        delta = float(delta)
        if not np.isfinite(delta) or not -self.rho <= delta <= self.rho:
            raise ValueError("delta is outside the deployed perturbation range")
        remaining = float(remaining_budget)
        b0 = self.proposal(stage_idx, remaining, production_weights)
        lower, upper = self.bounds(stage_idx, remaining, production_weights)
        raw = b0 * (1 + delta)
        action = float(np.clip(raw, 0, remaining))
        perturbation_density = 0.0 if self.rho == 0 else 1.0 / (2 * self.rho)
        perturbation_mass = 1.0 if self.rho == 0 else 0.0
        if self.rho == 0 or b0 == 0 or remaining == 0:
            action_density, action_mass = 0.0, 1.0
            lower_atom_mass = float(action == 0)
            upper_atom_mass = float(action == remaining)
        else:
            lower_atom_mass = float(np.clip((self.rho - 1) / (2 * self.rho), 0, 1))
            upper_threshold = remaining / b0 - 1
            upper_atom_mass = float(np.clip((self.rho - upper_threshold) / (2 * self.rho), 0, 1))
            action_mass = (upper_atom_mass if action == remaining else
                           lower_atom_mass if action == 0 else 0.0)
            action_density = 0.0 if action_mass > 0 else perturbation_density / b0
        return {
            "b0": b0,
            "delta": delta,
            "assigned_budget": action,
            "unclipped_budget": raw,
            "perturbation_density": perturbation_density,
            "perturbation_mass": perturbation_mass,
            "action_density": action_density,
            "action_mass": action_mass,
            "lower_atom_mass": lower_atom_mass,
            "upper_atom_mass": upper_atom_mass,
            "support_lower": lower,
            "support_upper": upper,
            "rho": self.rho,
            "executed_supported": True,
        }
