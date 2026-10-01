"""Support-aware search over smooth policies constructed during model rollout."""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from .contracts import Decision
from .support import ProductionSupport


@dataclass
class SearchConfig:
    num_basis: int = 4
    rounds: int = 2
    samples: int = 32
    elites: int = 8
    initial_std: float = 0.4
    coefficient_bound: float = 1.5
    eps0: float = 1e-8
    lambda_leftover: float = 0.01
    lambda_adjustment: float = 0.001
    seed: int = 0
    support_enabled: bool = True
    rollout_mode: str = "gru"
    basis_mode: str = "smooth"

    def __post_init__(self):
        if self.num_basis < 1 or self.rounds < 1 or not 2 <= self.elites <= self.samples:
            raise ValueError("Require num_basis >= 1, rounds >= 1, and 2 <= elites <= samples")
        if self.initial_std < 0 or self.coefficient_bound < 0 or self.eps0 <= 0:
            raise ValueError("Search scales must be nonnegative and eps0 positive")
        if self.lambda_leftover < 0 or self.lambda_adjustment < 0:
            raise ValueError("Score penalties must be nonnegative")
        if self.rollout_mode not in ("gru", "reencode"):
            raise ValueError("rollout_mode must be 'gru' or 'reencode'")
        if self.basis_mode not in ("smooth", "independent"):
            raise ValueError("basis_mode must be 'smooth' or 'independent'")


def temporal_basis(num_stages, num_basis):
    """Return a trend followed by alternating smooth cosine and sine functions."""
    x = np.linspace(0.0, 1.0, num_stages) if num_stages > 1 else np.array([0.5])
    columns = [2 * x - 1]
    for index in range(1, num_basis):
        frequency = (index + 1) // 2
        angle = np.pi * frequency * x
        columns.append(np.cos(angle) if index % 2 else np.sin(angle))
    return np.stack(columns, axis=1)


def rescaled_baseline(baseline, stage_idx, remaining_budget, eps0=1e-8):
    """Rescale the ORIGINAL unfinished baseline, including its paper epsilon."""
    tail = np.asarray(baseline, dtype=float)[stage_idx:].copy()
    if not len(tail) or not np.all(np.isfinite(tail)) or np.any(tail < 0):
        raise ValueError("The unfinished baseline must be finite and nonnegative")
    weights = tail + eps0
    weights /= weights.sum()
    return float(remaining_budget) * weights


def candidate_schedules(coefficients, baseline, stage_idx, remaining_budget,
                        eps0=1e-8, basis_mode="smooth"):
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
    base = rescaled_baseline(baseline, stage_idx, 1.0, eps0)
    basis = (np.eye(len(base)) if basis_mode == "independent" else
             temporal_basis(len(base), coefficients.shape[1]))
    logits = np.log(base)[None, :] + coefficients @ basis.T
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return float(remaining_budget) * weights


class SupportAwareBudgetSearch:
    def __init__(self, model, config=None, support=None):
        self.model = model
        self.config = config or SearchConfig()
        self.support = support or ProductionSupport()
        self.rng = np.random.default_rng(self.config.seed)

    def _evaluate(self, coefficients, plan_weights, context, initial_hidden, diagnostics):
        """Construct supported actions from predicted remaining budget, then score."""
        count, horizon = plan_weights.shape
        results = np.full(count, -np.inf)
        rollout_plans = np.full((count, horizon), np.nan, dtype=float)
        if count == 0:
            return results, rollout_plans
        hidden = initial_hidden.expand(count, -1).clone()
        remaining = hidden.new_full((count,), float(context.remaining_budget))
        daily = hidden.new_full((count,), float(context.daily_budget))
        values = hidden.new_zeros(count)
        live = np.arange(count)
        histories = ([list(context.history) for _ in range(count)]
                     if self.config.rollout_mode == "reencode" else None)
        for offset in range(horizon):
            if not len(live):
                break
            stage_idx = context.stage_idx + offset
            remaining_np = remaining.detach().cpu().numpy().astype(float)
            tolerance = 1e-6 * max(1.0, float(context.daily_budget))
            tail_mass = plan_weights[live, offset:].sum(axis=1)
            raw_actions = remaining_np * plan_weights[live, offset] / tail_mass
            actions = raw_actions.copy()
            if self.config.support_enabled:
                for index in range(len(actions)):
                    lower, upper = self.support.bounds(
                        stage_idx, max(0.0, remaining_np[index]),
                        context.production_weights)
                    projected = float(np.clip(actions[index], lower, upper))
                    if abs(projected - actions[index]) > tolerance:
                        diagnostics["support_projections_by_stage"][offset] += 1
                    actions[index] = projected
            business_ok = np.isfinite(actions) & (actions >= 0) & (actions <= remaining_np + tolerance)
            diagnostics["business_rejections_by_stage"][offset] += int((~business_ok).sum())
            supported = np.ones(len(live), dtype=bool)
            if self.config.support_enabled:
                diagnostics["support_checks"] += int(business_ok.sum())
                for index in np.flatnonzero(business_ok):
                    supported[index] = self.support.contains(
                        actions[index], stage_idx, max(0.0, remaining_np[index]),
                        context.production_weights, atol=tolerance)
                rejected = business_ok & ~supported
                diagnostics["support_rejections_by_stage"][offset] += int(rejected.sum())
            keep = business_ok & supported
            diagnostics["live_candidates_by_stage"][offset] += int(keep.sum())
            current_live = live
            rollout_plans[current_live[keep], offset] = actions[keep]
            live = live[keep]
            if not len(live):
                break
            mask = torch.as_tensor(keep, device=hidden.device)
            hidden, remaining, daily, values = hidden[mask], remaining[mask], daily[mask], values[mask]
            action_tensor = torch.as_tensor(actions[keep], dtype=hidden.dtype, device=hidden.device)
            if histories is not None and offset:
                hidden = torch.cat([self.model.encode_history(histories[index], context.daily_budget)
                                    for index in live], dim=0)
                diagnostics["history_encoding_calls"] += len(live)
            diagnostics["model_calls"] += 1
            diagnostics["candidate_stage_model_calls"] += len(live)
            outcomes, next_hidden, next_remaining = self.model.predict_step(
                hidden, action_tensor, remaining, daily, stage_idx)
            spend, value = outcomes["spend"], outcomes["value"]
            valid = (torch.isfinite(spend) & torch.isfinite(value) & torch.isfinite(next_remaining)
                     & torch.all(torch.isfinite(next_hidden), dim=1)
                     & (spend >= 0) & (spend <= action_tensor + tolerance)
                     & (value >= 0) & (next_remaining >= -tolerance)
                     & (torch.abs(next_remaining - (remaining - spend)) <= tolerance))
            for market_field in ("auction_volume", "market_price"):
                market = outcomes[market_field]
                valid &= torch.isfinite(market) & (market >= 0)
            valid_np = valid.detach().cpu().numpy()
            diagnostics["invalid_model_rejections"] += int((~valid_np).sum())
            if histories is not None:
                outcome_arrays = {key: val.detach().cpu().numpy() for key, val in outcomes.items()}
                next_remaining_np = next_remaining.detach().cpu().numpy()
                for local, candidate_index in enumerate(live):
                    if valid_np[local]:
                        histories[candidate_index].append({
                            "stage_idx": stage_idx,
                            "assigned_budget": float(rollout_plans[candidate_index, offset]),
                            "daily_budget": float(context.daily_budget),
                            "remaining_budget": float(next_remaining_np[local]),
                            **{key: float(val[local]) for key, val in outcome_arrays.items()},
                        })
            live = live[valid_np]
            hidden = next_hidden[valid]
            remaining = next_remaining[valid].clamp_min(0)
            daily = daily[valid]
            values = (values + value)[valid]
        if len(live):
            score = values - self.config.lambda_leftover * remaining
            results[live] = (score.detach().cpu().numpy()
                             - self.config.lambda_adjustment * np.square(coefficients[live]).sum(axis=1))
        return results, rollout_plans

    def decide(self, context):
        started = perf_counter()
        cfg = self.config
        remaining = float(context.remaining_budget)
        if not np.isfinite(remaining) or remaining < 0 or context.daily_budget <= 0:
            raise ValueError("Require finite nonnegative remaining and positive daily budget")
        horizon = context.num_stages - context.stage_idx
        base = rescaled_baseline(context.baseline, context.stage_idx, remaining, cfg.eps0)
        base_weights = rescaled_baseline(context.baseline, context.stage_idx, 1.0, cfg.eps0)
        dimensions = horizon if cfg.basis_mode == "independent" else cfg.num_basis
        diagnostics = {
            "policy": "sabs", "fallback": False, "fallback_used": False,
            "support_enabled": cfg.support_enabled, "support_checks": 0,
            "support_rejections_by_stage": [0] * horizon,
            "support_projections_by_stage": [0] * horizon,
            "business_rejections_by_stage": [0] * horizon,
            "live_candidates_by_stage": [0] * horizon,
            "model_calls": 0, "candidate_stage_model_calls": 0,
            "history_encoding_calls": 0, "invalid_model_rejections": 0,
            "coefficient_rejections": 0, "plan_rejections": 0,
            "sampled_candidates": 0, "feasible_candidates": 0,
            "baseline_checked": True, "baseline_supported": False,
            "baseline_valid": False, "baseline_support_checked": cfg.support_enabled,
            "round_trace": [], "config": asdict(cfg),
        }
        mean, std = np.zeros(dimensions), np.full(dimensions, cfg.initial_std)
        pool = []
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.inference_mode():
            initial_hidden = self.model.encode_history(context.history, context.daily_budget)
            diagnostics["history_encoding_calls"] = 1
            for round_idx in range(cfg.rounds):
                sampled = self.rng.normal(mean, std, size=(cfg.samples, dimensions))
                diagnostics["sampled_candidates"] += cfg.samples
                bounded = np.all(np.isfinite(sampled), axis=1) & (np.abs(sampled).max(axis=1) <= cfg.coefficient_bound)
                diagnostics["coefficient_rejections"] += int((~bounded).sum())
                coefficients = sampled[bounded]
                plan_weights = candidate_schedules(
                    coefficients, context.baseline, context.stage_idx, 1.0,
                    cfg.eps0, cfg.basis_mode) if len(coefficients) else np.empty((0, horizon))
                plan_ok = (np.all(np.isfinite(plan_weights), axis=1)
                           & np.all(plan_weights >= 0, axis=1)
                           & np.isclose(plan_weights.sum(axis=1), 1.0, rtol=1e-8, atol=1e-8))
                diagnostics["plan_rejections"] += int((~plan_ok).sum())
                coefficients, plan_weights = coefficients[plan_ok], plan_weights[plan_ok]
                scores, rollout_plans = self._evaluate(
                    coefficients, plan_weights, context, initial_hidden, diagnostics)
                feasible = np.flatnonzero(np.isfinite(scores))
                for index in feasible:
                    pool.append((float(scores[index]), coefficients[index].copy(),
                                 rollout_plans[index].copy(), False))
                diagnostics["round_trace"].append({
                    "round": round_idx, "feasible": len(feasible),
                    "proposal_mean": mean.tolist(), "proposal_std": std.tolist(),
                    "best_score": float(scores[feasible].max()) if len(feasible) else None,
                })
                if len(feasible) < cfg.elites:
                    break
                top = feasible[np.argsort(scores[feasible], kind="stable")[-cfg.elites:]]
                mean = coefficients[top].mean(axis=0)
                std = coefficients[top].std(axis=0, ddof=0)
            # Algorithm 1 explicitly checks beta=0 after all search rounds.
            zero = np.zeros((1, dimensions))
            baseline_scores, baseline_plans = self._evaluate(
                zero, base_weights[None, :], context, initial_hidden, diagnostics)
            baseline_score = baseline_scores[0]
            if np.isfinite(baseline_score):
                pool.append((float(baseline_score), zero[0], baseline_plans[0].copy(), True))
                diagnostics["baseline_valid"] = True
                diagnostics["baseline_supported"] = cfg.support_enabled
            if pool:
                score, coefficients, plan, is_baseline = max(pool, key=lambda entry: entry[0])
                diagnostics.update(best_score=score, selected_coefficients=coefficients.tolist(),
                                   selected_baseline=is_baseline)
            else:
                # Operational fallback is not scored and is not projected into support.
                plan = base
                diagnostics.update(fallback=True, fallback_used=True, best_score=None,
                                   selected_coefficients=None, selected_baseline=True)
        action = float(np.clip(plan[0], 0, remaining))
        diagnostics["feasible_candidates"] = len(pool)
        diagnostics["support_rejections"] = sum(diagnostics["support_rejections_by_stage"])
        diagnostics["executed_supported"] = self.support.contains(
            action, context.stage_idx, remaining, context.production_weights,
            atol=1e-6 * max(1.0, float(context.daily_budget)))
        diagnostics["latency_ms"] = (perf_counter() - started) * 1000
        return Decision(action, diagnostics, plan.copy())

    search = decide
    __call__ = decide
