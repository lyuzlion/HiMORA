"""Generate toy data at runtime; check training, prediction, and online decisions.

This small toy process is NOT the paper's experimental environment.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import DecisionContext
from .model import load_model
from .search import SearchConfig, SupportAwareBudgetSearch
from .support import ProductionSupport
from .training import train_model


def observe(stage, budget, remaining, daily, episode, rng, num_stages):
    """Toy feedback depends on stage and a persistent day-level market factor."""
    phase = (stage + 1) / num_stages
    market = 1.0 + 0.1 * np.sin(episode)
    volume = float(1.5 + np.sin(np.pi * phase) + rng.uniform(0, 0.05))
    price = float(0.5 + 0.2 * phase)
    spend = float(min(budget * rng.uniform(0.6, 0.9), 20 * volume))
    value = float(spend * (0.02 + 0.04 * phase) * market)
    return dict(episode=int(episode), player_index=0, stage_idx=int(stage),
                daily_budget=float(daily), assigned_budget=float(budget), spend=spend,
                value=value, remaining_budget=float(remaining - spend),
                auction_volume=volume, market_price=price)


def write_toy_log(path, episodes, rng, support, weights):
    """Write complete campaign-day trajectories with randomized stage budgets."""
    with Path(path).open("w", encoding="utf-8") as stream:
        for episode in episodes:
            daily, remaining = 100.0, 100.0
            for stage in range(len(weights)):
                delta = rng.uniform(-support.rho, support.rho)
                budget = support.perturb(stage, remaining, weights, delta)["assigned_budget"]
                record = observe(stage, budget, remaining, daily, episode, rng, len(weights))
                stream.write(json.dumps(record) + "\n")
                remaining = record["remaining_budget"]


def run_demo(output_dir, steps=20, seed=7):
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ("train.jsonl", "validation.jsonl", "online_context.json", "checkpoint")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Use a fresh output directory; existing demo artifacts are not overwritten")
    num_stages = 6
    weights = np.linspace(1, 2, num_stages)
    baseline = 100 * weights / weights.sum()
    support = ProductionSupport(rho=0.25)
    rng = np.random.default_rng(seed)
    write_toy_log(targets[0], range(8), rng, support, weights)
    write_toy_log(targets[1], range(100, 102), rng, support, weights)
    train_model(targets[0], targets[1], targets[3],
                dict(hidden_dim=16, num_heads=2, num_layers=1, pretrain_steps=steps,
                     finetune_steps=steps, batch_size=8, max_horizon=num_stages,
                     eval_interval=steps, seed=seed, lr=0.001, device="cpu"))
    model = load_model(targets[3] / "model.pt")
    search = SupportAwareBudgetSearch(
        model, SearchConfig(rounds=2, samples=8, elites=2, initial_std=0.1, seed=seed), support)
    history, remaining, decisions = [], 100.0, []
    for stage in range(num_stages):
        context = DecisionContext(stage, remaining, 100.0, baseline, weights, history)
        decision = search.decide(context)
        if not (0 <= decision.budget <= remaining and not decision.diagnostics["fallback"]
                and support.contains(decision.budget, stage, remaining, weights)):
            raise AssertionError("Toy decision failed feasibility or support checks")
        if stage == 2:
            payload = dict(stage_idx=stage, remaining_budget=remaining, daily_budget=100.0,
                           baseline=baseline.tolist(), production_weights=weights.tolist(), history=history.copy())
            targets[2].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            # Fixed-budget rollout is separate from adaptive supported SABS search.
            with torch.inference_mode():
                forecast = model.rollout(history, [remaining / 4] * 4, 100.0)
            if any(not torch.isfinite(values).all() for values in forecast.values()):
                raise AssertionError("Forecast must be finite")
        record = observe(stage, decision.budget, remaining, 100.0, 999, rng, num_stages)
        history.append(record)
        remaining = record["remaining_budget"]
        decisions.append({"stage_idx": stage, "next_budget": decision.budget,
                          "observed_spend": record["spend"], "observed_value": record["value"],
                          "remaining_budget": remaining})
    return {"status": "passed", "num_stages": num_stages, "decisions": decisions,
            "checkpoint": str(targets[3] / "model.pt"), "context": str(targets[2]),
            "note": "Toy execution check only; not a paper benchmark or production dataset."}
