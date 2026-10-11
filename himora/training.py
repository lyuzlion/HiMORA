"""One-step pretraining followed by free-running, end-to-end fine-tuning."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import file_sha256, fit_normalization, prefix_indices, read_trajectories, split_manifest
from .model import HiMORA, OUTCOME_FIELDS


DEFAULT_TRAINING = dict(hidden_dim=64, num_heads=4, num_layers=2, dropout=0.0,
                        history_mode="full", epsilon=1e-4, pretrain_steps=200,
                        finetune_steps=500, batch_size=32, max_horizon=8, lr=0.001,
                        lambda_ms=1.0, gamma=0.95, alpha_C=1.0, alpha_V=1.0,
                        alpha_N=1.0, huber_delta=1.0, grad_clip=5.0, seed=7,
                        eval_interval=50, device="cpu")


def _prepared(model, trajectories):
    prepared = []
    for trajectory in trajectories:
        records = trajectory.records
        prepared.append({"tokens": model.history_tokens(records, trajectory.daily_budget).detach(),
                         "outcomes": model._tensor([[r[name] for name in OUTCOME_FIELDS] for r in records]),
                         "budgets": model._tensor([r["assigned_budget"] for r in records]),
                         "remaining": model._tensor([trajectory.daily_budget] + [r["remaining_budget"] for r in records]),
                         "daily": trajectory.daily_budget})
    return prepared


def _batch(model, prepared, indices, max_horizon, rng=None):
    lengths = [prefix for _, prefix in indices]
    horizons = [min(max_horizon, len(prepared[index]["budgets"]) - prefix) for index, prefix in indices]
    if rng is not None:
        horizons = [int(rng.integers(1, length + 1)) for length in horizons]
    batch, time, horizon = len(indices), max(1, max(lengths)), max(horizons)
    tokens = model.cold_start.new_zeros((batch, time, 7))
    budgets = model.cold_start.new_zeros((batch, horizon))
    targets = model.cold_start.new_zeros((batch, horizon, 4))
    mask = torch.zeros((batch, horizon), dtype=torch.bool, device=model.device)
    remaining, daily = [], []
    for row, ((index, prefix), length) in enumerate(zip(indices, horizons)):
        trajectory = prepared[index]
        tokens[row, :prefix] = trajectory["tokens"][:prefix]
        budgets[row, :length] = trajectory["budgets"][prefix:prefix + length]
        targets[row, :length] = trajectory["outcomes"][prefix:prefix + length]
        mask[row, :length] = True
        remaining.append(trajectory["remaining"][prefix])
        daily.append(trajectory["daily"])
    return dict(tokens=tokens, lengths=lengths, budgets=budgets, targets=targets, mask=mask,
                remaining=torch.stack(remaining), daily=model._tensor(daily), stages=model._tensor(lengths))


def stage_loss(model, predicted, target, daily_budget, config):
    """Weighted Huber: spend/B, z-scored V, and mean z-scored N loss."""
    scaled_predicted = torch.cat([predicted[..., :1] / daily_budget[..., None],
                                  (predicted[..., 1:] - model.norm_mean) / model.norm_scale], dim=-1)
    scaled_target = torch.cat([target[..., :1] / daily_budget[..., None],
                               (target[..., 1:] - model.norm_mean) / model.norm_scale], dim=-1)
    terms = F.huber_loss(scaled_predicted, scaled_target, reduction="none", delta=config["huber_delta"])
    return (config["alpha_C"] * terms[..., 0] + config["alpha_V"] * terms[..., 1]
            + config["alpha_N"] * terms[..., 2:].mean(dim=-1))


def batch_objective(model, batch, config, multistage=True):
    """The paper's one-step term PLUS discounted multi-step sum.

    Subsequent observed budgets are supplied unchanged. No target outcome
    or true remaining budget after the prefix enters the recursive state.
    """
    hidden = model.encode_tokens(batch["tokens"], batch["lengths"])
    remaining = batch["remaining"]
    losses, predictions, remaining_path = [], [], []
    for offset in range(batch["budgets"].shape[1]):
        outcomes, hidden, remaining = model.predict_step(hidden, batch["budgets"][:, offset], remaining,
                                                         batch["daily"], batch["stages"] + offset)
        predicted = torch.stack([outcomes[name] for name in OUTCOME_FIELDS], dim=-1)
        predictions.append(predicted)
        remaining_path.append(remaining)
        losses.append(stage_loss(model, predicted, batch["targets"][:, offset], batch["daily"], config))
    losses = torch.stack(losses, dim=1)
    discount = config["gamma"] ** torch.arange(losses.shape[1], device=model.device)
    one_step = losses[:, 0].mean()
    multi_step = (losses * discount * batch["mask"]).sum(dim=1).mean()
    total = one_step + (config["lambda_ms"] * multi_step if multistage else 0.)
    return total, {"one_step": one_step, "multi_step": multi_step,
                   "predictions": torch.stack(predictions, dim=1),
                   "remaining": torch.stack(remaining_path, dim=1)}


def _evaluate(model, trajectories, config, max_horizon, batch_size):
    model.eval()
    prepared = _prepared(model, trajectories)
    indices = prefix_indices(trajectories)
    totals = dict(loss=0., one_step_loss=0., multistage_loss=0.)
    raw_errors = torch.zeros(4, device=model.device)
    raw_targets = torch.zeros(4, device=model.device)
    normalized_errors = torch.zeros_like(raw_errors)
    per_horizon = {k: {"count": 0, "raw_abs_error": [0.] * 4,
                       "raw_abs_target": [0.] * 4}
                   for k in range(1, max_horizon + 1)}
    observation_count, negative_remaining = 0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selection = indices[start:start + batch_size]
            batch = _batch(model, prepared, selection, max_horizon)
            loss, details = batch_objective(model, batch, config, multistage=True)
            totals["loss"] += float(loss) * len(selection)
            totals["one_step_loss"] += float(details["one_step"]) * len(selection)
            totals["multistage_loss"] += float(details["multi_step"]) * len(selection)
            errors = (details["predictions"] - batch["targets"]).abs()
            targets = batch["targets"].abs()
            scale = torch.cat([batch["daily"].unsqueeze(-1), model.norm_scale.expand(len(selection), -1)], dim=-1)
            raw_errors += (errors * batch["mask"].unsqueeze(-1)).sum(dim=(0, 1))
            raw_targets += (targets * batch["mask"].unsqueeze(-1)).sum(dim=(0, 1))
            normalized_errors += (errors / scale.unsqueeze(1) * batch["mask"].unsqueeze(-1)).sum(dim=(0, 1))
            observation_count += int(batch["mask"].sum())
            negative_remaining += int(((details["remaining"] < -1e-5) & batch["mask"]).sum())
            for offset in range(errors.shape[1]):
                active = batch["mask"][:, offset]
                horizon = per_horizon[offset + 1]
                horizon["count"] += int(active.sum())
                horizon["raw_abs_error"] = (np.asarray(horizon["raw_abs_error"])
                                             + errors[active, offset].sum(dim=0).cpu().numpy()).tolist()
                horizon["raw_abs_target"] = (np.asarray(horizon["raw_abs_target"])
                                              + targets[active, offset].sum(dim=0).cpu().numpy()).tolist()
    result = {name: value / len(indices) for name, value in totals.items()}
    result.update(num_prefixes=len(indices), num_predicted_stages=observation_count,
                  raw_mae=dict(zip(OUTCOME_FIELDS, (raw_errors / observation_count).cpu().tolist())),
                  raw_nmae=dict(zip(OUTCOME_FIELDS,
                      (raw_errors / raw_targets.clamp_min(1e-8)).cpu().tolist())),
                  normalized_mae=dict(zip(OUTCOME_FIELDS, (normalized_errors / observation_count).cpu().tolist())),
                  negative_predicted_remaining_count=negative_remaining,
                  negative_predicted_remaining_rate=negative_remaining / observation_count)
    result["by_horizon"] = {str(k): {
                                   "count": values["count"],
                                   "raw_mae": dict(zip(OUTCOME_FIELDS,
                                       (np.asarray(values["raw_abs_error"]) / values["count"]).tolist())),
                                   "raw_nmae": dict(zip(OUTCOME_FIELDS,
                                       (np.asarray(values["raw_abs_error"])
                                        / np.maximum(np.asarray(values["raw_abs_target"]), 1e-8)).tolist()))}
                            for k, values in per_horizon.items() if values["count"]}
    return result


def evaluate_model(model, log_path, max_horizon=8, batch_size=32, config=None):
    """Deterministic held-out forecast metrics, including errors by horizon."""
    settings = {**DEFAULT_TRAINING, **(config or {})}
    if max_horizon < 1 or batch_size < 1:
        raise ValueError("max_horizon and batch_size must be positive")
    trajectories = read_trajectories(log_path, model.num_stages)
    return _evaluate(model, trajectories, settings, int(max_horizon), int(batch_size))


def train_model(train_log, val_log, output_dir, config: dict):
    """Fit solely to train logs; select the best fine-tuned model on validation.

    Writes model.pt, metadata.json, and split_manifest.json. Counts are
    optimizer steps, not epochs. Caller controls CPU thread allocation.
    """
    settings = {**DEFAULT_TRAINING, **config}
    for name in ("batch_size", "max_horizon", "eval_interval"):
        if settings[name] < 1:
            raise ValueError(f"{name} must be positive")
    if settings["pretrain_steps"] < 0 or settings["finetune_steps"] < 0 or settings["pretrain_steps"] + settings["finetune_steps"] == 0:
        raise ValueError("At least one nonnegative training phase must have positive steps")
    if not 0 < settings["gamma"] <= 1 or settings["lambda_ms"] < 0:
        raise ValueError("gamma must be in (0,1] and lambda_ms must be nonnegative")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    train = read_trajectories(train_log, settings.get("num_stages"))
    num_stages = len(train[0].records)
    validation = read_trajectories(val_log, num_stages)
    manifest = split_manifest(train_log, val_log, train, validation)
    normalization = fit_normalization(train)
    model_keys = ("hidden_dim", "num_heads", "num_layers", "dropout", "history_mode", "epsilon")
    model = HiMORA(num_stages=num_stages, normalization=normalization,
                   **{key: settings[key] for key in model_keys}).to(settings["device"])
    prepared, indices = _prepared(model, train), prefix_indices(train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    history, best_state, best_score, best_step = [], None, float("inf"), None
    global_step = 0
    for phase, steps in (("pretrain", settings["pretrain_steps"]), ("finetune", settings["finetune_steps"])):
        for phase_step in range(1, steps + 1):
            global_step += 1
            model.train()
            selection = [indices[int(index)] for index in rng.integers(len(indices), size=settings["batch_size"])]
            horizon = 1 if phase == "pretrain" else settings["max_horizon"]
            batch = _batch(model, prepared, selection, horizon, rng if phase == "finetune" else None)
            optimizer.zero_grad(set_to_none=True)
            loss, details = batch_objective(model, batch, settings, multistage=phase == "finetune")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {global_step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if phase_step % settings["eval_interval"] == 0 or phase_step == steps:
                metrics = _evaluate(model, validation, settings, settings["max_horizon"], settings["batch_size"])
                entry = dict(step=global_step, phase=phase, phase_step=phase_step, train_loss=float(loss.detach()),
                             gradient_norm=float(grad_norm), validation=metrics)
                history.append(entry)
                print(json.dumps({"phase": phase, "step": phase_step, "train_loss": entry["train_loss"],
                                  "validation_loss": metrics["loss"]}), flush=True)
                # When fine-tuning is requested, selection considers its checkpoints only.
                eligible = phase == "finetune" or settings["finetune_steps"] == 0
                if eligible and metrics["loss"] < best_score:
                    best_score, best_step = metrics["loss"], global_step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = _evaluate(model, validation, settings, settings["max_horizon"], settings["batch_size"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), split_manifest=manifest,
                      training_config=settings, best_step=best_step,
                      sources={name: file_sha256(Path(__file__).with_name(name))
                               for name in ("model.py", "data.py", "training.py")})
    checkpoint = dict(format="himora-v1", model_config=model.config, normalization=normalization,
                      state_dict=best_state, provenance=provenance)
    torch.save(checkpoint, checkpoint_path)
    metadata = dict(checkpoint_path=checkpoint_path.name, model_config=model.config,
                    normalization=normalization, training_config=settings, best_step=best_step,
                    validation=metrics, training_history=history, provenance=provenance,
                    training_budget_semantics="Observed suffix budgets are never clipped to predicted remaining; "
                    "negative predicted remaining is retained. Action denominator uses max(remaining,0)+epsilon. "
                    "Validation reports the frequency of such model-inconsistent states.")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metadata
