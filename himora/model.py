"""Paper-faithful causal history encoder, outcome heads, and GRU rollouts."""
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


OUTCOME_FIELDS = ("spend", "value", "auction_volume", "market_price")


class HiMORA(nn.Module):
    """All public outcomes and budgets use raw physical units.

    V and N tokens use training-only z-scores. The MLP's N logits have a
    fixed, training-derived output scale for conditioning; the physical
    market head remains exactly softplus(o_N). Logged training budgets are
    never clipped during free-running prediction. Prediction error can
    make a logged suffix overspend *predicted* remaining budget. We retain
    that arithmetic and use max(B_t, 0) + epsilon in the action denominator
    outside the physical domain (equal to the paper in the physical domain).
    """
    def __init__(self, num_stages=48, hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.0, normalization=None, history_mode="full", epsilon=1e-4):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_stages <= 0 or epsilon <= 0:
            raise ValueError("num_stages and epsilon must be positive")
        if history_mode not in ("full", "last", "none"):
            raise ValueError("history_mode must be full, last, or none")
        self.config = dict(num_stages=int(num_stages), hidden_dim=int(hidden_dim), num_heads=int(num_heads),
                           num_layers=int(num_layers), dropout=float(dropout), history_mode=history_mode,
                           epsilon=float(epsilon))
        self.num_stages, self.hidden_dim = int(num_stages), int(hidden_dim)
        self.history_mode, self.epsilon = history_mode, float(epsilon)
        self.normalization = normalization or {"mean": [0., 0., 0.], "scale": [1., 1., 1.],
                                                "fit_split": "train", "count": 0, "mean_spend": 1.}
        means = torch.tensor(self.normalization["mean"], dtype=torch.float32)
        scales = torch.tensor(self.normalization["scale"], dtype=torch.float32)
        if means.shape != (3,) or scales.shape != (3,) or not torch.isfinite(means).all() or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Normalization must contain three finite means and positive scales")
        self.register_buffer("norm_mean", means)
        self.register_buffer("norm_scale", scales)
        self.register_buffer("output_logit_scale", torch.cat([torch.ones(2), scales[1:].clamp_min(1.)]))
        self.token_projection = nn.Linear(7, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout,
                                            batch_first=True, activation="gelu", norm_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(hidden_dim),
                                                     enable_nested_tensor=False)
        self.cold_start = nn.Parameter(torch.zeros(hidden_dim))
        self.outcome_mlp = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))
        self.state_update = nn.GRUCell(7, hidden_dim)
        self._initialize_heads()

    @property
    def device(self):
        return self.cold_start.device

    def _initialize_heads(self):
        last = self.outcome_mlp[-1]
        nn.init.normal_(last.weight, std=0.01)
        with torch.no_grad():
            ratio = max(float(self.norm_mean[0]) / max(self.normalization.get("mean_spend", 1.), 1e-6), 1e-4)
            targets = torch.tensor([ratio, max(float(self.norm_mean[1]), 1e-3), max(float(self.norm_mean[2]), 1e-3)])
            # Stable inverse softplus. The spend logit starts at zero.
            logits = targets + torch.log(-torch.expm1(-targets))
            last.bias.zero_()
            last.bias[1:] = logits / self.output_logit_scale[1:]

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.cold_start.dtype, device=self.device)

    def stage_token(self, budget, spend, remaining, value, volume, price, daily_budget, stage_idx):
        daily = self._tensor(daily_budget).clamp_min(self.epsilon)
        budget, spend, remaining = self._tensor(budget), self._tensor(spend), self._tensor(remaining)
        physical = torch.stack([self._tensor(value), self._tensor(volume), self._tensor(price)], dim=-1)
        normalized = (physical - self.norm_mean) / self.norm_scale
        stage = (self._tensor(stage_idx) + 1.) / self.num_stages
        stage = torch.broadcast_to(stage, budget.shape)
        return torch.cat([torch.stack([budget / daily, spend / daily, remaining / daily], dim=-1),
                          normalized, stage.unsqueeze(-1)], dim=-1)

    def history_tokens(self, history: List[dict], daily_budget):
        if not history:
            return self.cold_start.new_empty((0, 7))
        keys = ("assigned_budget", "spend", "remaining_budget", "value", "auction_volume", "market_price")
        data = [self._tensor([record[key] for record in history]) for key in keys]
        return self.stage_token(*data, daily_budget, [record["stage_idx"] for record in history])

    def encode_token_sequence(self, tokens, padding_mask=None):
        """Expose causal sequence states for diagnostics; input is [batch,time,7]."""
        length = tokens.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        return self.history_encoder(self.token_projection(tokens), mask=mask, src_key_padding_mask=padding_mask)

    def encode_tokens(self, tokens, lengths):
        """Batched variable-length observed prefixes; empty ones use cold_start."""
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        result = self.cold_start.unsqueeze(0).expand(len(lengths), -1).clone()
        if self.history_mode == "none":
            return result
        active = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if not len(active):
            return result
        selected_lengths = lengths[active]
        selected = tokens[active]
        if self.history_mode == "last":
            selected = selected[torch.arange(len(active), device=self.device), selected_lengths - 1].unsqueeze(1)
            selected_lengths = torch.ones_like(selected_lengths)
        padding = torch.arange(selected.shape[1], device=self.device).unsqueeze(0) >= selected_lengths.unsqueeze(1)
        states = self.encode_token_sequence(selected, padding)
        final = states[torch.arange(len(active), device=self.device), selected_lengths - 1]
        return result.index_copy(0, active, final)

    def encode_history(self, history: List[dict], daily_budget):
        tokens = self.history_tokens(history, daily_budget)
        return self.encode_tokens(tokens.unsqueeze(0), [len(history)])

    def predict_step(self, hidden, budget, remaining, daily_budget, stage_idx):
        """Return (raw outcome tensors, GRU next state, arithmetic remaining).

        Scalars are broadcast to hidden's batch; stage_idx can also be a
        per-row tensor when training mixed decision points in one batch.
        """
        batch = hidden.shape[0]
        budget = self._tensor(budget).expand(batch)
        remaining = self._tensor(remaining).expand(batch)
        daily_budget = self._tensor(daily_budget).expand(batch)
        if (budget < 0).any():
            raise ValueError("Candidate budget must be nonnegative")
        relative = budget / (remaining.clamp_min(0.) + self.epsilon)
        logits = self.outcome_mlp(torch.cat([hidden, relative.unsqueeze(-1)], dim=-1)) * self.output_logit_scale
        spend = budget * torch.sigmoid(logits[:, 0])
        value = spend * F.softplus(logits[:, 1])
        stats = F.softplus(logits[:, 2:])
        next_remaining = remaining - spend
        outcomes = dict(spend=spend, value=value, auction_volume=stats[:, 0], market_price=stats[:, 1])
        token = self.stage_token(budget, spend, next_remaining, value, stats[:, 0], stats[:, 1], daily_budget, stage_idx)
        next_hidden = self.state_update(token, hidden)
        return outcomes, next_hidden, next_remaining

    def rollout(self, history, budgets, daily_budget, remaining_budget=None, start_stage=None, rollout_mode="gru"):
        """Free-running rollout: no observed suffix feedback is ever accepted.

        reencode is the paper's comparison: append the *same predicted*
        stage tokens and fully re-encode instead of applying the GRU.
        """
        if rollout_mode not in ("gru", "reencode"):
            raise ValueError("rollout_mode must be gru or reencode")
        budgets = self._tensor(budgets)
        if budgets.ndim == 1:
            budgets = budgets.unsqueeze(0)
        if budgets.ndim != 2:
            raise ValueError("budgets must have shape [batch, horizon] or [horizon]")
        batch, horizon = budgets.shape
        start = len(history) if start_stage is None else int(start_stage)
        if start < 0 or start + horizon > self.num_stages:
            raise ValueError("Rollout goes beyond the configured day")
        daily = self._tensor(daily_budget).expand(batch)
        remaining = self._tensor(history[-1]["remaining_budget"] if history else daily_budget) if remaining_budget is None else self._tensor(remaining_budget)
        remaining = remaining.expand(batch)
        hidden = self.encode_history(history, daily[0]).expand(batch, -1)
        token_history = self.history_tokens(history, daily[0]).unsqueeze(0).expand(batch, -1, -1)
        collected = {name: [] for name in (*OUTCOME_FIELDS, "remaining_budget")}
        for offset in range(horizon):
            outcomes, hidden, remaining = self.predict_step(hidden, budgets[:, offset], remaining, daily, start + offset)
            for name in OUTCOME_FIELDS:
                collected[name].append(outcomes[name])
            collected["remaining_budget"].append(remaining)
            if rollout_mode == "reencode":
                token = self.stage_token(budgets[:, offset], outcomes["spend"], remaining, outcomes["value"],
                                         outcomes["auction_volume"], outcomes["market_price"], daily, start + offset)
                token_history = torch.cat([token_history, token.unsqueeze(1)], dim=1)
                hidden = self.encode_tokens(token_history, [token_history.shape[1]] * batch)
        return {name: torch.stack(items, dim=1) if items else budgets.new_empty((batch, 0))
                for name, items in collected.items()}


def load_model(checkpoint_path, device="cpu"):
    """Load a trusted local HiMORA checkpoint and freeze it for deployment."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if checkpoint.get("format") != "himora-v1":
        raise ValueError("Unsupported HiMORA checkpoint format")
    model = HiMORA(**checkpoint["model_config"], normalization=checkpoint["normalization"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval().requires_grad_(False)
    model.provenance = checkpoint.get("provenance", {})
    return model
