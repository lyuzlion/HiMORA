"""Validated campaign-day logs and train-only normalization for HiMORA.

The unit of splitting is an episode (a day), not a stage or a
campaign. All stages and campaigns sharing a day must remain in one split.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contracts import STAGE_FIELDS


NORMALIZED_FIELDS = ("value", "auction_volume", "market_price")


@dataclass
class Trajectory:
    episode: int
    player_index: int
    records: List[dict]

    @property
    def daily_budget(self):
        return float(self.records[0]["daily_budget"])

    @property
    def key(self):
        return (self.episode, self.player_index)


def read_trajectories(path, num_stages: Optional[int] = None) -> List[Trajectory]:
    """Read flat JSONL, rejecting duplicates, gaps, and inconsistent budgets."""
    path = Path(path)
    grouped = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                missing = set(STAGE_FIELDS) - set(record)
                if missing:
                    raise ValueError("missing fields: " + ", ".join(sorted(missing)))
                for field in STAGE_FIELDS:
                    if not isinstance(record[field], (int, float)) or not np.isfinite(record[field]):
                        raise ValueError("non-finite/non-numeric field " + field)
                for field in ("episode", "player_index", "stage_idx"):
                    if record[field] != int(record[field]) or record[field] < 0:
                        raise ValueError("expected nonnegative integer " + field)
                for field in STAGE_FIELDS[3:]:
                    if record[field] < -1e-7:
                        raise ValueError("negative observed " + field)
                if record["daily_budget"] <= 0:
                    raise ValueError("daily_budget must be positive")
                key = (int(record["episode"]), int(record["player_index"]))
                grouped.setdefault(key, []).append(record)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not grouped:
        raise ValueError(f"No stage records found in {path}")
    inferred_stages = max(int(r["stage_idx"]) for records in grouped.values() for r in records) + 1
    expected_stages = inferred_stages if num_stages is None else int(num_stages)
    trajectories = []
    for key, records in sorted(grouped.items()):
        records.sort(key=lambda r: r["stage_idx"])
        stages = [int(r["stage_idx"]) for r in records]
        if stages != list(range(expected_stages)):
            raise ValueError(f"Trajectory {key} has duplicate, missing, or incomplete stages: {stages}")
        daily = float(records[0]["daily_budget"])
        previous_remaining = daily
        tolerance = max(1e-5, daily * 1e-6)
        for record in records:
            if not np.isclose(record["daily_budget"], daily, rtol=0, atol=tolerance):
                raise ValueError(f"Daily budget changed in trajectory {key}")
            budget, spend = record["assigned_budget"], record["spend"]
            if budget > previous_remaining + tolerance or spend > budget + tolerance:
                raise ValueError(f"Observed stage cap violation in trajectory {key}, stage {record['stage_idx']}")
            expected_remaining = previous_remaining - spend
            if abs(record["remaining_budget"] - expected_remaining) > tolerance:
                raise ValueError(f"Remaining budget is not previous remaining minus spend in trajectory {key}")
            previous_remaining = float(record["remaining_budget"])
        trajectories.append(Trajectory(*key, records))
    return trajectories


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(train_path, val_path, train, validation) -> dict:
    train_days = {trajectory.episode for trajectory in train}
    validation_days = {trajectory.episode for trajectory in validation}
    overlap = train_days & validation_days
    if overlap:
        raise ValueError(f"Train/validation day leakage: episode IDs {sorted(overlap)} occur in both splits")
    return {
        "split_unit": "episode (entire day, across all players)",
        "normalization_fit_split": "train",
        "train": {"path": Path(train_path).name, "sha256": file_sha256(train_path),
                  "episode_ids": sorted(train_days), "trajectories": [list(t.key) for t in train],
                  "num_records": sum(len(t.records) for t in train)},
        "validation": {"path": Path(val_path).name, "sha256": file_sha256(val_path),
                       "episode_ids": sorted(validation_days), "trajectories": [list(t.key) for t in validation],
                       "num_records": sum(len(t.records) for t in validation)},
    }


def fit_normalization(train: Sequence[Trajectory]) -> Dict:
    """Z-score value/statistics using training records only; budgets use B."""
    rows = [record for trajectory in train for record in trajectory.records]
    if not rows:
        raise ValueError("Cannot fit normalization to an empty training split")
    values = np.asarray([[record[field] for field in NORMALIZED_FIELDS] for record in rows], dtype=np.float64)
    means, std = values.mean(axis=0), values.std(axis=0)
    # Constant features use unit scale; a tiny denominator would amplify noise.
    scales = np.where(std > 1e-6, std, 1.0)
    return {"fields": list(NORMALIZED_FIELDS), "mean": means.tolist(), "scale": scales.tolist(),
            "count": len(rows), "fit_split": "train", "mean_spend": float(np.mean([r["spend"] for r in rows]))}


def prefix_indices(trajectories: Sequence[Trajectory]):
    """Every decision point, including the empty pre-day history."""
    return [(index, prefix) for index, trajectory in enumerate(trajectories)
            for prefix in range(len(trajectory.records))]
