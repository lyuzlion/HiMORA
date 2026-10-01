# HiMORA + SABS


**HiMORA** stands for **HI**story-conditioned **M**ulti-stage **O**utcome Modeling
for Budget **R**e**A**llocation. It predicts future stage outcomes from completed
stage feedback and candidate budgets. **Support-Aware Budget Search (SABS)** uses
the trained model to select the next stage budget online.

This release contains the model, training objective, validated data loader,
randomized-budget support definition, and online search. 

## 1. Installation

Python 3.9 or later is required. Runtime dependencies are NumPy and PyTorch.

```bash
python -m pip install -e .
```

Alternatively, run directly from this directory after installing dependencies:

```bash
python -m pip install -r requirements.txt
python -m himora --help
```

Install a PyTorch build appropriate for your hardware if needed. The dependency
ranges are declared in `pyproject.toml`.

## 2. Quick start

Run an end-to-end execution check:

```bash
python -m himora demo --output outputs/demo --steps 20 --seed 7
python -m unittest discover -s tests -v
```

The demo generates a **tiny artificial dataset at runtime**, trains a small
HiMORA, reloads the checkpoint, checks a multi-stage forecast, and runs SABS at
every decision point of a toy day. Only the selected next budget is executed.
New observed feedback is appended before the next decision. The final JSON output
contains `"status": "passed"` when the execution checks succeed.

The demo writes its generated logs and checkpoint under `outputs/demo/`. These
are local runtime outputs, not files distributed with this release. Use a new
output directory for another run; the demo does not overwrite existing artifacts.
Its six-stage, 16-dimensional model is intentionally small. **The demo is not a
reproduction of the paper's experiments and does not establish performance gains.**

## 3. Training data interface

Provide JSON Lines files, with one campaign-stage record per line. There is no
bundled dataset. The required fields are:

| Field | Meaning |
| --- | --- |
| `episode` | Nonnegative integer day ID; globally consistent across splits |
| `player_index` | Nonnegative integer campaign ID |
| `stage_idx` | Zero-based stage index |
| `daily_budget` | Total budget for the day; positive and constant within a trajectory |
| `assigned_budget` | Budget assigned to this stage, not actual spend |
| `spend` | Actual spend during this stage |
| `value` | Nonnegative observed value in the task's chosen, consistent units |
| `remaining_budget` | Actual remaining daily budget **after** this stage |
| `auction_volume` | Observed nonnegative stage traffic/auction statistic |
| `market_price` | Observed nonnegative stage market-price statistic |

An illustrative record:

```json
{"episode": 0, "player_index": 0, "stage_idx": 0, "daily_budget": 100.0, "assigned_budget": 15.0, "spend": 10.0, "value": 0.4, "remaining_budget": 90.0, "auction_volume": 200.0, "market_price": 0.05}
```

Each `(episode, player_index)` trajectory must contain all stages from `0` to
`m - 1`, where `m` is the number of stages per day. All trajectories in a run use
the same `m`. The loader rejects missing stages, duplicates, nonfinite values,
negative observations, spending above the assigned budget, and inconsistent
remaining-budget updates. Assigned budgets cannot exceed the actual budget
remaining before their stage. Budgets and spend must use the same monetary unit.

Split by **day**, not by stage: every campaign sharing a day must remain in the
same split. Training and validation episode IDs must be disjoint; the loader
checks this. Keep test days separate as well. Outcome normalization is fitted
only on training records. A data file hash and day-level split manifest are saved
with each trained model. Generated metadata stores filenames, not machine-specific
absolute paths.

## 4. Train and evaluate HiMORA

```bash
python -m himora train --train data/train.jsonl --validation data/validation.jsonl --config config/default.json --output outputs/model --device cpu
python -m himora evaluate --checkpoint outputs/model/model.pt --log data/test.jsonl --horizon 12 --config config/default.json
```

Use `--device cuda` for GPU training or evaluation. `config/default.json` contains
the full-size model and training/search settings; the stage count is inferred
from complete training trajectories. Adjust `max_horizon` for your task. Training
counts are optimizer steps, not epochs.

HiMORA contains three jointly trained modules:

1. A causal Transformer history module encodes completed stage feedback.
2. An outcome module predicts spend, value, auction volume, and market price.
3. A GRU transition module propagates predicted states over future stages.

Training first uses one-step prediction and then adds a discounted multi-stage
loss. Future observed budgets are provided during training, but future observed
outcomes are **not** fed back into the recursive predictions. Validation selects
the best eligible checkpoint; no test data enters this selection.

Outputs are `model.pt`, `metadata.json`, and `split_manifest.json`. Evaluation
reports raw MAE, normalized errors, and errors by prediction horizon. Here, raw
NMAE is the sum of absolute prediction errors divided by the sum of absolute
observed targets, not the mean of per-record percentage errors. Multiplying raw
NMAE by 100 expresses that aggregate ratio as a percentage.

Only load checkpoints from trusted sources. Checkpoint loading uses PyTorch's
serialized Python object format.

## 5. Online SABS decisions

SABS is a **deployment-time decision algorithm**, not part of HiMORA training.
Call it before the next stage, after receiving real feedback from completed
stages. It searches with frozen model parameters.

```python
import numpy as np
from himora import (
    DecisionContext, ProductionSupport, SearchConfig,
    SupportAwareBudgetSearch, load_model,
)

model = load_model("outputs/model/model.pt", device="cpu")
m = model.num_stages
weights = np.ones(m)  # Replace with the actual logging system's weights.
baseline = 100.0 * weights / weights.sum()
context = DecisionContext(
    stage_idx=0,
    remaining_budget=100.0,
    daily_budget=100.0,
    baseline=baseline,
    production_weights=weights,
    history=[],
)
search = SupportAwareBudgetSearch(
    model,
    SearchConfig(num_basis=4, rounds=3, samples=64, elites=8,
                 initial_std=0.1, coefficient_bound=1.0, seed=7),
    ProductionSupport(rho=0.25),
)
decision = search.decide(context)
next_stage_budget = decision.budget
```

`baseline` is the original complete daily schedule and remains unchanged across
decisions. `production_weights` defines the logging policy's stage proportions;
it need not equal the baseline and must not be replaced with a candidate plan.
For a later decision, `history` contains all real completed-stage records in
order, `stage_idx` equals their count, and `remaining_budget` equals the latest
record's post-stage remaining budget. The context and checkpoint must use the
same number of stages. Keep the same search object for sequential decisions so
its random generator progresses naturally.

SABS samples smooth schedule adjustments, constructs each future budget using
predicted remaining budget, projects it into the corresponding logging-support
interval, and scores the predicted remaining-day outcomes. The score subtracts
penalties for leftover budget and adjustment magnitude. The projected baseline
candidate is also evaluated. If no candidate is valid, the algorithm falls back
to the original baseline proportionally rescaled to actual remaining budget.
The fallback is business-feasible but is not guaranteed to lie within logging
support; `diagnostics` explicitly records this case.

Execute **only `decision.budget`**. `decision.schedule` contains budgets constructed
along predicted states, not a fixed schedule to execute. Its sum need not equal
the current remaining budget, because later budgets incorporate predicted unspent
money. Re-run SABS with real feedback after the stage ends; do not use its predicted
suffix to update the original baseline.

The demo also creates an online context file for the command-line interface:

```bash
python -m himora search --checkpoint outputs/demo/checkpoint/model.pt --context outputs/demo/online_context.json --config config/default.json
```

This command returns the next budget, predicted stage budgets, and diagnostics.

### Randomized-budget support

The included support definition matches a logging policy that proportionally
allocates actual remaining budget using `production_weights`, multiplies the
proposal by `1 + delta`, and clips it to available budget. `delta` is uniformly
sampled from `[-rho, rho]`; `rho` is the perturbation radius in `[0, 1]`.
`ProductionSupport.perturb()` implements this collection rule and
`ProductionSupport.bounds()` returns its stage-specific interval.

Set `rho` and the production weights to the logging process that actually
collected your data. A search restriction alone does not establish causal
identification. Using the model to compare alternative budgets requires suitable
data collection and coverage. If your logging policy differs, replace this
support definition accordingly. The package does not perform offline policy
evaluation or claim outcomes for unobserved decisions as measured results.



