# HiMORA + SABS

HiMORA (**Hi**story-conditioned **M**ulti-stage **O**utcome Modeling for Budget
**Re**allocation) predicts future stage outcomes from completed feedback and
candidate budgets. **Support-Aware Budget Search (SABS)** uses the frozen model
to choose the next stage budget online.

## Install and check

Python 3.9+ and NumPy/PyTorch are required.

```bash
python -m pip install -e .
python -m himora demo --output outputs/demo --steps 20 --seed 7
python -m unittest discover -s tests -v
```

The demo creates a tiny synthetic dataset and writes a checkpoint under
`outputs/demo/`. It is an execution check, not a reproduction of paper results.

## Data format

Training and validation inputs are JSON Lines files with one record per
campaign-stage. Required fields:

```text
episode, player_index, stage_idx, daily_budget, assigned_budget,
spend, value, remaining_budget, auction_volume, market_price
```

Each `(episode, player_index)` trajectory must contain stages `0..m-1`, with the
same `m` across the run. Budgets and spend must use one monetary unit;
`assigned_budget` cannot exceed the actual pre-stage remainder, and
`remaining_budget` must be consistent with spend. The loader rejects missing or
duplicate stages, nonfinite/negative observations, and invalid budget updates.

Split by **day**, never by stage: all campaigns sharing an `episode` stay in the
same split. Keep test days separate. Normalization is fitted on training data
only. Training saves `model.pt`, `metadata.json`, and `split_manifest.json`.

## Train and evaluate

```bash
python -m himora train --train data/train.jsonl \
    --validation data/validation.jsonl --config config/default.json \
    --output outputs/model --device cpu

python -m himora evaluate --checkpoint outputs/model/model.pt \
    --log data/test.jsonl --horizon 12 --config config/default.json
```

Use `--device cuda` for GPU runs. The stage count is inferred from complete
training trajectories; training counts are optimizer steps, not epochs. HiMORA
combines a causal Transformer history encoder, an outcome predictor, and a GRU
transition module. Future observed outcomes are not fed into recursive forecasts.

## Online SABS

SABS runs after real feedback is received and is separate from model training:

```python
import numpy as np
from himora import DecisionContext, ProductionSupport, SearchConfig
from himora import SupportAwareBudgetSearch, load_model

model = load_model("outputs/model/model.pt", device="cpu")
m = model.num_stages
weights = np.ones(m)  # Use the actual logging-policy weights.
context = DecisionContext(
        stage_idx=0,
        remaining_budget=100.0,
        daily_budget=100.0,
        baseline=100.0 * weights / weights.sum(),
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

Keep `baseline` unchanged across decisions, pass only real completed records in
`history`, and reuse the same search object. Execute **only**
`decision.budget`; `decision.schedule` is a predicted suffix, not an executable
fixed schedule.

The default `ProductionSupport` models proportional allocation from the actual
remaining budget, multiplicative perturbation `delta ~ Uniform[-rho, rho]`, and
clipping to available budget. Set `rho` and `production_weights` to match the
logging policy that collected the data. A support restriction alone does not
establish causal identification, and this package does not perform offline policy
evaluation for unobserved decisions.

The demo also creates a CLI context:

```bash
python -m himora search --checkpoint outputs/demo/checkpoint/model.pt \
    --context outputs/demo/online_context.json --config config/default.json
```

Only load checkpoints from trusted sources.



