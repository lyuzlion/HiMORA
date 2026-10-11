"""Small dependency-free test suite (unittest is in Python's standard library)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from himora import DecisionContext, HiMORA, ProductionSupport, SearchConfig, SupportAwareBudgetSearch
from himora.data import read_trajectories, split_manifest
from himora.demo import write_toy_log
from himora.search import candidate_schedules


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = HiMORA(num_stages=6, hidden_dim=16, num_heads=2, num_layers=1).eval()
        self.weights = np.ones(6)
        self.support = ProductionSupport(0.25)
        self.context = DecisionContext(0, 100., 100., 100 * self.weights / 6, self.weights, [])

    def test_rollout_physical_outputs(self):
        with torch.inference_mode():
            result = self.model.rollout([], [10.] * 6, 100.)
        self.assertEqual(tuple(result["spend"].shape), (1, 6))
        self.assertTrue(torch.all((result["spend"] >= 0) & (result["spend"] <= 10)))
        self.assertTrue(torch.all(result["value"] >= 0))
        self.assertTrue(torch.allclose(result["remaining_budget"], 100 - result["spend"].cumsum(1)))

    def test_schedule_parameterization(self):
        plans = candidate_schedules(np.zeros((2, 4)), self.context.baseline, 0, 100)
        np.testing.assert_allclose(plans.sum(axis=1), [100, 100])
        np.testing.assert_allclose(plans[0], self.context.baseline)

    def test_support_bounds_and_perturbation(self):
        low, high = self.support.bounds(0, 100, self.weights)
        self.assertAlmostEqual(low, 12.5)
        self.assertAlmostEqual(high, 100 / 6 * 1.25)
        for delta in (-0.25, 0, 0.25):
            budget = self.support.perturb(0, 100, self.weights, delta)["assigned_budget"]
            self.assertTrue(self.support.contains(budget, 0, 100, self.weights))
        self.assertLessEqual(self.support.bounds(5, 10, self.weights)[1], 10)

    def test_search_feasibility_and_reproducibility(self):
        cfg = SearchConfig(rounds=2, samples=8, elites=2, initial_std=0.1, seed=19)
        first = SupportAwareBudgetSearch(self.model, cfg, self.support).decide(self.context)
        second = SupportAwareBudgetSearch(self.model, cfg, self.support).decide(self.context)
        self.assertFalse(first.diagnostics["fallback"])
        self.assertTrue(self.support.contains(first.budget, 0, 100, self.weights))
        self.assertEqual(first.budget, second.budget)
        np.testing.assert_allclose(first.schedule, second.schedule)
        with torch.inference_mode():
            remaining = 100.
            hidden = self.model.encode_history([], 100)
            for stage, budget in enumerate(first.schedule):
                self.assertTrue(self.support.contains(budget, stage, remaining, self.weights, atol=1e-4))
                _, hidden, predicted = self.model.predict_step(hidden, budget, remaining, 100, stage)
                remaining = float(predicted[0])

    def test_fallback_with_invalid_predictions(self):
        with torch.no_grad():
            self.model.outcome_mlp[-1].bias.fill_(float("nan"))
        decision = SupportAwareBudgetSearch(self.model, SearchConfig(samples=4, elites=2), self.support).decide(self.context)
        self.assertTrue(decision.diagnostics["fallback"])
        self.assertAlmostEqual(decision.budget, 100 / 6)
        self.assertGreaterEqual(decision.budget, 0)
        self.assertLessEqual(decision.budget, 100)

    def test_history_context_validation(self):
        with self.assertRaises(ValueError):
            DecisionContext(2, 100, 100, self.weights, self.weights, [])

    def test_data_validation_and_day_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            train, val = Path(directory) / "train.jsonl", Path(directory) / "validation.jsonl"
            rng = np.random.default_rng(7)
            write_toy_log(train, [0, 1], rng, self.support, self.weights)
            write_toy_log(val, [2], rng, self.support, self.weights)
            a, b = read_trajectories(train, 6), read_trajectories(val, 6)
            self.assertEqual(len(a), 2)
            manifest = split_manifest(train, val, a, b)
            self.assertEqual(manifest["train"]["path"], "train.jsonl")
            with self.assertRaises(ValueError):
                split_manifest(train, train, a, a)
            row = json.loads(train.read_text(encoding="utf-8").splitlines()[0])
            row["remaining_budget"] = 100
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_trajectories(train, 1)


if __name__ == "__main__":
    unittest.main()
