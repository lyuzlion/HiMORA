"""Small command-line interface with no simulator or external service dependency."""
import argparse
import json
from pathlib import Path

import torch

from .contracts import DecisionContext
from .model import load_model
from .search import SearchConfig, SupportAwareBudgetSearch
from .support import ProductionSupport
from .training import evaluate_model, train_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="HiMORA training and SABS budget decisions")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Train and run a complete toy closed-loop example")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--steps", type=int, default=20, help="Optimizer steps per training phase")
    demo.add_argument("--seed", type=int, default=7)
    train = commands.add_parser("train", help="Train on user-supplied stage logs")
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate", help="Evaluate a trusted local checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--log", required=True)
    evaluate.add_argument("--horizon", type=int, default=12)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--config", help="Use the same loss weights as training")
    evaluate.add_argument("--device", default="cpu")
    search = commands.add_parser("search", help="Select the next stage budget")
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--context", required=True)
    search.add_argument("--config", required=True)
    search.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.output, args.steps, args.seed)
    elif args.command == "train":
        result = train_model(args.train, args.validation, args.output,
                             {**read_json(args.config)["training"], "device": args.device})
        result = {"checkpoint": str(Path(args.output) / "model.pt"),
                  "best_step": result["best_step"], "validation": result["validation"]}
    else:
        model = load_model(args.checkpoint, args.device)
        if args.command == "evaluate":
            settings = read_json(args.config)["training"] if args.config else None
            result = evaluate_model(model, args.log, args.horizon, args.batch_size, settings)
        else:
            config = read_json(args.config)
            context = DecisionContext(**read_json(args.context))
            if context.num_stages != model.num_stages:
                parser.error("context and checkpoint must use the same number of stages")
            decision = SupportAwareBudgetSearch(
                model, SearchConfig(**config["search"]), ProductionSupport(**config["support"])
            ).decide(context)
            result = {"next_budget": decision.budget, "predicted_stage_budgets": decision.schedule.tolist(),
                      "diagnostics": decision.diagnostics}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
