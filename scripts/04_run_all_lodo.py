#!/usr/bin/env python3
"""Run all LODO folds sequentially."""

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)


def main():
    parser = argparse.ArgumentParser(description="Run all LODO folds")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subset of folds to run (default: all 8)")
    parser.add_argument("--resume", action="store_true", help="Resume crashed folds")
    parser.add_argument("--prompt-strategy", type=str, default=None)
    args = parser.parse_args()

    if args.prompt_strategy:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config["prompt_strategy"] = args.prompt_strategy
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=".") as f:
            yaml.dump(config, f)
            args.config = f.name

    from src.training.lodo_runner import LODORunner
    runner = LODORunner(args.config)
    summary = runner.run_all(fold_ids=args.folds, resume=args.resume)


if __name__ == "__main__":
    main()
