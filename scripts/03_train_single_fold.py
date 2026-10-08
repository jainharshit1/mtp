#!/usr/bin/env python3
"""Train a single LODO fold."""

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
    parser = argparse.ArgumentParser(description="Train single LODO fold")
    parser.add_argument("--fold", type=int, required=True, help="Fold ID (0-7)")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test: 5 epochs, 100 images")
    args = parser.parse_args()

    if args.smoke_test:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config["training"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 99

        # Write temp config
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=".") as f:
            yaml.dump(config, f)
            args.config = f.name

    from src.training.lodo_runner import LODORunner
    runner = LODORunner(args.config)
    results = runner.run_fold(args.fold, resume=args.resume)

    print(f"\nFold {args.fold} results:")
    print(f"  mAP:    {results.get('mAP', 'N/A')}")
    print(f"  mAP@50: {results.get('mAP_50', 'N/A')}")

    if args.smoke_test and os.path.exists(args.config) and args.config.startswith("/tmp"):
        os.unlink(args.config)


if __name__ == "__main__":
    main()
