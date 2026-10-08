#!/usr/bin/env python3
"""Aggregate LODO results and print report."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Aggregate and report LODO results")
    parser.add_argument("--results-dir", type=str, default="outputs",
                        help="Directory containing fold_* results")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report as JSON")
    args = parser.parse_args()

    from src.evaluation.aggregate import aggregate_results, print_report
    summary = aggregate_results(args.results_dir)

    if not summary:
        print("No results found.")
        return 1

    print_report(summary)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nReport saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
