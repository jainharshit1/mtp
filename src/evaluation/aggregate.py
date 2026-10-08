"""Aggregate LODO evaluation results across folds."""

import json
import logging
from pathlib import Path

import numpy as np

from src.data.taxonomy import UNIFIED_CLASSES

logger = logging.getLogger(__name__)


def aggregate_results(results_dir: str) -> dict:
    """Aggregate per-fold results into summary statistics."""
    results_path = Path(results_dir)
    fold_results = []

    for fold_dir in sorted(results_path.glob("fold_*")):
        results_file = fold_dir / "eval_results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            fold_results.append(json.load(f))

    if not fold_results:
        logger.warning(f"No results found in {results_dir}")
        return {}

    # Overall mAP stats
    maps = [r["mAP"] for r in fold_results]
    maps_50 = [r["mAP_50"] for r in fold_results]

    summary = {
        "num_folds": len(fold_results),
        "mAP_mean": float(np.mean(maps)),
        "mAP_std": float(np.std(maps)),
        "mAP_50_mean": float(np.mean(maps_50)),
        "mAP_50_std": float(np.std(maps_50)),
        "per_fold": [],
    }

    # Per-class aggregation
    per_class = {}
    for cls_name in UNIFIED_CLASSES:
        cls_aps = []
        cls_support = 0
        for r in fold_results:
            if cls_name in r.get("per_class", {}):
                cls_data = r["per_class"][cls_name]
                cls_aps.append(cls_data["AP"])
                cls_support += cls_data.get("support", 0)

        if cls_aps:
            per_class[cls_name] = {
                "AP_mean": float(np.mean(cls_aps)),
                "AP_std": float(np.std(cls_aps)),
                "total_support": cls_support,
                "num_folds_present": len(cls_aps),
                "low_support": cls_support < 50,
            }

    summary["per_class"] = per_class

    # Per-fold details
    for i, r in enumerate(fold_results):
        fold_info = {
            "fold": i,
            "held_out": r.get("held_out_dataset", f"fold_{i}"),
            "mAP": r["mAP"],
            "mAP_50": r["mAP_50"],
            "n_test_images": r.get("n_test_images", 0),
        }
        summary["per_fold"].append(fold_info)

    return summary


def print_report(summary: dict):
    """Print a formatted report of aggregated results."""
    print("\n" + "=" * 70)
    print("LODO EVALUATION REPORT")
    print("=" * 70)
    print(f"Folds evaluated: {summary['num_folds']}")
    print(f"Overall mAP:     {summary['mAP_mean']:.4f} ± {summary['mAP_std']:.4f}")
    print(f"Overall mAP@50:  {summary['mAP_50_mean']:.4f} ± {summary['mAP_50_std']:.4f}")

    print("\n--- Per-Class Results ---")
    print(f"{'Class':<20} {'AP Mean':>10} {'AP Std':>10} {'Support':>10} {'Note':>10}")
    print("-" * 60)
    for cls_name, data in summary.get("per_class", {}).items():
        note = "LOW" if data.get("low_support") else ""
        print(
            f"{cls_name:<20} {data['AP_mean']:>10.4f} {data['AP_std']:>10.4f} "
            f"{data['total_support']:>10d} {note:>10}"
        )

    print("\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Held-out':<30} {'mAP':>10} {'mAP@50':>10} {'Images':>10}")
    print("-" * 66)
    for fold in summary.get("per_fold", []):
        print(
            f"{fold['fold']:<6} {fold['held_out']:<30} "
            f"{fold['mAP']:>10.4f} {fold['mAP_50']:>10.4f} "
            f"{fold['n_test_images']:>10}"
        )
    print("=" * 70)
