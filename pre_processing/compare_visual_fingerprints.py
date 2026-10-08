import itertools
import heapq
import re
from contextlib import redirect_stdout
from collections import defaultdict
from pathlib import Path
from PIL import Image
import imagehash

DATASETS = {
    "Aircraft-AI-Dataset-4": Path("Aircraft-AI-Dataset-4"),
    "Danos_dataset": Path("Danos_dataset"),
    "Exp-29b-2": Path("Exp-29b-2"),
    "Panther-Combiné-3": Path("Panther-Combiné-3"),
    "a-2": Path("a-2"),
    "aircraft-dataset": Path("aircraft-dataset"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_RESULTS = 30
REPORT_PATH = Path("visual_fingerprint_report.txt")

def original_image_name(image_path):
    filename = image_path.stem
    return re.split(r"_jpg", filename, maxsplit=1, flags=re.IGNORECASE)[0].lower()

def collect_fingerprints(dataset_path):
    fingerprints = []
    image_paths = sorted(
        path for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                fingerprints.append((image_path, imagehash.phash(image)))
        except Exception as error:
            print(f"Could not process {image_path}: {error}")

    return fingerprints

def generate_report():
    fingerprints = {}
    pair_summaries = []
    
    # 1. Collect all hashes
    for dataset_name, dataset_path in DATASETS.items():
        print(f"Processing {dataset_name}...")
        fingerprints[dataset_name] = collect_fingerprints(dataset_path)
        print(f"  Images processed: {len(fingerprints[dataset_name])}")

    dataset_names = list(DATASETS)

    # 2. Compare pairs of datasets
    for first_idx, second_idx in itertools.combinations(range(len(dataset_names)), 2):
        first_name = dataset_names[first_idx]
        second_name = dataset_names[second_idx]
        
        first_data = fingerprints[first_name]
        second_data = fingerprints[second_name]

        # --- OPTIMIZATION 1: O(1) Exact Match Lookups ---
        # Group the second dataset by hash for instant lookups
        second_hash_dict = defaultdict(list)
        for path, img_hash in second_data:
            second_hash_dict[img_hash].append(path)

        zero_distance_matches = []
        for first_path, first_hash in first_data:
            if first_hash in second_hash_dict:
                for match_path in second_hash_dict[first_hash]:
                    zero_distance_matches.append((first_path, match_path))

        # --- OPTIMIZATION 2: Optimized Heap for Fuzzy Matches ---
        # Generate all distances lazily and let heapq.nsmallest handle the C-optimized sorting
        distance_counts = {"1-5": 0, "6-10": 0, "11+": 0}

        def distance_generator():
            for first_path, first_hash in first_data:
                for second_path, second_hash in second_data:
                    dist = first_hash - second_hash
                    if dist > 0: # Skip exact matches handled above
                        if dist <= 5:
                            distance_counts["1-5"] += 1
                        elif dist <= 10:
                            distance_counts["6-10"] += 1
                        else:
                            distance_counts["11+"] += 1
                        yield (dist, first_path, second_path)

        closest_fuzzy_matches = heapq.nsmallest(MAX_RESULTS, distance_generator(), key=lambda x: x[0])

        # --- Print Results ---
        print(f"\n=== Closest fuzzy matches: {first_name} vs {second_name} ===")
        for dist, p1, p2 in closest_fuzzy_matches:
            print(f"Distance: {dist}\n  {p1}\n  {p2}")

        print(f"\n=== Exact matches (Distance 0): {first_name} vs {second_name} ===")
        print(f"Total distance-0 pairs: {len(zero_distance_matches)}")
        
        matching_names = 0
        for p1, p2 in zero_distance_matches:
            n1, n2 = original_image_name(p1), original_image_name(p2)
            if n1 == n2:
                matching_names += 1
                print(f"Names match: True | {n1} | {n2}\n  {p1}\n  {p2}")
            else:
                print(f"Names match: False | {n1} | {n2}\n  {p1}\n  {p2}")

        print(
            f"Distance-0 pairs with matching names before _jpg: "
            f"{matching_names}/{len(zero_distance_matches)}"
        )

        pair_summaries.append(
            {
                "pair": f"{first_name} vs {second_name}",
                "confirmed": matching_names,
                "distance_zero": len(zero_distance_matches),
                "very_similar": distance_counts["1-5"],
                "possibly_similar": distance_counts["6-10"],
                "probably_different": distance_counts["11+"],
            }
        )

    print("\n\n=== FINAL DATASET OVERLAP SUMMARY ===")
    print("Confirmed match rule: pHash distance = 0 AND normalized names match before _jpg.")
    print("Distance 1-5: very similar candidates; distance 6-10: possible candidates; distance 11+: probably different.")
    print("\nDataset pair summary (all cross-dataset image pairs):")
    for summary in sorted(pair_summaries, key=lambda item: item["confirmed"], reverse=True):
        print(
            f"{summary['pair']} | "
            f"confirmed={summary['confirmed']} | "
            f"distance_0={summary['distance_zero']} | "
            f"distance_1_5={summary['very_similar']} | "
            f"distance_6_10={summary['possibly_similar']} | "
            f"distance_11_plus={summary['probably_different']}"
        )

    print("\nRecommended deduplication candidates:")
    for summary in sorted(pair_summaries, key=lambda item: item["confirmed"], reverse=True):
        if summary["confirmed"] > 0:
            print(f"{summary['pair']}: remove or reconcile {summary['confirmed']} confirmed overlaps")

def main():
    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        with redirect_stdout(report_file):
            generate_report()

    print(f"Comparison complete. Report written to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()