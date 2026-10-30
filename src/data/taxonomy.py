"""Unified class taxonomy for aircraft defect detection (6 classes).

2026-10-28: missing-head and fastener-damage merged into missing-fastener (both mark empty fastener
holes); rupture merged into crack (sydney's rupture boxes are mostly cracks and tears). See
configs/datasets.yaml for the per-dataset mapping.
"""

UNIFIED_CLASSES = [
    "crack",
    "dent",
    "corrosion",
    "scratch",
    "paint-off",
    "missing-fastener",
]

CLASS_TO_ID = {name: i for i, name in enumerate(UNIFIED_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(UNIFIED_CLASSES)}
NUM_CLASSES = len(UNIFIED_CLASSES)
