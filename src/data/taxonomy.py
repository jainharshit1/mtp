"""Unified class taxonomy for aircraft defect detection (8 classes, spot removed)."""

UNIFIED_CLASSES = [
    "crack",
    "dent",
    "corrosion",
    "scratch",
    "paint-off",
    "missing-head",
    "rupture",
    "fastener-damage",
]

CLASS_TO_ID = {name: i for i, name in enumerate(UNIFIED_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(UNIFIED_CLASSES)}
NUM_CLASSES = len(UNIFIED_CLASSES)
