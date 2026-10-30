"""
Leave-One-Dataset-Out (LODO) split generation.

Uses scene groups (see dedup.py) to keep every copy of a photo on one side:
  - A group "touches" dataset k if ANY image in it came from dataset k
  - Test: images of groups that touch the held-out dataset, taken from the held-out dataset
    itself (images with boxes only)
  - Train/val: groups that do NOT touch the held-out dataset. No held-out image and no
    copy of a test scene is ever trained on.
  - Val (default, lodo_v3): at least `val_fraction` of every training dataset's images, in whole scene
    groups (a group is either train or val), images selected like test. Val then covers every
    class, so the best checkpoint is chosen on all classes. Groups with a near-copy in another
    group (DINOv2 >= `proximity_filter`) go to val only together with it, as one cluster;
    isolated groups are preferred. (A whole val dataset per fold,
    `val_dataset_for`, is still supported, but each dataset annotates only a few classes, so it
    scored classes the test set does not measure.)
  - Only `test_datasets` get a fold; the rest (too small to test on) are always trained on.
  - `hold_out_together`: datasets from the same photo session (Defect-Detection-6 and closeup);
    when one is test or val, the others are not trained on in that fold.
  - Training uses only images with boxes.
  - The same photo (sub-scene) in two training datasets of a fold: only one dataset's copies are
    kept, the best labelled, except for datasets in `keep_copies_together` (korean-aerospace and
    RFDETR_DATASET: korean holds more augmented variants), whose copies are all kept in training,
    capped per dataset. Leakage is unaffected: a photo whose group touches test or val is never
    trained on in any dataset.

Which images of a group are kept (same rule for train/val and test):
  - best images first (boxes, labelled classes, box count, resolution)
  - thinning: skip an image whose DINOv2 cosine to an image already kept from the group is
    >= `thin_threshold` (0.95: about one frame in 3-4 of a video walk-around survives)
  - at most `max_per_group` (train/val, default 5) or `max_test_per_group` (test, default 2)
    images per sub-scene (copies linked by SHA, Roboflow name or pHash)
  - byte-identical files count once
The group decides the side of the split; thinning and caps only decide how many images stay.
Finally, per fold, near-copies across splits (DINOv2 cosine >= `proximity_filter` (0.80): re-exports
under another name with small rotations and recolouring score 0.80-0.90 and are linked by no
grouping rule) and photo-matched copies below 0.80 (scripts/fold_geometric_filter.py) are removed:
  - of a test image: the training (or val) copy is dropped
  - of a val image: the val image's whole scene group moves to train (nothing is lost, and the scene
    stays on one side)
"""

import hashlib
import logging
import random
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)

THIN_THRESHOLD = 0.95
PROXIMITY_FILTER = 0.80


def split_val_by_group(
    pool_ids: list,
    img_to_group: dict,
    val_fraction: float,
    rng: random.Random,
) -> tuple[list, list]:
    """Split pool_ids into (train, val) so that no scene group spans both sides.

    Whole groups are moved to val until it holds ~val_fraction of the images. A group that
    would push val past the target goes to train, so one large group (a whole video) cannot
    end up as most of val.
    """
    by_group = defaultdict(list)
    for img_id in pool_ids:
        by_group[img_to_group[img_id]].append(img_id)

    group_ids = sorted(by_group)
    rng.shuffle(group_ids)

    target = max(1, int(len(pool_ids) * val_fraction))
    val_ids, train_ids = [], []
    for gid in group_ids:
        if len(val_ids) + len(by_group[gid]) <= target:
            val_ids.extend(by_group[gid])
        else:
            train_ids.extend(by_group[gid])
    if not val_ids and group_ids:
        # Every group is larger than the target: use the smallest one as val.
        smallest = min(group_ids, key=lambda g: len(by_group[g]))
        val_ids = by_group[smallest]
        train_ids = [i for i in train_ids if img_to_group[i] != smallest]
    return train_ids, val_ids


def near_copy_clusters(group_images: dict, img_to_group: dict, embeddings, threshold: float) -> dict:
    """{group_id: cluster id}: scene groups joined when any image pair across them has DINOv2
    cosine >= threshold (other frames of a video, re-exports that no grouping rule links)."""
    parent = {g: g for g in group_images}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    ids = [i for g in sorted(group_images) for i in group_images[g]]
    grp = np.array([img_to_group[i] for i in ids])
    emb = embeddings[ids]
    for start in range(0, len(ids), 1024):
        a, b = np.nonzero(emb[start:start + 1024] @ emb.T >= threshold)
        a += start
        cross = grp[a] != grp[b]
        for x, y in np.unique(np.stack([grp[a[cross]], grp[b[cross]]], axis=1), axis=0):
            parent[find(x)] = find(y)
    return {g: find(g) for g in group_images}


def pick_val_groups(
    train_counts: dict,
    val_counts: dict,
    val_boxes: dict,
    val_fraction: float,
    clusters: dict | None = None,
    excluded: set = frozenset(),
    max_share: float = 1.2,
    min_boxes: int = 50,
    min_groups: int = 15,
    salt: str = "",
) -> tuple[set, dict]:
    """Val groups, by priority: (1) no relative in training, (2) ~val_fraction of every dataset,
    (3) every class with at least min_boxes boxes in at least min_groups groups, where possible.

    train_counts / val_counts: {group_id: Counter(dataset: images)} the group gives to train / to val
    (val keeps fewer copies per photo); val_boxes: {group_id: Counter(class: boxes)} of its val
    images; clusters: {group_id: cluster id} from near_copy_clusters (default: every group alone);
    excluded: clusters that may not go to val (a val image of theirs had a photo-matched copy in
    train in an earlier pass); other groups refill their place.

    (1) Only a group alone in its cluster (no near-copy in any other group) and not excluded can go
        to val; this is never relaxed.
    (2) A group belongs to the dataset most of its val images come from; shares are counted per
        image (val / (train + val) of each dataset). No group may push its dataset past
        max_share * val_fraction (one large group, a whole video, cannot become most of val).
    (3) First, while a class is short of min_boxes boxes or min_groups groups in val, the next
        eligible group with that class is added (most short class first). Then every dataset is
        filled with eligible groups up to val_fraction. A dataset or class with too few eligible
        groups stays short; the shortfall is returned, never forced.
    Groups are tried in a fixed pseudo-random order (hash of salt and group id), so a pass that
    excludes a group changes only the groups that replace it.

    Returns (val group ids, {"datasets": {ds: share}, "classes": {cls: (boxes, groups)}}).
    """
    clusters = clusters or {g: g for g in train_counts}
    cluster_n = Counter(clusters[g] for g in train_counts)

    def order(g):
        return hashlib.sha1(f"{salt}:{g}".encode()).hexdigest()

    eligible = sorted((g for g in train_counts
                       if cluster_n[clusters[g]] == 1 and clusters[g] not in excluded and sum(val_counts[g].values())),
                      key=order)
    main = {g: max(sorted(val_counts[g]), key=val_counts[g].get) for g in eligible}
    n_train, n_val = Counter(), Counter()
    for g in train_counts:
        n_train.update(train_counts[g])
    boxes, groups_with = Counter(), Counter()
    classes = sorted({c for g in train_counts for c in val_boxes.get(g, {})})

    def share(ds, g=None):
        t = n_train[ds] - (train_counts[g][ds] if g is not None else 0)
        v = n_val[ds] + (val_counts[g][ds] if g is not None else 0)
        return v / max(1, t + v)

    val = set()

    def take(g):
        val.add(g)
        n_train.subtract(train_counts[g])
        n_val.update(val_counts[g])
        boxes.update(val_boxes.get(g, {}))
        groups_with.update(c for c in val_boxes.get(g, {}) if val_boxes[g][c])

    def fits(g):
        return g not in val and share(main[g], g) <= max_share * val_fraction

    # (3) class floors first: they count towards the dataset shares.
    hopeless = set()
    while True:
        short = [c for c in classes if c not in hopeless and (boxes[c] < min_boxes or groups_with[c] < min_groups)]
        if not short:
            break
        c = min(short, key=lambda c: (min(boxes[c] / min_boxes, groups_with[c] / min_groups), c))
        g = next((g for g in eligible if val_boxes.get(g, {}).get(c) and fits(g)), None)
        if g is None:
            hopeless.add(c)
        else:
            take(g)
    # (2) fill every dataset up to val_fraction.
    for ds in sorted(set(main.values())):
        for g in eligible:
            if share(ds) >= val_fraction:
                break
            if main[g] == ds and fits(g):
                take(g)
    report = {"datasets": {ds: round(share(ds), 4) for ds in sorted(n_train | n_val)},
              "classes": {c: (boxes[c], groups_with[c]) for c in classes}}
    return val, report


def _check_fold(fold: dict, img_to_group: dict, img_by_id: dict) -> None:
    """Fail loudly if a scene spans two splits, a held-out image reaches train/val, or a
    val-dataset image reaches train."""
    held_out, val_ds = fold["held_out_dataset"], fold.get("val_dataset")
    split_groups = {
        name: {img_to_group[i] for i in fold[f"{name}_image_ids"]} for name in ("train", "val", "test")
    }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = split_groups[a] & split_groups[b]
        assert not shared, f"fold {fold['fold_id']}: {len(shared)} scene groups in both {a} and {b}"
    trained = fold["train_image_ids"] + fold["val_image_ids"]
    leaked = sum(img_by_id[i]["source_dataset"] == held_out for i in trained)
    assert not leaked, f"fold {fold['fold_id']}: {leaked} {held_out} images in train/val"
    for ds in fold.get("excluded_from_train", [val_ds]):
        leaked = sum(img_by_id[i]["source_dataset"] == ds for i in fold["train_image_ids"])
        assert not leaked, f"fold {fold['fold_id']}: {leaked} {ds} images in train (held out in this fold)"
        if ds != val_ds:
            # Without a val dataset, val is a slice of the training datasets: no held-out data in it.
            leaked = sum(img_by_id[i]["source_dataset"] == ds for i in fold["val_image_ids"])
            assert not leaked, f"fold {fold['fold_id']}: {leaked} {ds} images in val (held out in this fold)"


def generate_lodo_folds(
    unified_images: list[dict],
    merged_annotations: list[dict],
    annotated_classes: dict[int, list[int]],
    groups: dict[int, dict],
    dataset_names: list[str],
    val_fraction: float = 0.10,
    seed: int = 42,
    max_per_group: int = 5,
    max_test_per_group: int = 2,
    embeddings=None,
    thin_threshold: float | None = THIN_THRESHOLD,
    test_datasets: list[str] | None = None,
    val_dataset_for: dict[str, str] | None = None,
    proximity_filter: float | None = PROXIMITY_FILTER,
    train_drop_files: dict[int, set[str]] | None = None,
    hold_out_together: list[list[str]] | None = None,
    val_to_train_files: dict[int, set[str]] | None = None,
    keep_copies_together: list[list[str]] | None = None,
) -> list[dict]:
    """
    Generate LODO folds using group-level dataset membership.

    Args:
        unified_images: all images with group_id
        merged_annotations: merged annotation list
        annotated_classes: per-image annotated class sets
        groups: {group_id: {"source_datasets": set, "image_ids": list}}
        dataset_names: all dataset names
        test_datasets: datasets that get a fold (held out as test), in fold order; default all.
            The others are only ever trained on.
        val_dataset_for: {held-out dataset: val dataset}. That dataset is held out of training in
            the fold and selected like test; folds without an entry use a group split per dataset
        val_fraction: fraction of every training dataset's images used as validation (whole
            scene groups) when no val dataset is given
        seed: random seed for reproducibility
        max_per_group: images kept per sub-scene in train/val
        max_test_per_group: images kept per sub-scene in test
        embeddings: [n_images, D] L2-normalised, row = image id (from build_image_registry);
            None disables thinning
        thin_threshold: drop an image whose cosine to an image already kept from the same
            group is at least this (near-repeat frames); None disables thinning
        proximity_filter: drop a training image whose cosine to any test or val image of the
            fold is at least this; None disables
        train_drop_files: {fold_id: file names} removed from that fold's train and val (copies of
            test images found by scripts/fold_geometric_filter.py)
        val_to_train_files: {fold_id: val file names} with a copy in training (same script); their
            scene groups move from val to train
        keep_copies_together: lists of datasets whose copies of one photo are all kept in training
            (not only the best-labelled dataset's); the per-sub-scene caps count per dataset for them
        hold_out_together: lists of datasets photographed in the same session; when one is the
            test or val dataset of a fold, the others are left out of that fold's training set
            (their images are not used in the fold at all)
    """
    rng = random.Random(seed)

    img_by_id = {img["id"]: img for img in unified_images}
    img_to_group = {img["id"]: img["group_id"] for img in unified_images}
    n_boxes = Counter(ann["image_id"] for ann in merged_annotations)
    box_classes = defaultdict(Counter)  # image id -> {category id: boxes}
    for ann in merged_annotations:
        box_classes[ann["image_id"]][ann["category_id"]] += 1

    def keep_score(img_id):
        # Prefer images with boxes, then richer labelling, then more boxes, then resolution.
        img = img_by_id[img_id]
        return (n_boxes[img_id] > 0, len(annotated_classes.get(img_id, [])), n_boxes[img_id],
                img["width"] * img["height"], -img_id)

    thin = embeddings is not None and thin_threshold is not None
    kept_with = {ds: frozenset(grp) for grp in keep_copies_together or [] for ds in grp}

    def cap_key(img):
        # Copies of one photo in datasets kept together are capped per dataset.
        sub = img.get("subscene_id", img["group_id"])
        return (sub, img["source_dataset"]) if img["source_dataset"] in kept_with else sub

    def pick(img_ids, k):
        # Best-scoring images first. Skip a byte-identical file already kept, a near-repeat
        # (cosine >= thin_threshold) of an image already kept, and copies beyond k per sub-scene.
        chosen, seen_sha, per_subscene = [], set(), Counter()
        for i in sorted(img_ids, key=keep_score, reverse=True):
            img = img_by_id[i]
            sha, sub = img.get("sha256", i), cap_key(img)
            if sha in seen_sha or per_subscene[sub] >= k:
                continue
            if thin and chosen and (embeddings[chosen] @ embeddings[i]).max() >= thin_threshold:
                continue
            seen_sha.add(sha)
            per_subscene[sub] += 1
            chosen.append(i)
        return chosen

    def held_out_images(group, dataset):
        # A held-out dataset's own images with boxes; copies of the scene from other
        # datasets are dropped from the fold, not trained on.
        candidates = [i for i in group["image_ids"]
                      if img_by_id[i]["source_dataset"] == dataset and n_boxes[i]]
        return pick(candidates, max_test_per_group)

    folds = []
    test_datasets = test_datasets or dataset_names
    val_dataset_for = val_dataset_for or {}

    partners = defaultdict(set)
    for together in hold_out_together or []:
        for ds in together:
            partners[ds] |= set(together) - {ds}

    for fold_id, held_out in enumerate(test_datasets):
        val_ds = val_dataset_for.get(held_out)
        # Same-session datasets of the test/val dataset: never trained on in this fold.
        sidelined = (partners[held_out] | partners.get(val_ds, set())) - {held_out, val_ds}
        test_ids, val_ids, train_pool = [], [], []
        candidates = {}  # group_id -> images with boxes (train/val side)
        excluded = sidelined_n = dropped_empty = 0

        for group_id in sorted(groups):
            group = groups[group_id]
            if held_out in group["source_datasets"]:
                chosen = held_out_images(group, held_out)
                test_ids.extend(chosen)
                excluded += len(group["image_ids"]) - len(chosen)
            elif val_ds in group["source_datasets"]:
                # The val dataset is held out from training too, and selected like test.
                chosen = held_out_images(group, val_ds)
                val_ids.extend(chosen)
                excluded += len(group["image_ids"]) - len(chosen)
            elif sidelined & group["source_datasets"]:
                sidelined_n += len(group["image_ids"])
            else:
                # Images without boxes are left out: in a sample of them many show unlabelled
                # defects, which would train the model to miss them.
                boxed = [i for i in group["image_ids"] if n_boxes[i]]
                dropped_empty += len(group["image_ids"]) - len(boxed)
                if boxed:
                    candidates[group_id] = boxed

        # The same photo in two training datasets (a sub-scene across datasets: re-exports, datasets
        # merged from others): keep the copies of one dataset only, the one whose best copy is
        # labelled best (more annotated classes, then more boxes). Test keeps the held-out
        # dataset's own copies (above).
        by_sub = defaultdict(list)
        for ids in candidates.values():
            for i in ids:
                by_sub[img_by_id[i].get("subscene_id", img_by_id[i]["group_id"])].append(i)
        cross_copies = set()
        for ids in by_sub.values():
            ds_of = {i: img_by_id[i]["source_dataset"] for i in ids}
            if len(set(ds_of.values())) > 1:
                keep_ds = ds_of[max(ids, key=keep_score)]
                kept = kept_with.get(keep_ds, frozenset({keep_ds}))
                cross_copies.update(i for i in ids if ds_of[i] not in kept)
        if cross_copies:
            candidates = {g: [i for i in ids if i not in cross_copies] for g, ids in candidates.items()}
            candidates = {g: ids for g, ids in candidates.items() if ids}

        use_proximity = embeddings is not None and proximity_filter is not None
        moved_matched = moved_near = 0
        val_in_clusters = 0
        val_report = None
        if val_ds:
            for boxed in candidates.values():
                train_pool.extend(pick(boxed, max_per_group))
        else:
            # Val = ~val_fraction of every training dataset, in whole scene groups (a group is
            # either train or val), so val covers every dataset and every class. Val images are
            # selected like test (max_test_per_group per sub-scene). Only a group with no near-copy
            # (any image pair >= proximity_filter, e.g. neighbouring frames of a video) in any other
            # group can go to val (pick_val_groups), so val starts with no near-copy in train.
            as_train = {g: pick(boxed, max_per_group) for g, boxed in candidates.items()}
            as_val = {g: pick(boxed, max_test_per_group) for g, boxed in candidates.items()}
            clusters = (near_copy_clusters({g: sorted(set(as_train[g]) | set(as_val[g])) for g in as_train},
                                           img_to_group, embeddings, proximity_filter) if use_proximity else None)
            # A val image with a photo-matched copy in training (scripts/fold_geometric_filter.py,
            # an earlier pass) moves to train with its whole cluster, instead of the training image
            # being dropped: the scene stays on one side and no training data is lost. Such clusters
            # are not picked for val again; other clusters of the dataset refill val to val_fraction.
            move = (val_to_train_files or {}).get(fold_id, set())
            matched = {clusters[g] if clusters else g for g in candidates
                       if any(img_by_id[i]["file_name"] in move for i in candidates[g])}
            # Sizes without the photo-matched copies of test (dropped below), so the shares are final.
            drop = (train_drop_files or {}).get(fold_id, set())

            def per_ds(ids):
                return Counter(img_by_id[i]["source_dataset"] for i in ids if img_by_id[i]["file_name"] not in drop)

            def per_class(ids):
                c = Counter()
                for i in ids:
                    if img_by_id[i]["file_name"] not in drop:
                        c.update(box_classes[i])
                return c

            val_groups, val_report = pick_val_groups(
                {g: per_ds(ids) for g, ids in as_train.items()}, {g: per_ds(ids) for g, ids in as_val.items()},
                {g: per_class(ids) for g, ids in as_val.items()}, val_fraction, clusters, excluded=matched,
                salt=f"{seed}:{held_out}")
            short = {c: v for c, v in val_report["classes"].items() if v[0] < 50 or v[1] < 15}
            if short:
                logger.warning(f"Fold {fold_id} ({held_out}): classes short in val (boxes, groups) "
                               f"with no eligible group left: {short}")
            if clusters:
                cluster_n = Counter(clusters.values())
                val_in_clusters = sum(cluster_n[clusters[g]] > 1 for g in val_groups)
            val_by_group = {g: as_val[g] for g in sorted(val_groups)}
            moved_matched = sum((clusters[g] if clusters else g) in matched for g in candidates)
            train_pool = [i for g, ids in as_train.items() if g not in val_by_group for i in ids]
            # DINOv2 near-copies (>= proximity_filter): none by construction of the clusters; kept as
            # a safety net (repeat, since a moved group can be close to another val group).
            while use_proximity and val_by_group and train_pool:
                v_ids = [i for ids in val_by_group.values() for i in ids]
                best = np.zeros(len(v_ids))
                for start in range(0, len(train_pool), 4096):
                    chunk = embeddings[train_pool[start:start + 4096]]
                    best = np.maximum(best, (embeddings[v_ids] @ chunk.T).max(axis=1))
                near = {img_to_group[i] for i, s in zip(v_ids, best) if s >= proximity_filter}
                if not near:
                    break
                for g in near:
                    del val_by_group[g]
                    train_pool.extend(as_train[g])
                moved_near += len(near)
            val_ids = [i for ids in val_by_group.values() for i in ids]
        train_ids = train_pool

        # Copies that no grouping rule links (e.g. re-exports rotated a few degrees with black
        # corners and recoloured, under a new name) score 0.80-0.90. Drop any training image that
        # close to a test image (or to a whole-dataset val) of this fold. A per-fold filter, so it
        # cannot chain groups.
        dropped_near = 0
        held_ids = test_ids + (val_ids if val_ds else [])
        if use_proximity and train_ids and held_ids:
            held = embeddings[held_ids]
            keep = []
            for start in range(0, len(train_ids), 2048):
                chunk = train_ids[start:start + 2048]
                near = (embeddings[chunk] @ held.T).max(axis=1) >= proximity_filter
                keep.extend(i for i, n in zip(chunk, near) if not n)
            dropped_near = len(train_ids) - len(keep)
            train_ids = keep

        # Copies of test images found by photo matching (scripts/fold_geometric_filter.py) below the
        # cosine filter, in train or (for a val slice) in val.
        drop = (train_drop_files or {}).get(fold_id, set())
        dropped_matched = sum(img_by_id[i]["file_name"] in drop for i in train_ids + val_ids)
        train_ids = [i for i in train_ids if img_by_id[i]["file_name"] not in drop]
        val_ids = [i for i in val_ids if img_by_id[i]["file_name"] not in drop]

        if len(test_ids) < 10:
            logger.warning(
                f"Fold {fold_id} ({held_out}): only {len(test_ids)} "
                f"test images — results may be unreliable"
            )

        fold = {
            "fold_id": fold_id,
            "held_out_dataset": held_out,
            "val_dataset": val_ds,
            "train_image_ids": train_ids,
            "val_image_ids": val_ids,
            "test_image_ids": test_ids,
            "train_dropped_near_test_or_val": dropped_near,
            "train_dropped_photo_match": dropped_matched,
            "val_groups_moved_to_train": {"photo_match": moved_matched, "near_copy": moved_near},
            "val_groups_in_multi_group_clusters": val_in_clusters,
            "train_dropped_cross_dataset_copies": len(cross_copies),
            "val_selection": val_report,
            "excluded_from_train": sorted({held_out, val_ds, *sidelined} - {None}),
        }
        _check_fold(fold, img_to_group, img_by_id)
        folds.append(fold)

        logger.info(
            f"Fold {fold_id} ({held_out}): "
            f"train={len(train_ids)}, val={len(val_ids)}{f' ({val_ds})' if val_ds else ''}, "
            f"test={len(test_ids)} ({excluded} images of held-out scenes excluded, "
            f"{dropped_near} training images dropped as near-copies of test/val, "
            f"{dropped_matched} as photo-matched copies of test, "
            f"{len(cross_copies)} copies of a photo kept from another training dataset; "
            f"val groups moved to train: {moved_matched} photo-matched, {moved_near} near-copies; "
            f"{val_in_clusters} val groups in near-copy clusters of several groups; "
            f"{sidelined_n} images of {sorted(sidelined)} held out with the test/val dataset; "
            f"{dropped_empty} images without boxes not used)"
        )

    return folds
