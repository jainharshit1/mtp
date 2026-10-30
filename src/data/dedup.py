"""
Image deduplication and scene grouping.

Two outputs:
  1. Exact dedup (SHA256): byte-identical copies collapse to one image.
  2. Scene groups: every copy of one photo (Roboflow augmentations, flips, 90-degree
     rotations, crops, re-exports in another dataset) gets the same group_id. LODO splits
     assign whole groups, so one scene can never sit on both sides of a split.

Two images are linked if ANY rule below fires; groups are the connected components of
the links (union-find), so A~B and B~C puts A, B and C in one group:
  - same Roboflow source name within one dataset (`IMG_12_jpg.rf.<hash>.jpg` -> `img_12`).
    Roboflow keeps it on every augmented copy, so this catches any augmentation. Datasets
    exported from the same photo pool (shared_name_groups) are matched across each other too.
    Some datasets name their own variants of one photo (source_name_patterns: korean-aerospace
    `105_14` = photo 105, variant 14; all variants of a number are the same photo, recoloured,
    darkened or inverted: visual check 2026-10-30).
  - pHash distance <= PHASH_THRESHOLD under the best of the 8 flips/90-degree rotations.
    Plain pHash puts a flipped or rotated copy ~32 of 64 bits away, i.e. "unrelated".
    With embeddings on, the pair must also reach PHASH_MIN_COSINE (rejects chance matches).
  - same photo by local-feature matching (geometric_links, from scripts/geometric_links.py).
  - DINOv2 embedding cosine >= embed_threshold. Catches mild crops (about two thirds of 80%
    crops) but few heavy crops or rotations, and chains consecutive video frames together.

Each image also gets a subscene_id: the component under the first three rules only. Groups
decide which side of a split an image is on; sub-scenes are what the copy caps count.
"""

import hashlib
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.fftpack
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

PHASH_THRESHOLD = 8
PHASH_MIN_COSINE = 0.7  # a pHash link also needs this DINOv2 cosine (when embeddings are on)
EMBED_MODEL = "facebook/dinov2-small"
EMBED_THRESHOLD = 0.90

_RF_STEM = re.compile(r"^(.*)\.rf\.[0-9A-Za-z]+$")
_EXT_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp|tiff?)$", re.IGNORECASE)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def roboflow_source(path: str) -> str | None:
    """`IMG_12_jpg.rf.<hash>.jpg` -> `img_12`; None for files not exported by Roboflow."""
    m = _RF_STEM.match(Path(path).stem)
    return _EXT_SUFFIX.sub("", m.group(1)).lower() if m else None


def dihedral_phash(path: str) -> np.ndarray | None:
    """pHash of the image under all 8 flips/90-degree rotations, as uint64[8].

    Index 0 is the image as stored and equals `imagehash.phash`. Orientations are applied to
    the 32x32 thumbnail, which matches transforming the full image to within a bit or two.
    """
    try:
        with PILImage.open(path) as img:
            px = np.asarray(img.convert("L").resize((32, 32), PILImage.Resampling.LANCZOS), dtype=np.float64)
    except Exception as e:
        logger.warning(f"pHash failed for {path}: {e}")
        return None
    hashes = []
    for k in range(4):
        rot = np.rot90(px, k)
        for m in (rot, rot[:, ::-1]):
            low = scipy.fftpack.dct(scipy.fftpack.dct(m, axis=0), axis=1)[:8, :8]
            hashes.append(int.from_bytes(np.packbits(low > np.median(low)).tobytes(), "big"))
    return np.array(hashes, dtype=np.uint64)


def compute_phashes(paths: list[str], workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Returns (hashes uint64[N, 8], valid bool[N])."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(dihedral_phash, paths, chunksize=64))
    valid = np.array([r is not None for r in results])
    hashes = np.stack([r if r is not None else np.zeros(8, dtype=np.uint64) for r in results])
    return hashes, valid


def phash_pairs(hashes: np.ndarray, valid: np.ndarray, threshold: int = PHASH_THRESHOLD, chunk: int = 64):
    """Yield (i, j) with min-over-orientations Hamming distance <= threshold."""
    idx = np.flatnonzero(valid)
    h = hashes[idx]
    for start in range(0, len(idx), chunk):
        a = h[start:start + chunk, 0][:, None, None]
        dist = np.bitwise_count(a ^ h[None]).min(axis=2)
        ii, jj = np.nonzero(dist <= threshold)
        ii += start
        keep = ii != jj
        yield from zip(idx[ii[keep]].tolist(), idx[jj[keep]].tolist())


class _ImageDataset:
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with PILImage.open(self.paths[i]) as img:
                return self._pixels(img.convert("RGB")), True
        except Exception:
            return self._pixels(PILImage.new("RGB", (224, 224))), False

    def _pixels(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def compute_embeddings(paths: list[str], model_name: str = EMBED_MODEL, device: str | None = None,
                       batch_size: int = 64, workers: int = 8):
    """L2-normalised DINOv2 CLS embeddings, averaged with the horizontal flip. Returns (emb[N, D], valid[N])."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    loader = DataLoader(_ImageDataset(paths, processor), batch_size=batch_size, num_workers=workers)

    embs, valid = [], []
    with torch.no_grad():
        for x, ok in loader:
            x = x.to(device)
            f = model(pixel_values=x).pooler_output + model(pixel_values=x.flip(-1)).pooler_output
            f = F.normalize(f.float(), dim=-1)
            f[~ok.to(device)] = 0
            embs.append(f.cpu())
            valid.append(ok)
    return torch.cat(embs), torch.cat(valid).numpy()


def embedding_pairs(emb, threshold: float = EMBED_THRESHOLD, chunk: int = 1024):
    """Yield (i, j) with cosine similarity >= threshold."""
    import torch

    for start in range(0, len(emb), chunk):
        sim = emb[start:start + chunk] @ emb.T
        ii, jj = torch.nonzero(sim >= threshold, as_tuple=True)
        ii += start
        keep = ii != jj
        yield from zip(ii[keep].tolist(), jj[keep].tolist())


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def build_image_registry(
    ingested_datasets: list[dict],
    phash_dedup: bool = True,
    phash_threshold: int = PHASH_THRESHOLD,
    embed_model: str | None = EMBED_MODEL,
    embed_threshold: float = EMBED_THRESHOLD,
    device: str | None = None,
    workers: int = 8,
    embed_cache: dict | None = None,
    phash_min_cosine: float = PHASH_MIN_COSINE,
    cross_dataset_embed_threshold: float | None = None,
    shared_name_groups: list[list[str]] | None = None,
    geometric_links: list[tuple[str, str]] | None = None,
    source_name_patterns: dict[str, str] | None = None,
) -> dict:
    """
    Build image registry with scene grouping.

    phash_dedup=False disables the perceptual rules (pHash and embeddings); embed_model=None
    disables only the embedding rule. The Roboflow-name rule always runs. embed_cache
    ({file path: embedding}, from scripts/00_embed_images.py) replaces computing the embeddings.

    Returns:
        {
            "images": [all images with group_id, subscene_id, sha256 and contiguous new IDs],
            "old_to_new_id": {(dataset_name, old_img_id): new_img_id},
            "groups": {group_id: {"source_datasets": set, "image_ids": [new_ids]}},
            "stats": {...},
            "embeddings": float32[n_images, D] indexed by image id, or None
        }
    """
    logger.info("Pass 1: Computing SHA256 hashes for exact dedup...")
    sha_groups = defaultdict(list)
    for ds in ingested_datasets:
        for img in ds["images"]:
            sha = _sha256(img["file_name"])
            sha_groups[sha].append((ds["name"], img))

    total_input = sum(len(ds["images"]) for ds in ingested_datasets)
    exact_dups = total_input - len(sha_groups)
    logger.info(f"  {total_input} images → {len(sha_groups)} unique by SHA256 ({exact_dups} exact duplicates)")

    sha_list = list(sha_groups.keys())
    paths = [sha_groups[sha][0][1]["file_name"] for sha in sha_list]
    uf = _UnionFind(len(sha_list))
    links = Counter()

    logger.info("Pass 2: Linking Roboflow augmentations by source name...")
    name_space = {ds: "+".join(sorted(grp)) for grp in (shared_name_groups or []) for ds in grp}
    # {dataset: regex}: the first group of the source name is the photo (korean `105_14` -> `105`).
    photo_of = {ds: re.compile(p) for ds, p in (source_name_patterns or {}).items()}
    first_with_source = {}
    for i, sha in enumerate(sha_list):
        for ds_name, img in sha_groups[sha]:
            src = roboflow_source(img["file_name"])
            if src is None:
                continue
            if ds_name in photo_of and (m := photo_of[ds_name].match(src)):
                src = m.group(1)
            # Datasets exported from the same photo pool share a name space (shared_name_groups).
            j = first_with_source.setdefault((name_space.get(ds_name, ds_name), src), i)
            links["roboflow_name"] += uf.union(i, j)

    emb = None
    if phash_dedup and embed_model and embed_cache is not None:
        missing = [p for p in paths if p not in embed_cache]
        if missing:
            raise KeyError(f"{len(missing)} images missing from the embedding cache (first: {missing[0]})")
        logger.info(f"Using cached {embed_model} embeddings for {len(paths)} images")
        import torch
        emb = torch.from_numpy(np.stack([embed_cache[p] for p in paths]))
    elif phash_dedup and embed_model:
        logger.info(f"Computing {embed_model} embeddings (used by passes 3 and 4)...")
        emb, _ = compute_embeddings(paths, embed_model, device, workers=workers)

    if phash_dedup:
        logger.info("Pass 3: Computing pHash under 8 orientations...")
        hashes, valid = compute_phashes(paths, workers)
        for i, j in phash_pairs(hashes, valid, phash_threshold):
            # Low-texture images (plain panels) match unrelated photos under one of the 8
            # orientations. On all 8 datasets, pHash pairs below 0.7 cosine were mostly different
            # photos and chained 5,189 images; heavily recoloured real copies reach down to ~0.7.
            if emb is not None and float(emb[i] @ emb[j]) < phash_min_cosine:
                links["phash_rejected_by_embedding"] += 1
                continue
            links["phash_dihedral"] += uf.union(i, j)

    if geometric_links:
        # Same photo confirmed by local-feature matching (scripts/geometric_links.py): copies
        # re-exported by another dataset, rotated and recoloured, that no rule above links.
        index_of = {sha_groups[sha][k][1]["file_name"]: i for i, sha in enumerate(sha_list)
                    for k in range(len(sha_groups[sha]))}
        for a, b in geometric_links:
            if a in index_of and b in index_of:
                links["geometric"] += uf.union(index_of[a], index_of[b])

    # Sub-scenes: copies linked by the strong rules only (SHA, Roboflow name, pHash, geometric
    # matches). The embedding rule below can chain a whole video into one group; the group
    # decides the split side, while copy caps apply per sub-scene.
    strong_root = [uf.find(i) for i in range(len(sha_list))]

    if emb is not None:
        logger.info("Pass 4: Linking by embedding similarity...")
        if cross_dataset_embed_threshold is None or cross_dataset_embed_threshold >= embed_threshold:
            for i, j in embedding_pairs(emb, embed_threshold):
                links["embedding"] += uf.union(i, j)
        else:
            # Copies re-exported by another dataset (small rotations with black corners, recolouring,
            # new file names) score below embed_threshold, so links between datasets use a lower one.
            import torch
            names = sorted({ds for sha in sha_list for ds, _ in sha_groups[sha]})
            ds_id = torch.tensor([names.index(sha_groups[sha][0][0]) for sha in sha_list])
            for start in range(0, len(emb), 1024):
                sim = emb[start:start + 1024] @ emb.T
                cross = ds_id[start:start + 1024, None] != ds_id[None, :]
                hit = (sim >= embed_threshold) | (cross & (sim >= cross_dataset_embed_threshold))
                ii, jj = torch.nonzero(hit, as_tuple=True)
                ii += start
                for i, j, c in zip(ii.tolist(), jj.tolist(), cross[ii - start, jj].tolist()):
                    if i < j:
                        links["embedding_cross_dataset" if c else "embedding"] += uf.union(i, j)
        emb = emb.numpy()
    logger.info(f"  Group merges by rule: {dict(links)}")

    members = defaultdict(list)
    for i in range(len(sha_list)):
        members[uf.find(i)].append(sha_list[i])
    phash_groups = {sha_list[root]: shas for root, shas in sorted(members.items())}
    logger.info(f"  {len(sha_groups)} SHA-unique → {len(phash_groups)} scene groups")

    sha_index = {sha: i for i, sha in enumerate(sha_list)}
    subscene_of_root = {}
    emb_rows = []  # embedding row per new image id

    old_to_new_id = {}
    all_images = []
    groups = {}
    new_img_id = 0

    for group_id, (canonical_sha, sha_members) in enumerate(phash_groups.items()):
        group_datasets = set()
        group_image_ids = []

        for sha in sha_members:
            seen_paths = set()
            for ds_name, img in sha_groups[sha]:
                if img["file_name"] in seen_paths:
                    old_to_new_id[(ds_name, img["id"])] = None
                    continue
                seen_paths.add(img["file_name"])

                old_to_new_id[(ds_name, img["id"])] = new_img_id
                group_datasets.add(ds_name)
                group_image_ids.append(new_img_id)

                all_images.append({
                    "id": new_img_id,
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "source_dataset": ds_name,
                    "group_id": group_id,
                    "subscene_id": subscene_of_root.setdefault(
                        strong_root[sha_index[sha]], len(subscene_of_root)),
                    "sha256": sha,
                })
                emb_rows.append(sha_index[sha])
                new_img_id += 1

        groups[group_id] = {
            "source_datasets": group_datasets,
            "image_ids": group_image_ids,
        }

    none_count = sum(1 for v in old_to_new_id.values() if v is None)
    if none_count:
        logger.info(f"  {none_count} exact-duplicate entries mapped to None (same path in same dataset)")

    sizes = sorted((len(g["image_ids"]) for g in groups.values()), reverse=True)
    # Transitive linking can chain a whole video or photo session into one group. That is
    # correct for leakage but costs data, so flag it for a visual check.
    if sizes and sizes[0] > max(50, 0.01 * len(all_images)):
        logger.warning(f"  Largest scene group has {sizes[0]} images; inspect it before trusting the threshold")

    stats = {
        "total_input_images": total_input,
        "unique_sha256": len(sha_groups),
        "perceptual_groups": len(phash_groups),
        "subscenes": len(subscene_of_root),
        "final_images": len(all_images),
        "exact_duplicates_removed": none_count,
        "cross_dataset_groups": sum(
            1 for g in groups.values() if len(g["source_datasets"]) > 1
        ),
        "group_merges_by_rule": dict(links),
        "largest_groups": sizes[:10],
        "phash_threshold": phash_threshold if phash_dedup else None,
        "phash_min_cosine": phash_min_cosine if phash_dedup and embed_model else None,
        "embed_model": embed_model if phash_dedup else None,
        "embed_threshold": embed_threshold if phash_dedup and embed_model else None,
        "cross_dataset_embed_threshold": cross_dataset_embed_threshold if phash_dedup and embed_model else None,
    }
    logger.info(
        f"Registry: {len(all_images)} images in {len(groups)} groups "
        f"({stats['cross_dataset_groups']} cross-dataset groups)"
    )

    return {
        "images": all_images,
        "old_to_new_id": old_to_new_id,
        "groups": groups,
        "stats": stats,
        # [n_images, D], row = image id; None when the embedding rule is off
        "embeddings": emb[emb_rows] if emb is not None else None,
    }
