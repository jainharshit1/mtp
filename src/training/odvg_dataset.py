"""ODVG dataset with annotated_classes mask for loss masking."""

import io
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torchvision.transforms.functional as F

from src.data.taxonomy import NUM_CLASSES

logger = logging.getLogger(__name__)


def source_dataset(filename: str) -> str:
    """Dataset of an image in a portable build: the folder under images/ (images/<dataset>/...)."""
    parts = Path(filename).parts
    return parts[1] if len(parts) > 2 and parts[0] == "images" else parts[0]


def repeat_factors(lines: list[str], cap: float = 3.0) -> list[float]:
    """How often each training image is drawn relative to an average one (repeat-factor sampling).

    Per image: max(1, sqrt(t_ds / f_ds), sqrt(t_c / f_c) for each class c in the image), capped at
    `cap`. f_ds is the share of the training images from the image's dataset, t_ds an equal share
    (1 / number of datasets); f_c the share of images containing class c, t_c = 1 / number of
    classes. Images of small datasets or with rare classes are drawn up to `cap` times as often.
    """
    recs = [json.loads(line) for line in lines]
    ds = [source_dataset(r["filename"]) for r in recs]
    cls = [{i["label"] for i in r["detection"]["instances"] if i["label"] < NUM_CLASSES} for r in recs]
    n = len(recs)
    f_ds = Counter(ds)
    f_cl = Counter(c for s in cls for c in s)
    t_ds, t_cl = 1 / len(f_ds), 1 / max(1, len(f_cl))
    out = []
    for d, s in zip(ds, cls):
        r = max([1.0, math.sqrt(t_ds / (f_ds[d] / n))] + [math.sqrt(t_cl / (f_cl[c] / n)) for c in s])
        out.append(min(cap, r))
    return out


def balance_report(lines: list[str], factors: list[float], class_names: list[str] = None) -> str:
    """Share of each dataset and class among the images drawn per epoch, without -> with the factors."""
    recs = [json.loads(line) for line in lines]
    ds = [source_dataset(r["filename"]) for r in recs]
    cls = [{i["label"] for i in r["detection"]["instances"] if i["label"] < NUM_CLASSES} for r in recs]
    n, m = len(recs), sum(factors)
    d0, d1, c0, c1 = Counter(), Counter(), Counter(), Counter()
    for d, s, r in zip(ds, cls, factors):
        d0[d] += 1
        d1[d] += r
        for c in s:
            c0[c] += 1
            c1[c] += r
    name = (lambda c: class_names[c]) if class_names else str
    return ("datasets: " + ", ".join(f"{d} {d0[d] / n:.0%}->{d1[d] / m:.0%}" for d, _ in d0.most_common())
            + " | classes (images containing): "
            + ", ".join(f"{name(c)} {c0[c] / n:.0%}->{c1[c] / m:.0%}" for c, _ in c0.most_common())
            + f" | images with extra draws and light/colour changes: {sum(r > 1 for r in factors)}/{n}")


class PhotometricJitter:
    """Random light, colour and image-quality changes for oversampled training images (no geometry:
    boxes stay valid), RandAugment-style: every draw applies `ops_per_image` [min, max] changes chosen
    at random from `ops`. A graded change is [no change, low, high]: the direction (towards low or
    high; a one-sided range has one) is a coin flip, and the strength a share u of the way to that
    limit with u ~ Beta(strength_beta, 1): strong changes are the most likely, near "no change" the
    least (user, 2026-10-30: not a bell curve around no change). An on/off change is `true`; `false`
    or a missing entry disables a change.
    """

    DEFAULTS = {
        "ops_per_image": [3, 6],
        "strength_beta": 3.0,
        "ops": {
            "brightness": [1.0, 0.3, 2.0],
            "contrast": [1.0, 0.3, 1.6],
            "saturation": [1.0, 0.0, 1.8],
            "gamma": [1.0, 0.4, 2.5],
            "hue": [0.0, -0.5, 0.5],           # share of the colour wheel
            "tint": [0.0, -0.4, 0.4],          # > 0 warmer (red up, blue down), < 0 cooler
            "blur_sharpen": [0.0, -3.0, 3.0],  # < 0 Gaussian blur radius, > 0 sharpen amount
            "motion_blur": [0, 0, 15],         # camera shake: streak length in pixels
            "noise": [0, 0, 12],               # sensor noise: std in grey levels
            "jpeg": [100, 10, 100],            # compression quality
            "posterize": [8, 3, 8],            # bits per colour channel
            "shadow": [0.0, 0.0, 0.7],         # light-to-dark gradient: darkest side loses this share
            "haze": [0.0, 0.0, 0.5],           # blend with a light grey veil
            "greyscale": True,
            "equalize": [0.0, 0.0, 0.3],       # histogram equalisation, blended in up to this share
            "channel_shuffle": True,           # swap the colour channels
        },
    }

    # The defect must stay visible: inside every box, the fine detail (grey image minus a blurred copy,
    # which removes slow gradients such as shadow, haze or brightness) must still correlate with the
    # original's detail by at least MIN_DETAIL_CORR. A draw that fails (defect washed out, drowned in
    # noise or posterized away) is redrawn, up to TRIES times; then the original image is used.
    # (A plain contrast check let a strong shadow pass that had erased an AGDD scratch, 2026-10-30.)
    # The whole image must keep its texture too (MIN_IMAGE_CORR): equalisation of a flat panel plus noise
    # had turned a fuselage image into blotches while its tiny boxes still passed (user, 2026-10-30).
    # And every defect must still stand out from the background: (detail spread in the box / detail
    # spread of the whole image) must keep MIN_STANDOUT of its original value. Contrast, saturation and
    # equalisation of a flat panel amplified its faint texture until it drowned the scratches, while the
    # pattern checks above still passed (fuselage 196.jpg, 2026-10-30).
    MIN_DETAIL_CORR, MIN_IMAGE_CORR, MIN_STANDOUT, TRIES = 0.4, 0.5, 0.6, 5
    # Light/colour changes first, image-quality changes last: noise is never amplified by a later
    # contrast or equalisation step.
    ORDER = ("brightness", "contrast", "gamma", "equalize", "shadow", "haze", "saturation", "hue", "tint",
             "greyscale", "channel_shuffle", "blur_sharpen", "motion_blur", "posterize", "jpeg", "noise")

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = {**self.DEFAULTS, **{k: v for k, v in cfg.items() if k != "ops"}}
        self.ops = {**self.DEFAULTS["ops"], **cfg.get("ops", {})}
        self.ops = {k: v for k, v in self.ops.items() if v is not False and v is not None}

    def _draw(self, key):
        none, lo, hi = self.ops[key]
        u = random.betavariate(self.cfg["strength_beta"], 1)
        up = hi != none and (lo == none or random.random() < 0.5)
        return none + u * (hi - none) if up else none - u * (none - lo)

    @staticmethod
    def _detail(img: Image.Image) -> np.ndarray:
        grey = img.convert("L")
        return (np.asarray(grey, dtype=np.float32)
                - np.asarray(grey.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32))

    @staticmethod
    def _crops(a: np.ndarray, boxes):
        return [a[int(y0):max(int(y0) + 2, int(round(y1))), int(x0):max(int(x0) + 2, int(round(x1)))]
                for x0, y0, x1, y1 in boxes]

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray):
        a, b = a - a.mean(), b - b.mean()
        na, nb = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
        if na < 1e-3:
            return None  # no detail in the original to lose
        return 0.0 if nb < 1e-3 else float((a * b).sum()) / (na * nb)

    def _visible(self, before: list, before_img: np.ndarray, out: Image.Image, boxes) -> bool:
        detail = self._detail(out)
        c = self._corr(before_img, detail)
        if c is not None and c < self.MIN_IMAGE_CORR:
            return False
        s0, s1 = float(before_img.std()) + 1e-3, float(detail.std()) + 1e-3
        for a, b in zip(before, self._crops(detail, boxes)):
            c = self._corr(a, b)
            if c is not None and c < self.MIN_DETAIL_CORR:
                return False
            if c is not None and (float(b.std()) / s1) < self.MIN_STANDOUT * (float(a.std()) / s0):
                return False
        return True

    def __call__(self, img: Image.Image, boxes=()) -> Image.Image:
        detail = self._detail(img)
        before = self._crops(detail, boxes)
        for _ in range(self.TRIES):
            out = self._jitter(img)
            if self._visible(before, detail, out, boxes):
                return out
        return img

    def _jitter(self, img: Image.Image) -> Image.Image:
        lo, hi = self.cfg["ops_per_image"]
        names = list(self.ops)
        chosen = set(random.sample(names, min(len(names), random.randint(lo, hi))))
        for name in sorted(chosen, key=lambda n: self.ORDER.index(n) if n in self.ORDER else len(self.ORDER)):
            img = getattr(self, "_" + name)(img, None if self.ops[name] is True else self._draw(name))
        return img.convert("RGB")

    # --- the changes (image, strength) -> image ---
    def _brightness(self, img, v):
        return ImageEnhance.Brightness(img).enhance(v)

    def _contrast(self, img, v):
        return ImageEnhance.Contrast(img).enhance(v)

    def _saturation(self, img, v):
        return ImageEnhance.Color(img).enhance(v)

    def _gamma(self, img, v):
        lut = [min(255, int(255 * (i / 255) ** v + 0.5)) for i in range(256)]
        return img.point(lut * 3)

    def _hue(self, img, v):
        h, s, val = img.convert("HSV").split()
        shift = int(round(v * 255))
        return Image.merge("HSV", (h.point(lambda x: (x + shift) % 256), s, val)).convert("RGB")

    def _tint(self, img, v):
        r, g, b = img.split()
        return Image.merge("RGB", (r.point(lambda x: min(255, int(x * (1 + v)))), g,
                                   b.point(lambda x: min(255, int(x * (1 - v))))))

    def _blur_sharpen(self, img, v):
        return img.filter(ImageFilter.GaussianBlur(-v)) if v < 0 else ImageEnhance.Sharpness(img).enhance(1 + v)

    def _motion_blur(self, img, v):
        n = max(2, int(round(v)))
        a = np.asarray(img, dtype=np.float32)
        ang = random.uniform(0, math.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        acc = np.zeros_like(a)
        for i in range(n):
            t = i - (n - 1) / 2
            acc += np.roll(a, (int(round(t * dy)), int(round(t * dx))), axis=(0, 1))
        return Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8))

    def _noise(self, img, v):
        a = np.asarray(img, dtype=np.float32)
        rng = np.random.default_rng(random.getrandbits(32))  # seeded per data worker via `random`
        return Image.fromarray(np.clip(a + rng.normal(0, v, a.shape), 0, 255).astype(np.uint8))

    def _jpeg(self, img, v):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(round(v)))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    def _posterize(self, img, v):
        return ImageOps.posterize(img, max(1, int(round(v))))

    def _shadow(self, img, v):
        w, h = img.size
        ang = random.uniform(0, 2 * math.pi)
        xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        ramp = (xs * math.cos(ang) + ys * math.sin(ang) + 1.415) / 2.83  # 0..1 across the image
        a = np.asarray(img, dtype=np.float32) * (1 - v * ramp)[..., None]
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    def _haze(self, img, v):
        veil = Image.new("RGB", img.size, (random.randint(170, 235),) * 3)
        return Image.blend(img, veil, v)

    def _greyscale(self, img, _):
        return img.convert("L").convert("RGB")

    def _equalize(self, img, v):
        return Image.blend(img, ImageOps.equalize(img), 1.0 if v is None else v)

    def _channel_shuffle(self, img, _):
        bands = list(img.split())
        random.shuffle(bands)
        return Image.merge("RGB", bands)


class MaskedODVGDataset(Dataset):
    def __init__(self, odvg_path: str, prompt_builder, transforms=None, max_text_len: int = 256):
        self.prompt_builder = prompt_builder
        self.transforms = transforms
        self.max_text_len = max_text_len
        self.caption, self.cat_list = prompt_builder.build_caption_and_cat_list()
        # Relative image paths are relative to the folder holding fold_*/ (a self-contained build).
        self.root = Path(odvg_path).resolve().parent.parent
        # Light/colour changes for oversampled images (set_jitter; training only).
        self.jitter, self.jitter_index = None, set()

        self.lines = []
        with open(odvg_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
        logger.info(f"Loaded {len(self.lines)} samples from {odvg_path}")

    def set_jitter(self, jitter: PhotometricJitter, indices) -> None:
        """Apply `jitter` (fresh random strengths on every draw) to the images at `indices`."""
        self.jitter, self.jitter_index = jitter, set(indices)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        try:
            return self._load_sample(index)
        except Exception as e:
            logger.warning(f"Failed to load sample {index}: {e}, returning random other sample")
            return self._load_sample(torch.randint(0, len(self), (1,)).item())

    def _load_sample(self, index):
        data = json.loads(self.lines[index])

        img = Image.open(self.root / data["filename"]).convert("RGB")
        instances = data["detection"]["instances"]
        if self.jitter is not None and index in self.jitter_index:
            img = self.jitter(img, [inst["bbox"] for inst in instances])
        w, h = img.size

        boxes = []
        labels = []
        for inst in instances:
            bbox = inst["bbox"]  # [x1, y1, x2, y2] absolute
            label = inst["label"]
            if label >= NUM_CLASSES:
                continue
            boxes.append(bbox)
            labels.append(label)

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        ann_classes = data.get("annotated_classes", list(range(NUM_CLASSES)))
        annotated_mask = torch.zeros(NUM_CLASSES, dtype=torch.bool)
        for c in ann_classes:
            if c < NUM_CLASSES:
                annotated_mask[c] = True

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "caption": self.caption,
            "cat_list": self.cat_list,
            "annotated_classes": annotated_mask,
            "orig_size": torch.as_tensor([h, w]),
            "size": torch.as_tensor([h, w]),
            "image_id": torch.tensor([index]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target


def make_odvg_transforms(image_set: str, image_size: int = 800, fixed_square: bool = False,
                         pad_mask: bool = False):
    """Build transforms compatible with GroundingDINO expectations.

    fixed_square: pad each image on the right/bottom to a square, then resize to
    image_size x image_size, so every batch has one shape (no padding inside batches, cuDNN tunes
    once). Boxes keep their pixel coordinates through the padding and are scaled with the resize.
    Otherwise: short side = image_size, long side <= 1333 (shapes vary).
    pad_mask (with fixed_square): record the real image area (target["valid_size"]) so the batch
    mask marks the square padding as padding, and normalise boxes by that area (the DETR
    convention: predictions are relative to the real image, not the padded canvas).
    """
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    sizing = SquareResize(image_size, pad_mask) if fixed_square else RandomResize([image_size], max_size=1333)

    if image_set == "train":
        return Compose([RandomHorizontalFlip(), sizing, normalize])
    elif image_set in ("val", "test"):
        return Compose([sizing, normalize])
    raise ValueError(f"Unknown image_set: {image_set}")


def prepare_eval_image(image, image_size: int, fixed_square: bool):
    """Resize a PIL image exactly like the val transform (before normalisation)."""
    if fixed_square:
        return square_resize(image, {}, image_size)[0]
    return resize(image, {}, image_size, 1333)[0]


def box_frame(width: int, height: int, fixed_square: bool, pad_mask: bool) -> tuple[float, float]:
    """(x, y) scale that turns the model's normalised boxes into original-image pixels.

    With fixed_square and no pad mask the boxes are relative to the padded square (side
    max(w, h)); with a pad mask, or without fixed_square, relative to the image itself.
    """
    if fixed_square and not pad_mask:
        return (max(width, height),) * 2
    return width, height


# Minimal transform classes that handle both image and target
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        h, w = image.shape[-2:]
        # With a padding mask, boxes are relative to the real (unpadded) image area.
        bh, bw = target["valid_size"].tolist() if "valid_size" in target else (h, w)
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"]
            # Convert xyxy absolute to cxcywh normalized
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([bw, bh, bw, bh], dtype=torch.float32)
            target["boxes"] = boxes
        target["size"] = torch.as_tensor([h, w])
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if "boxes" in target and target["boxes"].numel() > 0:
                w = image.width if hasattr(image, 'width') else image.shape[-1]
                boxes = target["boxes"]
                boxes = boxes.clone()
                # Flip x1 and x2: new_x1 = w - old_x2, new_x2 = w - old_x1
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target):
        size = self.sizes[torch.randint(len(self.sizes), (1,)).item()]
        image, target = resize(image, target, size, self.max_size)
        return image, target


def resize(image, target, size, max_size=None):
    import torchvision.transforms.functional as F
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        w, h = image.size
    else:
        h, w = image.shape[-2:]

    if max_size is not None:
        min_orig = float(min(h, w))
        max_orig = float(max(h, w))
        if max_orig / min_orig * size > max_size:
            size = int(round(max_size * min_orig / max_orig))

    if h <= w:
        new_h = size
        new_w = int(round(size * w / h))
    else:
        new_w = size
        new_h = int(round(size * h / w))

    if isinstance(image, PILImage.Image):
        image = F.resize(image, (new_h, new_w))
    else:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)

    if "boxes" in target and target["boxes"].numel() > 0:
        ratio_w = new_w / w
        ratio_h = new_h / h
        boxes = target["boxes"]
        boxes = boxes * torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h], dtype=torch.float32)
        target["boxes"] = boxes

    target["size"] = torch.as_tensor([new_h, new_w])
    return image, target


class SquareResize:
    def __init__(self, size, pad_mask=False):
        self.size = size
        self.pad_mask = pad_mask

    def __call__(self, image, target):
        w, h = image.size
        image, target = square_resize(image, target, self.size)
        if self.pad_mask:
            target["valid_size"] = torch.as_tensor(square_valid_size(w, h, self.size), dtype=torch.float32)
        return image, target


def square_valid_size(w: int, h: int, size: int) -> tuple[float, float]:
    """(height, width) of the real image inside the size x size square (the rest is padding).

    Exact (not rounded), so boxes normalised by it map back to original pixels exactly; the mask
    rounds it to whole pixels."""
    s = size / max(w, h)
    return h * s, w * s


class _SimpleNested:
    """Stand-in for util.misc.NestedTensor when the vendor package is not importable."""

    def __init__(self, tensors, mask):
        self.tensors, self.mask = tensors, mask

    def to(self, device, **kw):
        return _SimpleNested(self.tensors.to(device, **kw), self.mask.to(device, **kw))

    def decompose(self):
        return self.tensors, self.mask


def make_nested(images, valid_sizes=None):
    """Batch images (zero-padded to the largest) with a mask that is True on padding.

    Without valid_sizes this is nested_tensor_from_tensor_list: only batch padding is masked.
    With valid_sizes ([h, w] per image, from SquareResize with pad_mask) the square padding inside
    each image is masked too, so position encodings, attention and query selection ignore it.
    """
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)
    batch = torch.zeros(len(images), images[0].shape[0], max_h, max_w,
                        dtype=images[0].dtype, device=images[0].device)
    mask = torch.ones(len(images), max_h, max_w, dtype=torch.bool, device=images[0].device)
    for i, img in enumerate(images):
        h, w = img.shape[-2:]
        batch[i, :, :h, :w] = img
        vh, vw = (max(1, round(float(v))) for v in valid_sizes[i]) if valid_sizes and valid_sizes[i] is not None else (h, w)
        mask[i, :vh, :vw] = False
    try:
        from util.misc import NestedTensor
        return NestedTensor(batch, mask)
    except ImportError:
        return _SimpleNested(batch, mask)


# ImageNet mean: the padding normalises to ~0, like the zero padding of a batch.
PAD_FILL = (124, 116, 104)


def square_resize(image, target, size):
    """Pad a PIL image on the right/bottom to a square, then resize to size x size.

    Padding only on the right/bottom leaves every box's pixel coordinates unchanged; the resize
    scales them by size / side.
    """
    from PIL import Image as PILImage

    w, h = image.size
    side = max(w, h)
    if (w, h) != (side, side):
        canvas = PILImage.new("RGB", (side, side), PAD_FILL)
        canvas.paste(image, (0, 0))
        image = canvas
    image = image.resize((size, size), PILImage.BILINEAR)

    if "boxes" in target and target["boxes"].numel() > 0:
        target["boxes"] = target["boxes"] * (size / side)
    target["size"] = torch.as_tensor([size, size])
    return image, target


def box_xyxy_to_cxcywh(boxes):
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def collate_fn(batch):
    """Custom collate that handles variable-size targets."""
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets
