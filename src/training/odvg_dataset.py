"""ODVG dataset with annotated_classes mask for loss masking."""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from src.data.taxonomy import NUM_CLASSES

logger = logging.getLogger(__name__)


class MaskedODVGDataset(Dataset):
    def __init__(self, odvg_path: str, prompt_builder, transforms=None, max_text_len: int = 256):
        self.prompt_builder = prompt_builder
        self.transforms = transforms
        self.max_text_len = max_text_len
        self.caption, self.cat_list = prompt_builder.build_caption_and_cat_list()

        self.lines = []
        with open(odvg_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)
        logger.info(f"Loaded {len(self.lines)} samples from {odvg_path}")

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

        img_path = data["filename"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        instances = data["detection"]["instances"]
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


def make_odvg_transforms(image_set: str, image_size: int = 800):
    """Build transforms compatible with GroundingDINO expectations."""
    import torchvision.transforms as T

    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if image_set == "train":
        return Compose([
            RandomHorizontalFlip(),
            RandomResize([image_size], max_size=1333),
            normalize,
        ])
    elif image_set in ("val", "test"):
        return Compose([
            RandomResize([image_size], max_size=1333),
            normalize,
        ])
    raise ValueError(f"Unknown image_set: {image_set}")


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
        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"]
            # Convert xyxy absolute to cxcywh normalized
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
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
