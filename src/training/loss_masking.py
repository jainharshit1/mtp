"""Loss masking for partial-annotation training with Grounding DINO.

Subclasses SetCriterion to inject an annotated_classes mask into the
focal classification loss. Token positions for unannotated classes
are zeroed so they contribute no gradient.

Box losses (L1, GIoU) are NOT masked — they only fire on matched
GT-prediction pairs, which by construction belong to annotated classes.
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.taxonomy import NUM_CLASSES

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "Open-GroundingDino"


def _ensure_vendor():
    v = str(VENDOR_DIR)
    if v not in sys.path:
        sys.path.insert(0, v)


def compute_class_token_spans(
    caption: str,
    cat_list: list[str],
    tokenizer,
) -> dict[int, tuple[int, int]]:
    """Map each class ID to its [start, end) token positions in the caption.

    Called once at setup. Result is cached and used by MaskedSetCriterion.
    """
    encoding = tokenizer(caption, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    class_to_span = {}

    for class_id, class_text in enumerate(cat_list):
        char_start = caption.find(class_text)
        if char_start == -1:
            raise ValueError(
                f"Class text '{class_text}' not found in caption: {caption}"
            )
        char_end = char_start + len(class_text)

        token_start = None
        token_end = None
        for tok_idx, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s < char_end and e > char_start:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx + 1

        if token_start is None:
            raise ValueError(
                f"Could not map class '{class_text}' to tokens in caption"
            )

        # Verify by decoding back
        decoded = tokenizer.decode(input_ids[token_start:token_end]).strip()
        clean_decoded = decoded.replace(" - ", "-").replace(" ##", "")
        if class_text not in decoded and class_text not in clean_decoded:
            logger.warning(
                f"Token decode mismatch for class {class_id} '{class_text}': "
                f"got '{decoded}'"
            )

        class_to_span[class_id] = (token_start, token_end)

    # Verify no overlapping spans
    all_positions = set()
    for cid, (s, e) in class_to_span.items():
        positions = set(range(s, e))
        overlap = all_positions & positions
        if overlap:
            raise ValueError(
                f"Class {cid} tokens overlap with another class at positions {overlap}"
            )
        all_positions.update(positions)

    logger.info(f"Token spans for {len(class_to_span)} classes:")
    for cid in sorted(class_to_span):
        s, e = class_to_span[cid]
        tokens = tokenizer.decode(input_ids[s:e])
        logger.info(f"  {cid} ({cat_list[cid]}): [{s},{e}) = '{tokens}'")

    return class_to_span


def build_annotated_token_mask(
    targets: list[dict],
    class_to_token_spans: dict[int, tuple[int, int]],
    max_text_len: int,
) -> torch.Tensor:
    """Build [batch, max_text_len] bool mask: True at annotated token positions."""
    bs = len(targets)
    mask = torch.zeros(bs, max_text_len, dtype=torch.bool)
    for b, tgt in enumerate(targets):
        ann = tgt["annotated_classes"]  # BoolTensor[NUM_CLASSES]
        for cid in range(NUM_CLASSES):
            if ann[cid] and cid in class_to_token_spans:
                s, e = class_to_token_spans[cid]
                if e <= max_text_len:
                    mask[b, s:e] = True
    return mask


class MaskedSetCriterion(nn.Module):
    """Wraps the vendor SetCriterion, injecting annotation masking into the focal loss.

    The approach: we override `token_sigmoid_binary_focal_loss` by monkey-patching
    the criterion instance. This is more robust than subclassing since the vendor
    code may change attributes after __init__.
    """

    def __init__(self, criterion, class_to_token_spans: dict[int, tuple[int, int]]):
        super().__init__()
        self.criterion = criterion
        self.class_to_token_spans = class_to_token_spans
        self._targets_ref = None  # set before each forward

        # Monkey-patch the focal loss method
        original_focal = criterion.token_sigmoid_binary_focal_loss

        def masked_focal_loss(outputs, targets, indices, num_boxes):
            """Focal loss with annotation masking injected."""
            pred_logits = outputs['pred_logits']
            new_targets = outputs['one_hot'].to(pred_logits.device)
            text_mask = outputs['text_mask']

            bs, n, max_text_len = pred_logits.shape
            alpha = criterion.focal_alpha
            gamma = criterion.focal_gamma

            # Build annotation mask [bs, max_text_len]
            ann_mask = build_annotated_token_mask(
                self._targets_ref, class_to_token_spans, max_text_len
            ).to(pred_logits.device)
            # Expand to [bs, n, max_text_len]
            ann_mask_expanded = ann_mask.unsqueeze(1).expand_as(pred_logits)

            # Combine text_mask (padding) AND annotation mask
            if text_mask is not None:
                combined_mask = text_mask.repeat(1, n).view(bs, -1, max_text_len) & ann_mask_expanded
            else:
                combined_mask = ann_mask_expanded

            # Select only masked positions
            pred_selected = torch.masked_select(pred_logits, combined_mask)
            tgt_selected = torch.masked_select(new_targets, combined_mask)

            tgt_selected = tgt_selected.float()
            p = torch.sigmoid(pred_selected)
            ce_loss = F.binary_cross_entropy_with_logits(
                pred_selected, tgt_selected, reduction="none"
            )
            p_t = p * tgt_selected + (1 - p) * (1 - tgt_selected)
            loss = ce_loss * ((1 - p_t) ** gamma)

            if alpha >= 0:
                alpha_t = alpha * tgt_selected + (1 - alpha) * (1 - tgt_selected)
                loss = alpha_t * loss

            total_num_pos = sum(len(bi[0]) for bi in indices)
            num_pos_avg = max(total_num_pos, 1.0)
            loss = loss.sum() / num_pos_avg

            return {'loss_ce': loss}

        criterion.token_sigmoid_binary_focal_loss = masked_focal_loss

    def forward(self, outputs, targets, cat_lists, captions):
        self._targets_ref = targets
        result = self.criterion(outputs, targets, cat_lists, captions)
        self._targets_ref = None
        return result

    def train(self, mode=True):
        self.criterion.train(mode)
        return self

    def eval(self):
        self.criterion.eval()
        return self

    def to(self, *args, **kwargs):
        self.criterion.to(*args, **kwargs)
        return self
