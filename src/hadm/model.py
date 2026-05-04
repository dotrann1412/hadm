"""
Full HADM detector: EVA-02 backbone + RPN + Cascade R-CNN.
"""

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, clip_boxes_to_image, roi_align

from .backbone import Backbone

LOCAL_CLASSES = ["face", "torso", "arm", "leg", "hand", "feet"]
GLOBAL_CLASSES = [
    "missing arm", "missing face", "missing feet",
    "missing hand", "missing leg", "missing torso",
    "extra arm", "extra face", "extra feet",
    "extra hand", "extra leg", "extra torso",
]


# ---------------------------------------------------------------------------
# Box encoding / decoding
# ---------------------------------------------------------------------------

def _decode_boxes(deltas, anchors, weights=(1., 1., 1., 1.)):
    wx, wy, ww, wh = weights
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh
    dw = dw.clamp(max=math.log(1000.0 / 16))
    dh = dh.clamp(max=math.log(1000.0 / 16))

    cx = dx * aw + ax
    cy = dy * ah + ay
    w = torch.exp(dw) * aw
    h = torch.exp(dh) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


# ---------------------------------------------------------------------------
# Anchor Generator
# ---------------------------------------------------------------------------

class BufferList(nn.Module):
    """Stores a list of tensors as numbered buffers (Detectron2 convention)."""
    def __init__(self, buffers: list[torch.Tensor]):
        super().__init__()
        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf)

    def __iter__(self):
        return iter(self._buffers.values())


def _generate_cell_anchors(sizes, aspect_ratios):
    """Generate anchor templates centred at origin for one FPN level."""
    anchors = []
    for s in sizes:
        area = s * s
        for ar in aspect_ratios:
            w = math.sqrt(area * ar)
            h = area / w
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(0.5, 1.0, 2.0),
        strides=(4, 8, 16, 32, 64),
    ):
        super().__init__()
        self.strides = strides
        self.cell_anchors = BufferList([
            _generate_cell_anchors(s, aspect_ratios) for s in sizes
        ])

    @torch.no_grad()
    def forward(self, feature_maps: list[torch.Tensor]):
        anchors_all = []
        for feat, stride, cell in zip(feature_maps, self.strides, self.cell_anchors):
            H, W = feat.shape[-2:]
            shifts_x = torch.arange(0, W, device=feat.device, dtype=torch.float32) * stride + stride / 2
            shifts_y = torch.arange(0, H, device=feat.device, dtype=torch.float32) * stride + stride / 2
            sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack([sx, sy, sx, sy], dim=-1).reshape(-1, 1, 4)
            anchors_all.append((shifts + cell.unsqueeze(0)).reshape(-1, 4)) # type: ignore
        return anchors_all


# ---------------------------------------------------------------------------
# RPN Head
# ---------------------------------------------------------------------------

class RPNConv(nn.Module):
    """Two 3×3 conv layers with ReLU (matching ``proposal_generator.rpn_head.conv.*``)."""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv1(F.relu(self.conv0(x))))


class RPNHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_anchors: int = 3):
        super().__init__()
        self.conv = RPNConv(in_channels)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, features: list[torch.Tensor]):
        obj_all, delta_all = [], []
        for feat in features:
            t = self.conv(feat)
            obj_all.append(self.objectness_logits(t))
            delta_all.append(self.anchor_deltas(t))
        return obj_all, delta_all


class ProposalGenerator(nn.Module):
    """Region Proposal Network (``proposal_generator.*`` in checkpoint)."""

    def __init__(self, in_channels: int = 256, num_anchors: int = 3,
                 pre_nms_topk: int = 2000, post_nms_topk: int = 1000,
                 nms_thresh: float = 0.7):
        super().__init__()
        self.rpn_head = RPNHead(in_channels, num_anchors)
        self.anchor_generator = AnchorGenerator()
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh

    @torch.no_grad()
    def forward(self, features: dict[str, torch.Tensor],
                image_size: tuple[int, int]) -> torch.Tensor:
        feat_list = [features[k] for k in ("p2", "p3", "p4", "p5", "p6")]
        obj_logits, anchor_deltas = self.rpn_head(feat_list)
        anchors_per_level = self.anchor_generator(feat_list)

        proposals = []
        scores = []
        for obj, delta, anch in zip(obj_logits, anchor_deltas, anchors_per_level):
            B = obj.shape[0]
            A = obj.shape[1]  # num_anchors
            H, W = obj.shape[2:]

            obj_flat = obj.permute(0, 2, 3, 1).reshape(B, -1)
            delta_flat = delta.permute(0, 2, 3, 1).reshape(B, -1, 4)

            boxes = _decode_boxes(
                delta_flat.reshape(-1, 4), anch.unsqueeze(0).expand(B, -1, -1).reshape(-1, 4)
            ).reshape(B, -1, 4)

            proposals.append(boxes)
            scores.append(obj_flat)

        proposals = torch.cat(proposals, dim=1)
        scores = torch.cat(scores, dim=1)

        result = []
        for b in range(proposals.shape[0]):
            p = clip_boxes_to_image(proposals[b], image_size)
            s = scores[b].sigmoid()

            topk = min(self.pre_nms_topk, s.shape[0])
            topk_idx = s.topk(topk).indices
            p, s = p[topk_idx], s[topk_idx]

            keep = batched_nms(p, s, torch.zeros_like(s, dtype=torch.int64), self.nms_thresh)
            keep = keep[: self.post_nms_topk]
            result.append(p[keep])

        return torch.stack(result)  # (B, K, 4)


# ---------------------------------------------------------------------------
# Cascade Box Head + Predictor
# ---------------------------------------------------------------------------

class CascadeBoxHead(nn.Module):
    """4× Conv(256, LN, ReLU) + FC(12544→1024) for one cascade stage.

    Key path: ``roi_heads.box_head.{stage}.conv{1..4}.*`` and ``fc1.*``.
    """

    def __init__(self, channels: int = 256, fc_dim: int = 1024, pool_size: int = 7):
        super().__init__()
        self.conv1 = Conv2dNorm(channels, channels, 3, padding=1, bias=False,
                                norm_layer=nn.LayerNorm(channels, eps=1e-6))
        self.conv2 = Conv2dNorm(channels, channels, 3, padding=1, bias=False,
                                norm_layer=nn.LayerNorm(channels, eps=1e-6))
        self.conv3 = Conv2dNorm(channels, channels, 3, padding=1, bias=False,
                                norm_layer=nn.LayerNorm(channels, eps=1e-6))
        self.conv4 = Conv2dNorm(channels, channels, 3, padding=1, bias=False,
                                norm_layer=nn.LayerNorm(channels, eps=1e-6))
        self.fc1 = nn.Linear(channels * pool_size * pool_size, fc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return F.relu(self.fc1(x.flatten(1)))


class BoxPredictor(nn.Module):
    """Classification + class-agnostic box regression for one cascade stage."""

    def __init__(self, in_features: int = 1024, num_classes: int = 7):
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, 4)

    def forward(self, x: torch.Tensor):
        return self.cls_score(x), self.bbox_pred(x)


from .backbone import Conv2dNorm  # noqa: already imported but make it explicit


class CascadeROIHeads(nn.Module):
    """Three-stage Cascade R-CNN detection head (``roi_heads.*`` in checkpoint)."""

    CASCADE_WEIGHTS = [(10, 10, 5, 5), (20, 20, 10, 10), (30, 30, 15, 15)]
    CASCADE_IOU_THRESHOLDS = [0.5, 0.6, 0.7]
    NUM_STAGES = 3

    def __init__(self, num_classes: int = 7, pool_size: int = 7,
                 use_sigmoid: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.use_sigmoid = use_sigmoid

        self.box_head = nn.ModuleList([
            CascadeBoxHead(256, 1024, pool_size) for _ in range(self.NUM_STAGES)
        ])
        self.box_predictor = nn.ModuleList([
            BoxPredictor(1024, num_classes) for _ in range(self.NUM_STAGES)
        ])

    def _pool_features(self, features: dict[str, torch.Tensor],
                       boxes: torch.Tensor) -> torch.Tensor:
        """Multi-scale ROI Align on p2–p5 (canonical FPN level assignment)."""
        levels = ["p2", "p3", "p4", "p5"]
        strides = [4, 8, 16, 32]

        B, N, _ = boxes.shape
        flat_boxes = boxes.reshape(-1, 4)
        batch_idx = torch.arange(B, device=boxes.device).unsqueeze(1).expand(B, N).reshape(-1)

        areas = (flat_boxes[:, 2] - flat_boxes[:, 0]) * (flat_boxes[:, 3] - flat_boxes[:, 1])
        target_lvl = torch.floor(4 + torch.log2(torch.sqrt(areas.clamp(min=1)) / 224 + 1e-8))
        target_lvl = target_lvl.clamp(min=2, max=5).long() - 2  # 0-indexed for p2-p5

        rois = torch.cat([batch_idx.unsqueeze(1).float(), flat_boxes], dim=1)

        pooled = torch.zeros(rois.shape[0], 256, self.pool_size, self.pool_size,
                             device=rois.device, dtype=rois.dtype)

        for lvl_idx, (lvl_name, stride) in enumerate(zip(levels, strides)):
            mask = target_lvl == lvl_idx
            if not mask.any():
                continue
            lvl_rois = rois[mask]
            pooled[mask] = roi_align(  # type: ignore
                features[lvl_name], lvl_rois,
                output_size=self.pool_size, spatial_scale=1.0 / stride,
                sampling_ratio=0,
            )
        return pooled

    @torch.no_grad()
    def forward(self, features: dict[str, torch.Tensor],
                proposals: torch.Tensor,
                image_size: tuple[int, int],
                score_thresh: float = 0.05,
                nms_thresh: float = 0.5,
                detections_per_img: int = 100):
        B, N, _ = proposals.shape
        boxes = proposals

        all_scores = []
        for stage in range(self.NUM_STAGES):
            pooled = self._pool_features(features, boxes)
            head_out = self.box_head[stage](pooled)
            cls_logits, box_deltas = self.box_predictor[stage](head_out)

            if self.use_sigmoid:
                scores = cls_logits.sigmoid()
            else:
                scores = F.softmax(cls_logits, dim=-1)

            all_scores.append(scores.reshape(B, N, -1))

            flat_boxes = boxes.reshape(-1, 4)
            flat_deltas = box_deltas
            new_boxes = _decode_boxes(flat_deltas, flat_boxes,
                                      self.CASCADE_WEIGHTS[stage])
            boxes = clip_boxes_to_image(
                new_boxes, image_size
            ).reshape(B, N, 4)

        avg_scores = sum(all_scores) / len(all_scores)

        results = []
        for b in range(B):
            result = self._postprocess(
                avg_scores[b], boxes[b], image_size,  # type: ignore
                score_thresh, nms_thresh, detections_per_img,
            )
            results.append(result)
        return results

    def _postprocess(self, scores, boxes, image_size,
                     score_thresh, nms_thresh, detections_per_img):
        nc = self.num_classes
        if self.use_sigmoid:
            # Sigmoid: first nc-1 columns are real classes, last is bg (ignore)
            all_boxes, all_scores, all_labels = [], [], []
            for j in range(nc - 1):
                s = scores[:, j]
                keep = s > score_thresh
                if not keep.any():
                    continue
                all_boxes.append(boxes[keep])
                all_scores.append(s[keep])
                all_labels.append(torch.full((keep.sum(),), j + 1,
                                             dtype=torch.int64, device=scores.device))
        else:
            # Softmax: indices 0..nc-2 are real classes, index nc-1 is bg
            all_boxes, all_scores, all_labels = [], [], []
            for j in range(nc - 1):
                s = scores[:, j]
                keep = s > score_thresh
                if not keep.any():
                    continue
                all_boxes.append(boxes[keep])
                all_scores.append(s[keep])
                all_labels.append(torch.full((keep.sum(),), j + 1,
                                             dtype=torch.int64, device=scores.device))

        if not all_boxes:
            z4 = torch.zeros((0, 4), device=scores.device)
            z1 = torch.zeros((0,), device=scores.device)
            return {"boxes": z4, "scores": z1,
                    "labels": torch.zeros((0,), dtype=torch.int64, device=scores.device)}

        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)

        keep = batched_nms(all_boxes, all_scores, all_labels, nms_thresh)
        keep = keep[:detections_per_img]
        return {"boxes": all_boxes[keep], "scores": all_scores[keep],
                "labels": all_labels[keep]}


# ---------------------------------------------------------------------------
# Full HADM Model
# ---------------------------------------------------------------------------

class HADM(nn.Module):
    """Full HADM detector matching the original checkpoint layout.

    Keys: ``backbone.*``, ``proposal_generator.*``, ``roi_heads.*``,
          ``pixel_mean``, ``pixel_std``.
    """

    def __init__(self, mode: str = "local", img_size: int = 1024):
        super().__init__()
        if mode == "local":
            num_classes = len(LOCAL_CLASSES) + 1   # 7 (6 + bg)
            use_sigmoid = False
        else:
            num_classes = len(GLOBAL_CLASSES) + 1  # 13 (12 + bg)
            use_sigmoid = True

        self.mode = mode
        self.img_size = img_size

        # Window blocks: 0,1, 3,4, 6,7, 9,10, 12,13, 15,16, 18,19, 21,22
        win_idxs = []
        for start in range(0, 24, 3):
            win_idxs.extend([start, start + 1])

        self.backbone = Backbone(
            img_size=img_size, patch_size=16, embed_dim=1024,
            depth=24, num_heads=16, mlp_ratio=4 * 2 / 3,
            drop_path_rate=0.0,  # no drop-path at inference
            window_size=16,
            window_block_indexes=tuple(win_idxs),
        )
        self.proposal_generator = ProposalGenerator()
        self.roi_heads = CascadeROIHeads(
            num_classes=num_classes, pool_size=7, use_sigmoid=use_sigmoid,
        )

        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.280, 103.530]).reshape(3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.1156, 57.375]).reshape(3, 1, 1),
        )

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor],
                score_thresh: float = 0.05,
                nms_thresh: float = 0.5,
                detections_per_img: int = 100) -> list[dict]:
        """Run inference on a list of RGB [0, 1] tensors."""
        device = self.pixel_mean.device
        preprocessed, orig_sizes = self._preprocess(images, device)

        features = self.backbone(preprocessed)
        proposals = self.proposal_generator(features, (self.img_size, self.img_size))
        detections = self.roi_heads(
            features, proposals, (self.img_size, self.img_size),
            score_thresh, nms_thresh, detections_per_img,
        )

        # Scale boxes back to original image coordinates
        for det, (oh, ow) in zip(detections, orig_sizes):
            if det["boxes"].numel() > 0:
                scale_w = ow / self.img_size
                scale_h = oh / self.img_size
                det["boxes"][:, [0, 2]] *= scale_w
                det["boxes"][:, [1, 3]] *= scale_h

        return detections

    def _preprocess(self, images: list[torch.Tensor], device):
        orig_sizes = []
        batch = []
        for img in images:
            img = img.to(device)
            if img.max() <= 1.0:
                img = img * 255.0

            _, h, w = img.shape
            orig_sizes.append((h, w))

            # Resize shortest edge to img_size (cap long edge at img_size too)
            scale = self.img_size / min(h, w)
            new_h = min(int(h * scale + 0.5), self.img_size)
            new_w = min(int(w * scale + 0.5), self.img_size)
            img = F.interpolate(
                img.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

            # Normalize (pixel_mean/std are in RGB order, matching input)
            img = (img - self.pixel_mean) / self.pixel_std

            # Pad to img_size × img_size
            pad_h = self.img_size - new_h
            pad_w = self.img_size - new_w
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h))

            batch.append(img)

        return torch.stack(batch), orig_sizes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_hadm_weights(model: HADM, checkpoint_path: str,
                      use_ema: bool = True) -> tuple[list[str], list[str]]:
    """Load original Detectron2 HADM checkpoint into our model.

    Args:
        model: An ``HADM`` instance.
        checkpoint_path: Path to ``HADM-L_0249999.pth`` or ``HADM-G_0249999.pth``.
        use_ema: If True (recommended), load EMA weights which are better.

    Returns:
        (missing_keys, unexpected_keys) — for diagnostics.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if use_ema and "ema_state" in ckpt:
        src = ckpt["ema_state"]
    elif "model" in ckpt:
        src = ckpt["model"]
    else:
        src = ckpt

    # The EMA state doesn't include per-block RoPE buffers (they're
    # deterministically computed), so we load with strict=False.
    missing, unexpected = model.load_state_dict(src, strict=False)

    # Filter out expected missing/unexpected keys
    expected_missing = {"pixel_mean", "pixel_std"}
    expected_unexpected_prefixes = ("proposal_generator.anchor_generator.",)

    missing_real = [k for k in missing
                    if k not in expected_missing
                    and not k.startswith("backbone.net.rope_")
                    and "rope" not in k]

    if missing_real:
        print(f"Missing keys: {missing_real}")

    unexpected_real = [k for k in unexpected
                       if not any(k.startswith(p) for p in expected_unexpected_prefixes)]

    return missing_real, unexpected_real


def get_class_names(mode: str) -> list[str]:
    return list(LOCAL_CLASSES if mode == "local" else GLOBAL_CLASSES)
