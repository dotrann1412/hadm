"""
Training, evaluation, and inference script for simplified HADM.

Usage:
    # Train HADM-L (local body-part artifact detector)
    python -m hadm_simple.train --mode local --data_root datasets/human_artifact_dataset

    # Train HADM-G (global anomaly detector, multi-label)
    python -m hadm_simple.train --mode global --data_root datasets/human_artifact_dataset

    # Evaluate a checkpoint
    python -m hadm_simple.train --mode local --eval_only --checkpoint best.pth \\
        --data_root datasets/human_artifact_dataset --val_split val_ALL

    # Run inference on a folder of images
    python -m hadm_simple.train --mode local --infer --checkpoint best.pth \\
        --infer_dir demo/images --output_dir demo/outputs
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import HADDataset, ImageFolderDataset, collate_fn
from .model import ModelEMA, build_model, get_class_names


# ---------------------------------------------------------------------------
# Mean Average Precision (simplified COCO-style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ap(predictions: list[dict], ground_truths: list[dict], iou_thresholds=None):
    """Compute mean Average Precision across IoU thresholds.

    Args:
        predictions: list of dicts per image with keys 'boxes', 'scores', 'labels'.
        ground_truths: list of dicts per image with keys 'boxes', 'labels'.
        iou_thresholds: IoU thresholds to evaluate at (default: COCO 0.50:0.05:0.95).

    Returns:
        dict with 'mAP', 'AP50', 'AP75', and per-class AP.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    all_classes = set()
    for gt in ground_truths:
        all_classes.update(gt["labels"].cpu().tolist())
    for pred in predictions:
        all_classes.update(pred["labels"].cpu().tolist())
    all_classes = sorted(all_classes)

    if not all_classes:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}

    ap_per_class = {}
    for cls in all_classes:
        aps_for_thresholds = []
        for iou_thresh in iou_thresholds:
            ap = _single_class_ap(predictions, ground_truths, cls, iou_thresh)
            aps_for_thresholds.append(ap)
        ap_per_class[cls] = sum(aps_for_thresholds) / len(aps_for_thresholds)

    ap50_per_class = {}
    ap75_per_class = {}
    for cls in all_classes:
        ap50_per_class[cls] = _single_class_ap(predictions, ground_truths, cls, 0.5)
        ap75_per_class[cls] = _single_class_ap(predictions, ground_truths, cls, 0.75)

    mAP = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    AP50 = sum(ap50_per_class.values()) / len(ap50_per_class) if ap50_per_class else 0.0
    AP75 = sum(ap75_per_class.values()) / len(ap75_per_class) if ap75_per_class else 0.0

    return {"mAP": mAP, "AP50": AP50, "AP75": AP75, "per_class": ap_per_class}


def _single_class_ap(predictions, ground_truths, cls, iou_thresh):
    scored_dets = []
    n_gt = 0

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        gt_mask = gt["labels"] == cls
        gt_boxes = gt["boxes"][gt_mask]
        n_gt += gt_boxes.shape[0]

        pred_mask = pred["labels"] == cls
        pred_boxes = pred["boxes"][pred_mask]
        pred_scores = pred["scores"][pred_mask]

        for j in range(pred_boxes.shape[0]):
            scored_dets.append((pred_scores[j].item(), img_idx, pred_boxes[j], j))

    if n_gt == 0:
        return 0.0

    scored_dets.sort(key=lambda x: x[0], reverse=True)

    matched = defaultdict(set)
    tp = []
    fp = []

    for score, img_idx, det_box, _ in scored_dets:
        gt = ground_truths[img_idx]
        gt_mask = gt["labels"] == cls
        gt_boxes = gt["boxes"][gt_mask]

        best_iou = 0.0
        best_gt_idx = -1
        for gi in range(gt_boxes.shape[0]):
            iou = _box_iou_single(det_box, gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_thresh and best_gt_idx not in matched[img_idx]:
            tp.append(1)
            fp.append(0)
            matched[img_idx].add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp_cum = torch.tensor(fp, dtype=torch.float32).cumsum(0)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # 101-point interpolated AP
    ap = 0.0
    for t in torch.linspace(0, 1, 101):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max().item()
    return ap / 101


def _box_iou_single(box_a, box_b):
    x1 = max(box_a[0].item(), box_b[0].item())
    y1 = max(box_a[1].item(), box_b[1].item())
    x2 = min(box_a[2].item(), box_b[2].item())
    y2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
    area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch, ema=None, log_every=50):
    model.train()
    running_loss = defaultdict(float)
    n_batches = 0
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        if not math.isfinite(total_loss.item()):
            print(f"WARNING: non-finite loss {total_loss.item()}, skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        for k, v in loss_dict.items():
            running_loss[k] += v.item()
        running_loss["total"] += total_loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg = {k: v / n_batches for k, v in running_loss.items()}
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            print(f"  [Epoch {epoch}] batch {batch_idx + 1}/{len(data_loader)} "
                  f"({elapsed:.0f}s) — {loss_str}")

    avg_losses = {k: v / max(n_batches, 1) for k, v in running_loss.items()}
    return avg_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_gts = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            all_preds.append({k: v.cpu() for k, v in out.items()})
            all_gts.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tgt.items()})

    metrics = compute_ap(all_preds, all_gts)
    return metrics


# ---------------------------------------------------------------------------
# Inference + visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, data_loader, device, output_dir, class_names, score_thresh=0.3):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    try:
        from PIL import Image, ImageDraw, ImageFont
        can_draw = True
    except ImportError:
        can_draw = False

    all_results = {}

    for images, paths in data_loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for img_tensor, path, out in zip(images, paths, outputs):
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            img_name = Path(path).name
            result = []
            for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                cls_name = class_names[l - 1] if 1 <= l <= len(class_names) else f"class_{l}"
                result.append({"bbox": b, "score": round(s, 4), "class": cls_name})
            all_results[img_name] = result

            if can_draw:
                img_pil = Image.fromarray(
                    (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
                )
                draw = ImageDraw.Draw(img_pil)
                for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    cls_name = class_names[l - 1] if 1 <= l <= len(class_names) else f"class_{l}"
                    color = _class_color(l)
                    draw.rectangle(b, outline=color, width=3)
                    label_text = f"{cls_name} {s:.2f}"
                    draw.text((b[0] + 2, b[1] + 2), label_text, fill=color)
                img_pil.save(os.path.join(output_dir, img_name))

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved {len(all_results)} results to {output_dir}/")
    return all_results


_COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan",
    "magenta", "yellow", "lime", "pink", "teal", "coral",
]


def _class_color(label_idx: int) -> str:
    return _COLORS[(label_idx - 1) % len(_COLORS)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HADM Simplified — Train / Eval / Infer")

    p.add_argument("--mode", choices=["local", "global"], default="local",
                   help="Detection mode: 'local' (6 body-part classes) or 'global' (12 anomaly classes)")
    p.add_argument("--data_root", type=str, default="datasets/human_artifact_dataset",
                   help="Root dir of the HAD dataset")
    p.add_argument("--train_split", type=str, default="train_ALL")
    p.add_argument("--val_split", type=str, default="val_ALL")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=1024)

    p.add_argument("--ema", action="store_true", default=True, help="Use EMA (default: on)")
    p.add_argument("--no_ema", dest="ema", action="store_false")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    p.add_argument("--eval_only", action="store_true", help="Skip training, evaluate checkpoint")
    p.add_argument("--infer", action="store_true", help="Run inference on --infer_dir images")
    p.add_argument("--infer_dir", type=str, default="demo/images")
    p.add_argument("--infer_thresh", type=float, default=0.3)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")

    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Use COCO-pretrained Faster R-CNN weights (default: on)")
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode} ({'6 classes' if args.mode == 'local' else '12 classes, multi-label'})")

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = get_class_names(args.mode)

    # ---- Build model ----
    model = build_model(
        mode=args.mode,
        pretrained_backbone=args.pretrained,
        min_size=args.img_size,
        max_size=args.img_size,
    )
    model.to(device)

    ema = None
    if args.ema:
        ema = ModelEMA(model, decay=args.ema_decay)
        ema.to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"])
            if ema is not None and "ema" in state:
                ema.module.load_state_dict(state["ema"])
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M")

    # ---- Inference mode ----
    if args.infer:
        infer_ds = ImageFolderDataset(args.infer_dir)
        infer_loader = DataLoader(infer_ds, batch_size=1, collate_fn=collate_fn)
        eval_model = ema.module if ema is not None else model
        eval_model.eval()
        run_inference(eval_model, infer_loader, device, args.output_dir, class_names, args.infer_thresh)
        return

    # ---- Datasets ----
    train_ds = HADDataset(args.data_root, split=args.train_split, mode=args.mode, train=True)
    val_ds = HADDataset(args.data_root, split=args.val_split, mode=args.mode, train=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Eval-only mode ----
    if args.eval_only:
        eval_model = ema.module if ema is not None else model
        metrics = evaluate(eval_model, val_loader, device)
        print(f"\nEvaluation results on {args.val_split}:")
        print(f"  mAP:  {metrics['mAP']:.3f}")
        print(f"  AP50: {metrics['AP50']:.3f}")
        print(f"  AP75: {metrics['AP75']:.3f}")
        if "per_class" in metrics:
            for cls_id, ap in metrics["per_class"].items():
                idx = cls_id - 1
                name = class_names[idx] if 0 <= idx < len(class_names) else f"class_{cls_id}"
                print(f"    {name}: {ap:.3f}")
        return

    # ---- Optimizer + LR schedule ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    warmup_iters = args.warmup_epochs * len(train_loader)
    total_iters = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_iters:
            return max(step / max(warmup_iters, 1), 0.001)
        progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        print(f"{'='*60}")

        losses = train_one_epoch(model, optimizer, train_loader, device, epoch, ema=ema)
        scheduler.step()

        loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
        print(f"  Train — {loss_str}")

        eval_model = ema.module if ema is not None else model
        metrics = evaluate(eval_model, val_loader, device)
        print(f"  Val   — mAP: {metrics['mAP']:.3f} | AP50: {metrics['AP50']:.3f} | AP75: {metrics['AP75']:.3f}")

        is_best = metrics["mAP"] > best_map
        if is_best:
            best_map = metrics["mAP"]

        if epoch % args.save_every == 0 or is_best:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "args": vars(args),
            }
            if ema is not None:
                ckpt["ema"] = ema.module.state_dict()

            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(ckpt, save_path)

            if is_best:
                best_path = os.path.join(args.output_dir, "best.pth")
                torch.save(ckpt, best_path)
                print(f"  ** New best mAP: {best_map:.3f} — saved to {best_path}")

    print(f"\nTraining complete. Best mAP: {best_map:.3f}")


if __name__ == "__main__":
    main()
