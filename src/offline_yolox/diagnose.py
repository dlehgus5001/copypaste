"""Inspect checkpoint predictions on a small real/online sample; no training."""
import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .bootstrap import local_path
from .experiment import create_exp, read_config, seed_everything
from .startup import report, startup_stage
from .train import load_checkpoint
from yolox.data.data_augment import preproc
from yolox.utils import postprocess


def match_boxes(labels, predictions, iou_threshold=0.5, *, class_aware=True):
    """Greedy one-to-one matches in score order. predictions: xyxy,score,class."""
    used, matched = set(), 0
    for index in np.argsort(-predictions[:, 4], kind="stable"):
        prediction = predictions[index]
        best, best_iou = None, iou_threshold
        for gt_index, label in enumerate(labels):
            if gt_index in used or (class_aware and label[4] != prediction[5]):
                continue
            lo = np.maximum(label[:2], prediction[:2])
            hi = np.minimum(label[2:4], prediction[2:4])
            intersection = float(np.maximum(0, hi - lo).prod())
            union = float(np.maximum(0, label[2:4] - label[:2]).prod()
                          + np.maximum(0, prediction[2:4] - prediction[:2]).prod() - intersection)
            iou = intersection / union if union > 0 else 0.0
            if iou >= best_iou:
                best, best_iou = gt_index, iou
        if best is not None:
            used.add(best)
            matched += 1
    return dict(tp=matched, fp=len(predictions) - matched, fn=len(labels) - matched)


def inspect_sample(model, image, labels, *, input_size, num_classes, confidence,
                   nms_threshold, device, max_candidates=1000):
    """Use official letterboxing/NMS; restore predictions to the raw sample frame."""
    tensor, ratio = preproc(image, input_size)
    with torch.inference_mode():
        raw = model(torch.from_numpy(tensor).unsqueeze(0).to(device))
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Checkpoint produced nonfinite predictions")
        scores = raw[0, :, 4] * raw[0, :, 5:5 + num_classes].max(dim=1).values
        indices = torch.nonzero(scores >= confidence, as_tuple=False).flatten()
        above_threshold = len(indices)
        if len(indices) > max_candidates:
            indices = indices[torch.topk(scores[indices], max_candidates).indices]
        maximum = float(scores.max()) if scores.numel() else 0.0
        predictions = np.zeros((0, 6), dtype=np.float32)
        if len(indices):
            candidates = raw[:, indices].clone()
            # Upstream postprocess squeezes a one-candidate boolean mask to scalar.
            # A zero-score dummy avoids that edge case and never passes confidence.
            if len(indices) == 1:
                candidates = torch.cat([candidates, torch.zeros_like(candidates)], dim=1)
            detections = postprocess(candidates, num_classes, confidence, nms_threshold)[0]
            if detections is not None:
                detections = detections.detach().cpu().numpy()
                predictions = np.column_stack((detections[:, :4] / ratio,
                                                detections[:, 4] * detections[:, 5],
                                                detections[:, 6])).astype(np.float32)
    return dict(
        max_raw_score=maximum, candidates_above_threshold=above_threshold,
        nms_candidates_used=len(indices), candidates_capped=above_threshold > max_candidates,
        gt_count=len(labels), prediction_count=len(predictions),
        class_aware_iou50=match_boxes(labels, predictions),
        class_agnostic_iou50=match_boxes(labels, predictions, class_aware=False),
        predictions=predictions.tolist(), ground_truth=labels.tolist(),
    )


def _labels_at(dataset, index):
    return (dataset.records[index]["labels"] if hasattr(dataset, "records")
            else dataset.annotations[index][0])


def select_samples(dataset, count):
    """Spread across metadata; prefer positives with up to 25% empty samples."""
    positive, empty = [], []
    for index in range(len(dataset)):
        (positive if len(_labels_at(dataset, index)) else empty).append(index)
    count = min(count, len(dataset))
    n_empty = min(len(empty), count // 4)
    n_positive = min(len(positive), count - n_empty)
    n_empty = min(len(empty), count - n_positive)
    n_positive = min(len(positive), count - n_empty)

    def spread(indices, number):
        return [indices[i] for i in np.linspace(0, len(indices) - 1, number, dtype=int)] if number else []

    return (spread(positive, n_positive) + spread(empty, n_empty),
            dict(total_images=len(dataset), positive_images=len(positive), empty_images=len(empty)))


def _draw(image, boxes, names, *, predictions=False):
    canvas = image.copy()
    height, width = image.shape[:2]
    for box in boxes:
        xyxy = np.rint(box[:4]).astype(int)
        xyxy[0::2] = np.clip(xyxy[0::2], 0, width - 1)
        xyxy[1::2] = np.clip(xyxy[1::2], 0, height - 1)
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        class_index = int(box[5] if predictions else box[4])
        label = names[class_index] + (f" {box[4]:.3f}" if predictions else "")
        color = (0, 0, 255) if predictions else (0, 220, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.putText(canvas, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)
    return canvas


def diagnose(config, output, *, checkpoint=None, device="cuda", count=8,
             confidence=None, raw_model=False, max_candidates=1000):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if confidence is not None and (not math.isfinite(confidence) or not 0 < confidence <= 1):
        raise ValueError("confidence must be finite and in (0, 1]")
    if device not in ("cuda", "cpu") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("A usable CUDA GPU is required for cuda; use cpu explicitly for diagnostics")
    output = local_path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Diagnostic output must be new or empty: {output}")
    values = read_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    # Do not decode the whole asset bank just to inspect a small sample.
    if isinstance(values.get("copy_paste"), dict):
        values["copy_paste"]["validate_assets_on_start"] = False
    with startup_stage("Preparing diagnostic config, classes and file metadata"):
        exp = create_exp(values)
    confidence = exp.test_conf if confidence is None else confidence
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("Configured test_conf must be in (0, 1]; use --confidence to override")
    checkpoint = local_path(checkpoint or Path(exp.output_dir) / exp.exp_name / "latest_ckpt.pth",
                            must_exist=True)
    with startup_stage(f"Loading diagnostic checkpoint: {checkpoint}"):
        saved = load_checkpoint(checkpoint)
        seed_everything(exp.seed)
        model = exp.get_model()
        use_ema = not raw_model and bool(saved.get("ema_model"))
        model.load_state_dict(saved["ema_model"] if use_ema else saved["model"], strict=True)
        model.to(device).eval()
    previous = saved.get("config") or {}
    old_settings, current_settings = previous.get("copy_paste", {}), values.get("copy_paste", {})
    differences = {key: dict(checkpoint=old_settings.get(key), current=current_settings.get(key))
                   for key in sorted(set(old_settings) | set(current_settings))
                   if key != "validate_assets_on_start" and old_settings.get(key) != current_settings.get(key)}
    if differences:
        report("Current copy_paste differs from checkpoint: " + ", ".join(differences)
               + ". Fresh synthetic diagnostics use current config; inspect summary.json.")
    validation = exp.get_eval_dataset()
    real_indices, real_metadata = select_samples(validation, count)
    training = exp.get_dataset()
    online = values.get("mode") in ("online", "online_patch")
    training_indices = (list(range(count)) if online else select_samples(training, count)[0])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, dataset, indices in [("real", validation, real_indices),
                                     ("synthetic" if online else "training", training, training_indices)]:
        for number, index in enumerate(indices):
            if split == "synthetic":
                sample = dataset.sample_raw()
                image, labels, metadata = sample["image"], sample["labels"], sample["metadata"]
            else:
                image, labels, _, image_id = dataset.pull_item(index)
                metadata = dict(dataset_index=index, image_id=int(image_id[0]))
                if hasattr(dataset, "records"):
                    record = dataset.records[index]
                    metadata.update(source_image=record["path"], source_image_id=record["source_image_id"],
                                    crop_xy=list(record["xy"]))
            result = inspect_sample(model, image, labels, input_size=exp.test_size,
                                    num_classes=exp.num_classes, confidence=confidence,
                                    nms_threshold=exp.nmsthre, device=device, max_candidates=max_candidates)
            stem = f"{split}_{number:04d}"
            row = dict(split=split, metadata=metadata, **result)
            for suffix, canvas in [("", image), ("_gt", _draw(image, labels, exp.class_names)),
                                    ("_pred", _draw(image, np.asarray(result["predictions"]).reshape(-1, 6),
                                                    exp.class_names, predictions=True))]:
                filename = f"{stem}{suffix}.png"
                if not cv2.imwrite(str(output / filename), canvas):
                    raise OSError(f"Cannot write diagnostic image: {filename}")
            row["files"] = dict(image=stem + ".png", ground_truth=stem + "_gt.png",
                                 predictions=stem + "_pred.png")
            (output / f"{stem}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            rows.append(row)
            report(f"{stem}: gt={len(labels)}, predictions={result['prediction_count']}, "
                   f"max_score={result['max_raw_score']:.6f}, TP@0.5={result['class_aware_iou50']['tp']}")
    groups = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        groups[split] = dict(
            sampled_images=len(selected), gt_count=sum(row["gt_count"] for row in selected),
            prediction_count=sum(row["prediction_count"] for row in selected),
            max_raw_score=max(row["max_raw_score"] for row in selected),
            tp_iou50=sum(row["class_aware_iou50"]["tp"] for row in selected),
            tp_ignoring_class_iou50=sum(row["class_agnostic_iou50"]["tp"] for row in selected),
        )
    result = dict(
        checkpoint=str(checkpoint), checkpoint_epoch=saved.get("start_epoch"),
        checkpoint_config=previous, current_config=exp.config,
        copy_paste_differences=differences,
        optimizer_lrs=[group.get("lr") for group in (saved.get("optimizer") or {}).get("param_groups", [])],
        weights="ema_model" if use_ema else "model", confidence=confidence,
        evaluation_confidence=exp.test_conf, max_candidates=max_candidates,
        class_mapping=[dict(class_index=i, category_id=value, name=exp.class_names[i])
                       for i, value in enumerate(exp.class_ids)],
        real_dataset=real_metadata, groups=groups,
        sample_selection="spread positive samples plus up to 25 percent empty, with availability fallback",
        diagnostic_only=True, computes_full_dataset_ap=False,
        samples=rows,
    )
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online_patch.json")
    parser.add_argument("--checkpoint", help="Default: latest_ckpt.pth under the configured output/experiment")
    parser.add_argument("--output", default="outputs/diagnose")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--count", type=int, default=8, help="Maximum images per source; synthetic samples are freshly generated")
    parser.add_argument("--confidence", type=float, help="Default: actual evaluation test_conf")
    parser.add_argument("--raw-model", action="store_true", help="Inspect raw training weights instead of EMA")
    parser.add_argument("--max-candidates", type=int, default=1000, help="Visualization NMS limit; capped counts are reported")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    result = diagnose(args.config, args.output, checkpoint=args.checkpoint, device=args.device,
                      count=args.count, confidence=args.confidence, raw_model=args.raw_model,
                      max_candidates=args.max_candidates)
    print(json.dumps(dict(summary=str(local_path(args.output) / "summary.json"), groups=result["groups"])))
    return result


if __name__ == "__main__":
    main()
