"""Predict on local images/folders and save boxes in original image coordinates."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from .asset_discovery import IMAGE_SUFFIXES
from .bootstrap import local_path
from .experiment import PRESETS, read_config, seed_everything
from .patch_data import crop_canvas, dimensions, starts
from .startup import report, startup_stage
from .train import load_checkpoint
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess


def _threshold(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value > 1 or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be finite and between {'0' if allow_zero else '>0'} and 1")
    return float(value)


def _class_metadata(config, num_classes):
    names, ids = config.get("class_names"), config.get("class_ids")
    if names is None and config.get("data_dir"):
        annotation = (local_path(config["data_dir"]) / "annotations"
                      / config.get("val_ann", "instances_val2017.json"))
        if annotation.is_file():
            categories = sorted(json.loads(annotation.read_text(encoding="utf-8"))["categories"],
                                key=lambda category: category["id"])
            names = [category["name"] for category in categories]
            if ids is None:
                ids = [category["id"] for category in categories]
    if names is None:
        report("Class names unavailable; using class_0, class_1, ... . Set class_names to label them.")
        names = [f"class_{index}" for index in range(num_classes)]
    if (not isinstance(names, (list, tuple)) or len(names) != num_classes
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError("class_names/COCO categories must match num_classes in internal class order")
    if ids is not None and (not isinstance(ids, (list, tuple)) or len(ids) != num_classes
                            or len(set(ids)) != len(ids)
                            or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)):
        raise ValueError("class_ids must contain num_classes unique integer category IDs")
    return list(names), list(ids) if ids is not None else [None] * num_classes


def load_predictor(config, checkpoint=None, *, device="cuda", raw_model=False):
    """Build only the detector; never initialize training datasets or asset banks."""
    mode, preset = config.get("mode", "original"), config.get("preset", "s")
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    checkpoint = local_path(checkpoint or Path(config.get("output_dir", f"outputs/{mode}"))
                            / config.get("exp_name", f"{mode}_{preset}") / "latest_ckpt.pth",
                            must_exist=True)
    saved = load_checkpoint(checkpoint)
    previous = saved.get("config") or {}
    count = config.get("num_classes", previous.get("num_classes"))
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("num_classes must be a positive integer in config/checkpoint config")
    names_config = {**previous, **config}
    for key in ("class_names", "class_ids"):
        if key not in names_config and saved.get(key) is not None:
            names_config[key] = saved[key]
    names, category_ids = _class_metadata(names_config, count)
    size = dimensions(config.get("test_size", config.get("input_size", [640, 640])), "test_size")
    if any(side < 64 or side % 32 for side in size):
        raise ValueError("test_size must contain multiples of 32, at least 64")
    seed_everything(int(config.get("seed", 42)))
    exp = get_exp(exp_name=f"yolox-{preset}")
    exp.num_classes = count
    model = exp.get_model()
    use_ema = not raw_model and bool(saved.get("ema_model"))
    model.load_state_dict(saved["ema_model"] if use_ema else saved["model"], strict=True)
    model.to(device).eval()
    return model, dict(checkpoint=str(checkpoint), weights="ema_model" if use_ema else "model",
                       preset=preset, num_classes=count, class_names=names, class_ids=category_ids,
                       input_size=list(size))


def predict_image(model, image, *, input_size, num_classes, confidence=0.25, nms_threshold=0.65,
                  device="cuda", tile_size=None, stride=None, batch_size=8):
    """Decode an image once, batch its tiles, restore offsets and merge with class-aware NMS."""
    confidence = _threshold(confidence, "confidence")
    nms_threshold = _threshold(nms_threshold, "nms", allow_zero=True)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if tile_size is None and stride is not None:
        raise ValueError("stride requires tiled prediction")
    ih, iw = image.shape[:2]
    if tile_size is not None:
        tile_size = dimensions(tile_size, "patch_size")
        stride = dimensions(stride or tile_size, "stride")
        if any(step > side for step, side in zip(stride, tile_size)):
            raise ValueError("stride must not exceed patch_size")
        windows = [(x, y, min(tile_size[0], ih - y), min(tile_size[1], iw - x))
                   for y in starts(ih, tile_size[0], stride[0])
                   for x in starts(iw, tile_size[1], stride[1])]
    else:
        windows = [(0, 0, ih, iw)]
    collected = []
    for first in range(0, len(windows), batch_size):
        batch_windows = windows[first:first + batch_size]
        tensors, ratios = [], []
        for x, y, height, width in batch_windows:
            chip = crop_canvas(image, x, y, tile_size) if tile_size is not None else image
            tensor, ratio = preproc(chip, input_size)
            tensors.append(tensor)
            ratios.append(ratio)
        with torch.inference_mode():
            raw = model(torch.from_numpy(np.stack(tensors)).to(device))
            if not torch.isfinite(raw).all():
                raise FloatingPointError("Model produced nonfinite predictions")
            if raw.shape[1] == 1:
                raw = torch.cat([raw, torch.zeros_like(raw)], dim=1)
            predictions = postprocess(raw, num_classes, confidence, nms_threshold)
        for detections, ratio, (x, y, height, width) in zip(predictions, ratios, batch_windows, strict=True):
            if detections is None:
                continue
            values = detections.detach().cpu().numpy()
            boxes = values[:, :4] / ratio
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes += np.array([x, y, x, y], dtype=np.float32)
            collected.append(np.column_stack((boxes[valid], values[valid, 4] * values[valid, 5],
                                               values[valid, 6])).astype(np.float32))
    if not collected:
        return np.zeros((0, 6), dtype=np.float32), len(windows)
    combined = np.concatenate(collected)
    if not len(combined):
        return np.zeros((0, 6), dtype=np.float32), len(windows)
    tensor = torch.as_tensor(combined, device=device)
    keep = batched_nms(tensor[:, :4], tensor[:, 4], tensor[:, 5].long(), nms_threshold)
    return tensor[keep].cpu().numpy(), len(windows)


def draw_predictions(image, predictions, names):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(max(height, width) / 1000))
    font_scale = max(0.4, min(1.2, max(height, width) / 2000))
    for box in predictions:
        coords = np.rint(box[:4]).astype(int)
        coords[0::2] = np.clip(coords[0::2], 0, width - 1)
        coords[1::2] = np.clip(coords[1::2], 0, height - 1)
        x1, y1, x2, y2 = coords
        category = int(box[5])
        color = ((37 * category + 50) % 200, (97 * category + 170) % 256, (61 * category + 230) % 256)
        label = f"{names[category]} {box[4]:.3f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(canvas, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
    return canvas


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def predict(config, source, output, *, checkpoint=None, device="cuda", confidence=0.25,
            nms_threshold=None, tiled=False, patch_size=None, stride=None, batch_size=8,
            raw_model=False, ignore_exif_orientation=False):
    confidence = _threshold(confidence, "confidence")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not tiled and (patch_size is not None or stride is not None):
        raise ValueError("patch_size/stride require --tiled")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("CUDA is unavailable; use --device cpu explicitly")
    source = local_path(source, must_exist=True)
    output = local_path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Prediction output must be new or empty: {output}")
    if source.is_file():
        files = [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
        root = source.parent
    else:
        files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        root = source
    if not files:
        raise ValueError(f"No supported image files found: {source}")
    values = read_config(config) if isinstance(config, (str, Path)) else dict(config)
    nms_threshold = _threshold(values.get("nmsthre", 0.65) if nms_threshold is None else nms_threshold,
                               "nms", allow_zero=True)
    with startup_stage("Loading inference model (without training data initialization)"):
        model, model_info = load_predictor(values, checkpoint, device=device, raw_model=raw_model)
    if tiled:
        settings = values.get("patch") or {}
        patch_size = dimensions(patch_size or settings.get("size", model_info["input_size"]), "patch_size")
        stride = dimensions(stride or settings.get("stride", patch_size), "stride")
        if any(step > side for step, side in zip(stride, patch_size)):
            raise ValueError("stride must not exceed patch_size")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, path in enumerate(files):
        flags = cv2.IMREAD_COLOR | (cv2.IMREAD_IGNORE_ORIENTATION if ignore_exif_orientation else 0)
        image = cv2.imread(str(path), flags)
        if image is None:
            raise ValueError(f"File exists but OpenCV cannot decode image: {path}")
        boxes, tiles = predict_image(model, image, input_size=model_info["input_size"],
                                     num_classes=model_info["num_classes"], confidence=confidence,
                                     nms_threshold=nms_threshold, device=device,
                                     tile_size=patch_size if tiled else None, stride=stride if tiled else None,
                                     batch_size=batch_size)
        relative = path.relative_to(root)
        image_path = output / (relative.as_posix() + ".pred.png")
        json_path = output / (relative.as_posix() + ".pred.json")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = draw_predictions(image, boxes, model_info["class_names"])
        success, encoded = cv2.imencode(".png", canvas)
        if not success:
            raise OSError(f"Cannot encode prediction image: {image_path}")
        with image_path.open("xb") as handle:
            handle.write(encoded.tobytes())
        detections = [dict(bbox_xyxy=box[:4].tolist(), score=float(box[4]), class_index=int(box[5]),
                           class_name=model_info["class_names"][int(box[5])],
                           category_id=model_info["class_ids"][int(box[5])]) for box in boxes]
        item = dict(input=str(path), output_image=str(image_path.relative_to(output)),
                    output_json=str(json_path.relative_to(output)), width=image.shape[1], height=image.shape[0],
                    tiles=tiles, detection_count=len(boxes))
        _write_json(json_path, {**item, "detections": detections})
        results.append(item)
        report(f"Predict {number + 1}/{len(files)}: {relative} -> {len(boxes)} boxes ({tiles} tiles)")
    summary = dict(**model_info, source=str(source), strategy="tiled" if tiled else "whole_image",
                   confidence=confidence, nms_threshold=nms_threshold, patch_size=patch_size if tiled else None,
                   stride=stride if tiled else None, batch_size=batch_size, device=device,
                   ignore_exif_orientation=ignore_exif_orientation, image_count=len(results), images=results)
    _write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online_patch.json")
    parser.add_argument("--checkpoint", help="Local model; default: configured output/exp_name/latest_ckpt.pth")
    parser.add_argument("--input", required=True, help="Image file or folder (searched recursively)")
    parser.add_argument("--output", default="outputs/predict")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms", type=float, help="Class-aware IoU threshold within/between patches")
    parser.add_argument("--tiled", action="store_true", help="Predict patches and merge boxes on the original image")
    parser.add_argument("--patch-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--stride", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8, help="Maximum patches per forward pass")
    parser.add_argument("--raw-model", action="store_true", help="Use raw weights instead of saved EMA")
    parser.add_argument("--ignore-exif-orientation", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    result = predict(args.config, args.input, args.output, checkpoint=args.checkpoint, device=args.device,
                     confidence=args.confidence, nms_threshold=args.nms, tiled=args.tiled,
                     patch_size=args.patch_size, stride=args.stride, batch_size=args.batch_size,
                     raw_model=args.raw_model, ignore_exif_orientation=args.ignore_exif_orientation)
    print(json.dumps(dict(output=str(local_path(args.output)), images=result["image_count"])))
    return result


if __name__ == "__main__":
    main()
