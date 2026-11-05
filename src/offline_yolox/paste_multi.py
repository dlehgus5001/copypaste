"""Compose multiple object/mask pairs, create masks and harmonize in one command."""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from .bootstrap import local_path
from .composition_options import add_composition_arguments, composition_config
from .patch_data import OnlinePatchDataset, dimensions, patch_settings
from .paste_check import _comparison, _finite, _write_png, harmonize_image, load_harmonizer
from .startup import report, startup_stage


class _PreviewDataset(OnlinePatchDataset):
    def __init__(self, *args, background=None, **kwargs):
        self.background_override = local_path(background, must_exist=True) if background is not None else None
        super().__init__(*args, **kwargs)

    def _read_backgrounds(self, name):
        if self.background_override is None:
            return super()._read_backgrounds(name)
        if not self.background_override.is_file():
            raise ValueError("--background must be one image file")
        if self.validate_assets_on_start:
            self._read_image(self.background_override)
        return [{"image": str(self.background_override), "image_path": str(self.background_override)}]

    def _background_canvas(self, rng, background):
        canvas = super()._background_canvas(rng, background)
        self.last_background = canvas.copy()
        return canvas


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_multi_paste(config, output, count=4, *, background=None, repository="models/Harmonizer",
                     checkpoint=None, device="auto", strength=1.0, plain_only=False, model=None):
    """No training model/COCO file is required. A single union mask feeds Harmonizer."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    strength = _finite(strength, "strength", minimum=0, maximum=1)
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    values = (json.loads(local_path(config, must_exist=True).read_text(encoding="utf-8"))
              if isinstance(config, (str, Path)) else copy.deepcopy(config))
    if not isinstance(values, dict) or values.get("mode") != "online_patch":
        raise ValueError("paste-multi requires an online_patch-style config; use configs/paste_multi.json")
    ids, names = values.get("class_ids"), values.get("class_names")
    if not isinstance(ids, list) or not isinstance(names, list) or not ids or len(ids) != len(names):
        raise ValueError("Standalone config requires class_ids and class_names; use configs/paste_multi.json")
    if not isinstance(values.get("copy_paste"), dict):
        raise ValueError("copy_paste must be an object")
    seed = values.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = local_path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    dataset = _PreviewDataset(config=values["copy_paste"],
        input_size=dimensions(values.get("input_size", [640, 640]), "input_size"),
        class_ids=ids, class_names=names, seed=seed, patch=patch_settings(values), background=background)
    output.mkdir(parents=True, exist_ok=True)
    records, model_info = [], None
    for index in range(count):
        sample = dataset.sample_raw()
        before, masks = sample["image"], sample["masks"]
        union = np.zeros(before.shape[:2], dtype=bool)
        for mask in masks:
            if np.any(union & mask):
                raise ValueError("Final visible instance masks must not overlap")
            union |= mask
        after, correction, applied = None, None, False
        if not plain_only:
            if union.any():
                if model_info is None:
                    started = perf_counter()
                    if model is None:
                        with startup_stage("Loading local Harmonizer once for all composites"):
                            model, model_info = load_harmonizer(repository, checkpoint, device)
                    else:
                        selected = device
                        if selected == "auto":
                            parameter = next(model.parameters(), None) if hasattr(model, "parameters") else None
                            selected = str(parameter.device) if parameter is not None else "cpu"
                        model_info = dict(device=selected, backend="test_or_custom_model")
                    model_info["load_seconds"] = perf_counter() - started
                alpha = union.astype(np.float32)
                if dataset.blend_sigma:
                    alpha *= cv2.GaussianBlur(alpha, (0, 0), dataset.blend_sigma, borderType=cv2.BORDER_CONSTANT)
                report(f"Harmonizing composite {index + 1}/{count}: {len(masks)} objects")
                after, correction = harmonize_image(model, before, union, alpha,
                    device=model_info["device"], strength=strength)
                applied = True
            else:
                after = before.copy()
                correction = dict(skipped_empty_mask=True, seconds=0)
        folder = output / f"sample_{index:04}"
        (folder / "masks").mkdir(parents=True)
        files = dict(background="background.png", before="before.png", mask="mask.png", boxes="boxes.png")
        _write_png(folder / files["background"], dataset.last_background)
        _write_png(folder / files["before"], before)
        _write_png(folder / files["mask"], union.astype(np.uint8) * 255)
        instances = copy.deepcopy(sample["metadata"]["instances"])
        for item_index, (mask, instance) in enumerate(zip(masks, instances, strict=True)):
            filename = f"masks/object_{item_index + 1:03}.png"
            _write_png(folder / filename, mask.astype(np.uint8) * 255)
            instance["composite_mask"] = filename
        if after is not None:
            files.update(after="after.png", comparison="comparison.png")
            _write_png(folder / files["after"], after)
            label = "AFTER: Harmonizer" if applied else "AFTER: unchanged (empty mask)"
            _write_png(folder / files["comparison"], _comparison(before, after, label))
        boxed = (after if after is not None else before).copy()
        for instance in instances:
            x1, y1, x2, y2 = instance["bbox_xyxy"]
            cv2.rectangle(boxed, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 1)
            cv2.putText(boxed, str(instance["category_id"]), (x1, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        _write_png(folder / files["boxes"], boxed)
        delta = np.abs(after.astype(np.int16) - before.astype(np.int16)) if applied else None
        metadata = dict(**{key: value for key, value in sample["metadata"].items() if key not in ("instances", "mask_semantics")},
            instances=instances, labels_xyxy_class_index=sample["labels"].tolist(), files=files,
            mask_semantics="Union of final visible objects, white foreground 255, background 0",
            harmonizer_applied=applied, correction=correction,
            background_pixels_preserved=bool(np.array_equal((after if after is not None else before)[~union], before[~union])),
            foreground_mean_absolute_change=float(delta[union].mean()) if delta is not None else None,
            note="Support masks and boxes are generated during paste, not predicted by segmentation/detection.")
        _write_json(folder / "result.json", metadata)
        records.append(f"sample_{index:04}/result.json")
        report(f"Saved {index + 1}/{count}: {len(masks)} objects, union mask and {len(masks)} instance masks")
    manifest = dict(count=count, config=values, model=model_info, plain_only=plain_only,
        background_override=str(dataset.background_override) if dataset.background_override else None,
        samples=records, training_invoked=False, mask_semantics="white foreground in composite pixel coordinates")
    _write_json(output / "manifest.json", manifest)
    return dict(output=str(output), count=count, manifest=str(output / "manifest.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paste_multi.json")
    parser.add_argument("--output", default="outputs/paste_multi")
    parser.add_argument("--count", type=int, default=4, help="Number of complete composite images")
    parser.add_argument("--background", help="Optional single background file, instead of the background bank")
    parser.add_argument("--harmonizer-repo", default="models/Harmonizer", dest="repository")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--plain-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    add_composition_arguments(parser)
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    values = json.loads(local_path(args.config, must_exist=True).read_text(encoding="utf-8"))
    values = composition_config(values, args)
    result = save_multi_paste(values, args.output, args.count, background=args.background,
        repository=args.repository, checkpoint=args.checkpoint, device=args.device,
        strength=args.strength, plain_only=args.plain_only)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
