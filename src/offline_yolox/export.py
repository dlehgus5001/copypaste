"""Save online compositions as images and standard COCO instance annotations."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .bootstrap import ROOT, local_path
from .experiment import create_exp, read_config


def _fresh_output(value):
    """Accept only a new or empty real directory, without following symlinks."""
    output = local_path(value)
    original = Path(value).expanduser()
    if not original.is_absolute():
        original = ROOT / original
    for candidate in (original, *original.parents):
        if candidate.is_symlink():
            raise ValueError(f"Export output must not use symlinks: {candidate}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Export output must be empty to protect existing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _atomic_json(path, value):
    """Publish a complete JSON file only after serialization succeeds."""
    partial = path.with_name(path.name + ".partial")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    partial.replace(path)


def _write_image(path, image):
    encoded, png = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"Cannot encode image: {path}")
    with path.open("xb") as handle:
        handle.write(png.tobytes())


def export_coco(config: dict, output, count=100, split="train2017") -> dict:
    """Export the same fresh samples used online, with final visible-mask labels.

    The destination must be new or empty. If composition or writing fails, any
    files already written remain for inspection; the final annotation JSON is
    published only after every sample has succeeded.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if split not in ("train2017", "val2017"):
        raise ValueError("split must be 'train2017' or 'val2017'")
    if not isinstance(config, dict) or config.get("mode") != "online":
        raise ValueError("COCO export requires an online configuration")
    exp = create_exp(config)
    dataset = exp.get_dataset()
    validation_annotation = validation_images = None
    if split == "train2017":
        validation_annotation = local_path(Path(exp.data_dir) / "annotations" / exp.val_ann,
                                           must_exist=True)
        validation_images = local_path(Path(exp.data_dir) / exp.val_name, must_exist=True)
    output = _fresh_output(output)
    images_dir = output / split
    annotations_dir = output / "annotations"
    metadata_dir = output / "metadata"
    for directory in (images_dir, annotations_dir, metadata_dir):
        directory.mkdir()
    coco = {
        "info": {
            "description": "Online Copy-Paste compositions", "seed": exp.seed,
            "mask_semantics": "final_visible_foreground_ownership",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in zip(exp.class_ids, exp.class_names, strict=True)
        ],
    }
    annotation_id = 1
    for image_id in range(1, count + 1):
        sample = dataset.sample_raw()
        image = sample["image"]
        height, width = image.shape[:2]
        stem = f"{image_id:012d}"
        image_name = stem + ".png"
        coco["images"].append({
            "id": image_id, "file_name": image_name, "height": height, "width": width,
        })
        for mask, instance in zip(sample["masks"], sample["metadata"]["instances"], strict=True):
            mask = np.asfortranarray(mask, dtype=np.uint8)
            if mask.shape != (height, width) or not mask.any():
                raise ValueError("Export masks must be nonempty and match the image dimensions")
            rle = mask_utils.encode(mask)
            rle["counts"] = rle["counts"].decode("ascii")
            x1, y1, x2, y2 = instance["bbox_xyxy"]
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": instance["category_id"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": int(mask.sum()),
                "iscrowd": 0,
                "segmentation": rle,
            })
            annotation_id += 1
        _write_image(images_dir / image_name, image)
        _atomic_json(metadata_dir / (stem + ".json"), {
            **sample["metadata"], "image_id": image_id,
            "file_name": image_name, "split": split,
        })
    annotation_path = annotations_dir / f"instances_{split}.json"
    _atomic_json(annotation_path, coco)
    training_config_path = None
    if split == "train2017":
        # Official COCODataset joins these paths with data_dir; an absolute
        # annotation/image path takes precedence. Keep the held-out validation
        # data at its original location instead of evaluating on training data.
        training_config = copy.deepcopy(config)
        training_config.pop("copy_paste", None)
        training_config.update({
            "mode": "original", "data_dir": exp.data_dir,
            "train_ann": str(annotation_path), "train_name": str(images_dir),
            "val_ann": str(validation_annotation), "val_name": str(validation_images),
            # Keep the exported object count and pixel sizes when switching to
            # Original: its default Mosaic/MixUp would merge several images.
            "mosaic_prob": 0.0, "enable_mixup": False, "mixup_prob": 0.0,
            "multiscale_range": 0,
            "output_dir": str(output / "training-output"),
            "exp_name": "exported_" + exp.preset,
        })
        training_config_path = output / "train_config.json"
        _atomic_json(training_config_path, training_config)
    return {
        "output": str(output), "count": count,
        "annotations": str(annotation_path), "images_dir": str(images_dir),
        "training_config": str(training_config_path) if training_config_path else None,
    }


def main(argv=None):
    from .composition_options import add_composition_arguments, composition_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online.json")
    parser.add_argument("--output", default="outputs/synthetic")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=("train2017", "val2017"), default="train2017")
    add_composition_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = composition_config(read_config(args.config), args)
    except ValueError as error:
        parser.error(str(error))
    result = export_coco(config, args.output, args.count, args.split)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
