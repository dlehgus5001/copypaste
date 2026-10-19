"""Explicitly export a few online compositions and their visible-mask labels."""
import argparse
import json

import cv2
import numpy as np

from .bootstrap import local_path
from .composition_options import add_composition_arguments, composition_config
from .experiment import create_exp, read_config


def save_preview(config, output, count=4):
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be positive")
    exp = create_exp(config)
    mode = exp.config.get("mode")
    if mode not in ("online", "online_patch", "original_patch"):
        raise ValueError("Preview requires an online or original_patch configuration")
    dataset = exp.get_dataset()
    if mode == "original_patch":
        count = min(count, len(dataset))
    output = local_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preview output must be empty to protect existing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"count": count, "seed": exp.seed, "config": exp.config,
               "training_writes_synthetic_images": False, "samples": []}
    for index in range(count):
        if mode == "original_patch":
            image, labels, _, _ = dataset.pull_item(index)
            record = dataset.records[index]
            sample = {"image": image, "labels": labels, "masks": [], "metadata": {
                "source_image": record["path"], "source_image_id": record["source_image_id"],
                "crop_xy": list(record["xy"]), "patch_size": list(dataset.settings["size"]),
                "instances": [{"bbox_xyxy": label[:4].tolist(),
                               "class_name": exp.class_names[int(label[4])]}
                              for label in labels],
            }}
        else:
            sample = dataset.sample_raw()
        image_name, overlay_name = f"sample_{index:04}.png", f"sample_{index:04}_boxes.png"
        annotated = sample["image"].copy()
        mask_paths = []
        for instance_index, mask in enumerate(sample["masks"]):
            mask_name = f"sample_{index:04}_mask_{instance_index:03}.png"
            if not cv2.imwrite(str(output / mask_name), mask.astype(np.uint8) * 255):
                raise OSError(f"Cannot write {mask_name}")
            mask_paths.append(mask_name)
        for instance in sample["metadata"]["instances"]:
            x1, y1, x2, y2 = [int(round(v)) for v in instance["bbox_xyxy"]]
            cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 230, 255), 1)
            cv2.putText(annotated, instance["class_name"], (x1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 230, 255), 1, cv2.LINE_AA)
        for name, image in ((image_name, sample["image"]), (overlay_name, annotated)):
            if not cv2.imwrite(str(output / name), image):
                raise OSError(f"Cannot write {name}")
        metadata = {**sample["metadata"], "image": image_name, "preview": overlay_name,
                    "masks": mask_paths, "labels_xyxy_class_index": sample["labels"].tolist()}
        metadata_name = f"sample_{index:04}.json"
        (output / metadata_name).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["samples"].append(metadata_name)
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"output": str(output), "count": count, "manifest": str(output / "manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online.json")
    parser.add_argument("--output", default="outputs/preview")
    parser.add_argument("--count", type=int, default=4)
    add_composition_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = composition_config(read_config(args.config), args)
    except ValueError as error:
        parser.error(str(error))
    result = save_preview(config, args.output, args.count)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
