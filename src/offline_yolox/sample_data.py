"""Create deterministic tiny COCO images and exported object/mask examples."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .bootstrap import local_path


def _background(rng, size):
    image = np.full((size, size, 3), (65, 88, 62), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + rng.integers(-10, 11, image.shape), 0, 255).astype(np.uint8)
    road = max(10, size // 8)
    cv2.rectangle(image, (0, size // 3), (size - 1, size // 3 + road), (92, 92, 92), -1)
    cv2.rectangle(image, (size // 2, 0), (size // 2 + road, size - 1), (92, 92, 92), -1)
    for x, y in ((8, 8), (size * 3 // 4, 8), (8, size * 3 // 4)):
        cv2.rectangle(image, (x, y), (min(x + size // 8, size - 1), min(y + size // 8, size - 1)),
                      (105, 120, 125), -1)
    return image


def make_sample_data(output, *, seed=42, image_size=128, train_images=8, val_images=4, force=False):
    output = local_path(output)
    if image_size < 64 or train_images <= 0 or val_images <= 0:
        raise ValueError("image_size must be >=64 and both splits must be nonempty")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"Refusing to overwrite existing data: {output}; use --force for generated samples")
    for directory in ("annotations", "train2017", "val2017", "objects", "backgrounds"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    categories = [{"id": 1, "name": "sample_vehicle", "supercategory": "vehicle"}]
    manifest = []
    for index in range(6):
        height, width = 18 + index, 28 + index * 2
        image = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (2, 2), (width - 3, height - 3), 255, -1)
        image[mask > 0] = (35 + index * 20, 45, 190 - index * 10)
        cv2.rectangle(image, (width // 4, 3), (width // 2, height - 4), (180, 200, 210), -1)
        image_name = f"objects/object_{index}.png"
        mask_name = f"objects/mask_{index}.png"
        if not cv2.imwrite(str(output / image_name), image) or not cv2.imwrite(str(output / mask_name), mask):
            raise OSError("Cannot write generated object sample")
        manifest.append({"image": image_name, "mask": mask_name, "category_id": 1})
    (output / "objects" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    background_paths = []
    for index in range(4):
        name = f"backgrounds/background_{index}.png"
        if not cv2.imwrite(str(output / name), _background(rng, image_size)):
            raise OSError("Cannot write generated background sample")
        background_paths.append(name)
    (output / "backgrounds" / "manifest.json").write_text(json.dumps(background_paths, indent=2) + "\n")
    counts = {}
    for split, count in (("train2017", train_images), ("val2017", val_images)):
        images, annotations = [], []
        annotation_id = 1
        for image_index in range(count):
            image = _background(rng, image_size)
            for object_index in range(1 + image_index % 2):
                source = manifest[(image_index + object_index) % len(manifest)]
                obj = cv2.imread(str(output / source["image"]))
                mask = cv2.imread(str(output / source["mask"]), cv2.IMREAD_GRAYSCALE)
                height, width = mask.shape
                x = int(rng.integers(0, image_size - width + 1))
                y = int(rng.integers(0, image_size - height + 1))
                region = image[y:y + height, x:x + width]
                region[mask > 0] = obj[mask > 0]
                ys, xs = np.nonzero(mask)
                bbox = [x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1)]
                annotations.append({"id": annotation_id, "image_id": image_index + 1,
                                    "category_id": 1, "bbox": bbox,
                                    "area": bbox[2] * bbox[3], "iscrowd": 0})
                annotation_id += 1
            name = f"sample_{image_index:04}.png"
            if not cv2.imwrite(str(output / split / name), image):
                raise OSError("Cannot write generated training sample")
            images.append({"id": image_index + 1, "file_name": name,
                           "height": image_size, "width": image_size})
        dataset = {"info": {"description": "Deterministic synthetic diagnostic sample, not SAM3 output"},
                   "licenses": [], "images": images, "annotations": annotations, "categories": categories}
        (output / "annotations" / f"instances_{split}.json").write_text(json.dumps(dataset, indent=2) + "\n")
        counts[split] = {"images": len(images), "annotations": len(annotations)}
    report = {"root": str(output), "seed": seed, "image_size": image_size, "splits": counts,
              "objects_manifest": str(output / "objects/manifest.json"),
              "backgrounds_manifest": str(output / "backgrounds/manifest.json"),
              "manifest_path_base": "dataset root", "categories": categories,
              "synthetic_sample": True, "sam3_inference_used": False}
    (output / "sample_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-images", type=int, default=8)
    parser.add_argument("--val-images", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report = make_sample_data(args.output, seed=args.seed, image_size=args.image_size,
                              train_images=args.train_images, val_images=args.val_images, force=args.force)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    main()
