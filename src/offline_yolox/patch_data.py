"""Lazy image crops and in-memory COCO boxes; source files are never changed."""
import copy
import math
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

from .copy_paste import OnlineCopyPasteDataset
from yolox.data.datasets.datasets_wrapper import Dataset


def dimensions(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value)):
        raise ValueError(f"{name} must contain positive integer [height, width]")
    return tuple(value)


def patch_settings(config):
    value = config.get("patch")
    if not isinstance(value, dict):
        raise ValueError("Patch modes require a patch configuration object")
    size = dimensions(value.get("size", [640, 640]), "patch.size")
    stride = dimensions(value.get("stride", size), "patch.stride")
    if any(step > side for step, side in zip(stride, size)):
        raise ValueError("patch.stride cannot exceed patch.size (would leave gaps)")
    fraction = value.get("min_visible_fraction", 0.0)
    side = value.get("min_box_side", 2.0)
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError("patch.min_visible_fraction must be between 0 and 1")
    if (isinstance(side, bool) or not isinstance(side, (int, float))
            or not math.isfinite(side) or side <= 0):
        raise ValueError("patch.min_box_side must be positive")
    keep_empty = value.get("keep_empty", True)
    if not isinstance(keep_empty, bool):
        raise ValueError("patch.keep_empty must be boolean")
    max_labels = value.get("max_labels", 500)
    if isinstance(max_labels, bool) or not isinstance(max_labels, int) or max_labels <= 0:
        raise ValueError("patch.max_labels must be a positive integer")
    return dict(size=size, stride=stride, min_visible_fraction=fraction,
                min_box_side=side, keep_empty=keep_empty, max_labels=max_labels)


def starts(length, side, stride):
    """Cover both ends; the last window can overlap even when stride == side."""
    result = list(range(0, max(0, length - side) + 1, stride))
    end = max(0, length - side)
    if result[-1] != end:
        result.append(end)
    return result


def crop_canvas(image, x, y, size):
    height, width = size
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    crop = image[y:y + height, x:x + width]
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    return canvas


class CocoPatchDataset(Dataset):
    """Enumerate a COCO split's crops without saving patches or a new JSON.

    Patch IDs are local to this dataset. COCO evaluation uses the same clipped
    patch-space ground truth; these are patch metrics, not stitched image metrics.
    """
    def __init__(self, *, data_dir, json_file, name, input_size, settings,
                 preproc=None, training=True):
        super().__init__(input_size, mosaic=False)
        source = COCO(str(Path(data_dir) / "annotations" / json_file))
        self.class_ids = sorted(source.getCatIds())
        self._classes = tuple(source.cats[value]["name"] for value in self.class_ids)
        class_index = {value: i for i, value in enumerate(self.class_ids)}
        self.settings, self.preproc = settings, preproc
        self.records, self.ids = [], []
        images, annotations = [], []
        height, width = settings["size"]
        for image_id in source.getImgIds():
            image = source.imgs[image_id]
            ih, iw = dimensions([image["height"], image["width"]], "COCO image dimensions")
            path = Path(data_dir) / name / image["file_name"]
            objects = source.imgToAnns[image_id]
            for y in starts(ih, height, settings["stride"][0]):
                for x in starts(iw, width, settings["stride"][1]):
                    clipped, labels = [], []
                    for obj in objects:
                        bx, by, bw, bh = obj["bbox"]
                        if not all(math.isfinite(v) for v in (bx, by, bw, bh)) or bw <= 0 or bh <= 0:
                            continue
                        left, top = max(0, bx), max(0, by)
                        right, bottom = min(iw, bx + bw), min(ih, by + bh)
                        full_area = max(0, right - left) * max(0, bottom - top)
                        x1, y1 = max(x, left), max(y, top)
                        x2, y2 = min(x + width, right), min(y + height, bottom)
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if (min(x2 - x1, y2 - y1) < settings["min_box_side"] or full_area <= 0
                                or area / full_area < settings["min_visible_fraction"]):
                            continue
                        box = [x1 - x, y1 - y, x2 - x1, y2 - y1]
                        clipped.append(dict(category_id=obj["category_id"], bbox=box,
                                            area=area, iscrowd=int(obj.get("iscrowd", 0))))
                        if not obj.get("iscrowd", 0):
                            labels.append([x1 - x, y1 - y, x2 - x, y2 - y, class_index[obj["category_id"]]])
                    # Keep validation coverage regardless of training background sampling.
                    if training and not settings["keep_empty"] and not labels:
                        continue
                    if training and len(labels) > settings["max_labels"]:
                        raise ValueError("Patch has more boxes than patch.max_labels; increase it to avoid truncation")
                    patch_id = len(self.records) + 1
                    self.ids.append(patch_id)
                    self.records.append(dict(path=str(path), source_image_id=image_id,
                                             source_size=(ih, iw), xy=(x, y),
                                             labels=np.asarray(labels, dtype=np.float32).reshape(-1, 5)))
                    images.append(dict(id=patch_id, file_name=image["file_name"], height=height, width=width,
                                       source_image_id=image_id, crop_xy=[x, y]))
                    for annotation in clipped:
                        annotations.append(dict(**annotation, id=len(annotations) + 1, image_id=patch_id))
        if not self.records:
            raise ValueError("Patch dataset has no images; check annotations/keep_empty settings")
        self.coco = COCO()
        self.coco.dataset = dict(info=copy.deepcopy(source.dataset.get("info", {})),
                                 images=images, annotations=annotations,
                                 categories=copy.deepcopy(source.dataset["categories"]))
        self.coco.createIndex()

    def __len__(self):
        return len(self.records)

    def pull_item(self, index):
        record = self.records[index]
        image = cv2.imread(record["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode patch source image: {record['path']}")
        if image.shape[:2] != record["source_size"]:
            raise ValueError(f"COCO dimensions differ from source image: {record['path']}")
        image = crop_canvas(image, *record["xy"], self.settings["size"])
        return image, record["labels"].copy(), self.settings["size"], np.array([self.ids[index]])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        image, labels, info, image_id = self.pull_item(index)
        if self.preproc is not None:
            image, labels = self.preproc(image, labels, self.input_dim)
        return image, labels, info, image_id


class OnlinePatchDataset(OnlineCopyPasteDataset):
    def __init__(self, *args, patch, **kwargs):
        self.patch = patch
        super().__init__(*args, **kwargs)

    def _background_canvas(self, rng, background):
        image = self._read_image(background["image_path"])
        ih, iw = image.shape[:2]
        height, width = self.patch["size"]
        x = int(rng.integers(max(0, iw - width) + 1))
        y = int(rng.integers(max(0, ih - height) + 1))
        self._last_crop = dict(source_size=[ih, iw], xy=[x, y], size=[height, width])
        canvas = crop_canvas(image, x, y, self.patch["size"])
        return cv2.resize(canvas, (self.input_dim[1], self.input_dim[0]), interpolation=cv2.INTER_LINEAR)

    def sample_raw(self):
        result = super().sample_raw()
        result["metadata"]["background_patch"] = copy.deepcopy(self._last_crop)
        return result
