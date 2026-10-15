"""In-memory online composition of local, pre-extracted objects and backgrounds.

SAM3 is an upstream extraction step: this module consumes exported images and
binary masks, and performs no inference, authentication, downloads or writes.
Boxes use half-open pixel bounds, calculated from final visible ownership masks.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from .bootstrap import local_path
from yolox.data.datasets.datasets_wrapper import Dataset


def bbox_from_mask(mask):
    """Return tight xyxy bounds (exclusive right/bottom), or None for no pixels."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def transform_object(image, mask, long_side_pixels, rotation_degrees):
    """Scale tight foreground bounds, rotate without clipping, then trim padding."""
    bounds = bbox_from_mask(mask)
    if bounds is None:
        raise ValueError("Object mask contains no foreground")
    x1, y1, x2, y2 = bounds
    image, mask = image[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    scale = float(long_side_pixels) / max(mask.shape)
    height = max(1, int(round(mask.shape[0] * scale)))
    width = max(1, int(round(mask.shape[1] * scale)))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), float(rotation_degrees), 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    rotated_width = max(1, int(np.ceil(height * sine + width * cosine - 1e-8)))
    rotated_height = max(1, int(np.ceil(height * cosine + width * sine - 1e-8)))
    matrix[0, 2] += (rotated_width - width) / 2
    matrix[1, 2] += (rotated_height - height) / 2
    image = cv2.warpAffine(image, matrix, (rotated_width, rotated_height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (rotated_width, rotated_height), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    bounds = bbox_from_mask(mask)
    if bounds is None:
        raise ValueError("Scaling removed all object foreground pixels; increase object sizes")
    x1, y1, x2, y2 = bounds
    return np.ascontiguousarray(image[y1:y2, x1:x2]), np.ascontiguousarray(mask[y1:y2, x1:x2])


def _pair(value, name, *, integer=False, minimum=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain a minimum and maximum")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain finite numbers")
    if integer and any(int(v) != v for v in value):
        raise ValueError(f"{name} must contain integers")
    if value[0] > value[1] or (minimum is not None and value[0] < minimum):
        raise ValueError(f"Invalid {name} range: {value}")
    return tuple(int(v) if integer else float(v) for v in value)


class OnlineCopyPasteDataset(Dataset):
    """Generate a fresh sample on every call, including repeated calls to an index.

    Finite epoch loaders are recreated with the same seed/epoch. Each worker owns
    an independent random stream. Repeatability requires identical worker count,
    batch size, order, seed, epoch, dependencies and input assets. It does not mean
    that changing workers reproduces the same samples, or that native infinite
    loader prefetch state is restored by a model checkpoint.
    """

    def __init__(self, config, input_size, class_ids, class_names, seed=42, epoch=0, preproc=None):
        super().__init__(tuple(input_size), mosaic=False)
        self.config = dict(config)
        self.asset_root = local_path(config["asset_root"], must_exist=True)
        if not self.asset_root.is_dir():
            raise ValueError("asset_root must be a local directory")
        self.preproc = preproc
        self.seed, self.epoch = int(seed), int(epoch)
        self.class_ids, self.class_names = list(class_ids), list(class_names)
        if len(self.class_ids) != len(self.class_names) or len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("Class IDs and names must form a unique shared taxonomy")
        self.class_index = {category_id: index for index, category_id in enumerate(self.class_ids)}
        self.epoch_length = self._positive_int(config.get("epoch_length", 1000), "epoch_length")
        self.object_count = _pair(config.get("object_count", [1, 5]), "object_count", integer=True, minimum=0)
        self.unique_objects = self._boolean(config.get("unique_objects", False), "unique_objects")
        self.require_object_count = self._boolean(config.get("require_object_count", False), "require_object_count")
        self.composition_attempts = self._positive_int(config.get("composition_attempts", 20), "composition_attempts")
        self.max_labels = self._positive_int(config.get("max_labels", 120), "max_labels")
        if self.object_count[1] > self.max_labels:
            raise ValueError("object_count maximum must not exceed max_labels")
        self.rotation = _pair(config.get("rotation_degrees", [-180, 180]), "rotation_degrees")
        self.object_size_range = None
        self.object_sizes = self.size_probabilities = None
        if "object_long_side_range" in config:
            if "object_long_side_pixels" in config or "object_size_weights" in config:
                raise ValueError("object_long_side_range cannot be combined with object_long_side_pixels or object_size_weights")
            self.object_size_range = _pair(config["object_long_side_range"], "object_long_side_range", integer=True, minimum=2)
        else:
            sizes = config.get("object_long_side_pixels", [12, 18, 24, 32])
            if not isinstance(sizes, list) or not sizes or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 2
                    for v in sizes):
                raise ValueError("object_long_side_pixels must contain finite sizes >= 2")
            self.object_sizes = np.asarray(sizes, dtype=np.float64)
            weights = config.get("object_size_weights", [1.0] * len(sizes))
            if not isinstance(weights, list) or len(weights) != len(sizes) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0
                    for v in weights):
                raise ValueError("object_size_weights must have one finite positive weight per size")
            weights = np.asarray(weights, dtype=np.float64)
            self.size_probabilities = weights / weights.sum()
        self.distribution_source = str(config.get("distribution_source", "uncalibrated_sample_pixels"))
        self.min_visible_fraction = self._nonnegative(config.get("min_visible_fraction", 0.25), "min_visible_fraction")
        if self.min_visible_fraction > 1:
            raise ValueError("min_visible_fraction must be between 0 and 1")
        self.min_visible_pixels = self._positive_int(config.get("min_visible_pixels", 4), "min_visible_pixels")
        self.min_box_side = self._positive_int(config.get("min_box_side_pixels", 2), "min_box_side_pixels")
        self.placement_attempts = self._positive_int(config.get("placement_attempts", 30), "placement_attempts")
        self.allow_partial_objects = self._boolean(config.get("allow_partial_objects", False), "allow_partial_objects")
        self.blend_sigma = self._nonnegative(config.get("gaussian_blend_sigma", 0.0), "gaussian_blend_sigma")
        self.blur_sigma = self._nonnegative(config.get("blur_sigma", 0.0), "blur_sigma")
        self.noise_std = self._nonnegative(config.get("noise_std", 0.0), "noise_std")
        self.brightness = _pair(config.get("brightness_range", [1.0, 1.0]), "brightness_range", minimum=0.01)
        self.objects = self._read_objects(config.get("objects_manifest", "objects/manifest.json"))
        if self.unique_objects and self.object_count[1] > len(self.objects):
            raise ValueError("object_count maximum must not exceed available distinct object image/mask sources when unique_objects is true")
        self.backgrounds = self._read_backgrounds(config.get("backgrounds_manifest", "backgrounds/manifest.json"))
        self._rng = None
        self._stream_key = None
        self._calls = 0

    @staticmethod
    def _boolean(value, name):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
        return value

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _nonnegative(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        return float(value)

    def _asset_path(self, value):
        if not isinstance(value, str) or not value or "://" in value:
            raise ValueError("Manifest paths must be nonempty local paths")
        path = Path(value)
        path = (path if path.is_absolute() else self.asset_root / path).resolve()
        if not path.is_relative_to(self.asset_root):
            raise ValueError(f"Asset path escapes asset_root: {value}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _read_manifest(self, name):
        with self._asset_path(name).open(encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Manifest must contain a nonempty list: {name}")
        return entries

    @staticmethod
    def _read_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Image must be readable uint8 BGR/BGRA: {path}")
        # The separately supplied binary mask, rather than PNG alpha, is authoritative.
        return np.ascontiguousarray(image[:, :, :3])

    @staticmethod
    def _read_mask(path, shape):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError(f"Mask must be readable uint8 single-channel binary data: {path}")
        if mask.shape != shape:
            raise ValueError(f"Image and mask dimensions differ: {path}")
        if not np.isin(mask, [0, 1, 255]).all():
            raise ValueError(f"Mask must be binary (0 and 1 or 255): {path}")
        mask = mask > 0
        if not mask.any():
            raise ValueError(f"Mask contains no foreground: {path}")
        return mask

    def _read_objects(self, name):
        result = []
        seen_sources = set()
        for entry in self._read_manifest(name):
            if not isinstance(entry, dict) or not all(k in entry for k in ("image", "mask", "category_id")):
                raise ValueError("Object entries require image, mask and category_id")
            category = entry["category_id"]
            if isinstance(category, bool) or not isinstance(category, int) or category not in self.class_index:
                raise ValueError(f"Object category_id {category} is absent from the validation COCO taxonomy")
            image_path, mask_path = self._asset_path(entry["image"]), self._asset_path(entry["mask"])
            source_key = (image_path, mask_path)
            if self.unique_objects and source_key in seen_sources:
                raise ValueError("unique_objects requires distinct image/mask sources; the object manifest contains a duplicate pair")
            seen_sources.add(source_key)
            image = self._read_image(image_path)
            self._read_mask(mask_path, image.shape[:2])
            result.append({"image": entry["image"], "mask": entry["mask"], "category_id": category,
                           "image_path": str(image_path), "mask_path": str(mask_path)})
        return result

    def _read_backgrounds(self, name):
        result = []
        for entry in self._read_manifest(name):
            path = self._asset_path(entry)
            self._read_image(path)
            result.append({"image": entry, "image_path": str(path)})
        return result

    def __len__(self):
        return self.epoch_length

    def _random_stream(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else -1
        worker_seed = worker.seed if worker else 0
        key = (worker_id, worker_seed, self.epoch)
        if key != self._stream_key:
            sequence = np.random.SeedSequence([self.seed, self.epoch, worker_seed])
            self._stream_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
            self._rng = np.random.default_rng(self._stream_seed)
            self._stream_key, self._calls = key, 0
        return self._rng, worker_id

    def _visible_enough(self, mask, full_area):
        area = int(mask.sum())
        if area < self.min_visible_pixels or area / full_area < self.min_visible_fraction:
            return False
        bounds = bbox_from_mask(mask)
        return bounds is not None and min(bounds[2] - bounds[0], bounds[3] - bounds[1]) >= self.min_box_side

    def _compose(self, rng, background, count):
        """Attempt one complete composition without changing the requested count."""
        height, width = self.input_dim
        image = cv2.resize(self._read_image(background["image_path"]), (width, height), interpolation=cv2.INTER_LINEAR)
        masks, instances = [], []
        rejected = 0
        source_indices = rng.choice(len(self.objects), size=count, replace=False) if self.unique_objects else None
        for source_index in range(count):
            source = self.objects[int(source_indices[source_index]) if self.unique_objects else int(rng.integers(len(self.objects)))]
            obj = self._read_image(source["image_path"])
            mask = self._read_mask(source["mask_path"], obj.shape[:2])
            if self.object_size_range is None:
                long_side = float(rng.choice(self.object_sizes, p=self.size_probabilities))
            else:
                long_side = float(rng.integers(self.object_size_range[0], self.object_size_range[1] + 1))
            angle = float(rng.uniform(*self.rotation))
            obj, mask = transform_object(obj, mask, long_side, angle)
            obj_height, obj_width = mask.shape
            full_area = int(mask.sum())
            if not self.allow_partial_objects and (obj_height > height or obj_width > width):
                rejected += 1
                continue
            accepted = False
            for _attempt in range(self.placement_attempts):
                if self.allow_partial_objects:
                    x = int(rng.integers(-obj_width + 1, width))
                    y = int(rng.integers(-obj_height + 1, height))
                else:
                    x = int(rng.integers(0, width - obj_width + 1))
                    y = int(rng.integers(0, height - obj_height + 1))
                left, top, right, bottom = max(0, x), max(0, y), min(width, x + obj_width), min(height, y + obj_height)
                source_x, source_y = left - x, top - y
                crop = mask[source_y:source_y + bottom - top, source_x:source_x + right - left]
                visible = np.zeros((height, width), dtype=bool)
                visible[top:bottom, left:right] = crop
                if not self._visible_enough(visible, full_area):
                    continue
                surviving = [previous & ~visible for previous in masks]
                if any(not self._visible_enough(remaining, instance["full_transformed_area"])
                       for remaining, instance in zip(surviving, instances)):
                    continue
                alpha = mask.astype(np.float32)
                if self.blend_sigma:
                    # Inward feathering keeps rendered foreground within the annotated support.
                    alpha *= cv2.GaussianBlur(alpha, (0, 0), self.blend_sigma, borderType=cv2.BORDER_CONSTANT)
                alpha = alpha[source_y:source_y + bottom - top, source_x:source_x + right - left, None]
                region = image[top:bottom, left:right]
                foreground = obj[source_y:source_y + bottom - top, source_x:source_x + right - left]
                region[:] = np.clip(np.rint(foreground * alpha + region * (1 - alpha)), 0, 255).astype(np.uint8)
                masks = surviving + [visible]
                class_index = self.class_index[source["category_id"]]
                instances.append({"image": source["image"], "mask": source["mask"],
                                  "category_id": source["category_id"], "class_index": class_index,
                                  "class_name": self.class_names[class_index],
                                  "requested_long_side_pixels": long_side, "rotation_degrees": angle,
                                  "position_xy": [x, y], "full_transformed_area": full_area})
                accepted = True
                break
            if not accepted:
                rejected += 1
        return image, masks, instances, rejected

    def sample_raw(self):
        """Return image, aligned xyxy/class labels, final masks and JSON-safe metadata.

        Strict count mode retries the whole composition a bounded number of times.
        The requested count is drawn once and successful returned samples advance
        the call index once, regardless of the number of composition attempts.
        """
        rng, worker_id = self._random_stream()
        # Retain the original background/count draw order for existing configs.
        background = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
        count = int(rng.integers(self.object_count[0], self.object_count[1] + 1))
        attempt_limit = self.composition_attempts if self.require_object_count else 1
        for composition_attempt in range(1, attempt_limit + 1):
            if composition_attempt > 1:
                background = self.backgrounds[int(rng.integers(len(self.backgrounds)))]
            image, masks, instances, rejected = self._compose(rng, background, count)
            if not self.require_object_count or len(instances) == count:
                break
        else:
            raise RuntimeError(
                f"Could not place requested {count} objects after {attempt_limit} composition attempts "
                f"(last attempt placed {len(instances)}). Reduce object_count or object sizes, "
                "increase the input size, or adjust placement/visibility settings."
            )
        labels = []
        for mask, instance in zip(masks, instances):
            bounds = bbox_from_mask(mask)
            instance.update(visible_area=int(mask.sum()), visible_fraction=float(mask.sum()) / instance["full_transformed_area"],
                            bbox_xyxy=bounds)
            labels.append([*bounds, instance["class_index"]])
        image = image.astype(np.float32) * float(rng.uniform(*self.brightness))
        if self.blur_sigma:
            image = cv2.GaussianBlur(image, (0, 0), self.blur_sigma)
        if self.noise_std:
            image += rng.normal(0, self.noise_std, image.shape).astype(np.float32)
        image = np.ascontiguousarray(np.clip(np.rint(image), 0, 255).astype(np.uint8))
        metadata = {"background": background["image"], "instances": instances,
                    "requested_object_count": count, "placed_object_count": len(instances),
                    "composition_attempts_used": composition_attempt,
                    "rejected_object_count": rejected, "distribution_source": self.distribution_source,
                    "worker_id": worker_id, "stream_seed": self._stream_seed,
                    "epoch": self.epoch, "call_index": self._calls,
                    "bbox_format": "xyxy_half_open", "mask_semantics": "final_visible_foreground_ownership"}
        self._calls += 1
        return {"image": image, "labels": np.asarray(labels, dtype=np.float32).reshape(-1, 5),
                "masks": masks, "metadata": metadata}

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        sample = self.sample_raw()
        image, labels = sample["image"], sample["labels"]
        if self.preproc is not None:
            # Original TrainTransform uses global RNGs. Isolate those draws so fresh
            # zero-worker loaders replay without changing training's RNG state.
            rng, _ = self._random_stream()
            augmentation_seed = int(rng.integers(0, 2**32))
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                random.seed(augmentation_seed)
                np.random.seed(augmentation_seed)
                image, labels = self.preproc(image, labels, self.input_dim)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        return image, labels, tuple(self.input_dim), np.array([int(index)], dtype=np.int64)
