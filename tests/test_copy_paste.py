"""Verify fresh online composition, visible-mask labels, workers and real training."""
import copy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from offline_yolox.copy_paste import OnlineCopyPasteDataset
from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data
from offline_yolox.train import load_checkpoint, run_cpu_diagnostic


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    make_sample_data(data, train_images=4, val_images=2)
    return {
        "mode": "online", "preset": "nano", "num_classes": 1,
        "data_dir": str(data), "train_ann": "instances_train2017.json",
        "val_ann": "instances_val2017.json", "input_size": [128, 128],
        "test_size": [128, 128], "batch_size": 2, "max_epoch": 2,
        "warmup_epochs": 0, "no_aug_epochs": 0, "num_workers": 0,
        "seed": 1729, "mosaic_prob": 0.0, "hsv_prob": 0.0,
        "flip_prob": 0.0, "enable_mixup": False, "multiscale_range": 0,
        "eval_interval": 1, "output_dir": str(tmp_path / "out"),
        "exp_name": "online_test",
        "copy_paste": {
            "asset_root": str(data), "objects_manifest": "objects/manifest.json",
            "backgrounds_manifest": "backgrounds/manifest.json", "epoch_length": 4,
            "object_count": [2, 4], "object_long_side_pixels": [14, 22, 30],
            "object_size_weights": [1, 2, 1], "rotation_degrees": [-180, 180],
            "min_visible_fraction": 0.25, "min_visible_pixels": 4,
            "placement_attempts": 30, "gaussian_blend_sigma": 0.6,
            "blur_sigma": 0.3, "noise_std": 1.0, "brightness_range": [0.9, 1.1],
        },
    }


def make_dataset(config, *, epoch=0, cls=OnlineCopyPasteDataset):
    return cls(config=copy.deepcopy(config["copy_paste"]), input_size=(128, 128),
               class_ids=[1], class_names=["sample_vehicle"], seed=config["seed"],
               epoch=epoch, preproc=None)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def assert_visible_labels(sample):
    assert sample["image"].shape == (128, 128, 3)
    assert sample["image"].dtype == np.uint8
    labels = sample["labels"]
    assert labels.dtype == np.float32
    assert labels.ndim == 2 and labels.shape[1] == 5
    assert len(labels) > 0
    assert len(sample["masks"]) == len(labels) == len(sample["metadata"]["instances"])
    coverage = np.zeros((128, 128), dtype=np.uint16)
    for label, mask, instance in zip(labels, sample["masks"], sample["metadata"]["instances"]):
        assert mask.shape == (128, 128)
        assert set(np.unique(mask)).issubset({0, 1, 255, False, True})
        visible = mask > 0
        ys, xs = np.nonzero(visible)
        assert len(xs) >= 4
        expected = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        np.testing.assert_array_equal(label[:4], expected)
        np.testing.assert_array_equal(instance["bbox_xyxy"], expected)
        assert label[4] == 0
        assert instance["visible_area"] == len(xs)
        assert instance["visible_fraction"] == pytest.approx(
            len(xs) / instance["full_transformed_area"])
        assert 0 < instance["visible_fraction"] <= 1
        assert instance["class_index"] == 0
        coverage += visible
    # Earlier objects must lose the pixels covered by later pasted objects.
    assert coverage.max() == 1


def file_snapshot(path):
    return {str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in path.rglob("*") if item.is_file()}


def test_rotation_and_occlusion_labels_follow_visible_masks(config):
    config["copy_paste"].update({"object_count": [6, 6],
                                "object_long_side_pixels": [80],
                                "object_size_weights": [1],
                                "rotation_degrees": [37, 37],
                                "min_visible_fraction": 0.05})
    dataset = make_dataset(config)
    saw_occlusion = False
    for _ in range(12):
        sample = dataset.sample_raw()
        assert_visible_labels(sample)
        for instance in sample["metadata"]["instances"]:
            assert instance["rotation_degrees"] == 37
            assert instance["requested_long_side_pixels"] == 80
            assert instance["visible_fraction"] >= 0.05
            saw_occlusion |= instance["visible_area"] < instance["full_transformed_area"]
    assert saw_occlusion, "The crowded fixture must exercise actual partial occlusion"


def test_same_index_is_recomposed_and_seeded_replays_are_exact(config):
    first, second = make_dataset(config), make_dataset(config)
    samples = [first.sample_raw() for _ in range(5)]
    for expected in samples:
        assert_nested_equal(expected, second.sample_raw())
    assert any(not np.array_equal(samples[0]["image"], item["image"])
               for item in samples[1:])
    assert [item["metadata"]["call_index"] for item in samples] == list(range(5))
    indexed = make_dataset(config)
    first_item, second_item = indexed[0], indexed[0]
    assert not np.array_equal(first_item[0], second_item[0])


def test_epoch_changes_composition_stream(config):
    zero, one = make_dataset(config, epoch=0), make_dataset(config, epoch=1)
    sample_zero, sample_one = zero.sample_raw(), one.sample_raw()
    assert sample_zero["metadata"]["stream_seed"] != sample_one["metadata"]["stream_seed"]
    assert not np.array_equal(sample_zero["image"], sample_one["image"])


def test_compositing_never_writes_training_images(config, monkeypatch):
    from pathlib import Path
    data = Path(config["data_dir"])
    before = file_snapshot(data)

    def forbidden_write(*args, **kwargs):
        raise AssertionError("Online composition must not write an image")

    monkeypatch.setattr(cv2, "imwrite", forbidden_write)
    dataset = make_dataset(config)
    for index in range(12):
        assert_visible_labels(dataset.sample_raw())
        dataset[index % len(dataset)]
    assert file_snapshot(data) == before


def test_original_and_online_modes_share_official_model(config):
    from yolox.models import YOLOX
    online = create_exp(config)
    original = create_exp({**config, "mode": "original"})
    assert isinstance(online.get_model(), YOLOX)
    assert isinstance(original.get_model(), YOLOX)
    assert type(online.get_model()) is type(original.get_model())
    assert isinstance(online.get_dataset(), OnlineCopyPasteDataset)
    assert online.get_model().head.num_classes == 1


def test_yolox_finite_loader_has_valid_targets_and_epoch_length(config):
    exp = create_exp(config)
    loader = exp.get_data_loader(2, finite=True, epoch=0)
    visited = 0
    for images, targets, info, image_ids in loader:
        assert images.shape == (2, 3, 128, 128)
        assert targets.shape[0] == 2 and targets.shape[-1] == 5
        # TrainTransform converts xyxy,class to class,cx,cy,width,height.
        valid = targets[..., 3] > 0
        assert valid.any(dim=1).all()
        assert (targets[..., 4][valid] > 0).all()
        assert (targets[..., 0][valid] == 0).all()
        visited += len(images)
    assert len(loader) == 2
    assert visited == len(loader.dataset) == config["copy_paste"]["epoch_length"]


@pytest.mark.parametrize("workers", [0, 2])
def test_training_loader_replays_with_yolox_color_and_flip_augmentation(config, workers):
    augmented = {**config, "num_workers": workers, "hsv_prob": 1.0, "flip_prob": 0.5}
    first = list(create_exp(augmented).get_data_loader(2, finite=True, epoch=3))
    second = list(create_exp(augmented).get_data_loader(2, finite=True, epoch=3))
    assert_nested_equal(first, second)


def test_native_yolox_sampler_and_no_mosaic_transition_keep_online_compositing(config):
    loader = create_exp(config).get_data_loader(2, finite=False)
    iterator = iter(loader)
    initial = next(iterator)
    loader.close_mosaic()
    following = next(iterator)
    for images, targets, _, _ in (initial, following):
        assert images.shape == (2, 3, 128, 128)
        assert (targets[..., 3] > 0).any(dim=1).all()
    assert not torch.equal(initial[0], following[0])


class RawDataset(OnlineCopyPasteDataset):
    """Top-level class and collator keep the spawn path genuinely picklable."""
    def __getitem__(self, index):
        return self.sample_raw()


def raw_collate(samples):
    return samples


def collect_raw(config, workers, context=None):
    kwargs = {"multiprocessing_context": context} if workers and context else {}
    loader = torch.utils.data.DataLoader(
        make_dataset(config, cls=RawDataset), batch_size=1, num_workers=workers,
        collate_fn=raw_collate, generator=torch.Generator().manual_seed(4107),
        **kwargs,
    )
    return [batch[0] for batch in loader]


@pytest.mark.parametrize("workers,context", [(0, None), (2, "fork"), (2, "spawn")])
def test_worker_reproducibility_and_independent_streams(config, workers, context):
    first = collect_raw(config, workers, context)
    second = collect_raw(config, workers, context)
    assert_nested_equal(first, second)
    for sample in first:
        assert_visible_labels(sample)
    if workers:
        by_worker = {}
        for sample in first:
            metadata = sample["metadata"]
            by_worker.setdefault(metadata["worker_id"], []).append(metadata)
        assert set(by_worker) == {0, 1}
        seeds = {items[0]["stream_seed"] for items in by_worker.values()}
        assert len(seeds) == 2
        for items in by_worker.values():
            assert [item["call_index"] for item in items] == [0, 1]


@pytest.mark.parametrize("invalid", ["missing_image", "unreadable_image", "wrong_mask_size",
                                    "empty_mask", "nonbinary_mask", "unknown_category"])
def test_bad_exported_assets_fail_before_training(config, invalid):
    from pathlib import Path
    data = Path(config["data_dir"])
    manifest_path = data / "objects/manifest.json"
    objects = json.loads(manifest_path.read_text())
    if invalid == "missing_image":
        objects[0]["image"] = "objects/does_not_exist.png"
    elif invalid == "unreadable_image":
        (data / objects[0]["image"]).write_bytes(b"not an image")
    elif invalid == "wrong_mask_size":
        cv2.imwrite(str(data / objects[0]["mask"]), np.ones((2, 3), dtype=np.uint8) * 255)
    elif invalid == "empty_mask":
        mask = cv2.imread(str(data / objects[0]["mask"]), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(data / objects[0]["mask"]), np.zeros_like(mask))
    elif invalid == "nonbinary_mask":
        mask = cv2.imread(str(data / objects[0]["mask"]), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(data / objects[0]["mask"]), np.full_like(mask, 7))
    else:
        objects[0]["category_id"] = 999
    manifest_path.write_text(json.dumps(objects))
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        make_dataset(config)


def test_noncontiguous_export_category_maps_to_contiguous_class_index(config):
    from pathlib import Path
    data = Path(config["data_dir"])
    path = data / "objects/manifest.json"
    objects = json.loads(path.read_text())
    for entry in objects:
        entry["category_id"] = 17
    path.write_text(json.dumps(objects))
    dataset = OnlineCopyPasteDataset(config=config["copy_paste"], input_size=(128, 128),
                                    class_ids=[17], class_names=["vehicle"], seed=1729,
                                    epoch=0, preproc=None)
    assert_visible_labels(dataset.sample_raw())


def test_true_online_training_and_exact_epoch_resume(config, tmp_path):
    full = run_cpu_diagnostic(create_exp({**config, "output_dir": str(tmp_path / "full")}))
    full_state = load_checkpoint(full["checkpoint"])
    split_config = {**config, "output_dir": str(tmp_path / "split")}
    first = run_cpu_diagnostic(create_exp(split_config), stop_after_epoch=1)
    first_state = load_checkpoint(first["checkpoint"])
    resumed = run_cpu_diagnostic(create_exp(split_config), resume=first["checkpoint"])
    resumed_state = load_checkpoint(resumed["checkpoint"])
    assert full["completed_epochs"] == resumed["completed_epochs"] == 2
    assert resumed["start_epoch"] == 1
    assert full_state["ema_updates"] == 4
    for name in ("model", "optimizer", "ema_model", "scaler", "rng"):
        assert_nested_equal(full_state[name], resumed_state[name])
    assert any(not torch.equal(first_state["model"][name], value)
               for name, value in full_state["model"].items() if value.is_floating_point())
    assert all(epoch["iterations"] == 2 and epoch["images"] == 4
               and np.isfinite(epoch["mean_loss"]) and epoch["mean_loss"] > 0
               for epoch in full["epochs"])
    assert full["epochs"][-1]["evaluation"]["images"] == 2
    assert full["native_cuda_trainer"] is False


def test_online_requires_only_validation_coco_and_exported_assets(config):
    from pathlib import Path
    import shutil
    data = Path(config["data_dir"])
    (data / "annotations" / config.pop("train_ann")).unlink()
    shutil.rmtree(data / "train2017")
    exp = create_exp(config)
    assert isinstance(exp.get_dataset(), OnlineCopyPasteDataset)
    assert_visible_labels(exp.get_dataset().sample_raw())
    assert len(exp.get_eval_dataset()) == 2
    images, targets, _, _ = next(iter(exp.get_data_loader(2, finite=True)))
    assert images.shape == (2, 3, 128, 128)
    assert (targets[..., 3] > 0).any(dim=1).all()


@pytest.mark.parametrize("angle,expected_shape", [(0, (5, 9)), (90, (9, 5)),
                                                  (180, (5, 9)), (270, (9, 5))])
def test_cardinal_rotations_preserve_nonsquare_foreground(angle, expected_shape):
    from offline_yolox.copy_paste import bbox_from_mask, transform_object
    source = np.full((5, 9, 3), (37, 81, 129), dtype=np.uint8)
    source_mask = np.ones((5, 9), dtype=bool)
    image, mask = transform_object(source, source_mask, long_side_pixels=9,
                                   rotation_degrees=angle)
    assert mask.shape == expected_shape
    assert image.shape == (*expected_shape, 3)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == int(source_mask.sum()) == 45
    assert mask.all()
    assert bbox_from_mask(mask) == [0, 0, expected_shape[1], expected_shape[0]]
    np.testing.assert_array_equal(image, np.full(image.shape, (37, 81, 129), dtype=np.uint8))


def test_preview_exports_images_visible_masks_and_matching_metadata(config, tmp_path):
    from pathlib import Path
    from offline_yolox.preview import save_preview
    data = Path(config["data_dir"])
    before = file_snapshot(data)
    output = tmp_path / "preview"
    result = save_preview(config, output, count=2)
    assert result["count"] == 2
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["count"] == len(manifest["samples"]) == 2
    assert manifest["training_writes_synthetic_images"] is False
    expected_files = {"manifest.json"}
    for index, metadata_name in enumerate(manifest["samples"]):
        expected_files.add(metadata_name)
        metadata = json.loads((output / metadata_name).read_text())
        assert metadata["call_index"] == index
        assert len(metadata["instances"]) == len(metadata["masks"]) > 0
        assert len(metadata["instances"]) == len(metadata["labels_xyxy_class_index"])
        for key in ("image", "preview"):
            expected_files.add(metadata[key])
            image = cv2.imread(str(output / metadata[key]))
            assert image.shape == (128, 128, 3)
        coverage = np.zeros((128, 128), dtype=np.uint16)
        for instance, mask_name, label in zip(metadata["instances"], metadata["masks"],
                                              metadata["labels_xyxy_class_index"]):
            expected_files.add(mask_name)
            mask = cv2.imread(str(output / mask_name), cv2.IMREAD_UNCHANGED)
            assert mask.dtype == np.uint8 and mask.shape == (128, 128)
            assert set(np.unique(mask)) == {0, 255}
            ys, xs = np.nonzero(mask)
            bounds = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            assert instance["bbox_xyxy"] == bounds
            assert label[:4] == bounds
            assert label[4] == instance["class_index"] == 0
            assert len(xs) == instance["visible_area"]
            coverage += mask > 0
        assert coverage.max() == 1
    assert {path.name for path in output.iterdir()} == expected_files
    assert file_snapshot(data) == before
    with pytest.raises(FileExistsError):
        save_preview(config, output, count=2)
