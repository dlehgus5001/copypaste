"""Verify fixed-count composition and exported COCO data through real training."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import pytest
import torch

from offline_yolox.experiment import create_exp
from offline_yolox.export import export_coco
from offline_yolox.sample_data import make_sample_data
from offline_yolox.train import run_cpu_diagnostic


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "source data"
    make_sample_data(data, train_images=2, val_images=1)
    # A COCO category ID is not a zero-based class index. Use two nonconsecutive
    # IDs to catch exporters that accidentally save the training class index.
    categories = [{"id": 7, "name": "vehicle_a"}, {"id": 29, "name": "vehicle_b"}]
    for path in (data / "annotations").glob("*.json"):
        annotations = json.loads(path.read_text())
        annotations["categories"] = categories
        for index, annotation in enumerate(annotations["annotations"]):
            annotation["category_id"] = categories[index % 2]["id"]
        path.write_text(json.dumps(annotations))
    manifest_path = data / "objects/manifest.json"
    objects = json.loads(manifest_path.read_text())
    for index, entry in enumerate(objects):
        entry["category_id"] = categories[index % 2]["id"]
    manifest_path.write_text(json.dumps(objects))
    return {
        "mode": "online", "preset": "nano", "num_classes": 2,
        "data_dir": str(data), "train_ann": "instances_train2017.json",
        "val_ann": "instances_val2017.json", "input_size": [128, 128],
        "test_size": [128, 128], "batch_size": 2, "max_epoch": 1,
        "warmup_epochs": 0, "no_aug_epochs": 0, "num_workers": 0,
        "seed": 1729, "mosaic_prob": 0.0, "hsv_prob": 0.0,
        "flip_prob": 0.0, "enable_mixup": False, "multiscale_range": 0,
        "eval_interval": 1, "output_dir": str(tmp_path / "training"),
        "exp_name": "export_training",
        "copy_paste": {
            "asset_root": str(data), "objects_manifest": "objects/manifest.json",
            "backgrounds_manifest": "backgrounds/manifest.json", "epoch_length": 2,
            "object_count": [5, 5], "require_object_count": True,
            "unique_objects": True, "object_long_side_range": [12, 18],
            "rotation_degrees": [-180, 180], "placement_attempts": 30,
            "min_visible_fraction": 0.25, "min_visible_pixels": 4,
        },
    }


def assert_same_sample(left, right):
    np.testing.assert_array_equal(left["image"], right["image"])
    np.testing.assert_array_equal(left["labels"], right["labels"])
    assert left["metadata"] == right["metadata"]
    for left_mask, right_mask in zip(left["masks"], right["masks"], strict=True):
        np.testing.assert_array_equal(left_mask, right_mask)


def test_five_unique_objects_have_controlled_sizes_and_seeded_replay(config):
    first = create_exp(config).get_dataset()
    replay = create_exp(copy.deepcopy(config)).get_dataset()
    seen_sizes = set()
    previous = None
    for index in range(8):
        sample = first.sample_raw()
        assert_same_sample(sample, replay.sample_raw())
        instances = sample["metadata"]["instances"]
        assert len(instances) == len(sample["masks"]) == len(sample["labels"]) == 5
        assert sample["metadata"]["requested_object_count"] == 5
        assert sample["metadata"]["placed_object_count"] == 5
        assert sample["metadata"]["call_index"] == index
        assert len({(item["image"], item["mask"]) for item in instances}) == 5
        assert {item["category_id"] for item in instances} == {7, 29}
        for item in instances:
            size = item["requested_long_side_pixels"]
            assert int(size) == size and 12 <= size <= 18
            seen_sizes.add(size)
        if previous is not None:
            assert not np.array_equal(sample["image"], previous)
        previous = sample["image"]
    assert len(seen_sizes) > 1


@pytest.mark.parametrize("invalid", [[18, 12], [1, 12], [12.5, 18], [True, 18],
                                   [12, float("inf")], [12, 18, 24]])
def test_invalid_object_size_ranges_fail_before_training(config, invalid):
    config["copy_paste"]["object_long_side_range"] = invalid
    with pytest.raises(ValueError, match="object_long_side_range"):
        create_exp(config)


@pytest.mark.parametrize("extra", [{"object_long_side_pixels": [12]},
                                  {"object_size_weights": [1]}])
def test_range_and_discrete_size_settings_cannot_silently_conflict(config, extra):
    config["copy_paste"].update(extra)
    with pytest.raises(ValueError, match="object_long_side_range"):
        create_exp(config)


def test_unique_object_count_cannot_exceed_the_source_bank(config):
    config["copy_paste"]["object_count"] = [7, 7]
    with pytest.raises(ValueError, match="object_count"):
        create_exp(config)


def test_duplicate_manifest_pairs_cannot_bypass_unique_objects(config):
    path = Path(config["data_dir"]) / "objects/manifest.json"
    objects = json.loads(path.read_text())
    objects.append(dict(objects[0]))
    path.write_text(json.dumps(objects))
    with pytest.raises(ValueError, match="duplicate"):
        create_exp(config)


def test_impossible_strict_count_fails_after_bounded_composition_attempts(config):
    config["copy_paste"].update(
        object_count=[1, 1], object_long_side_range=[1000, 1000],
        allow_partial_objects=False, composition_attempts=2,
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match=r"requested 1 objects after 2 composition attempts"):
            create_exp(config).get_dataset().sample_raw()


def test_exported_coco_preserves_pixels_categories_and_visible_masks(config, tmp_path):
    initial_config = copy.deepcopy(config)
    expected = create_exp(config).get_dataset()
    result = export_coco(config, tmp_path / "export", count=3)
    assert config == initial_config
    assert result["count"] == 3
    coco = COCO(result["annotations"])
    assert coco.getCatIds() == [7, 29]
    assert len(coco.imgs) == 3 and len(coco.anns) == 15
    assert len(set(coco.anns)) == 15
    output = Path(result["output"])
    for image_id, image_info in coco.imgs.items():
        sample = expected.sample_raw()
        image_path = output / "train2017" / image_info["file_name"]
        assert image_path.suffix == ".png"
        np.testing.assert_array_equal(cv2.imread(str(image_path)), sample["image"])
        metadata = json.loads((output / "metadata" / (image_path.stem + ".json")).read_text())
        assert metadata["instances"] == sample["metadata"]["instances"]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        assert len(annotations) == 5
        coverage = np.zeros((128, 128), dtype=np.uint8)
        for annotation, expected_mask, instance in zip(
                annotations, sample["masks"], sample["metadata"]["instances"], strict=True):
            assert annotation["category_id"] == instance["category_id"]
            assert annotation["category_id"] in (7, 29)
            assert annotation["iscrowd"] == 0
            assert isinstance(annotation["segmentation"]["counts"], str)
            decoded = mask_utils.decode(annotation["segmentation"])
            np.testing.assert_array_equal(decoded, expected_mask)
            np.testing.assert_array_equal(coco.annToMask(annotation), expected_mask)
            ys, xs = np.nonzero(decoded)
            expected_bbox = [int(xs.min()), int(ys.min()),
                             int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            assert annotation["bbox"] == expected_bbox
            assert annotation["area"] == int(decoded.sum()) == instance["visible_area"]
            coverage += decoded
        assert coverage.max() == 1


def test_exported_training_config_trains_and_uses_original_held_out_validation(config, tmp_path):
    # Online ignores Mosaic/MixUp fields. Switching to ordinary training must
    # not suddenly merge several exported five-object images into one input.
    config.update(mosaic_prob=1.0, enable_mixup=True, mixup_prob=1.0, multiscale_range=5)
    result = export_coco(config, tmp_path / "trainable", count=2)
    training_config_path = Path(result["training_config"])
    assert training_config_path.name == "train_config.json"
    training_config = json.loads(training_config_path.read_text())
    assert training_config["mode"] == "original"
    exp = create_exp(training_config_path)
    dataset = exp.get_dataset()
    assert len(dataset) == 2
    assert dataset.class_ids == [7, 29]
    for index in range(len(dataset)):
        assert len(dataset.load_anno(index)) == 5
    evaluation = exp.get_eval_dataset()
    assert len(evaluation) == 1
    source_image = Path(config["data_dir"]) / "val2017/sample_0000.png"
    np.testing.assert_array_equal(evaluation.load_image(0), cv2.imread(str(source_image)))
    source_annotations = Path(config["data_dir"]) / "annotations/instances_val2017.json"
    actual_annotation_path = Path(evaluation.data_dir) / "annotations" / evaluation.json_file
    assert actual_annotation_path.resolve() == source_annotations.resolve()
    assert actual_annotation_path.resolve() != Path(result["annotations"]).resolve()
    images, targets, _, _ = next(iter(exp.get_data_loader(2, finite=True)))
    assert images.shape == (2, 3, 128, 128)
    assert ((targets[..., 3] > 0) & (targets[..., 4] > 0)).sum(dim=1).tolist() == [5, 5]
    report = run_cpu_diagnostic(exp)
    assert report["completed_epochs"] == 1
    assert report["epochs"][0]["images"] == 2
    assert report["epochs"][0]["iterations"] == 1
    assert math.isfinite(report["epochs"][0]["mean_loss"])
    assert report["epochs"][0]["mean_loss"] > 0
    assert report["epochs"][0]["evaluation"]["images"] == 1
    assert Path(report["checkpoint"]).is_file()


@pytest.mark.parametrize("size_arguments,minimum,maximum", [
    (["--object-size", "12"], 12, 12),
    (["--object-size-range", "12", "18"], 12, 18),
])
def test_python_export_cli_overrides_discrete_sizes_and_saves_trainable_json(
        config, tmp_path, size_arguments, minimum, maximum):
    # The shipped settings use weighted discrete sizes. Explicit CLI sizes must
    # replace both old fields, rather than trigger the range conflict check.
    config["copy_paste"].pop("object_long_side_range")
    config["copy_paste"].update(object_count=[1, 2], object_long_side_pixels=[50, 60],
                                object_size_weights=[1, 3], require_object_count=False,
                                unique_objects=False)
    config_path = tmp_path / "online.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "CLI generated data"
    unrelated_working_dir = tmp_path / "unrelated working directory"
    unrelated_working_dir.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["YOLOX_PYTHON"] = str(tmp_path / "missing-python")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "export", "--config", str(config_path),
         "--output", str(output), "--count", "2", "--objects-per-image", "5",
         *size_arguments, "--unique-objects", "--require-object-count"],
        cwd=unrelated_working_dir, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Verified: IPv4/IPv6 sockets denied" in completed.stderr
    coco = COCO(str(output / "annotations/instances_train2017.json"))
    assert len(coco.imgs) == 2 and len(coco.anns) == 10
    for image_id in coco.imgs:
        assert len(coco.getAnnIds(imgIds=[image_id])) == 5
    for metadata_path in (output / "metadata").glob("*.json"):
        instances = json.loads(metadata_path.read_text())["instances"]
        assert len(instances) == 5
        assert len({(item["image"], item["mask"]) for item in instances}) == 5
        assert all(minimum <= item["requested_long_side_pixels"] <= maximum for item in instances)
    assert len(create_exp(output / "train_config.json").get_dataset()) == 2


def test_export_never_replaces_existing_outputs_or_uses_symlink_destinations(config, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("user data")
    with pytest.raises(FileExistsError):
        export_coco(config, output, count=1)
    assert existing.read_text() == "user data"
    assert set(output.iterdir()) == {existing}
    target = tmp_path / "empty target"
    target.mkdir()
    link = tmp_path / "linked export"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_coco(config, link, count=1)
    assert not list(target.iterdir())


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_export_requires_positive_integer_image_count(config, tmp_path, count):
    with pytest.raises(ValueError, match="count"):
        export_coco(config, tmp_path / "invalid", count=count)
    assert not (tmp_path / "invalid").exists()
