"""Class-level uniqueness across raw samples, workers and the public workflows."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from offline_yolox.composition_options import add_composition_arguments, composition_config
from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = [1, 3, 7, 29, 51]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "source"
    make_sample_data(data, train_images=2, val_images=1)
    categories = [{"id": value, "name": f"truck_{value}"} for value in CLASS_IDS]
    for path in (data / "annotations").glob("*.json"):
        coco = json.loads(path.read_text())
        coco["categories"] = categories
        for index, annotation in enumerate(coco["annotations"]):
            annotation["category_id"] = CLASS_IDS[index % len(CLASS_IDS)]
        path.write_text(json.dumps(coco))
    manifest = data / "objects/manifest.json"
    entries = json.loads(manifest.read_text())
    assert len(entries) == 6
    for entry, category_id in zip(entries, [1, 1, 3, 7, 29, 51], strict=True):
        entry["category_id"] = category_id
    manifest.write_text(json.dumps(entries))
    values = json.loads((ROOT / "configs/online.json").read_text())
    values.update(mode="online_patch", num_classes=5, data_dir=str(data),
                  input_size=[64, 64], test_size=[64, 64], max_epoch=1,
                  scheduler="cos", batch_size=2, hsv_prob=0, flip_prob=0,
                  patch={"size": [64, 64]}, output_dir=str(tmp_path / "training"),
                  exp_name="unique_classes")
    values["copy_paste"].update(asset_root=str(data), epoch_length=4,
                                object_count=[5, 5], unique_classes=True,
                                unique_objects=False, require_object_count=True,
                                object_long_side_pixels=[8], object_size_weights=[1],
                                min_visible_pixels=2, min_visible_fraction=0.1)
    return values


def assert_unique_labels(sample, expected_count=None):
    instances = sample["metadata"]["instances"]
    if expected_count is not None:
        assert len(instances) == expected_count
    ids = [item["category_id"] for item in instances]
    assert len(ids) == len(set(ids))
    assert len(sample["labels"]) == len(sample["masks"]) == len(instances)
    for label, mask, item in zip(sample["labels"], sample["masks"], instances, strict=True):
        assert int(label[4]) == CLASS_IDS.index(item["category_id"])
        ys, xs = np.nonzero(mask)
        np.testing.assert_array_equal(label[:4], [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])


@pytest.mark.parametrize("mode", ["online", "online_patch"])
@pytest.mark.parametrize("unique_objects", [False, True])
@pytest.mark.parametrize("source_kind", ["manifest", "folders"])
def test_five_classes_once_each_with_random_sources_and_exact_replay(config, mode, unique_objects, source_kind):
    config["mode"] = mode
    settings = config["copy_paste"]
    settings["unique_objects"] = unique_objects
    if source_kind == "folders":
        data = Path(config["data_dir"])
        entries = json.loads((data / "objects/manifest.json").read_text())
        for index, entry in enumerate(entries):
            for folder, key in [("object", "image"), ("mask", "mask")]:
                target = data / folder / str(entry["category_id"]) / f"{index}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((data / entry[key]).read_bytes())
        settings.update(objects_dir="object", masks_dir="mask")
        settings.pop("objects_manifest", None)
    first, replay = [create_exp(copy.deepcopy(config)).get_dataset() for _ in range(2)]
    seen_sources = set()
    for _ in range(10):
        sample, repeated = first.sample_raw(), replay.sample_raw()
        assert_unique_labels(sample, expected_count=5)
        assert {item["category_id"] for item in sample["metadata"]["instances"]} == set(CLASS_IDS)
        seen_sources.update(item["image"] for item in sample["metadata"]["instances"] if item["category_id"] == 1)
        np.testing.assert_array_equal(sample["image"], repeated["image"])
        np.testing.assert_array_equal(sample["labels"], repeated["labels"])
        assert sample["metadata"] == repeated["metadata"]
    assert len(seen_sources) == 2


def test_random_subsets_and_empty_patches_keep_class_uniqueness(config):
    config["copy_paste"]["object_count"] = [0, 5]
    dataset = create_exp(config).get_dataset()
    counts, subsets = set(), set()
    for _ in range(40):
        sample = dataset.sample_raw()
        assert_unique_labels(sample)
        counts.add(len(sample["labels"]))
        subsets.add(tuple(sorted(item["category_id"] for item in sample["metadata"]["instances"])))
        assert len(sample["labels"]) == sample["metadata"]["requested_object_count"]
    assert {0, 5}.issubset(counts)
    assert len(subsets) > 5


def test_disabled_option_preserves_previous_file_uniqueness_behavior(config):
    settings = config["copy_paste"]
    settings.pop("unique_classes")
    settings.update(object_count=[6, 6], unique_objects=True)
    explicit = copy.deepcopy(config)
    explicit["copy_paste"]["unique_classes"] = False
    first, replay = [create_exp(values).get_dataset() for values in (config, explicit)]
    for _ in range(3):
        a, b = first.sample_raw(), replay.sample_raw()
        np.testing.assert_array_equal(a["image"], b["image"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        assert a["metadata"] == b["metadata"]
        instances = a["metadata"]["instances"]
        assert len(instances) == len({item["image"] for item in instances}) == 6
        assert len({item["category_id"] for item in instances}) == 5


def test_count_limit_uses_classes_with_assets_not_validation_category_count(config):
    path = Path(config["data_dir"]) / "objects/manifest.json"
    entries = json.loads(path.read_text())
    entries[-1]["category_id"] = 29
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match=r"unique_classes.*maximum 5.*4 classes"):
        create_exp(config)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_unique_classes_requires_boolean(config, value):
    config["copy_paste"]["unique_classes"] = value
    with pytest.raises(ValueError, match="unique_classes must be boolean"):
        create_exp(config)


def test_unique_classes_cli_overrides_copy_settings_and_reject_original_mode(config):
    parser = argparse.ArgumentParser()
    add_composition_arguments(parser)
    before = copy.deepcopy(config)
    disabled = composition_config(config, parser.parse_args(["--no-unique-classes"]))
    assert disabled["copy_paste"]["unique_classes"] is False
    enabled = composition_config(disabled, parser.parse_args(["--unique-classes"]))
    assert enabled["copy_paste"]["unique_classes"] is True
    assert config == before
    with pytest.raises(ValueError, match="online"):
        composition_config({**config, "mode": "original_patch"}, parser.parse_args(["--unique-classes"]))


@pytest.mark.parametrize("workers", [0, 2])
def test_class_uniqueness_in_training_loaders_and_worker_replay(config, workers):
    config["num_workers"] = workers
    exp = create_exp(config)
    first, replay = [exp.get_data_loader(2, finite=True, epoch=1) for _ in range(2)]
    for a, b in zip(first, replay, strict=True):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for targets in a[1]:
            valid = targets[(targets[:, 3] > 0) & (targets[:, 4] > 0)]
            assert len(valid) == 5 and set(valid[:, 0].tolist()) == set(range(5))


@pytest.mark.parametrize("mode", ["online", "online_patch"])
def test_public_preview_training_and_export_use_class_unique_override(config, tmp_path, mode):
    config["mode"] = mode
    config["copy_paste"]["unique_classes"] = False
    path = tmp_path / "online.json"
    path.write_text(json.dumps(config))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    preview, exported = tmp_path / "preview", tmp_path / "exported"
    commands = [("preview", ["--output", str(preview), "--count", "3"]),
                ("train", ["--device", "cpu"])]
    # The existing COCO exporter accepts online mode, not online_patch mode.
    if mode == "online":
        commands.append(("export", ["--output", str(exported), "--count", "2"]))
    for command, extra in commands:
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), command,
                                 "--config", str(path), "--unique-classes", *extra],
                                cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((preview / "manifest.json").read_text())
    assert manifest["config"]["copy_paste"]["unique_classes"] is True
    for filename in manifest["samples"]:
        items = json.loads((preview / filename).read_text())["instances"]
        assert len(items) == 5 and {item["category_id"] for item in items} == set(CLASS_IDS)
    if mode == "online":
        coco = json.loads((exported / "annotations/instances_train2017.json").read_text())
        for image in coco["images"]:
            items = [item for item in coco["annotations"] if item["image_id"] == image["id"]]
            assert len(items) == 5 and {item["category_id"] for item in items} == set(CLASS_IDS)
    output = Path(config["output_dir"]) / config["exp_name"]
    checkpoint = torch.load(output / "latest_ckpt.pth", weights_only=True)
    assert checkpoint["config"]["copy_paste"]["unique_classes"] is True
    report = json.loads((output / "diagnostic_report.json").read_text())
    assert report["epochs"][0]["images"] == 4
    assert math.isfinite(report["epochs"][0]["mean_loss"]) and report["epochs"][0]["mean_loss"] > 0
