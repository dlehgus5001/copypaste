"""Per-category copy-paste sizes, fallback, CLI priority and real CPU training."""
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
from offline_yolox.export import export_coco
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]
SIZE_KEY = "object_long_side_range_by_category"


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "source"
    make_sample_data(data, train_images=2, val_images=1)
    categories = [{"id": 7, "name": "small_truck"}, {"id": 29, "name": "large_truck"}]
    for path in (data / "annotations").glob("*.json"):
        coco = json.loads(path.read_text())
        coco["categories"] = categories
        for index, annotation in enumerate(coco["annotations"]):
            annotation["category_id"] = categories[index % 2]["id"]
        path.write_text(json.dumps(coco))
    path = data / "objects/manifest.json"
    entries = json.loads(path.read_text())
    for index, entry in enumerate(entries):
        entry["category_id"] = categories[index % 2]["id"]
    path.write_text(json.dumps(entries))
    values = json.loads((ROOT / "configs/samples/online.json").read_text())
    values.update(mode="online_patch", data_dir=str(data), num_classes=2,
                  input_size=[128, 128], test_size=[128, 128], patch={"size": [64, 64]},
                  max_epoch=1, scheduler="cos", hsv_prob=0, flip_prob=0,
                  output_dir=str(tmp_path / "training"), exp_name="class_sizes", use_fast_coco_eval=False)
    settings = values["copy_paste"]
    settings.pop("object_long_side_pixels", None)
    settings.pop("object_size_weights", None)
    settings.update(asset_root=str(data), epoch_length=4, object_count=[2, 2],
                    unique_classes=True, require_object_count=True,
                    rotation_degrees=[0, 0], allow_partial_objects=False, min_visible_fraction=1.0,
                    object_long_side_range=[8, 10],
                    object_long_side_range_by_category={"7": [12, 12], "29": [28, 28]})
    return values


@pytest.mark.parametrize("mode", ["online", "online_patch"])
def test_category_sizes_follow_coco_ids_and_replay_with_valid_masks(config, mode):
    config["mode"] = mode
    first, replay = [create_exp(copy.deepcopy(config)).get_dataset() for _ in range(2)]
    for _ in range(8):
        sample, repeated = first.sample_raw(), replay.sample_raw()
        np.testing.assert_array_equal(sample["image"], repeated["image"])
        np.testing.assert_array_equal(sample["labels"], repeated["labels"])
        assert sample["metadata"] == repeated["metadata"]
        assert len(sample["labels"]) == 2
        for instance, label, mask in zip(sample["metadata"]["instances"], sample["labels"], sample["masks"], strict=True):
            expected = 12 if instance["category_id"] == 7 else 28
            assert instance["requested_long_side_pixels"] == expected
            ys, xs = np.nonzero(mask)
            np.testing.assert_array_equal(label[:4], [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            assert expected - 1 <= max(label[2] - label[0], label[3] - label[1]) <= expected


@pytest.mark.parametrize("fallback", ["range", "discrete"])
def test_unmapped_class_uses_global_fallback(config, fallback):
    settings = config["copy_paste"]
    settings[SIZE_KEY] = {"7": [16, 16]}
    if fallback == "range":
        settings["object_long_side_range"] = [9, 9]
    else:
        settings.pop("object_long_side_range")
        settings.update(object_long_side_pixels=[9], object_size_weights=[1])
    sample = create_exp(config).get_dataset().sample_raw()
    assert {item["category_id"]: item["requested_long_side_pixels"]
            for item in sample["metadata"]["instances"]} == {7: 16, 29: 9}


def test_empty_category_sizes_preserve_previous_pixels_and_rng(config):
    config["copy_paste"].pop(SIZE_KEY)
    explicit = copy.deepcopy(config)
    explicit["copy_paste"][SIZE_KEY] = {}
    first, replay = [create_exp(values).get_dataset() for values in (config, explicit)]
    for _ in range(4):
        a, b = first.sample_raw(), replay.sample_raw()
        np.testing.assert_array_equal(a["image"], b["image"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        assert a["metadata"] == b["metadata"]


@pytest.mark.parametrize("mapping", [None, [], {"0": [8, 12]}, {"small_truck": [8, 12]},
                                    {"07": [8, 12]}, {"7": [16, 8]}, {"7": [True, 12]}, {"7": [1, 12]}])
def test_invalid_per_category_ranges_rejected_before_training(config, mapping):
    config["copy_paste"][SIZE_KEY] = mapping
    with pytest.raises(ValueError, match=SIZE_KEY):
        create_exp(config)


@pytest.mark.parametrize("options", [["--object-size", "14"], ["--object-size-range", "14", "18"]])
def test_explicit_global_cli_sizes_override_category_json_without_mutating_it(config, options, capsys):
    parser = argparse.ArgumentParser()
    add_composition_arguments(parser)
    original = copy.deepcopy(config)
    changed = composition_config(config, parser.parse_args(options))
    assert config == original and SIZE_KEY not in changed["copy_paste"]
    assert SIZE_KEY in capsys.readouterr().err
    for instance in create_exp(changed).get_dataset().sample_raw()["metadata"]["instances"]:
        assert 14 <= instance["requested_long_side_pixels"] <= (14 if options[0] == "--object-size" else 18)
    unrelated = composition_config(config, parser.parse_args(["--unique-classes"]))
    assert unrelated["copy_paste"][SIZE_KEY] == original["copy_paste"][SIZE_KEY]


@pytest.mark.parametrize("workers", [0, 2])
def test_category_sizes_in_real_training_loader_and_worker_replay(config, workers):
    config["num_workers"] = workers
    exp = create_exp(config)
    first, replay = [exp.get_data_loader(2, finite=True, epoch=1) for _ in range(2)]
    for a, b in zip(first, replay, strict=True):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for targets in a[1]:
            valid = targets[(targets[:, 3] > 0) & (targets[:, 4] > 0)]
            assert len(valid) == 2
            for target in valid:
                expected = 12 if int(target[0]) == 0 else 28
                assert expected - 1 <= max(target[3:5]) <= expected


@pytest.mark.parametrize("mode", ["online", "online_patch"])
def test_public_preview_and_training_save_class_sizes_and_actual_scale(config, tmp_path, mode):
    config["mode"] = mode
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    preview = tmp_path / "preview"
    for command, extra in [("preview", ["--count", "2", "--output", str(preview)]),
                           ("train", ["--device", "cpu"])]:
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), command,
                                 "--config", str(path), *extra], cwd=tmp_path, env=environment,
                                text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    sample = json.loads((preview / "sample_0000.json").read_text())
    assert {item["category_id"]: item["requested_long_side_pixels"] for item in sample["instances"]} == {7: 12, 29: 28}
    output = Path(config["output_dir"]) / config["exp_name"]
    checkpoint = torch.load(output / "latest_ckpt.pth", weights_only=True)
    assert checkpoint["config"]["copy_paste"][SIZE_KEY] == config["copy_paste"][SIZE_KEY]
    report = json.loads((output / "diagnostic_report.json").read_text())
    assert report["epochs"][0]["images"] == 4
    assert math.isfinite(report["epochs"][0]["mean_loss"]) and report["epochs"][0]["mean_loss"] > 0


def test_online_export_uses_the_same_per_class_sizes(config, tmp_path):
    config["mode"] = "online"
    result = export_coco(config, tmp_path / "exported", count=2)
    coco = json.loads(Path(result["annotations"]).read_text())
    assert len(coco["images"]) == 2 and len(coco["annotations"]) == 4
    for annotation in coco["annotations"]:
        expected = 12 if annotation["category_id"] == 7 else 28
        assert expected - 1 <= max(annotation["bbox"][2:]) <= expected
