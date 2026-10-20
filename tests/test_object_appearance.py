"""Foreground color controls preserve labels, background pixels and replay."""
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
from offline_yolox.copy_paste import adjust_object_appearance
from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "source"
    make_sample_data(data, train_images=2, val_images=1)
    config = json.loads((ROOT / "configs/online.json").read_text())
    config.update(mode="online_patch", data_dir=str(data), input_size=[64, 64],
                  test_size=[64, 64], max_epoch=1, scheduler="cos", patch={"size": [64, 64]},
                  output_dir=str(tmp_path / "output"), exp_name="appearance_test")
    config["copy_paste"].update(asset_root=str(data), validate_assets_on_start=False,
                                epoch_length=4, object_count=[5, 5],
                                object_long_side_pixels=[8], object_size_weights=[1],
                                min_visible_pixels=2, min_visible_fraction=0.1,
                                brightness_range=[1, 1], noise_std=0, blur_sigma=0)
    return config


def test_foreground_saturation_and_contrast_preserve_outside_pixels_and_mask():
    image = np.array([[[0, 20, 240], [20, 220, 0]],
                      [[90, 90, 90], [255, 0, 0]]], dtype=np.uint8)
    mask = np.array([[True, True], [False, False]])
    original, original_mask = image.copy(), mask.copy()
    assert adjust_object_appearance(image, mask) is image
    gray = adjust_object_appearance(image, mask, saturation=0)
    assert np.all(gray[mask, 0] == gray[mask, 1])
    assert np.all(gray[mask, 1] == gray[mask, 2])
    np.testing.assert_array_equal(gray[~mask], original[~mask])
    softened = adjust_object_appearance(image, mask, saturation=0.5, contrast=0.8)
    assert np.ptp(softened[mask].astype(float), axis=1).mean() < np.ptp(image[mask].astype(float), axis=1).mean()
    flat = adjust_object_appearance(image, mask, contrast=0)
    assert np.unique(flat[mask]).size == 1
    np.testing.assert_array_equal(flat[~mask], original[~mask])
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("mode", ["online", "online_patch"])
@pytest.mark.parametrize("count", [0, 5])
def test_color_changes_keep_sample_geometry_background_and_identity_compatibility(config, mode, count):
    config["mode"] = mode
    config["copy_paste"]["object_count"] = [count, count]
    identity = copy.deepcopy(config)
    identity["copy_paste"].update(object_saturation_range=[1, 1], object_contrast_range=[1, 1])
    muted = copy.deepcopy(config)
    muted["copy_paste"].update(object_saturation_range=[0.4, 0.7], object_contrast_range=[0.8, 0.95])
    original_ds, identity_ds, muted_ds = [create_exp(c).get_dataset() for c in (config, identity, muted)]
    replay_ds = create_exp(muted).get_dataset()
    for _ in range(3):
        original, unchanged, changed, replay = [d.sample_raw() for d in
                                              (original_ds, identity_ds, muted_ds, replay_ds)]
        np.testing.assert_array_equal(original["image"], unchanged["image"])
        assert original["metadata"] == unchanged["metadata"]
        np.testing.assert_array_equal(changed["image"], replay["image"])
        assert changed["metadata"] == replay["metadata"]
        np.testing.assert_array_equal(original["labels"], changed["labels"])
        for a, b in zip(original["masks"], changed["masks"], strict=True):
            np.testing.assert_array_equal(a, b)
        union = np.zeros((64, 64), dtype=bool)
        for mask in changed["masks"]:
            union |= mask
        np.testing.assert_array_equal(original["image"][~union], changed["image"][~union])
        if count:
            assert np.any(original["image"][union] != changed["image"][union])
        metadata = copy.deepcopy(changed["metadata"])
        for item in metadata["instances"]:
            appearance = item.pop("appearance")
            assert 0.4 <= appearance["saturation"] <= 0.7
            assert 0.8 <= appearance["contrast"] <= 0.95
        assert metadata == original["metadata"]


@pytest.mark.parametrize("workers", [0, 2])
def test_color_augmentation_replays_in_training_workers(config, workers):
    config["num_workers"] = workers
    config["copy_paste"].update(object_saturation_range=[0.4, 0.7], object_contrast_range=[0.8, 0.95])
    exp = create_exp(config)
    first = exp.get_data_loader(2, finite=True, epoch=1)
    replay = exp.get_data_loader(2, finite=True, epoch=1)
    for a, b in zip(first, replay, strict=True):
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert torch.isfinite(a[0]).all()


@pytest.mark.parametrize("field", ["object_saturation_range", "object_contrast_range"])
@pytest.mark.parametrize("value", [[-0.1, 0.5], [0.8, 0.4], [1, float("nan")], [True, 1]])
def test_invalid_foreground_color_ranges_rejected(config, field, value):
    config["copy_paste"][field] = value
    with pytest.raises(ValueError, match=field):
        create_exp(config)


def test_color_cli_overrides_are_copied_validated_and_online_only(config):
    parser = argparse.ArgumentParser()
    add_composition_arguments(parser)
    args = parser.parse_args(["--object-saturation-range", "0.4", "0.7",
                              "--object-contrast-range", "0.8", "0.95"])
    original = copy.deepcopy(config)
    changed = composition_config(config, args)
    assert config == original
    assert changed["copy_paste"]["object_saturation_range"] == [0.4, 0.7]
    assert changed["copy_paste"]["object_contrast_range"] == [0.8, 0.95]
    with pytest.raises(ValueError, match="online"):
        composition_config({**config, "mode": "original_patch"}, args)
    for invalid in ["nan", "inf", "-1"]:
        with pytest.raises(ValueError, match="finite"):
            composition_config(config, parser.parse_args(["--object-saturation-range", invalid, "1"]))


def test_public_cli_color_preview_and_real_training(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    color_args = ["--object-saturation-range", "0.5", "0.5", "--object-contrast-range", "0.9", "0.9"]
    preview = tmp_path / "preview"
    for command, extra in [("preview", ["--output", str(preview), "--count", "2"]),
                           ("train", ["--device", "cpu"])]:
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), command,
                                 "--config", str(path), *extra, *color_args],
                                cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((preview / "sample_0000.json").read_text())
    assert len(metadata["instances"]) == 5
    for item in metadata["instances"]:
        assert item["appearance"] == {"saturation": 0.5, "contrast": 0.9}
    output = Path(config["output_dir"]) / config["exp_name"]
    report = json.loads((output / "diagnostic_report.json").read_text())
    assert report["epochs"][0]["images"] == 4
    assert math.isfinite(report["epochs"][0]["mean_loss"])
    saved = torch.load(output / "latest_ckpt.pth", map_location="cpu", weights_only=True)
    assert saved["config"]["copy_paste"]["object_saturation_range"] == [0.5, 0.5]
    assert saved["config"]["copy_paste"]["object_contrast_range"] == [0.9, 0.9]
