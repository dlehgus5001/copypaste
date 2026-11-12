"""Resolution degradation removes foreground detail without changing geometry."""
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
from offline_yolox.object_effects import apply_object_surface
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]
MODES = ("online", "online_patch", "online_harmonized", "online_patch_harmonized")


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "source"
    make_sample_data(data, train_images=2, val_images=1)
    values = json.loads((ROOT / "configs/samples/online.json").read_text())
    values.update(mode="online_patch", data_dir=str(data), input_size=[64, 64],
                  test_size=[64, 64], max_epoch=1, scheduler="cos", patch={"size": [64, 64]},
                  output_dir=str(tmp_path / "training"), exp_name="resolution_test")
    values["copy_paste"].update(asset_root=str(data), validate_assets_on_start=False,
        epoch_length=2, object_count=[5, 5], object_long_side_pixels=[8],
        object_size_weights=[1], min_visible_pixels=2, min_visible_fraction=0.1,
        brightness_range=[1, 1], noise_std=0, blur_sigma=0)
    return values


def test_downsample_then_restore_loses_detail_without_changing_canvas_or_mask():
    mask = np.zeros((48, 64), bool)
    mask[4:44, 6:58] = True
    mask[17:23, 25:29] = False
    image = np.full((48, 64, 3), [13, 24, 35], np.uint8)
    checker = (np.indices(mask.shape).sum(axis=0) % 2 * 200 + 25).astype(np.uint8)
    image[mask] = np.repeat(checker[:, :, None], 3, axis=2)[mask]
    original, original_mask = image.copy(), mask.copy()
    degraded = apply_object_surface(image, mask, resolution_scale=0.25)
    assert degraded.shape == image.shape and degraded.dtype == image.dtype
    assert degraded[mask].std() < image[mask].std() / 4
    np.testing.assert_array_equal(degraded[~mask], image[~mask])
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("scale", [0.25, 0.4, 0.7])
def test_normalized_resize_keeps_uniform_foreground_color_without_exterior_bleed(scale):
    mask = np.zeros((35, 53), bool)
    mask[3:32, 7:46] = True
    mask[12:19, 23:29] = False
    # Exterior color is deliberately very different from the foreground.
    image = np.full((35, 53, 3), [255, 0, 0], np.uint8)
    image[mask] = [80, 140, 200]
    degraded = apply_object_surface(image, mask, resolution_scale=scale, blur_sigma=0.8)
    np.testing.assert_array_equal(degraded, image)


def test_identity_resolution_preserves_existing_surface_pixels_and_rng_exactly():
    image = np.full((32, 40, 3), 180, np.uint8)
    mask = np.zeros((32, 40), bool)
    mask[4:28, 5:35] = True
    first_rng, second_rng = np.random.default_rng(42), np.random.default_rng(42)
    before = copy.deepcopy(first_rng.bit_generator.state)
    assert apply_object_surface(image, mask, resolution_scale=1.0, rng=first_rng) is image
    assert first_rng.bit_generator.state == before
    kwargs = dict(blur_sigma=0.8, dirt_strength=0.3)
    first = apply_object_surface(image, mask, rng=first_rng, **kwargs)
    second = apply_object_surface(image, mask, resolution_scale=1.0, rng=second_rng, **kwargs)
    np.testing.assert_array_equal(first, second)
    assert first_rng.bit_generator.state == second_rng.bit_generator.state


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("surface_enabled", [False, True])
def test_disabled_resolution_preserves_original_sample_metadata_and_random_stream(config, mode, surface_enabled):
    config["mode"] = mode
    if mode.endswith("_harmonized"):
        config["harmonizer"] = dict(device="cpu", strength=0, repository="absent-optional-model")
    if surface_enabled:
        config["copy_paste"].update(object_blur_sigma_range=[0.4, 0.8],
            object_dirt_strength_range=[0.1, 0.3], object_saturation_range=[0.6, 0.9])
    explicit = copy.deepcopy(config)
    explicit["copy_paste"]["object_resolution_scale_range"] = [1, 1]
    first, second = [create_exp(values).get_dataset() for values in (config, explicit)]
    for _ in range(3):
        a, b = first.sample_raw(), second.sample_raw()
        np.testing.assert_array_equal(a["image"], b["image"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        assert a["metadata"] == b["metadata"]
        for mask_a, mask_b in zip(a["masks"], b["masks"], strict=True):
            np.testing.assert_array_equal(mask_a, mask_b)


@pytest.mark.parametrize("mode", MODES)
def test_enabled_resolution_replays_and_preserves_geometry_background_and_previous_effect_draws(config, mode):
    config["mode"] = mode
    if mode.endswith("_harmonized"):
        config["harmonizer"] = dict(device="cpu", strength=0, repository="absent-optional-model")
    config["copy_paste"].update(object_blur_sigma_range=[0.4, 0.8],
        object_dirt_strength_range=[0.1, 0.3], object_saturation_range=[0.6, 0.9])
    degraded = copy.deepcopy(config)
    degraded["copy_paste"]["object_resolution_scale_range"] = [0.25, 0.5]
    baseline, first, replay = [create_exp(values).get_dataset() for values in (config, degraded, degraded)]
    for _ in range(3):
        a, b, c = baseline.sample_raw(), first.sample_raw(), replay.sample_raw()
        np.testing.assert_array_equal(b["image"], c["image"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        assert b["metadata"] == c["metadata"]
        union = np.zeros((64, 64), bool)
        for mask_a, mask_b in zip(a["masks"], b["masks"], strict=True):
            np.testing.assert_array_equal(mask_a, mask_b)
            union |= mask_b
        np.testing.assert_array_equal(a["image"][~union], b["image"][~union])
        assert np.any(a["image"][union] != b["image"][union])
        metadata = copy.deepcopy(b["metadata"])
        for instance in metadata["instances"]:
            assert 0.25 <= instance["appearance"].pop("resolution_scale") <= 0.5
        assert metadata == a["metadata"]


@pytest.mark.parametrize("value", [[0, 1], [-0.1, 0.5], [0.3, 1.1], [0.8, 0.4],
    [0.3, float("nan")], [0.3, float("inf")], [True, 1], [0.5], None])
def test_bad_resolution_ranges_rejected_before_training(config, value):
    config["copy_paste"]["object_resolution_scale_range"] = value
    with pytest.raises(ValueError, match="object_resolution_scale_range"):
        create_exp(config)


def test_resolution_cli_override_copies_config_and_rejects_out_of_bounds_or_offline_mode(config):
    parser = argparse.ArgumentParser()
    add_composition_arguments(parser)
    original = copy.deepcopy(config)
    arguments = parser.parse_args(["--object-resolution-scale-range", "0.25", "0.5"])
    modified = composition_config(config, arguments)
    assert config == original
    assert modified["copy_paste"]["object_resolution_scale_range"] == [0.25, 0.5]
    with pytest.raises(ValueError, match="online"):
        composition_config({**config, "mode": "original_patch"}, arguments)
    for low, high in [("0", "1"), ("0.3", "1.1"), ("0.8", "0.4"), ("nan", "1"), ("0.3", "inf")]:
        with pytest.raises(ValueError):
            composition_config(config, parser.parse_args(["--object-resolution-scale-range", low, high]))


@pytest.mark.parametrize("mode", ["online_patch", "online_patch_harmonized"])
def test_public_preview_and_training_share_resolution_override_and_checkpoint_settings(config, tmp_path, mode):
    config["mode"] = mode
    if mode.endswith("_harmonized"):
        config["harmonizer"] = dict(device="cpu", strength=0, repository="absent-optional-model")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    preview = tmp_path / "preview"
    for command, extra in [("preview", ["--output", str(preview), "--count", "2"]),
                           ("train", ["--device", "cpu"])]:
        completed = subprocess.run([sys.executable, str(ROOT / "run.py"), command,
            "--config", str(path), "--object-resolution-scale-range", "0.3", "0.3", *extra],
            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    metadata = json.loads((preview / "sample_0000.json").read_text())
    assert len(metadata["instances"]) == 5
    assert all(item["appearance"]["resolution_scale"] == 0.3 for item in metadata["instances"])
    output = Path(config["output_dir"]) / config["exp_name"]
    report = json.loads((output / "diagnostic_report.json").read_text())
    assert report["epochs"][0]["images"] == 2
    assert math.isfinite(report["epochs"][0]["mean_loss"]) and report["epochs"][0]["mean_loss"] > 0
    checkpoint = torch.load(output / "latest_ckpt.pth", map_location="cpu", weights_only=True)
    assert checkpoint["config"]["copy_paste"]["object_resolution_scale_range"] == [0.3, 0.3]
