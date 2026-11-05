"""Multi-object composite masks and one-shot local Harmonizer integration."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from offline_yolox.paste_multi import save_multi_paste
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    root = tmp_path / "assets"
    make_sample_data(root, train_images=2, val_images=1, image_size=256)
    for index, entry in enumerate(json.loads((root / "objects/manifest.json").read_text())):
        folder = f"object_{index % 5 + 1}"
        for directory, key in [("object", "image"), ("mask", "mask")]:
            target = root / directory / folder / f"{index}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / entry[key]).read_bytes())
    (root / "background").mkdir()
    for index, entry in enumerate(json.loads((root / "backgrounds/manifest.json").read_text())):
        (root / "background" / f"{index}.png").write_bytes((root / entry).read_bytes())
    # Standalone composition must not need annotation files at all.
    shutil.rmtree(root / "annotations")
    values = json.loads((ROOT / "configs/paste_multi.json").read_text())
    values.update(input_size=[128, 128], patch={"size": [96, 96]})
    values["copy_paste"].update(asset_root=str(root),
        object_long_side_range_by_category={str(i): [6 + i * 2, 10 + i * 2] for i in range(1, 6)})
    return values


class FakeHarmonizer:
    def __init__(self):
        self.masks = []

    def predict_arguments(self, image, mask):
        self.masks.append(mask.detach().cpu().numpy().copy())
        return [torch.tensor([0.1])] * 6

    def restore_image(self, image, mask, arguments):
        return [torch.clamp(image + 0.05, 0, 1)]


def assert_masks_and_labels(folder, expected_count):
    info = json.loads((folder / "result.json").read_text())
    union = cv2.imread(str(folder / "mask.png"), cv2.IMREAD_UNCHANGED) > 0
    ownership = np.zeros(union.shape, np.uint16)
    assert len(info["instances"]) == len(info["labels_xyxy_class_index"]) == expected_count
    for instance, label in zip(info["instances"], info["labels_xyxy_class_index"], strict=True):
        mask = cv2.imread(str(folder / instance["composite_mask"]), cv2.IMREAD_UNCHANGED) > 0
        assert mask.shape == union.shape and mask.any()
        ys, xs = np.nonzero(mask)
        assert instance["bbox_xyxy"] == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        assert label[:4] == instance["bbox_xyxy"] and label[4] == instance["class_index"]
        ownership += mask
    assert ownership.max() <= 1
    np.testing.assert_array_equal(ownership > 0, union)
    return info, union


def test_composite_union_and_instance_masks_match_random_size_objects_and_preserve_background(config, tmp_path):
    root = Path(config["copy_paste"]["asset_root"])
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    original = copy.deepcopy(config)
    model = FakeHarmonizer()
    output = tmp_path / "results"
    result = save_multi_paste(config, output, count=3, model=model, device="cpu")
    assert result["count"] == 3 and len(model.masks) == 3
    assert config == original
    for index in range(3):
        folder = output / f"sample_{index:04}"
        info, union = assert_masks_and_labels(folder, 5)
        assert {item["category_id"] for item in info["instances"]} == {1, 2, 3, 4, 5}
        for item in info["instances"]:
            low, high = config["copy_paste"]["object_long_side_range_by_category"][str(item["category_id"])]
            assert low <= item["requested_long_side_pixels"] <= high
        np.testing.assert_array_equal(model.masks[index][0, 0] > 0, union)
        background, before, after = [cv2.imread(str(folder / name)) for name in ("background.png", "before.png", "after.png")]
        np.testing.assert_array_equal(before[~union], background[~union])
        np.testing.assert_array_equal(after[~union], before[~union])
        assert info["background_pixels_preserved"] and info["harmonizer_applied"]
        assert info["foreground_mean_absolute_change"] > 0
        assert (folder / "comparison.png").is_file()
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        save_multi_paste(config, output, plain_only=True)


def test_random_composites_replay_with_same_seed_but_change_across_samples(config, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    save_multi_paste(config, first, count=3, plain_only=True)
    save_multi_paste(config, second, count=3, plain_only=True)
    frames = []
    for index in range(3):
        relative = Path(f"sample_{index:04}")
        assert json.loads((first / relative / "result.json").read_text()) == json.loads((second / relative / "result.json").read_text())
        a, b = [cv2.imread(str(path / relative / "before.png")) for path in (first, second)]
        np.testing.assert_array_equal(a, b)
        frames.append(a)
    assert any(not np.array_equal(frames[0], other) for other in frames[1:])


def test_public_cli_overrides_global_random_sizes_count_and_single_background_without_model(config, tmp_path):
    config["copy_paste"]["backgrounds_dir"] = "absent_bank"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    background = tmp_path / "single background.png"
    assert cv2.imwrite(str(background), np.full((256, 256, 3), 60, np.uint8))
    output = tmp_path / "cli"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "paste-multi", "--config", str(path),
        "--background", str(background), "--count", "2", "--objects-per-image", "3", "--object-size-range", "14", "18",
        "--plain-only", "--harmonizer-repo", str(tmp_path / "absent_model"), "--output", str(output)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "overrides object_long_side_range_by_category" in result.stderr
    for index in range(2):
        info, _ = assert_masks_and_labels(output / f"sample_{index:04}", 3)
        assert info["background"] == str(background)
        assert all(14 <= item["requested_long_side_pixels"] <= 18 for item in info["instances"])
        assert not (output / f"sample_{index:04}/after.png").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"] is None and not manifest["training_invoked"]


def test_empty_composites_keep_black_mask_and_skip_model(config, tmp_path):
    config["copy_paste"]["object_count"] = [0, 0]
    model = FakeHarmonizer()
    output = tmp_path / "empty"
    save_multi_paste(config, output, count=1, model=model)
    info, mask = assert_masks_and_labels(output / "sample_0000", 0)
    assert not mask.any() and not model.masks
    assert not info["harmonizer_applied"] and info["correction"]["skipped_empty_mask"]
    a, b = [cv2.imread(str(output / "sample_0000" / name)) for name in ("before.png", "after.png")]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("changes", [{"mode": "original"}, {"class_ids": []}, {"class_names": None}, {"seed": True}])
def test_invalid_standalone_config_fails_before_output(config, tmp_path, changes):
    with pytest.raises(ValueError):
        save_multi_paste({**config, **changes}, tmp_path / "bad", plain_only=True)
    assert not (tmp_path / "bad").exists()


@pytest.mark.skipif(not (ROOT / "models/Harmonizer/pretrained/harmonizer.pth").is_file()
                    or importlib.util.find_spec("kornia") is None, reason="Local optional Harmonizer source/weights not prepared")
def test_real_harmonizer_cli_loads_once_for_multiple_five_object_composites(config, tmp_path):
    path, output = tmp_path / "config.json", tmp_path / "real"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "paste-multi", "--config", str(path),
        "--count", "2", "--device", "cpu", "--output", str(output)],
        cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("Loading local Harmonizer once for all composites: done") == 1
    for index in range(2):
        info, _ = assert_masks_and_labels(output / f"sample_{index:04}", 5)
        assert info["harmonizer_applied"] and info["background_pixels_preserved"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"]["checkpoint_sha256"] == "401dd03706fc6fa37b4600ae4c643a2fa9ad0f11351acfbf030cb5d3490de6ca"
