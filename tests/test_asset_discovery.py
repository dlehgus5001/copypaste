"""Exercise folder pairing and real online training without asset manifests."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from offline_yolox.asset_discovery import discover_backgrounds, discover_objects
from offline_yolox.copy_paste import OnlineCopyPasteDataset
from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data
from offline_yolox.train import run_cpu_diagnostic


ROOT = Path(__file__).resolve().parents[1]


def write_pair(root, relative, *, image_extension=".png", mask_suffix=""):
    image = root / "object" / (relative + image_extension)
    mask = root / "mask" / (relative + mask_suffix + ".png")
    image.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((20, 32, 3), (29, 71, 203), dtype=np.uint8)
    foreground = np.zeros((20, 32), dtype=np.uint8)
    foreground[2:-2, 2:-2] = 255
    assert cv2.imwrite(str(image), pixels)
    assert cv2.imwrite(str(mask), foreground)
    return image, mask


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def bank(tmp_path):
    root = tmp_path / "asset bank"
    # Deliberately create out of order, using the same filename in five folders.
    for index in (4, 1, 3, 0, 2):
        write_pair(root, f"object_{index}/001")
    backgrounds = root / "background" / "nested"
    backgrounds.mkdir(parents=True)
    assert cv2.imwrite(str(backgrounds / "second.png"), np.full((128, 128, 3), 77, np.uint8))
    assert cv2.imwrite(str(root / "background" / "first.png"), np.full((128, 128, 3), 101, np.uint8))
    (root / "object" / "README.txt").write_text("not an object")
    assert cv2.imwrite(str(root / "mask" / "unused.png"), np.full((3, 4), 255, np.uint8))
    return root


def objects(root, **kwargs):
    return discover_objects(root, "object", "mask", [17], ["vehicle"], **kwargs)


def test_png_pairs_in_five_nested_folders_are_sorted_without_a_manifest(bank):
    before = snapshot(bank)
    expected = [{"image": f"object/object_{index}/001.png",
                 "mask": f"mask/object_{index}/001.png", "category_id": 17}
                for index in range(5)]
    assert objects(bank) == expected
    assert objects(bank) == expected
    assert discover_backgrounds(bank, "background") == [
        "background/first.png", "background/nested/second.png"]
    assert not list(bank.rglob("*.json"))
    assert snapshot(bank) == before


def test_extensions_can_differ_and_optional_mask_suffix_is_respected(tmp_path):
    root = tmp_path / "assets"
    write_pair(root, "nested/001", image_extension=".JPG", mask_suffix="_mask")
    assert objects(root, mask_suffix="_mask") == [{
        "image": "object/nested/001.JPG", "mask": "mask/nested/001_mask.png", "category_id": 17}]
    with pytest.raises(FileNotFoundError):
        objects(root)


def test_multiple_classes_map_names_and_numeric_folders_to_real_category_ids(tmp_path):
    root = tmp_path / "assets"
    write_pair(root, "car/nested/001")
    write_pair(root, "29/001")
    entries = discover_objects(root, "object", "mask", [7, 29], ["car", "truck"])
    assert {entry["image"]: entry["category_id"] for entry in entries} == {
        "object/29/001.png": 29, "object/car/nested/001.png": 7}


def test_explicit_folder_categories_support_arbitrary_folder_names(tmp_path):
    root = tmp_path / "assets"
    write_pair(root, "set_a/001")
    write_pair(root, "set_b/001")
    with pytest.raises(ValueError):
        discover_objects(root, "object", "mask", [7, 29], ["car", "truck"])
    entries = discover_objects(root, "object", "mask", [7, 29], ["car", "truck"],
                               folder_categories={"set_a": 29, "set_b": 7})
    assert [entry["category_id"] for entry in entries] == [29, 7]
    with pytest.raises(ValueError):
        discover_objects(root, "object", "mask", [7, 29], ["car", "truck"],
                         folder_categories={"set_a": 0, "set_b": 7})


def test_multiclass_flat_files_do_not_silently_get_an_arbitrary_class(tmp_path):
    root = tmp_path / "assets"
    write_pair(root, "001")
    with pytest.raises(ValueError):
        discover_objects(root, "object", "mask", [7, 29], ["car", "truck"])


@pytest.mark.parametrize("invalid", ["missing", "ambiguous", "duplicate_object"])
def test_bad_pairing_is_rejected_before_training(bank, invalid):
    image, mask = bank / "object/object_0/001.png", bank / "mask/object_0/001.png"
    if invalid == "missing":
        # A same-named mask in another folder must not satisfy this pair.
        mask.unlink()
    elif invalid == "ambiguous":
        assert cv2.imwrite(str(mask.with_suffix(".bmp")), cv2.imread(str(mask), 0))
    else:
        assert cv2.imwrite(str(image.with_suffix(".jpg")), cv2.imread(str(image)))
    with pytest.raises((ValueError, FileNotFoundError)):
        objects(bank)


@pytest.mark.parametrize("kind", ["objects", "backgrounds"])
def test_empty_folder_banks_fail_early(tmp_path, kind):
    for name in ("object", "mask", "background"):
        (tmp_path / name).mkdir()
    (tmp_path / "object" / "README.txt").write_text("not an image")
    with pytest.raises(ValueError):
        if kind == "objects":
            objects(tmp_path)
        else:
            discover_backgrounds(tmp_path, "background")


@pytest.mark.parametrize("directory", ["https://example.invalid/images", "../outside"])
def test_folder_paths_must_remain_local_and_inside_asset_root(bank, directory):
    with pytest.raises(ValueError):
        discover_objects(bank, directory, "mask", [17], ["vehicle"])
    with pytest.raises(ValueError):
        discover_backgrounds(bank, directory)


@pytest.mark.parametrize("kind", ["object", "background"])
def test_symlinks_cannot_load_images_outside_asset_root(bank, tmp_path, kind):
    outside = tmp_path / "outside.png"
    assert cv2.imwrite(str(outside), np.full((20, 32, 3), 1, np.uint8))
    (bank / kind / "escape.png").symlink_to(outside)
    with pytest.raises(ValueError):
        if kind == "object":
            objects(bank)
        else:
            discover_backgrounds(bank, "background")


@pytest.mark.parametrize("masks_dir", ["object", "object/nested_masks"])
def test_object_and_mask_directories_cannot_overlap(bank, masks_dir):
    (bank / masks_dir).mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        discover_objects(bank, "object", masks_dir, [17], ["vehicle"])


@pytest.fixture
def folder_config(tmp_path):
    torch.set_num_threads(2)
    data, assets = tmp_path / "held out data", tmp_path / "training assets"
    make_sample_data(data, train_images=2, val_images=1)
    manifest = json.loads((data / "objects/manifest.json").read_text())
    for index, entry in enumerate(manifest):
        # Six distinct PNG pairs across five folders; no source manifest in assets.
        relative = Path(f"object_{min(index, 4)}") / f"{index:03}.png"
        for key, directory in (("image", "object"), ("mask", "mask")):
            destination = assets / directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data / entry[key], destination)
    shutil.copytree(data / "backgrounds", assets / "background")
    (assets / "background" / "manifest.json").unlink()
    # Online training needs only the real held-out COCO split for evaluation.
    shutil.rmtree(data / "train2017")
    shutil.rmtree(data / "objects")
    shutil.rmtree(data / "backgrounds")
    (data / "annotations/instances_train2017.json").unlink()
    (data / "sample_manifest.json").unlink()
    return {
        "mode": "online", "preset": "nano", "num_classes": 1,
        "data_dir": str(data), "val_ann": "instances_val2017.json",
        "input_size": [128, 128], "test_size": [128, 128], "batch_size": 2,
        "max_epoch": 1, "warmup_epochs": 0, "no_aug_epochs": 0,
        "scheduler": "cos", "num_workers": 0, "seed": 1729,
        "mosaic_prob": 0.0, "hsv_prob": 0.0, "flip_prob": 0.0,
        "enable_mixup": False, "multiscale_range": 0, "eval_interval": 1,
        "output_dir": str(tmp_path / "training"), "exp_name": "folder_test",
        "copy_paste": {
            "asset_root": str(assets), "objects_dir": "object", "masks_dir": "mask",
            "backgrounds_dir": "background", "epoch_length": 2,
            "object_count": [5, 5], "require_object_count": True,
            "unique_objects": True, "object_long_side_range": [16, 24],
            "rotation_degrees": [-180, 180], "placement_attempts": 30,
            "min_visible_fraction": 0.25, "min_visible_pixels": 4,
        },
    }


def test_folder_dataset_makes_five_aligned_labels_without_writing_assets(folder_config, monkeypatch):
    settings = copy.deepcopy(folder_config["copy_paste"])
    # Stale manifest paths in an existing config must not override folder inputs.
    settings.update(objects_manifest="missing-objects.json", backgrounds_manifest="missing-backgrounds.json")
    assets = Path(settings["asset_root"])
    before = snapshot(assets)

    def no_image_writes(*args, **kwargs):
        raise AssertionError("Online folder composition must not write images")

    monkeypatch.setattr(cv2, "imwrite", no_image_writes)
    dataset = OnlineCopyPasteDataset(settings, (128, 128), [1], ["sample_vehicle"], seed=1729)
    images = []
    for _ in range(4):
        sample = dataset.sample_raw()
        instances = sample["metadata"]["instances"]
        assert sample["metadata"]["placed_object_count"] == 5
        assert len(instances) == len(sample["labels"]) == len(sample["masks"]) == 5
        assert len({instance["image"] for instance in instances}) == 5
        coverage = np.zeros((128, 128), dtype=np.uint8)
        for label, mask, instance in zip(sample["labels"], sample["masks"], instances, strict=True):
            ys, xs = np.nonzero(mask)
            bounds = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            np.testing.assert_array_equal(label[:4], bounds)
            assert instance["bbox_xyxy"] == bounds
            assert label[4] == instance["class_index"] == 0
            assert instance["category_id"] == 1
            assert 16 <= instance["requested_long_side_pixels"] <= 24
            assert instance["mask"] == instance["image"].replace("object/", "mask/", 1)
            coverage += mask > 0
        assert coverage.max() == 1
        images.append(sample["image"])
    assert any(not np.array_equal(images[0], image) for image in images[1:])
    assert snapshot(assets) == before
    assert not list(assets.rglob("*.json"))


def test_folder_only_config_runs_real_cpu_epoch_and_held_out_coco_evaluation(folder_config):
    sources = {name: snapshot(Path(folder_config[name])) for name in ("data_dir",)}
    assets = Path(folder_config["copy_paste"]["asset_root"])
    before_assets = snapshot(assets)
    report = run_cpu_diagnostic(create_exp(folder_config))
    assert report["completed_epochs"] == 1
    epoch = report["epochs"][0]
    assert epoch["iterations"] == 1 and epoch["images"] == 2
    assert math.isfinite(epoch["mean_loss"]) and epoch["mean_loss"] > 0
    assert epoch["evaluation"]["images"] == 1
    assert Path(report["checkpoint"]).is_file()
    assert snapshot(assets) == before_assets
    assert snapshot(Path(folder_config["data_dir"])) == sources["data_dir"]


def test_public_cli_reads_folder_overrides_from_an_unrelated_working_directory(folder_config, tmp_path):
    config = copy.deepcopy(folder_config)
    config["output_dir"] = str(tmp_path / "cli training")
    config["copy_paste"] = {
        "asset_root": "does-not-exist", "objects_manifest": "obsolete-objects.json",
        "backgrounds_manifest": "obsolete-backgrounds.json", "epoch_length": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    assets = Path(folder_config["copy_paste"]["asset_root"])
    before = snapshot(assets)
    working = tmp_path / "unrelated working directory"
    working.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["YOLOX_PYTHON"] = str(tmp_path / "wrong-python")
    result = subprocess.run([
        sys.executable, str(ROOT / "run.py"), "train", "--mode", "online",
        "--config", str(config_path), "--device", "cpu", "--asset-root", str(assets),
        "--objects-dir", "object", "--masks-dir", "mask", "--backgrounds-dir", "background",
        "--objects-per-image", "5", "--object-size-range", "16", "24",
        "--unique-objects", "--require-object-count",
    ], cwd=working, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report_path = Path(config["output_dir"]) / config["exp_name"] / "diagnostic_report.json"
    report = json.loads(report_path.read_text())
    assert report["completed_epochs"] == 1
    assert report["epochs"][0]["images"] == 2
    assert math.isfinite(report["epochs"][0]["mean_loss"])
    assert snapshot(assets) == before
    assert not list(assets.rglob("*.json"))
