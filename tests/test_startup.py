"""Avoid decoding unused assets at setup while retaining checks on selected data."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    data = tmp_path / "data"
    make_sample_data(data, train_images=2, val_images=1)
    config = json.loads((ROOT / "configs/online.json").read_text())
    config.update(data_dir=str(data), output_dir=str(tmp_path / "output"),
                  max_epoch=1, scheduler="cos", exp_name="startup_test")
    config["copy_paste"].update(asset_root=str(data), epoch_length=2,
                               object_count=[1, 1], require_object_count=True)
    return config


def decoded_files(monkeypatch):
    original = cv2.imread
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counted)
    return calls


@pytest.mark.parametrize("mode", ["online", "online_patch"])
def test_lazy_startup_decodes_no_assets_and_composition_matches_eager(config, monkeypatch, capsys, mode):
    config.update(mode=mode, patch={"size": [128, 128]})
    calls = decoded_files(monkeypatch)
    # The omitted option preserves the existing eager behavior.
    eager = create_exp(config).get_dataset()
    assert len(calls) == 6 * 2 + 4
    eager_sample = eager.sample_raw()
    calls.clear()

    lazy_config = copy.deepcopy(config)
    lazy_config["copy_paste"]["validate_assets_on_start"] = False
    exp = create_exp(lazy_config)
    lazy = exp.get_dataset()
    assert calls == []
    exp.get_dataset(epoch=1)
    assert calls == []  # Template reuse does not re-read input pixels either.
    lazy_sample = lazy.sample_raw()
    assert len(calls) == 3  # One background, one object and one mask.
    np.testing.assert_array_equal(eager_sample["image"], lazy_sample["image"])
    np.testing.assert_array_equal(eager_sample["labels"], lazy_sample["labels"])
    assert eager_sample["metadata"] == lazy_sample["metadata"]
    messages = capsys.readouterr().err
    assert "Indexing object/mask files" in messages
    assert "Indexing background files" in messages
    assert "6 object/mask pairs, 4 backgrounds" in messages
    assert "pixel checks=when selected" in messages


def only_first_object(config):
    root = Path(config["copy_paste"]["asset_root"])
    path = root / "objects/manifest.json"
    entry = json.loads(path.read_text())[0]
    path.write_text(json.dumps([entry]))
    return root, entry


def test_lazy_mask_content_still_validated_when_selected(config):
    root, entry = only_first_object(config)
    path = root / entry["mask"]
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    mask[0, 0] = 17
    assert cv2.imwrite(str(path), mask)
    with pytest.raises(ValueError, match="Mask must be binary"):
        create_exp(config)
    config["copy_paste"]["validate_assets_on_start"] = False
    dataset = create_exp(config).get_dataset()
    with pytest.raises(ValueError, match="Mask must be binary"):
        dataset.sample_raw()


def test_lazy_corrupt_background_fails_when_selected(config):
    root = Path(config["copy_paste"]["asset_root"])
    path = root / "backgrounds/manifest.json"
    entry = json.loads(path.read_text())[0]
    path.write_text(json.dumps([entry]))
    (root / entry).write_bytes(b"not an image")
    config["copy_paste"]["validate_assets_on_start"] = False
    dataset = create_exp(config).get_dataset()
    with pytest.raises(ValueError, match="Image must be readable"):
        dataset.sample_raw()


def test_lazy_missing_mask_still_fails_at_startup(config, monkeypatch):
    root, entry = only_first_object(config)
    (root / entry["mask"]).unlink()
    config["copy_paste"]["validate_assets_on_start"] = False
    calls = decoded_files(monkeypatch)
    with pytest.raises(FileNotFoundError):
        create_exp(config)
    assert calls == []


def test_lazy_folder_taxonomy_and_pairing_still_checked(config, monkeypatch):
    root, entry = only_first_object(config)
    object_dir = root / "object" / "object_1"
    mask_dir = root / "mask" / "object_1"
    object_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (object_dir / "1.png").write_bytes((root / entry["image"]).read_bytes())
    (mask_dir / "1.png").write_bytes((root / entry["mask"]).read_bytes())
    annotation = root / "annotations/instances_val2017.json"
    coco = json.loads(annotation.read_text())
    coco["categories"].append({"id": 29, "name": "another_class"})
    annotation.write_text(json.dumps(coco))
    config.update(num_classes=2)
    config["copy_paste"].update(objects_dir="object", masks_dir="mask", validate_assets_on_start=False)
    calls = decoded_files(monkeypatch)
    with pytest.raises(ValueError, match="unresolved"):
        create_exp(config)
    assert calls == []
    config["copy_paste"]["folder_categories"] = {"object_1": 1}
    create_exp(config)
    assert calls == []
    (mask_dir / "1.png").unlink()
    with pytest.raises(FileNotFoundError, match="No matching mask"):
        create_exp(config)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_asset_validation_option_requires_boolean(config, value):
    config["copy_paste"]["validate_assets_on_start"] = value
    with pytest.raises(ValueError, match="validate_assets_on_start must be boolean"):
        create_exp(config)


def test_lazy_public_cli_runs_real_epoch_and_reports_preparation_stages(config, tmp_path):
    config.update(mode="online_patch", patch={"size": [128, 128]})
    config["copy_paste"]["validate_assets_on_start"] = False
    path = tmp_path / "lazy.json"
    path.write_text(json.dumps(config))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "train", "--config", str(path), "--device", "cpu"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    messages = completed.stderr
    assert f"config={path}" in messages
    assert "pixel checks=when selected" in messages
    assert "Building validation patch index" in messages
    assert "Validation ready: 1 patches" in messages
    output = Path(config["output_dir"]) / config["exp_name"]
    report = json.loads((output / "diagnostic_report.json").read_text())
    epoch = report["epochs"][0]
    assert epoch["images"] == 2 and epoch["iterations"] == 1
    assert math.isfinite(epoch["mean_loss"]) and epoch["mean_loss"] > 0
    assert epoch["evaluation"]["images"] == 1
    assert (output / "latest_ckpt.pth").stat().st_size > 0
