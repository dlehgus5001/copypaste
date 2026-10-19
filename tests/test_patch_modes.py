"""Patch geometry, matching evaluation coordinates and real CLI optimizer steps."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
from pycocotools.cocoeval import COCOeval

from offline_yolox.experiment import create_exp
from offline_yolox.patch_data import CocoPatchDataset, patch_settings, starts
from offline_yolox.preview import save_preview
from offline_yolox.sample_data import make_sample_data
from yolox.evaluators import COCOEvaluator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "큰 이미지"
    make_sample_data(data, image_size=128, train_images=2, val_images=1)
    source = json.loads((ROOT / "configs/online.json").read_text())
    source.update(mode="original_patch", data_dir=str(data), input_size=[64, 64],
                  test_size=[64, 64], max_epoch=1, scheduler="cos", batch_size=2,
                  hsv_prob=0.0, flip_prob=0.0, output_dir=str(tmp_path / "outputs"),
                  patch={"size": [64, 64], "stride": [64, 64], "max_labels": 50})
    source["copy_paste"].update(asset_root=str(data), epoch_length=4,
                                object_count=[5, 5], require_object_count=True,
                                object_long_side_pixels=[8], object_size_weights=[1],
                                min_visible_pixels=2, min_visible_fraction=0.1)
    return source


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def test_grid_clipped_boxes_sparse_classes_and_patch_evaluation(tmp_path):
    image = np.zeros((96, 160, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(160, dtype=np.uint8)
    image[:, :, 1] = np.arange(96, dtype=np.uint8)[:, None]
    path = tmp_path / "좌표.png"
    assert cv2.imwrite(str(path), image)
    source = dict(images=[dict(id=19, file_name=str(path), height=96, width=160)],
                  categories=[dict(id=29, name="b"), dict(id=7, name="a")],
                  annotations=[dict(id=101, image_id=19, category_id=29,
                                    bbox=[50, 20, 30, 20], area=600, iscrowd=0)])
    annotation = tmp_path / "original.json"
    annotation.write_text(json.dumps(source))
    before = snapshot(tmp_path)
    settings = patch_settings({"patch": {"size": [64, 64], "stride": [64, 64]}})
    dataset = CocoPatchDataset(data_dir=tmp_path, json_file=str(annotation), name="unused",
                               input_size=(64, 64), settings=settings)
    assert starts(160, 64, 64) == [0, 64, 96]
    assert starts(96, 64, 64) == [0, 32]
    assert len(dataset) == 6 and dataset.class_ids == [7, 29]
    crop, labels, info, image_id = dataset.pull_item(0)
    np.testing.assert_array_equal(crop, image[:64, :64])
    np.testing.assert_array_equal(labels, [[50, 20, 64, 40, 1]])
    np.testing.assert_array_equal(dataset.pull_item(1)[1], [[0, 20, 16, 40, 1]])
    np.testing.assert_array_equal(dataset.pull_item(3)[1], [[50, 0, 64, 8, 1]])
    assert info == (64, 64) and image_id.tolist() == [1]
    assert snapshot(tmp_path) == before
    assert source == json.loads(annotation.read_text())
    assert {img["source_image_id"] for img in dataset.coco.dataset["images"]} == {19}

    # Exercise the actual YOLOX converter at a different model size, then COCO AP.
    converter = COCOEvaluator(type("Loader", (), {"dataset": dataset})(), (128, 128), 0.01, 0.65, 2)
    predictions = []
    for index in range(len(dataset)):
        _, boxes, _, ids = dataset.pull_item(index)
        outputs = torch.zeros((len(boxes), 7))
        outputs[:, :4] = torch.from_numpy(boxes[:, :4] * 2)
        outputs[:, 4:6] = 1
        outputs[:, 6] = torch.from_numpy(boxes[:, 4])
        predictions.extend(converter.convert_to_coco_format(
            [outputs], (torch.tensor([64]), torch.tensor([64])), [ids[0]]))
    evaluation = COCOeval(dataset.coco, dataset.coco.loadRes(predictions), "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    assert evaluation.stats[0] == pytest.approx(1.0)

    filtered_settings = {**settings, "min_visible_fraction": 0.6, "keep_empty": False}
    with pytest.raises(ValueError, match="no images"):
        CocoPatchDataset(data_dir=tmp_path, json_file=str(annotation), name="unused",
                         input_size=(64, 64), settings=filtered_settings)
    filtered = CocoPatchDataset(data_dir=tmp_path, json_file=str(annotation), name="unused",
                                input_size=(64, 64), settings={**filtered_settings, "min_visible_fraction": 0.5})
    assert len(filtered) == 1 and filtered.records[0]["xy"] == (64, 0)


def test_empty_patch_policy_partial_threshold_and_small_image_padding(tmp_path):
    path = tmp_path / "small.png"
    assert cv2.imwrite(str(path), np.full((24, 36, 3), 50, dtype=np.uint8))
    annotation = tmp_path / "small.json"
    source = dict(images=[dict(id=5, file_name=str(path), height=24, width=36)],
                  categories=[dict(id=1, name="vehicle")], annotations=[])
    annotation.write_text(json.dumps(source))
    settings = patch_settings({"patch": {"size": [64, 64]}})
    kwargs = dict(data_dir=tmp_path, json_file=str(annotation), name="unused", input_size=(64, 64))
    dataset = CocoPatchDataset(**kwargs, settings=settings)
    image, labels, _, _ = dataset.pull_item(0)
    assert len(dataset) == 1 and labels.shape == (0, 5)
    assert np.all(image[:24, :36] == 50) and np.all(image[24:] == 114)
    with pytest.raises(ValueError, match="no images"):
        CocoPatchDataset(**kwargs, settings={**settings, "keep_empty": False})
    assert len(CocoPatchDataset(**kwargs, settings={**settings, "keep_empty": False}, training=False)) == 1
    source["annotations"] = [dict(id=i, image_id=5, category_id=1, bbox=[-4, 0, 12, 10], area=120)
                             for i in [1, 2]]
    annotation.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="max_labels"):
        CocoPatchDataset(**kwargs, settings={**settings, "max_labels": 1})
    dataset = CocoPatchDataset(**kwargs, settings=settings)
    np.testing.assert_array_equal(dataset.pull_item(0)[1][:, :4], [[0, 0, 8, 10]] * 2)
    source["images"][0]["height"] = 25
    annotation.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="dimensions"):
        CocoPatchDataset(**kwargs, settings=settings).pull_item(0)


@pytest.mark.parametrize("patch", [None, {"size": [0, 64]}, {"size": [64.0, 64]},
                                  {"size": [64, 64], "stride": [65, 64]},
                                  {"min_visible_fraction": 1.1}, {"min_box_side": 0},
                                  {"keep_empty": "false"}, {"max_labels": 0}])
def test_invalid_patch_settings(patch):
    with pytest.raises(ValueError, match="[Pp]atch"):
        patch_settings({"patch": patch})


def test_online_patch_crops_before_composing_and_replays(config):
    config["mode"] = "online_patch"
    exp = create_exp(config)
    first = exp.get_dataset()
    replay = exp.get_dataset()
    seen = set()
    for _ in range(4):
        a, b = first.sample_raw(), replay.sample_raw()
        np.testing.assert_array_equal(a["image"], b["image"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        assert a["metadata"] == b["metadata"]
        assert a["image"].shape == (64, 64, 3) and len(a["labels"]) == 5
        assert a["metadata"]["placed_object_count"] == 5
        crop = a["metadata"]["background_patch"]
        assert crop["source_size"] == [128, 128] and crop["size"] == [64, 64]
        seen.add(tuple(crop["xy"]))
        for mask, label in zip(a["masks"], a["labels"]):
            ys, xs = np.nonzero(mask)
            np.testing.assert_array_equal(label[:4], [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    assert len(seen) > 1
    # With no objects/photometric changes, verify exact unscaled background pixels.
    clean = copy.deepcopy(config)
    clean["copy_paste"].update(object_count=[0, 0], blur_sigma=0, noise_std=0, brightness_range=[1, 1])
    sample = create_exp(clean).get_dataset().sample_raw()
    image = cv2.imread(str(Path(clean["copy_paste"]["asset_root"]) / sample["metadata"]["background"]))
    x, y = sample["metadata"]["background_patch"]["xy"]
    np.testing.assert_array_equal(sample["image"], image[y:y + 64, x:x + 64])


@pytest.mark.parametrize("mode", ["original_patch", "online_patch"])
def test_patch_workers_infinite_loader_and_previews(config, tmp_path, mode):
    config.update(mode=mode, num_workers=2)
    exp = create_exp(config)
    loader_a = exp.get_data_loader(2, finite=True, epoch=1)
    loader_b = exp.get_data_loader(2, finite=True, epoch=1)
    for a, b in zip(loader_a, loader_b):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert a[0].shape == (2, 3, 64, 64)
    native_loader = exp.get_data_loader(2)
    assert len(native_loader) == (4 if mode == "original_patch" else 2)
    assert next(iter(native_loader))[0].shape == (2, 3, 64, 64)
    native_loader.close_mosaic()
    assert exp.random_resize(None, 0, 0, False) == (64, 64)
    output = tmp_path / "preview"
    save_preview(config, output, count=2)
    assert cv2.imread(str(output / "sample_0000_boxes.png")).shape == (64, 64, 3)
    metadata = json.loads((output / "sample_0000.json").read_text())
    assert ("crop_xy" if mode == "original_patch" else "background_patch") in metadata


@pytest.mark.parametrize("mode", ["original_patch", "online_patch"])
def test_public_cli_patch_training_and_evaluation(config, tmp_path, mode):
    config["mode"] = mode
    before = snapshot(Path(config["data_dir"]))
    path = tmp_path / f"{mode}.json"
    path.write_text(json.dumps(config))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    extra = ["--objects-per-image", "5", "--object-size", "8"] if mode == "online_patch" else []
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "train", "--mode", mode,
                             "--config", str(path), "--device", "cpu", *extra],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    output = Path(config["output_dir"]) / config["exp_name"]
    report = json.loads((output / "diagnostic_report.json").read_text())
    epoch = report["epochs"][0]
    assert epoch["iterations"] == (4 if mode == "original_patch" else 2)
    assert epoch["images"] == (8 if mode == "original_patch" else 4)
    assert math.isfinite(epoch["mean_loss"]) and epoch["mean_loss"] > 0
    assert epoch["evaluation"]["images"] == 4
    assert math.isfinite(epoch["evaluation"]["ap50_95"])
    assert (output / "latest_ckpt.pth").stat().st_size > 0
    assert snapshot(Path(config["data_dir"])) == before
