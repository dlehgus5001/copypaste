"""Plain inference: input coordinates, tile offsets/NMS and standalone CLI."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from yolox.exp import get_exp

from offline_yolox.predict import load_predictor, predict, predict_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def predictor_files(tmp_path):
    torch.set_num_threads(2)
    exp = get_exp(exp_name="yolox-nano")
    exp.num_classes = 1
    model = exp.get_model()
    state = copy.deepcopy(model.state_dict())
    for level in range(3):
        for kind in ("obj_preds", "cls_preds"):
            state[f"head.{kind}.{level}.weight"].zero_()
            state[f"head.{kind}.{level}.bias"].fill_(-10)
        state[f"head.reg_preds.{level}.weight"].zero_()
        state[f"head.reg_preds.{level}.bias"][:] = torch.tensor([0.5, 0.5, 0, 0])
    ema = copy.deepcopy(state)
    for level in range(3):
        for kind in ("obj_preds", "cls_preds"):
            ema[f"head.{kind}.{level}.bias"].fill_(4)
    directory = tmp_path / "weights/test"
    directory.mkdir(parents=True)
    checkpoint = directory / "latest_ckpt.pth"
    config = dict(mode="online_patch", preset="nano", num_classes=1, class_names=["vehicle"],
                  test_size=[64, 64], data_dir=str(tmp_path / "missing_dataset"),
                  output_dir=str(tmp_path / "weights"), exp_name="test",
                  copy_paste={"asset_root": str(tmp_path / "missing_assets")},
                  patch={"size": [64, 64], "stride": [32, 32]})
    torch.save(dict(model=state, ema_model=ema, config=config), checkpoint)
    return config, checkpoint


def test_whole_image_restores_letterbox_coordinates_and_handles_one_candidate():
    class Fixed(torch.nn.Module):
        def forward(self, images):
            return torch.tensor([[[16, 12, 8, 8, 0.9, 0.9]]]).repeat(len(images), 1, 1)

    image = np.zeros((64, 128, 3), dtype=np.uint8)
    boxes, tiles = predict_image(Fixed(), image, input_size=(64, 64), num_classes=1,
                                 confidence=0.5, device="cpu")
    assert tiles == 1 and boxes.shape == (1, 6)
    np.testing.assert_allclose(boxes[0, :4], [24, 16, 40, 32])
    assert boxes[0, 4] == pytest.approx(0.81) and boxes[0, 5] == 0
    empty, _ = predict_image(Fixed(), image, input_size=(64, 64), num_classes=1,
                             confidence=0.9, device="cpu")
    assert empty.shape == (0, 6)


class AcrossTiles(torch.nn.Module):
    def __init__(self, different_classes=False):
        super().__init__()
        self.batches = []
        self.different_classes = different_classes

    def forward(self, images):
        self.batches.append(len(images))
        origin = images[:, 0, 0, 0]
        classes = 2 if self.different_classes else 1
        result = torch.zeros((len(images), 1, 5 + classes))
        result[:, 0, 0] = 48 - origin
        result[:, 0, 1:5] = torch.tensor([28, 16, 16, 0.9])
        if self.different_classes:
            for index, x in enumerate(origin):
                result[index, 0, 5 + int(x > 0)] = 0.9
        else:
            result[:, 0, 5] = 0.9
        return result


@pytest.mark.parametrize("batch_size", [1, 2])
def test_tiled_offsets_and_global_nms_merge_duplicate_boxes(batch_size):
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(96, dtype=np.uint8)
    before = image.copy()
    model = AcrossTiles()
    boxes, tiles = predict_image(model, image, input_size=(64, 64), num_classes=1,
                                 tile_size=(64, 64), stride=(32, 32), batch_size=batch_size,
                                 confidence=0.5, device="cpu")
    assert tiles == 2 and len(boxes) == 1
    np.testing.assert_allclose(boxes[0, :4], [40, 20, 56, 36])
    assert model.batches == ([1, 1] if batch_size == 1 else [2])
    np.testing.assert_array_equal(image, before)


def test_global_nms_retains_different_classes():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(96, dtype=np.uint8)
    boxes, _ = predict_image(AcrossTiles(different_classes=True), image, input_size=(64, 64),
                             num_classes=2, tile_size=(64, 64), stride=(32, 32), device="cpu")
    assert len(boxes) == 2 and set(boxes[:, 5]) == {0, 1}


def test_small_image_padding_does_not_create_boxes_outside_real_image():
    class Fixed(torch.nn.Module):
        def forward(self, images):
            return torch.tensor([[[10, 8, 4, 4, 0.9, 0.9], [48, 48, 8, 8, 0.9, 0.9]]])

    boxes, tiles = predict_image(Fixed(), np.zeros((16, 24, 3), np.uint8), input_size=(64, 64),
                                 num_classes=1, tile_size=(64, 64), device="cpu")
    assert tiles == 1 and len(boxes) == 1
    np.testing.assert_allclose(boxes[0, :4], [8, 6, 12, 10])


def test_public_cli_predicts_folder_draws_boxes_and_needs_no_training_data(predictor_files, tmp_path):
    config, checkpoint = predictor_files
    source = tmp_path / "사진 폴더"
    (source / "nested").mkdir(parents=True)
    for name in ("same.jpg", "same.png", "nested/same.png"):
        assert cv2.imwrite(str(source / name), np.full((64, 96, 3), 130, np.uint8))
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*") if path.is_file()}
    weights_before = checkpoint.read_bytes()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "results"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "predict", "--config", str(config_path),
         "--input", str(source), "--output", str(output), "--device", "cpu", "--tiled"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["image_count"] == 3 and summary["weights"] == "ema_model"
    assert summary["strategy"] == "tiled" and summary["patch_size"] == [64, 64]
    assert summary["checkpoint"] == str(checkpoint)
    assert {row["output_image"] for row in summary["images"]} == {
        "same.jpg.pred.png", "same.png.pred.png", "nested/same.png.pred.png"}
    for row in summary["images"]:
        assert row["tiles"] == 2 and row["detection_count"] > 0
        original = cv2.imread(row["input"])
        annotated = cv2.imread(str(output / row["output_image"]))
        assert annotated.shape == original.shape and not np.array_equal(annotated, original)
        metadata = json.loads((output / row["output_json"]).read_text())
        assert metadata["width"] == 96 and metadata["height"] == 64
        for detection in metadata["detections"]:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            assert 0 <= x1 < x2 <= 96 and 0 <= y1 < y2 <= 64
            assert detection["class_name"] == "vehicle" and detection["score"] > 0.9
    for path, digest in before.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert checkpoint.read_bytes() == weights_before
    with pytest.raises(FileExistsError, match="new or empty"):
        predict(config, source, output, device="cpu")


def test_raw_model_and_empty_predictions_still_save_image(predictor_files, tmp_path):
    config, checkpoint = predictor_files
    image = np.full((64, 64, 3), 150, np.uint8)
    source = tmp_path / "one.png"
    assert cv2.imwrite(str(source), image)
    output = tmp_path / "raw"
    summary = predict(config, source, output, checkpoint=checkpoint, device="cpu", raw_model=True)
    assert summary["weights"] == "model" and summary["images"][0]["detection_count"] == 0
    np.testing.assert_array_equal(cv2.imread(str(output / "one.png.pred.png")), image)


def test_class_metadata_from_categories_and_generic_fallback(predictor_files, tmp_path):
    config, checkpoint = predictor_files
    config = copy.deepcopy(config)
    config.pop("class_names")
    # Remove saved names as well, to exercise standalone deployment of old weights.
    saved = torch.load(checkpoint, weights_only=True)
    saved["config"].pop("class_names")
    torch.save(saved, checkpoint)
    _, info = load_predictor(config, checkpoint, device="cpu")
    assert info["class_names"] == ["class_0"] and info["class_ids"] == [None]
    annotations = tmp_path / "metadata/annotations"
    annotations.mkdir(parents=True)
    (annotations / "instances_val2017.json").write_text(json.dumps({"categories": [{"id": 29, "name": "truck"}]}))
    config["data_dir"] = str(annotations.parent)
    _, info = load_predictor(config, checkpoint, device="cpu")
    assert info["class_names"] == ["truck"] and info["class_ids"] == [29]


@pytest.mark.parametrize("ignore,expected_size", [(False, (40, 80)), (True, (80, 40))])
def test_exif_orientation_policy_is_explicit(predictor_files, tmp_path, ignore, expected_size):
    config, checkpoint = predictor_files
    source = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 40), (100, 120, 140)).save(source, exif=exif)
    summary = predict(config, source, tmp_path / "out", checkpoint=checkpoint,
                      device="cpu", ignore_exif_orientation=ignore)
    row = summary["images"][0]
    assert (row["width"], row["height"]) == expected_size


def test_invalid_options_rejected_before_inference(tmp_path):
    for kwargs in [dict(confidence=0), dict(confidence=float("nan")), dict(batch_size=0), dict(stride=[32, 32])]:
        with pytest.raises(ValueError):
            predict({}, tmp_path / "missing", tmp_path / "out", device="cpu", **kwargs)
    with pytest.raises(ValueError, match="stride"):
        predict_image(torch.nn.Identity(), np.zeros((64, 64, 3), np.uint8), input_size=(64, 64),
                       num_classes=1, tile_size=(64, 64), stride=(65, 64), device="cpu")


def test_corrupt_input_reports_decode_failure_without_touching_source(predictor_files, tmp_path):
    config, checkpoint = predictor_files
    source = tmp_path / "bad.jpg"
    source.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError, match="OpenCV cannot decode"):
        predict(config, source, tmp_path / "out", checkpoint=checkpoint, device="cpu")
    assert source.read_bytes() == b"not a jpeg"
