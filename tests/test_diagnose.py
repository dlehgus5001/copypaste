"""Diagnostic inference separates absent predictions, wrong classes and geometry."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from offline_yolox.diagnose import diagnose, inspect_sample, match_boxes, select_samples
from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]


def test_class_aware_matches_and_duplicate_predictions():
    labels = np.array([[10, 8, 30, 28, 1]], dtype=np.float32)
    predictions = np.array([[10, 8, 30, 28, 0.9, 0], [10, 8, 30, 28, 0.8, 0]], dtype=np.float32)
    assert match_boxes(labels, predictions) == dict(tp=0, fp=2, fn=1)
    assert match_boxes(labels, predictions, class_aware=False) == dict(tp=1, fp=1, fn=0)
    predictions[:, 5] = 1
    assert match_boxes(labels, predictions) == dict(tp=1, fp=1, fn=0)
    assert match_boxes(labels, predictions[:0]) == dict(tp=0, fp=0, fn=1)
    assert match_boxes(labels[:0], predictions) == dict(tp=0, fp=2, fn=0)


def test_official_preprocessing_and_nms_restore_box_coordinates_and_cap_candidates():
    class FixedModel(torch.nn.Module):
        def forward(self, images):
            # Eval-mode YOLOX emits center-x/y,width/height,objectness,class probabilities.
            return torch.tensor([[[40, 36, 40, 40, 0.9, 0.1, 0.9],
                                  [96, 96, 20, 20, 0.8, 0.8, 0.1]]], dtype=torch.float32)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    labels = np.array([[10, 8, 30, 28, 1]], dtype=np.float32)
    kwargs = dict(input_size=(128, 128), num_classes=2, nms_threshold=0.65, device="cpu", max_candidates=1)
    result = inspect_sample(FixedModel(), image, labels, confidence=0.5, **kwargs)
    assert result["max_raw_score"] == pytest.approx(0.81)
    assert result["candidates_above_threshold"] == 2 and result["candidates_capped"]
    assert result["nms_candidates_used"] == 1 and result["prediction_count"] == 1
    np.testing.assert_allclose(result["predictions"][0][:4], labels[0, :4])
    assert result["class_aware_iou50"] == dict(tp=1, fp=0, fn=0)
    assert result["class_agnostic_iou50"] == dict(tp=1, fp=0, fn=0)
    empty = inspect_sample(FixedModel(), image, labels, confidence=0.9, **kwargs)
    assert empty["prediction_count"] == 0 and empty["max_raw_score"] == pytest.approx(0.81)
    assert empty["class_aware_iou50"] == dict(tp=0, fp=0, fn=1)


def test_selection_spreads_positive_samples_and_keeps_empty_diagnostics():
    class Data:
        def __init__(self, positive_count, empty_count):
            self.records = ([dict(labels=np.ones((1, 5))) for _ in range(positive_count)]
                            + [dict(labels=np.zeros((0, 5))) for _ in range(empty_count)])

        def __len__(self):
            return len(self.records)

    indices, metadata = select_samples(Data(20, 20), 8)
    assert len(set(indices)) == 8
    assert sum(index < 20 for index in indices) == 6
    assert 0 in indices and 19 in indices
    assert metadata == dict(total_images=40, positive_images=20, empty_images=20)
    assert len(select_samples(Data(0, 20), 8)[0]) == 8
    assert len(select_samples(Data(20, 0), 8)[0]) == 8
    assert len(select_samples(Data(1, 3), 8)[0]) == 4


def test_public_diagnostic_cli_loads_checkpoint_and_saves_real_synthetic_predictions(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    make_sample_data(data, train_images=2, val_images=1)
    config = json.loads((ROOT / "configs/online.json").read_text())
    config.update(mode="online_patch", data_dir=str(data), input_size=[64, 64], test_size=[64, 64],
                  patch={"size": [64, 64]}, output_dir=str(tmp_path / "weights"), exp_name="test")
    config["copy_paste"].update(asset_root=str(data), validate_assets_on_start=False, epoch_length=8,
                                object_count=[1, 1], object_long_side_pixels=[8], object_size_weights=[1])
    model = create_exp(config).get_model()
    output_dir = tmp_path / "weights/test"
    output_dir.mkdir(parents=True)
    saved_config = copy.deepcopy(config)
    saved_config["copy_paste"]["epoch_length"] = 10000
    checkpoint = output_dir / "latest_ckpt.pth"
    torch.save(dict(model=model.state_dict(), ema_model=model.state_dict(), config=saved_config,
                    start_epoch=2, optimizer={"param_groups": [{"lr": 0.0001}]}), checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    before = checkpoint.read_bytes()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    output = tmp_path / "diagnostic"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "diagnose", "--config", str(config_path),
         "--output", str(output), "--device", "cpu", "--count", "2", "--confidence", "1"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["checkpoint"] == str(checkpoint) and summary["checkpoint_epoch"] == 2
    assert summary["checkpoint_config"]["copy_paste"]["epoch_length"] == 10000
    assert summary["optimizer_lrs"] == [0.0001] and summary["weights"] == "ema_model"
    assert summary["copy_paste_differences"]["epoch_length"] == dict(checkpoint=10000, current=8)
    assert summary["diagnostic_only"] and not summary["computes_full_dataset_ap"]
    assert summary["real_dataset"]["total_images"] == 4
    assert summary["groups"]["real"]["gt_count"] > 0
    assert summary["groups"]["synthetic"]["gt_count"] == 2
    for group in summary["groups"].values():
        assert group["sampled_images"] == 2 and group["prediction_count"] == 0
        assert 0 < group["max_raw_score"] < 1
    for row in summary["samples"]:
        for filename in row["files"].values():
            assert (output / filename).stat().st_size > 0
    assert checkpoint.read_bytes() == before
    with pytest.raises(FileExistsError, match="new or empty"):
        diagnose(config, output, checkpoint=checkpoint, device="cpu")


def test_invalid_diagnostic_arguments_fail_before_accessing_data(tmp_path):
    for kwargs in [dict(count=0), dict(confidence=float("nan")), dict(confidence=0),
                   dict(confidence=1.1), dict(max_candidates=0)]:
        with pytest.raises(ValueError):
            diagnose({}, tmp_path / "unused", device="cpu", **kwargs)
    assert not (tmp_path / "unused").exists()
