"""Meaningful offline checks of genuine official training, resume and COCO paths."""
import json
import random

import cv2
import numpy as np
import pytest
import torch

from offline_yolox.bootstrap import local_path
from offline_yolox.evaluate import evaluate_model
from offline_yolox.experiment import create_exp, seed_worker
from offline_yolox.sample_data import make_sample_data
from offline_yolox.train import initialize_from_checkpoint, load_checkpoint, main, run_cpu_diagnostic


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    make_sample_data(data, train_images=4, val_images=2)
    return {"preset": "nano", "num_classes": 1, "data_dir": str(data),
            "train_ann": "instances_train2017.json", "val_ann": "instances_val2017.json",
            "input_size": [128, 128], "test_size": [128, 128], "batch_size": 2,
            "max_epoch": 2, "warmup_epochs": 0, "no_aug_epochs": 0,
            "num_workers": 0, "seed": 1729, "mosaic_prob": 0.0, "hsv_prob": 0.0,
            "flip_prob": 0.0, "enable_mixup": False, "multiscale_range": 0,
            "eval_interval": 1, "output_dir": str(tmp_path / "out"), "exp_name": "test"}


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_sample_annotations_and_exported_masks(config):
    data = local_path(config["data_dir"])
    annotations = json.loads((data / "annotations/instances_train2017.json").read_text())
    assert len(annotations["images"]) == 4
    assert annotations["categories"][0]["id"] == 1
    for annotation in annotations["annotations"]:
        x, y, width, height = annotation["bbox"]
        assert 0 <= x < x + width <= 128
        assert 0 <= y < y + height <= 128
        assert annotation["area"] == width * height
    manifest = json.loads((data / "objects/manifest.json").read_text())
    for entry in manifest:
        image = cv2.imread(str(data / entry["image"]))
        mask = cv2.imread(str(data / entry["mask"]), cv2.IMREAD_GRAYSCALE)
        assert image.shape[:2] == mask.shape
        assert set(np.unique(mask)) == {0, 255}
        assert entry["category_id"] == 1
    with pytest.raises(FileExistsError):
        make_sample_data(data)


def test_loader_visits_finite_dataset_exactly_once(config):
    exp = create_exp(config)
    loader = exp.get_data_loader(2, finite=True, epoch=0)
    ids = []
    for images, labels, _, image_ids in loader:
        assert images.shape == (2, 3, 128, 128)
        assert labels.shape[2] == 5
        assert (labels[:, :, 3:5] > 0).any()
        ids.extend(image_ids.reshape(-1).tolist())
    assert len(loader) == 2
    assert sorted(ids) == [1, 2, 3, 4]


def test_true_training_and_exact_epoch_boundary_resume(config, tmp_path):
    augmented_config = {**config, "mosaic_prob": 0.5, "hsv_prob": 1.0, "flip_prob": 0.5}
    full_config = {**augmented_config, "output_dir": str(tmp_path / "full")}
    full = run_cpu_diagnostic(create_exp(full_config))
    full_state = load_checkpoint(full["checkpoint"])
    split_config = {**augmented_config, "output_dir": str(tmp_path / "split")}
    first = run_cpu_diagnostic(create_exp(split_config), stop_after_epoch=1)
    first_state = load_checkpoint(first["checkpoint"])
    resumed = run_cpu_diagnostic(create_exp(split_config), resume=first["checkpoint"])
    split_state = load_checkpoint(resumed["checkpoint"])
    assert first["epochs"][0]["iterations"] == 2
    assert full["completed_epochs"] == resumed["completed_epochs"] == 2
    assert resumed["start_epoch"] == 1
    assert full_state["ema_updates"] == 4
    for key in ("model", "optimizer", "ema_model", "scaler", "rng"):
        assert_nested_equal(full_state[key], split_state[key])
    assert any(not torch.equal(first_state["model"][key], full_state["model"][key])
               for key in full_state["model"] if full_state["model"][key].is_floating_point())
    assert all(np.isfinite(epoch["mean_loss"]) and epoch["mean_loss"] > 0 for epoch in full["epochs"])
    assert full["epochs"][-1]["evaluation"]["images"] == 2
    assert full["native_cuda_trainer"] is False


def test_zero_predictions_are_evaluated_as_empty_coco_result(config):
    class ZeroDetectionModel(torch.nn.Module):
        def forward(self, images):
            return torch.zeros((len(images), 21, 6), device=images.device)

    metrics = evaluate_model(create_exp(config), ZeroDetectionModel())
    assert metrics["images"] == 2
    assert metrics["detections"] == 0
    assert metrics["ap50_95"] == 0.0
    assert metrics["ap50"] == 0.0
    assert "Average Precision" in metrics["summary"]


def test_coco_conversion_preserves_perfect_boxes_and_category_mapping(config):
    annotations = json.loads((local_path(config["data_dir"]) / "annotations/instances_val2017.json").read_text())
    by_image = {image["id"]: [] for image in annotations["images"]}
    for annotation in annotations["annotations"]:
        x, y, width, height = annotation["bbox"]
        by_image[annotation["image_id"]].append([x + width / 2, y + height / 2, width, height, 1.0, 1.0])

    class PerfectBoxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.next_image = 0

        def forward(self, images):
            detections = torch.zeros((len(images), 2, 6), device=images.device)
            for row in range(len(images)):
                image_id = annotations["images"][self.next_image]["id"]
                boxes = by_image[image_id]
                detections[row, :len(boxes)] = torch.tensor(boxes, device=images.device)
                self.next_image += 1
            return detections

    metrics = evaluate_model(create_exp({**config, "nmsthre": 1.0}), PerfectBoxModel())
    assert metrics["detections"] == len(annotations["annotations"])
    assert metrics["ap50_95"] == pytest.approx(1.0)
    assert metrics["ap50"] == pytest.approx(1.0)


def test_cuda_requires_actual_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        main(["--device", "cuda"])


def test_no_checkpoint_downloads():
    with pytest.raises(ValueError, match="Only local"):
        load_checkpoint("https://example.invalid/weights.pth")


def test_local_coco80_pretrained_weights_restore_backbone_but_keep_one_class_head(config, tmp_path):
    from yolox.exp import get_exp
    source = get_exp(exp_name="yolox-nano").get_model()
    target = create_exp(config).get_model()
    assert source.head.num_classes == 80
    assert target.head.num_classes == 1
    with torch.no_grad():
        next(source.backbone.parameters()).fill_(0.125)
        for layer in source.head.cls_preds:
            layer.weight.fill_(0.75)
            layer.bias.fill_(0.5)
    initialized_head = {name: value.clone() for name, value in target.head.cls_preds.state_dict().items()}
    local_weights = tmp_path / "coco80_pretrained.pth"
    torch.save({"model": source.state_dict()}, local_weights)
    initialize_from_checkpoint(target, local_weights)
    assert_nested_equal(source.backbone.state_dict(), target.backbone.state_dict())
    assert_nested_equal(initialized_head, target.head.cls_preds.state_dict())
    assert_nested_equal(source.head.reg_preds.state_dict(), target.head.reg_preds.state_dict())
    assert_nested_equal(source.head.obj_preds.state_dict(), target.head.obj_preds.state_dict())


def test_fine_tuning_rejects_checkpoint_with_no_compatible_tensors(config, tmp_path):
    checkpoint = tmp_path / "unrelated.pth"
    torch.save({"model": {"unrelated.weight": torch.ones(3, 4)}}, checkpoint)
    with pytest.raises(ValueError, match="no compatible model tensor"):
        initialize_from_checkpoint(create_exp(config).get_model(), checkpoint)


def test_reproducible_worker_seed():
    torch.manual_seed(123)
    seed_worker(0)
    expected = (random.random(), np.random.random())
    torch.manual_seed(123)
    seed_worker(0)
    assert expected == (random.random(), np.random.random())


def test_category_mismatch_is_rejected(config):
    with pytest.raises(ValueError, match="differs from COCO"):
        create_exp({**config, "num_classes": 2})


def test_same_count_different_taxonomy_is_rejected(config):
    path = local_path(config["data_dir"]) / "annotations/instances_val2017.json"
    annotations = json.loads(path.read_text())
    annotations["categories"][0]["id"] = 2
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError, match="category IDs and names must agree"):
        create_exp(config)


def test_all_model_presets_use_official_classes(config):
    from yolox.models import YOLOX
    # Build small and nano; validate remaining presets without expensive model allocations.
    assert isinstance(create_exp(config).get_model(), YOLOX)
    nano = create_exp(config)
    small = create_exp({**config, "preset": "s"})
    nano_model = nano.get_model()
    small_model = small.get_model()
    assert sum(p.numel() for p in nano_model.parameters()) < sum(p.numel() for p in small_model.parameters())
    assert nano_model.head.num_classes == 1
    for preset, depth, width in (("tiny", 0.33, 0.375), ("m", 0.67, 0.75),
                                 ("l", 1.0, 1.0), ("x", 1.33, 1.25)):
        exp = create_exp({**config, "preset": preset})
        assert exp.depth == depth
        assert exp.width == width
