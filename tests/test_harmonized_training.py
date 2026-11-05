"""Optional Harmonizer training preserves geometry, validation and plain modes."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from offline_yolox.experiment import create_exp
from offline_yolox.sample_data import make_sample_data

ROOT = Path(__file__).resolve().parents[1]
MODES = ["online_harmonized", "online_patch_harmonized"]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "source"
    make_sample_data(data, train_images=2, val_images=1)
    # Online training uses the real validation taxonomy, never a COCO train file.
    (data / "annotations/instances_train2017.json").unlink()
    values = json.loads((ROOT / "configs/online.json").read_text())
    values.update(mode="online_patch_harmonized", data_dir=str(data),
        input_size=[64, 64], test_size=[64, 64], patch={"size": [64, 64]},
        max_epoch=1, scheduler="cos", batch_size=2, hsv_prob=0, flip_prob=0,
        output_dir=str(tmp_path / "training"), exp_name="harmonized_test")
    values["copy_paste"].update(asset_root=str(data), epoch_length=2,
        object_count=[2, 2], object_long_side_pixels=[8], object_size_weights=[1],
        min_visible_pixels=2, min_visible_fraction=0.1,
        brightness_range=[1, 1], blur_sigma=0, noise_std=0)
    values["harmonizer"] = dict(repository=str(ROOT / "models/Harmonizer"),
        checkpoint=None, device="cpu", strength=1.0)
    return values


class FakeHarmonizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.1))
        self.masks = []

    def predict_arguments(self, image, mask):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert not any(p.requires_grad for p in self.parameters())
        self.masks.append(mask.detach().cpu().numpy().copy())
        return [self.marker.reshape(1)] * 6

    def restore_image(self, image, mask, arguments):
        # Deliberately changes background: integration must restore it exactly.
        return [torch.clamp(image + 0.1, 0, 1)]


@pytest.fixture
def fake_loader(monkeypatch):
    import offline_yolox.harmonized_data as module
    models = []

    def load(repository, checkpoint=None, device="auto"):
        torch.rand(7)  # Constructing a real module also consumes Torch RNG.
        model = FakeHarmonizer()
        models.append(model)
        return model, dict(device="cpu", repository=str(repository), checkpoint=str(checkpoint),
                           checkpoint_sha256="fake-checkpoint-for-unit-tests")

    monkeypatch.setattr(module, "load_harmonizer", load)
    return models


def small_sample():
    image = np.full((32, 40, 3), 40, np.uint8)
    first = np.zeros((32, 40), bool)
    second = first.copy()
    first[4:12, 6:18] = True
    second[20:25, 28:35] = True
    image[first] = [80, 110, 160]
    image[second] = [160, 80, 110]
    labels = np.array([[6, 4, 18, 12, 0], [28, 20, 35, 25, 0]], np.float32)
    return dict(image=image, masks=[first, second], labels=labels, metadata={"instances": []})


def test_transform_loads_once_freezes_model_preserves_rng_and_background(fake_loader):
    from offline_yolox.harmonized_data import HarmonizerTransform
    transform = HarmonizerTransform(dict(device="cpu", strength=0.5))
    original = small_sample()
    rng = torch.get_rng_state().clone()
    sample = transform(copy.deepcopy(original), blend_sigma=0)
    assert torch.equal(torch.get_rng_state(), rng)
    transform(copy.deepcopy(original), blend_sigma=0)
    assert len(fake_loader) == 1
    union = np.logical_or.reduce(original["masks"])
    np.testing.assert_array_equal(sample["image"][~union], original["image"][~union])
    assert np.any(sample["image"][union] != original["image"][union])
    np.testing.assert_array_equal(fake_loader[0].masks[0][0, 0] > 0, union)
    np.testing.assert_array_equal(sample["labels"], original["labels"])
    for before, after in zip(original["masks"], sample["masks"], strict=True):
        np.testing.assert_array_equal(before, after)
    assert sample["metadata"]["harmonizer"]["applied"]
    assert fake_loader[0].marker.grad is None


@pytest.mark.parametrize("reason", ["empty", "strength_zero"])
def test_empty_or_disabled_strength_skips_weights_entirely(fake_loader, reason):
    from offline_yolox.harmonized_data import HarmonizerTransform
    sample = small_sample()
    if reason == "empty":
        sample["masks"] = []
        sample["labels"] = np.empty((0, 5), np.float32)
    original = sample["image"].copy()
    transform = HarmonizerTransform(dict(repository="absent-local-model", device="cpu",
                                         strength=0 if reason == "strength_zero" else 1))
    result = transform(sample)
    assert not fake_loader
    np.testing.assert_array_equal(result["image"], original)
    assert not result["metadata"]["harmonizer"]["applied"]


@pytest.mark.parametrize("changes", [dict(device="invalid"), dict(strength=-0.1),
    dict(strength=1.1), dict(strength=float("nan")), dict(strength=True)])
def test_bad_harmonizer_settings_fail_immediately(changes):
    from offline_yolox.harmonized_data import HarmonizerTransform
    with pytest.raises(ValueError):
        HarmonizerTransform({"device": "cpu", **changes})


def test_serializing_transform_drops_cached_model_and_reloads(fake_loader):
    from offline_yolox.harmonized_data import HarmonizerTransform
    transform = HarmonizerTransform(dict(device="cpu"))
    transform(small_sample())
    restored = pickle.loads(pickle.dumps(transform))
    restored(small_sample())
    assert len(fake_loader) == 2


@pytest.mark.parametrize("mode", MODES)
def test_harmonized_samples_match_plain_geometry_and_cache_model_across_epochs(config, fake_loader, mode):
    config["mode"] = mode
    plain = copy.deepcopy(config)
    plain["mode"] = mode.removesuffix("_harmonized")
    plain.pop("harmonizer")
    plain_exp, exp = create_exp(plain), create_exp(config)
    assert not fake_loader  # Constructing/evaluating the detector needs no Harmonizer weights.
    for epoch in [0, 1]:
        a, b = [e.get_dataset(epoch=epoch).sample_raw() for e in (plain_exp, exp)]
        np.testing.assert_array_equal(a["labels"], b["labels"])
        union = np.zeros((64, 64), bool)
        for mask_a, mask_b in zip(a["masks"], b["masks"], strict=True):
            np.testing.assert_array_equal(mask_a, mask_b)
            union |= mask_a
        np.testing.assert_array_equal(a["image"][~union], b["image"][~union])
        assert np.any(a["image"][union] != b["image"][union])
        assert b["metadata"]["harmonizer"]["applied"]
    assert len(fake_loader) == 1
    for a, b in zip(plain_exp.get_eval_loader(2), exp.get_eval_loader(2), strict=True):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_cuda_harmonizer_cannot_run_inside_dataloader_workers(config, device):
    config.update(num_workers=2)
    config["harmonizer"]["device"] = device
    with pytest.raises(ValueError, match="num_workers"):
        create_exp(config)


@pytest.mark.parametrize("mode", MODES)
def test_cpu_harmonizer_workers_and_infinite_loader_remain_compatible(config, fake_loader, mode):
    config.update(mode=mode, num_workers=2)
    exp = create_exp(config)
    batches = list(exp.get_data_loader(2, finite=True, epoch=0))
    assert len(batches) == 1 and batches[0][0].shape == (2, 3, 64, 64)
    assert torch.isfinite(batches[0][0]).all()
    infinite = exp.get_data_loader(2)
    assert next(iter(infinite))[0].shape == (2, 3, 64, 64)
    infinite.close_mosaic()


@pytest.mark.parametrize("mode", ["online", "online_patch"])
def test_plain_modes_never_load_optional_harmonizer(config, fake_loader, mode):
    config["mode"] = mode
    config.pop("harmonizer")
    exp = create_exp(config)
    exp.get_dataset().sample_raw()
    next(iter(exp.get_data_loader(2, finite=True)))
    assert not fake_loader


@pytest.mark.skipif(not (ROOT / "models/Harmonizer/pretrained/harmonizer.pth").is_file()
    or importlib.util.find_spec("kornia") is None, reason="Optional local Harmonizer source/weights not prepared")
@pytest.mark.parametrize("mode", MODES)
def test_real_public_cli_train_preview_and_unmodified_validation(config, tmp_path, mode):
    config["mode"] = mode
    path = tmp_path / "harmonized.json"
    path.write_text(json.dumps(config))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    preview = tmp_path / "preview"
    overrides = ["--harmonizer-device", "cpu", "--harmonizer-strength", "0.5",
        "--harmonizer-repo", str(ROOT / "models/Harmonizer"),
        "--harmonizer-checkpoint", str(ROOT / "models/Harmonizer/pretrained/harmonizer.pth")]
    for command, extra in [("preview", ["--count", "2", "--output", str(preview)]),
                           ("train", ["--mode", mode, "--device", "cpu"])]:
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), command,
            "--config", str(path), *overrides, *extra], cwd=tmp_path, env=environment,
            capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    for index in range(2):
        metadata = json.loads((preview / f"sample_{index:04}.json").read_text())
        assert metadata["harmonizer"]["applied"]
    output = Path(config["output_dir"]) / config["exp_name"]
    report = json.loads((output / "diagnostic_report.json").read_text())
    assert report["epochs"][0]["images"] == 2
    assert math.isfinite(report["epochs"][0]["mean_loss"]) and report["epochs"][0]["mean_loss"] > 0
    saved = torch.load(output / "latest_ckpt.pth", map_location="cpu", weights_only=True)
    assert saved["config"]["mode"] == mode
    assert saved["config"]["harmonizer"]["strength"] == 0.5
    assert saved["config"]["harmonizer"]["device"] == "cpu"
    # Evaluating a trained detector must not need a Harmonizer model on disk.
    absent = copy.deepcopy(config)
    absent["harmonizer"]["repository"] = str(tmp_path / "absent-model")
    absent["harmonizer"]["checkpoint"] = str(tmp_path / "absent-checkpoint.pth")
    path.write_text(json.dumps(absent))
    evaluation_path = tmp_path / "evaluation.json"
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "evaluate", "--config", str(path),
        "--checkpoint", str(output / "latest_ckpt.pth"), "--device", "cpu", "--output", str(evaluation_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["images"] == (1 if mode == "online_harmonized" else 4)
    assert math.isfinite(evaluation["ap50_95"])


def assert_same_state(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_same_state(a, b)
    else:
        assert first == second


@pytest.mark.skipif(not (ROOT / "models/Harmonizer/pretrained/harmonizer.pth").is_file()
    or importlib.util.find_spec("kornia") is None, reason="Optional local Harmonizer source/weights not prepared")
def test_real_harmonizer_epoch_resume_matches_full_detector_and_optimizer_state(config, tmp_path):
    from offline_yolox.train import run_cpu_diagnostic
    config.update(mode="online_patch_harmonized", max_epoch=2)
    config["harmonizer"]["strength"] = 0.5
    original = copy.deepcopy(config)
    sources = [*Path(config["data_dir"]).rglob("*"),
               ROOT / "models/Harmonizer/pretrained/harmonizer.pth"]
    digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()}
    full = run_cpu_diagnostic(create_exp(config), evaluate=False)
    full_state = torch.load(full["checkpoint"], weights_only=True, map_location="cpu")
    resumed_config = {**copy.deepcopy(config), "output_dir": str(tmp_path / "resumed")}
    first = run_cpu_diagnostic(create_exp(resumed_config), stop_after_epoch=1, evaluate=False)
    # A new experiment must lazily reconstruct Harmonizer without perturbing RNG.
    resumed = run_cpu_diagnostic(create_exp(resumed_config), resume=first["checkpoint"], evaluate=False)
    resumed_state = torch.load(resumed["checkpoint"], weights_only=True, map_location="cpu")
    assert full["epochs"][1]["losses"] == resumed["epochs"][0]["losses"]
    for key in ["model", "optimizer", "ema_model", "ema_updates", "rng"]:
        assert_same_state(full_state[key], resumed_state[key])
    plain = copy.deepcopy(config)
    plain.update(mode="online_patch")
    plain.pop("harmonizer")
    assert full_state["model"].keys() == create_exp(plain).get_model().state_dict().keys()
    assert config == original
    for path, digest in digests.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
