"""Standalone paste geometry, foreground-only correction and genuine local weights."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from offline_yolox.paste_check import color_filters_without_flash_attention, harmonize_image, load_harmonizer, paste_object, save_paste_check

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(2)
    bg = np.full((64, 80, 3), 60, np.uint8)
    bg[:, :, 0] = np.arange(80, dtype=np.uint8)
    obj = np.full((8, 16, 3), (180, 50, 220), np.uint8)
    mask = np.zeros((8, 16), np.uint8)
    mask[1:7, 2:14] = 255
    paths = []
    for name, pixels in [("배경.png", bg), ("객체.png", obj), ("마스크.png", mask)]:
        path = tmp_path / name
        assert cv2.imwrite(str(path), pixels)
        paths.append(path)
    return bg, obj, mask > 0, paths


class FakeHarmonizer:
    def predict_arguments(self, image, mask):
        return [torch.tensor([0.2])] * 6

    def restore_image(self, image, mask, arguments):
        # Deliberately changes background too: the integration must guard it.
        return [torch.clamp(image + 0.2, 0, 1)]


def test_paste_keeps_crop_pixels_and_scales_tight_foreground_exactly(inputs):
    bg, obj, mask, _ = inputs
    original = bg.copy()
    base, pasted, support, alpha, meta = paste_object(bg, obj, mask, patch_size=(32, 48),
        crop_xy=(10, 20), object_size=24, position=(7, 9), feather=0)
    np.testing.assert_array_equal(base, bg[20:52, 10:58])
    np.testing.assert_array_equal(bg, original)
    assert meta["bbox_xyxy"] == [7, 9, 31, 21] and support.sum() == 24 * 12
    np.testing.assert_array_equal(pasted[support], np.tile([180, 50, 220], (support.sum(), 1)))
    np.testing.assert_array_equal(pasted[~support], base[~support])
    np.testing.assert_array_equal(alpha, support.astype(np.float32))


def test_feather_stays_inside_support_and_full_background_mode(inputs):
    bg, obj, mask, _ = inputs
    base, pasted, support, alpha, meta = paste_object(bg, obj, mask, patch_size=None,
        object_size=24, feather=1)
    assert pasted.shape == bg.shape and meta["background_crop_xy"] == [0, 0]
    np.testing.assert_array_equal(pasted[~support], bg[~support])
    assert alpha[~support].max() == 0 and np.any(alpha[support] < 1)


def test_partial_object_requires_explicit_opt_in_and_has_correct_visible_box(inputs):
    bg, obj, mask, _ = inputs
    with pytest.raises(ValueError, match="does not fit"):
        paste_object(bg, obj, mask, patch_size=None, object_size=24, position=(-8, 5))
    _, _, support, _, meta = paste_object(bg, obj, mask, patch_size=None,
        object_size=24, position=(-8, 5), allow_partial=True)
    assert meta["bbox_xyxy"] == [0, 5, 16, 17]
    assert support.sum() == 16 * 12 and meta["visible_fraction"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("strength", [0, 0.5, 1])
def test_harmonization_strength_and_background_guard(inputs, strength):
    bg, obj, mask, _ = inputs
    _, pasted, support, alpha, _ = paste_object(bg, obj, mask, patch_size=None, object_size=24, feather=0)
    result, info = harmonize_image(FakeHarmonizer(), pasted, support, alpha, strength=strength)
    np.testing.assert_array_equal(result[~support], pasted[~support])
    original = pasted[support].astype(float)
    expected = np.rint(original * (1 - strength) + np.clip(original + 51, 0, 255) * strength)
    np.testing.assert_allclose(result[support], expected, atol=1)
    assert info["strength"] == strength and len(info["filter_parameters"]) == 6


def test_custom_harmonization_saves_review_mask_and_support_box_without_changing_inputs(inputs, tmp_path):
    _, _, _, paths = inputs
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    output = tmp_path / "result"
    result = save_paste_check(*paths, output, model=FakeHarmonizer(), device="cpu",
                              patch_size=(64, 80), object_size=24)
    assert result["harmonizer_applied"] and result["background_pixels_preserved"]
    assert result["foreground_mean_absolute_change"] > 0
    assert set(result["files"]) == {"background", "before", "after", "mask", "alpha", "boxes", "comparison"}
    before, after, mask = [cv2.imread(str(output / name), cv2.IMREAD_UNCHANGED)
                           for name in ("before.png", "after.png", "mask.png")]
    np.testing.assert_array_equal(before[mask == 0], after[mask == 0])
    assert cv2.imread(str(output / "comparison.png")).shape[:2] == (92, 160)
    ys, xs = np.nonzero(mask)
    assert result["bbox_xyxy"] == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    for path, digest in zip(paths, hashes, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        save_paste_check(*paths, output, plain_only=True)


def test_plain_public_cli_needs_no_coco_config_or_harmonizer_repo(inputs, tmp_path):
    _, _, _, paths = inputs
    output = tmp_path / "plain"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, str(ROOT / "run.py"), "paste-check",
        "--background", str(paths[0]), "--object", str(paths[1]), "--mask", str(paths[2]),
        "--patch-size", "64", "80", "--object-size", "24", "--output", str(output),
        "--harmonizer-repo", str(tmp_path / "absent"), "--plain-only"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "IPv4/IPv6 sockets denied" in completed.stderr
    report = json.loads((output / "result.json").read_text())
    assert not report["harmonizer_applied"] and report["model"] is None
    assert not (output / "after.png").exists() and not (output / "comparison.png").exists()


def test_small_background_padding_and_invalid_positions(inputs):
    bg, obj, mask, _ = inputs
    base, _, _, _, _ = paste_object(bg[:8, :8], obj, mask, patch_size=(32, 32), object_size=8)
    assert np.all(base[8:, 8:] == 114)
    for options in [dict(position=(10000, 10000), allow_partial=True), dict(crop_xy=(-1, 0)),
                    dict(position=(1.5, 2)), dict(patch_size=None, crop_xy=(0, 0)),
                    dict(object_size=1), dict(feather=-1)]:
        with pytest.raises(ValueError):
            paste_object(bg, obj, mask, **options)


def test_invalid_model_output_does_not_write_results(inputs, tmp_path):
    class Invalid(FakeHarmonizer):
        def restore_image(self, image, mask, arguments):
            return [torch.full_like(image, float("nan"))]
    with pytest.raises(ValueError, match="nonfinite"):
        save_paste_check(*inputs[3], tmp_path / "bad", model=Invalid(), device="cpu", object_size=24)
    assert not (tmp_path / "bad").exists()


def test_wrong_harmonizer_source_fails_before_executing_it(tmp_path):
    source = tmp_path / "wrong/src/model"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="unexpected Harmonizer source"):
        load_harmonizer(tmp_path / "wrong", device="cpu")


@pytest.mark.parametrize("already_loaded", [True, False])
def test_color_filter_import_guard_restores_existing_optional_module_on_error(monkeypatch, already_loaded):
    previous = object()
    if already_loaded:
        monkeypatch.setitem(sys.modules, "flash_attn", previous)
    else:
        monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    with pytest.raises(RuntimeError, match="test"):
        with color_filters_without_flash_attention():
            assert sys.modules["flash_attn"] is None
            raise RuntimeError("test")
    if already_loaded:
        assert sys.modules["flash_attn"] is previous
    else:
        assert "flash_attn" not in sys.modules


@pytest.mark.parametrize("good", [True, False])
def test_online_preparation_hashes_downloads_and_commits_only_complete_folder(tmp_path, monkeypatch, good):
    spec = importlib.util.spec_from_file_location("prepare_harmonizer_test", ROOT / "scripts/prepare_harmonizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "configs").mkdir()
    body = b"pinned source bytes"
    manifest = {"revision": "test", "files": {"src/model/__init__.py": hashlib.sha256(body).hexdigest()}}
    (tmp_path / "configs/harmonizer-source.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body if good else b"wrong"))
    output = tmp_path / "ready"
    if good:
        module.prepare(output)
        assert (output / "src/model/__init__.py").read_bytes() == body
        with pytest.raises(FileExistsError):
            module.prepare(output)
    else:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            module.prepare(output)
        assert not output.exists()


@pytest.mark.skipif(not (ROOT / "models/Harmonizer/pretrained/harmonizer.pth").is_file()
                    or importlib.util.find_spec("kornia") is None, reason="Local optional Harmonizer source/weights not prepared")
def test_genuine_checkpoint_public_cpu_cli_and_pixel_preservation(inputs, tmp_path):
    output = tmp_path / "real"
    environment = os.environ.copy()
    # Kornia imports an unused LightGlue feature module. An incompatible system
    # FlashAttention must not prevent the color-only Harmonizer from loading.
    poison = tmp_path / "optional_extensions/flash_attn"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text("raise ImportError('flash_attn_2_cuda: undefined symbol')\n")
    environment["PYTHONPATH"] = str(poison.parent)
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "paste-check",
        "--background", str(inputs[3][0]), "--object", str(inputs[3][1]), "--mask", str(inputs[3][2]),
        "--object-size", "24", "--patch-size", "64", "80", "--device", "cpu", "--output", str(output)],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "result.json").read_text())
    assert report["model"]["checkpoint_sha256"] == "401dd03706fc6fa37b4600ae4c643a2fa9ad0f11351acfbf030cb5d3490de6ca"
    assert report["model"]["device"] == "cpu" and report["correction"]["seconds"] > 0
    assert report["foreground_mean_absolute_change"] > 0 and report["background_pixels_preserved"]
