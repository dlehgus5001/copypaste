"""Offline editing orchestration tests; fake pixels do not validate Qwen quality."""
import copy
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from offline_yolox.asset_discovery import discover_objects
from offline_yolox.copy_paste import OnlineCopyPasteDataset
from offline_yolox.qwen_edit import adapt_objects, load_pipeline, native_attention_imports, read_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "자산 폴더"
    for folder in ("object", "mask", "aerial_crop"):
        for category in (1, 3):
            (root / folder / f"object_{category}").mkdir(parents=True)
    for category in (1, 3):
        pixels = np.full((96, 160, 3), (30, 70, 110), dtype=np.uint8)
        pixels[15:75, 40:120] = (120, 180, 230)
        mask = np.zeros((96, 160), dtype=np.uint8)
        mask[15:75, 40:120] = 255
        mask[40:55, 40:60] = 0
        assert cv2.imwrite(str(root / "object" / f"object_{category}" / "차량 001.png"), pixels)
        assert cv2.imwrite(str(root / "mask" / f"object_{category}" / "차량 001.png"), mask)
        assert cv2.imwrite(str(root / "aerial_crop" / f"object_{category}" / "reference.png"),
                           np.full((24, 40, 3), category * 25, dtype=np.uint8))
    model = tmp_path / "local_model"
    index = {"_class_name": "QwenImageEditPlusPipeline"}
    for component in ("transformer", "text_encoder", "tokenizer", "processor", "vae", "scheduler"):
        (model / component).mkdir(parents=True)
        index[component] = ["diffusers", "TestComponent"]
    (model / "model_index.json").write_text(json.dumps(index))
    return dict(model_dir=str(model), asset_root=str(root), objects_dir="object", masks_dir="mask",
                references_dir="aerial_crop", class_ids=[1, 3], class_names=["truck_a", "truck_b"],
                folder_categories={"object_1": 1, "object_3": 3},
                output=str(root / "qwen_adapted"), work_size=[96, 128], crop_padding=0.15,
                variants_per_object=2, num_inference_steps=2, seed=29,
                device="cpu", dtype="float32", offload="none",
                prompt="Preserve image 1 {class_name} ({category_id}); use image 2 camera appearance.")


class FakePipeline:
    """Deliberately edits the whole canvas to test the mask boundary, not image realism."""
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        seed = kwargs["generator"].initial_seed()
        color = (20 + seed % 180, 91, 205)
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), color)])


def snapshot(root):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


def test_adapted_bank_preserves_masks_background_coordinates_and_replays(config):
    root, output = Path(config["asset_root"]), Path(config["output"])
    before, original_config = snapshot(root), copy.deepcopy(config)
    pipeline = FakePipeline()
    summary = adapt_objects(config, pipeline=pipeline)
    assert config == original_config and summary["count"] == len(pipeline.calls) == 4
    assert summary["review_required"] and not summary["semantic_shape_preservation_verified"]
    for name, digest in before.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    for metadata_path, call in zip(summary["samples"], pipeline.calls, strict=True):
        record = json.loads((output / metadata_path).read_text())
        source = np.array(Image.open(root / record["image"]).convert("RGB"))
        mask = np.array(Image.open(root / record["mask"])) > 0
        exported = np.array(Image.open(output / record["output_image"]))
        final_mask = np.array(Image.open(output / record["output_mask"])) > 0
        assert exported.shape == source.shape == (96, 160, 3)
        np.testing.assert_array_equal(final_mask, mask)
        np.testing.assert_array_equal(exported[~mask], source[~mask])
        expected = [20 + record["seed"] % 180, 91, 205]
        np.testing.assert_array_equal(exported[mask], np.tile(expected, (mask.sum(), 1)))
        assert f"object_{record['category_id']}" in record["reference"]
        assert record["class_name"] in call["prompt"] and len(call["image"]) == 2
        assert call["image"][0].size == call["image"][1].size == (128, 96)
        assert call["generator"].device.type == "cpu"
        assert call["num_inference_steps"] == 2 and call["output_type"] == "pil"
        assert Image.open(output / record["review_image"]).size == (512, 120)
        assert Image.open(output / record["raw_image"]).size == (128, 96)
    changed = {**config, "output": str(root / "replay")}
    repeated = adapt_objects(changed, pipeline=FakePipeline())
    for a, b in zip(summary["samples"], repeated["samples"], strict=True):
        first = json.loads((output / a).read_text())
        second = json.loads((root / "replay" / b).read_text())
        assert first == second
        np.testing.assert_array_equal(np.array(Image.open(output / first["output_image"])),
                                      np.array(Image.open(root / "replay" / second["output_image"])))
    settings = json.loads((output / "copy_paste_settings.json").read_text())["copy_paste"]
    bank = discover_objects(root, settings["objects_dir"], settings["masks_dir"], [1, 3],
                            ["truck_a", "truck_b"], folder_categories=settings["folder_categories"])
    assert len(bank) == 4
    (root / "background").mkdir()
    assert cv2.imwrite(str(root / "background/bg.png"), np.full((128, 128, 3), 70, np.uint8))
    settings.update(backgrounds_dir="background", epoch_length=2, object_count=[2, 2],
                    object_long_side_range=[8, 12], unique_classes=True, require_object_count=True)
    dataset = OnlineCopyPasteDataset(settings, (128, 128), [1, 3], ["truck_a", "truck_b"])
    sample = dataset.sample_raw()
    assert len(sample["labels"]) == 2
    assert {item["category_id"] for item in sample["metadata"]["instances"]} == {1, 3}


def test_check_inputs_public_cli_uses_no_optional_model_and_creates_no_output(config, tmp_path):
    path = tmp_path / "qwen.json"
    path.write_text(json.dumps(config))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "adapt-objects", "--config", str(path),
                             "--check-inputs", "--limit", "1", "--variants-per-object", "1"],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IPv4/IPv6 sockets denied" in result.stderr
    status = json.loads(result.stdout.strip().splitlines()[-1])
    assert status["status"] == "inputs_checked_model_not_loaded" and status["generations"] == 1
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("offload", ["none", "model", "sequential"])
def test_pipeline_load_is_local_safetensors_offline_and_configures_offload(config, monkeypatch, offload):
    settings = read_settings({**config, "device": "cuda", "dtype": "bfloat16", "offload": offload})
    events = []
    pipeline = SimpleNamespace(
        to=lambda device: events.append(("to", device)),
        enable_model_cpu_offload=lambda **kwargs: events.append(("model", kwargs)),
        enable_sequential_cpu_offload=lambda **kwargs: events.append(("sequential", kwargs)),
        transformer=SimpleNamespace(set_attention_backend=lambda backend: events.append(("backend", backend))),
        vae=SimpleNamespace(enable_tiling=lambda: events.append(("tiling",)),
                            enable_slicing=lambda: events.append(("slicing",))))
    encoder = object()
    class EncoderLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            events.append(("encoder", path, kwargs))
            return encoder
    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            events.append(("load", path, kwargs))
            return pipeline
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(QwenImageEditPlusPipeline=Loader))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Qwen2_5_VLForConditionalGeneration=EncoderLoader))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DIFFUSERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DIFFUSERS_ATTN_BACKEND", "flash")
    assert load_pipeline(settings, Path(config["model_dir"])) is pipeline
    assert events[0] == ("encoder", str(Path(config["model_dir"]) / "text_encoder"),
                         dict(local_files_only=True, use_safetensors=True,
                              torch_dtype=torch.bfloat16, attn_implementation="sdpa"))
    assert events[1] == ("load", config["model_dir"], dict(local_files_only=True, use_safetensors=True,
                                                         torch_dtype=torch.bfloat16, text_encoder=encoder))
    assert events[2] == ("backend", "native")
    assert events[3] == (("to", "cuda") if offload == "none" else (offload, {"gpu_id": 0}))
    assert ("tiling",) in events and ("slicing",) in events
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["DIFFUSERS_ATTN_BACKEND"] == "native"


@pytest.mark.parametrize("fail_inside", [False, True])
def test_native_attention_imports_hide_optional_extensions_and_restore_lookup(tmp_path, monkeypatch, fail_inside):
    (tmp_path / "flash_attn").mkdir()
    (tmp_path / "flash_attn/__init__.py").write_text("raise ImportError('incompatible CUDA extension')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = importlib.util.find_spec
    assert before("flash_attn") is not None
    try:
        with native_attention_imports():
            assert importlib.util.find_spec("flash_attn") is None
            assert importlib.util.find_spec("flash_attn.bert_padding") is None
            assert importlib.util.find_spec("flash_attn_2_cuda") is None
            assert importlib.util.find_spec("xformers") is None
            assert importlib.util.find_spec("torch") is not None
            if fail_inside:
                raise RuntimeError("import failure")
    except RuntimeError as error:
        assert fail_inside and str(error) == "import failure"
    assert importlib.util.find_spec is before
    assert before("flash_attn") is not None


def test_reference_class_missing_fails_before_inference(config):
    root = Path(config["asset_root"])
    (root / "aerial_crop/object_3/reference.png").unlink()
    pipeline = FakePipeline()
    with pytest.raises(ValueError, match="No aerial reference crop for category_id=3"):
        adapt_objects(config, pipeline=pipeline)
    assert not pipeline.calls and not Path(config["output"]).exists()


@pytest.mark.parametrize("exception_type", [ImportError, RuntimeError])
def test_pipeline_import_failure_preserves_dependency_conflict_and_fix(config, monkeypatch, exception_type):
    original_import = builtins.__import__
    conflict = "peft>=0.17.0 is required, but found peft==0.15.2"

    def failing_import(name, *args, **kwargs):
        if name == "diffusers":
            raise exception_type(conflict)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DIFFUSERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError) as error:
        load_pipeline(read_settings(config), Path(config["model_dir"]))
    assert conflict in str(error.value)
    assert "requirements-qwen.txt" in str(error.value)
    assert "peft==0.17.1" in str(error.value)
    assert isinstance(error.value.__cause__, exception_type)


@pytest.mark.parametrize("kind", ["nonbinary", "empty", "wrong_size"])
def test_bad_source_masks_fail_check_without_writes(config, kind):
    path = Path(config["asset_root"]) / "mask/object_1/차량 001.png"
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if kind == "nonbinary":
        pixels[20, 50] = 128
    elif kind == "empty":
        pixels[:] = 0
    else:
        pixels = pixels[:30]
    assert cv2.imwrite(str(path), pixels)
    with pytest.raises(ValueError, match="Mask|dimensions"):
        adapt_objects(config, check_inputs=True)
    assert not Path(config["output"]).exists()


def test_output_protection_remote_paths_and_missing_model_components(config, tmp_path):
    root = Path(config["asset_root"])
    for output in (root / "object/new", tmp_path / "outside"):
        with pytest.raises(ValueError, match="output"):
            adapt_objects({**config, "output": str(output)}, check_inputs=True)
    output = Path(config["output"])
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        adapt_objects(config, pipeline=FakePipeline())
    assert (output / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="local"):
        adapt_objects({**config, "model_dir": "https://example.com/model"}, check_inputs=True)
    (Path(config["model_dir"]) / "processor").rmdir()
    with pytest.raises(FileNotFoundError, match="missing component"):
        adapt_objects(config, check_inputs=True)


def test_wrong_generated_size_is_not_silently_resized_or_exported(config):
    pipeline = lambda **kwargs: SimpleNamespace(images=[Image.new("RGB", (64, 64))])
    with pytest.raises(ValueError, match="dimensions differ"):
        adapt_objects(config, pipeline=pipeline)
    assert not list(Path(config["output"]).rglob("*.png"))


@pytest.mark.parametrize("changes", [dict(work_size=[65, 128]), dict(variants_per_object=0),
                                    dict(num_inference_steps=0), dict(seed=True), dict(prompt=""),
                                    dict(crop_padding=float("nan")), dict(offload="remote")])
def test_invalid_edit_settings(config, changes):
    with pytest.raises(ValueError):
        read_settings({**config, **changes})
