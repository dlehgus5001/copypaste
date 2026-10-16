"""Check DRM exclusion preserves COCO relationships and source files."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from pycocotools.coco import COCO

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/filter_coco.py"
SPEC = importlib.util.spec_from_file_location("filter_coco", SCRIPT)
filter_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(filter_tool)


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "이미지 폴더"
    (images / "가평").mkdir(parents=True)
    assert cv2.imwrite(str(images / "가평/plain.png"), np.full((8, 12, 3), 100, np.uint8))
    (images / "가평/protected.jpg").write_bytes(b"\x9b DRMONE  This Document is encrypted" + b"x" * 100)
    data = {
        "info": {"description": "metadata preserved"}, "licenses": [{"id": 1}],
        "images": [{"id": 2, "file_name": "가평/plain.png", "width": 12, "height": 8},
                   {"id": 19, "file_name": "가평/protected.jpg", "width": 12, "height": 8}],
        "categories": [{"id": 7, "name": "vehicle"}, {"id": 29, "name": "person"}],
        "annotations": [{"id": 104, "image_id": 2, "category_id": 7, "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": 0},
                        {"id": 108, "image_id": 19, "category_id": 29, "bbox": [2, 2, 3, 3], "area": 9, "iscrowd": 0},
                        {"id": 109, "image_id": 19, "category_id": 7, "bbox": [5, 2, 3, 3], "area": 9, "iscrowd": 0}],
    }
    source = tmp_path / "original.json"
    source.write_text(json.dumps(data, ensure_ascii=False))
    return source, images, data


def test_drm_filter_preserves_ids_categories_and_removes_all_related_annotations(dataset, tmp_path):
    source, images, original = dataset
    before_json, before_drm = source.read_bytes(), (images / "가평/protected.jpg").read_bytes()
    output = tmp_path / "clean.json"
    result = filter_tool.filter_coco(source, images, output, check_readable=True)
    cleaned = json.loads(output.read_text())
    assert cleaned == {**original, "images": original["images"][:1], "annotations": original["annotations"][:1]}
    assert result["removed"] == [{"id": 19, "file_name": "가평/protected.jpg", "reason": "DRMONE"}]
    assert source.read_bytes() == before_json and (images / "가평/protected.jpg").read_bytes() == before_drm
    assert COCO(str(output)).getImgIds() == [2]
    assert COCO(str(output)).getAnnIds() == [104]


def test_missing_files_are_errors_not_silently_filtered(dataset, tmp_path):
    source, images, _ = dataset
    (images / "가평/plain.png").unlink()
    output = tmp_path / "clean.json"
    with pytest.raises(OSError, match="id=2"):
        filter_tool.filter_coco(source, images, output)
    assert not output.exists()


def test_explicit_image_ids_can_be_removed_without_reading_their_files(dataset, tmp_path):
    source, images, _ = dataset
    (images / "가평/protected.jpg").unlink()
    output = tmp_path / "clean.json"
    result = filter_tool.filter_coco(source, images, output, exclude_ids=[19])
    assert result["images_after"] == result["annotations_after"] == 1
    assert result["removed"][0]["reason"] == "excluded_id"


def test_decode_check_omits_invalid_non_drm_images(dataset, tmp_path):
    source, images, original = dataset
    data = copy.deepcopy(original)
    data["images"].append({"id": 40, "file_name": "broken.jpg", "width": 12, "height": 8})
    (images / "broken.jpg").write_bytes(b"not an image")
    source.write_text(json.dumps(data))
    result = filter_tool.filter_coco(source, images, tmp_path / "clean.json", check_readable=True)
    assert [(item["id"], item["reason"]) for item in result["removed"]] == [(19, "DRMONE"), (40, "decode_failed")]


def test_filter_refuses_overwrites_or_empty_result(dataset, tmp_path):
    source, images, original = dataset
    with pytest.raises(FileExistsError):
        filter_tool.filter_coco(source, images, source)
    output = tmp_path / "existing.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        filter_tool.filter_coco(source, images, output)
    assert output.read_text() == "keep"
    output = tmp_path / "empty.json"
    with pytest.raises(ValueError, match="No images"):
        filter_tool.filter_coco(source, images, output, exclude_ids=[2, 19])
    assert not output.exists()
    assert json.loads(source.read_text()) == original


def test_config_cli_uses_absolute_validation_image_root(dataset, tmp_path):
    source, images, _ = dataset
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_dir": str(tmp_path), "val_ann": str(source), "val_name": str(images)}))
    output = tmp_path / "clean.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output),
                                "--check-readable"], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["image_root"] == str(images)
    assert COCO(str(output)).getImgIds() == [2]


def test_filtered_json_loads_in_the_yolox_validation_loader(dataset, tmp_path):
    from offline_yolox.experiment import create_exp

    source, images, _ = dataset
    output = tmp_path / "clean.json"
    filter_tool.filter_coco(source, images, output)
    exp = create_exp({
        "preset": "nano", "num_classes": 2, "data_dir": str(tmp_path),
        "train_ann": str(output), "val_ann": str(output),
        "train_name": str(images), "val_name": str(images),
        "input_size": [64, 64], "test_size": [64, 64],
        "batch_size": 1, "num_workers": 0,
    })
    loader = exp.get_eval_loader(1)
    assert loader.dataset.class_ids == [7, 29]
    batch, _, _, ids = next(iter(loader))
    assert tuple(batch.shape) == (1, 3, 64, 64)
    assert ids.reshape(-1).tolist() == [2]
