"""Exercise the installed binary dependencies and real COCO evaluator behavior."""
import contextlib
import importlib
import importlib.util
import io

import cv2
import numpy as np
import pytest
import torch
import torchvision
from pycocotools import mask as mask_api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def test_numpy_opencv_pycocotools_mask_roundtrip(tmp_path):
    mask = np.zeros((47, 63), dtype=np.uint8)
    cv2.ellipse(mask, (30, 22), (19, 13), 27, 0, 360, 1, thickness=-1)
    # Include a hole so encoding a solid rectangle cannot accidentally pass.
    cv2.circle(mask, (30, 22), 4, 0, thickness=-1)
    path = tmp_path / "mask.png"
    assert cv2.imwrite(str(path), mask * 255)
    loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    binary = (loaded > 0).astype(np.uint8)
    np.testing.assert_array_equal(binary, mask)

    encoded = mask_api.encode(np.asfortranarray(binary))
    decoded = mask_api.decode(encoded)
    np.testing.assert_array_equal(decoded, mask)
    assert int(mask_api.area(encoded)) == int(mask.sum())
    ys, xs = np.nonzero(mask)
    expected_box = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1)]
    np.testing.assert_array_equal(mask_api.toBbox(encoded), expected_box)
    # Verify NumPy/Torch interop for the actual decoded binary extension output.
    tensor = torch.from_numpy(decoded.copy())
    assert tensor.dtype == torch.uint8
    np.testing.assert_array_equal(tensor.numpy(), mask)


def test_tensorboard_summary_writer_produces_readable_scalar(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    log_dir = tmp_path / "tensorboard"
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        writer.add_scalar("train/total_loss", 1.25, global_step=7)
        writer.add_scalar("train/total_loss", 0.75, global_step=8)
        writer.flush()
    events = list(log_dir.glob("events.out.tfevents.*"))
    assert events and all(event.stat().st_size > 0 for event in events)
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    scalars = accumulator.Scalars("train/total_loss")
    assert [(item.step, item.value) for item in scalars] == [(7, 1.25), (8, 0.75)]


def test_torchvision_cpu_nms_binary_operator():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert torchvision.ops.nms(boxes, scores, iou_threshold=0.5).tolist() == [0, 2]


def _perfect_coco_fixture():
    # All three area ranges, two images, and nonconsecutive category IDs exercise
    # evaluator bookkeeping as well as the underlying C++ arithmetic.
    annotations = [
        {"id": 1, "image_id": 7, "category_id": 3, "bbox": [4, 5, 16, 18],
         "area": 288, "iscrowd": 0},
        {"id": 2, "image_id": 7, "category_id": 9, "bbox": [30, 40, 40, 50],
         "area": 2000, "iscrowd": 0},
        {"id": 3, "image_id": 41, "category_id": 3, "bbox": [20, 25, 100, 110],
         "area": 11000, "iscrowd": 0},
        {"id": 4, "image_id": 41, "category_id": 9, "bbox": [180, 185, 16, 18],
         "area": 288, "iscrowd": 0},
    ]
    truth = COCO()
    # Intentionally omit info, just as upstream COCODataset does. The selected
    # pycocotools 2.0.11 wheel must handle loadRes without a source patch.
    truth.dataset = {
        "images": [{"id": image_id, "width": 256, "height": 256}
                   for image_id in (7, 41)],
        "categories": [{"id": 3, "name": "vehicle"}, {"id": 9, "name": "aircraft"}],
        "annotations": annotations,
    }
    truth.createIndex()
    detections = [{"image_id": row["image_id"], "category_id": row["category_id"],
                   "bbox": list(row["bbox"]), "score": 1.0} for row in annotations]
    return truth, truth.loadRes(detections)


def _evaluate(evaluator_type):
    with contextlib.redirect_stdout(io.StringIO()):
        truth, detections = _perfect_coco_fixture()
        evaluator = evaluator_type(truth, detections, "bbox")
        evaluator.params.imgIds = [7, 41]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def test_standard_coco_perfect_predictions_and_missing_info():
    evaluator = _evaluate(COCOeval)
    np.testing.assert_allclose(evaluator.stats, np.ones(12), rtol=0, atol=1e-12)


def test_fast_coco_extension_matches_standard_perfect_predictions():
    from offline_yolox.bootstrap import activate_vendor

    activate_vendor()
    module_name = "yolox.layers.fast_cocoeval"
    if importlib.util.find_spec(module_name) is None:
        pytest.skip("Official FastCOCOEval extension has not been built; run scripts/build_yolox.sh")
    # ABI/load failures of an existing extension must fail this test, not skip it.
    extension = importlib.import_module(module_name)
    assert hasattr(extension, "COCOevalEvaluateImages")
    assert hasattr(extension, "COCOevalAccumulate")
    from yolox.layers import COCOeval_opt

    standard = _evaluate(COCOeval)
    fast = _evaluate(COCOeval_opt)
    np.testing.assert_allclose(fast.stats, standard.stats, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast.eval["precision"], standard.eval["precision"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast.eval["recall"], standard.eval["recall"], rtol=0, atol=1e-12)
    assert fast.stats[0] == pytest.approx(1.0, abs=1e-12)
