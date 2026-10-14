"""Local COCO evaluation, using official preprocessing and postprocessing."""
import argparse
import contextlib
import copy
import io
import json

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .bootstrap import local_path
from .experiment import create_exp
from .train import load_checkpoint
from yolox.evaluators import COCOEvaluator
from yolox.utils import postprocess


def empty_coco_result(ground_truth):
    """Construct a genuine empty result (loadRes([]) raises IndexError)."""
    result = COCO()
    result.dataset = {"info": copy.deepcopy(ground_truth.dataset.get("info", {})),
                      "images": copy.deepcopy(ground_truth.dataset["images"]),
                      "categories": copy.deepcopy(ground_truth.dataset["categories"]),
                      "annotations": []}
    result.createIndex()
    return result


def evaluate_model(exp, model, *, device="cpu", evaluator="standard"):
    loader = exp.get_eval_loader(exp.batch_size)
    converter = COCOEvaluator(loader, exp.test_size, exp.test_conf, exp.nmsthre, exp.num_classes)
    was_training = model.training
    model.to(device).eval()
    predictions = []
    image_ids = []
    try:
        with torch.inference_mode():
            for images, _, info, ids in loader:
                outputs = model(images.to(device=device, dtype=torch.float32))
                outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
                predictions.extend(converter.convert_to_coco_format(outputs, info, ids))
                image_ids.extend(int(item) for item in ids.reshape(-1))
    finally:
        model.train(was_training)
    if len(image_ids) != len(loader.dataset) or len(set(image_ids)) != len(image_ids):
        raise RuntimeError("Evaluation must visit every COCO image exactly once")
    gt = loader.dataset.coco
    gt.dataset.setdefault("info", {})
    result = gt.loadRes(predictions) if predictions else empty_coco_result(gt)
    evaluation_type = COCOeval
    if evaluator == "fast":
        from yolox.layers import COCOeval_opt
        evaluation_type = COCOeval_opt
    elif evaluator != "standard":
        raise ValueError("evaluator must be standard or fast")
    evaluation = evaluation_type(gt, result, "bbox")
    evaluation.params.imgIds = sorted(image_ids)
    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return {"ap50_95": float(evaluation.stats[0]), "ap50": float(evaluation.stats[1]),
            "images": len(image_ids), "detections": len(predictions), "backend": evaluator,
            "stats": [float(item) for item in evaluation.stats], "summary": summary.getvalue()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/original.json")
    parser.add_argument("--checkpoint", required=True, help="Required local checkpoint; downloads are disabled")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--evaluator", choices=("standard", "fast"), default="standard")
    parser.add_argument("--raw-model", action="store_true", help="Evaluate training weights instead of saved EMA")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for diagnostic evaluation")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    exp = create_exp(args.config)
    saved = load_checkpoint(args.checkpoint)
    model = exp.get_model()
    weights = saved["model"] if args.raw_model else saved.get("ema_model") or saved["model"]
    model.load_state_dict(weights, strict=True)
    report = evaluate_model(exp, model, device=args.device, evaluator=args.evaluator)
    report["checkpoint"] = str(local_path(args.checkpoint, must_exist=True))
    report["device"] = args.device
    path = local_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    main()
