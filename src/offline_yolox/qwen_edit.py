"""Local-only Qwen reference editing of object assets before Copy-Paste training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from .asset_discovery import discover_objects, discover_references
from .bootstrap import local_path
from .copy_paste import OnlineCopyPasteDataset
from .startup import report, startup_stage


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def read_settings(config):
    if isinstance(config, (str, Path)):
        config = json.loads(local_path(config, must_exist=True).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Qwen configuration must be a JSON object")
    values = dict(config)
    values.setdefault("work_size", [512, 512])
    size = values["work_size"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 64 or v % 32 for v in size)):
        raise ValueError("work_size must be [height, width], multiples of 32 and >=64")
    for key, default in [("variants_per_object", 1), ("num_inference_steps", 40), ("seed", 42)]:
        values[key] = _integer(values.get(key, default), key, 0 if key == "seed" else 1)
    values["crop_padding"] = _number(values.get("crop_padding", 0.15), "crop_padding")
    for key, default in [("true_cfg_scale", 4.0), ("guidance_scale", 1.0)]:
        values[key] = _number(values.get(key, default), key)
    for key, choices, default in [("device", ("cuda", "cpu"), "cuda"),
                                   ("dtype", ("bfloat16", "float16", "float32"), "bfloat16"),
                                   ("offload", ("none", "model", "sequential"), "sequential")]:
        values.setdefault(key, default)
        if values[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    if values["device"] == "cpu" and (values["dtype"] != "float32" or values["offload"] != "none"):
        raise ValueError("CPU editing requires dtype=float32 and offload=none; real Qwen inference is very large")
    for key, default in [("prompt", None), ("negative_prompt", " ")]:
        values.setdefault(key, default)
        if not isinstance(values[key], str) or (key == "prompt" and not values[key].strip()):
            raise ValueError(f"{key} must be a {'nonempty ' if key == 'prompt' else ''}string")
    return values


def prepare_jobs(settings, limit=None):
    """Validate local paths and build class-matched jobs without loading Qwen."""
    root = local_path(settings["asset_root"], must_exist=True)
    model = local_path(settings["model_dir"], must_exist=True)
    index_path = model / "model_index.json"
    if not model.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"model_dir must be a complete local Diffusers model folder: {model}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("_class_name") != "QwenImageEditPlusPipeline":
        raise ValueError("model_dir must contain Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline)")
    for key in ("transformer", "text_encoder", "tokenizer", "processor", "vae", "scheduler"):
        if key not in index or not (model / key).is_dir():
            raise FileNotFoundError(f"Incomplete local Qwen snapshot; missing component: {model / key}")
    output = local_path(settings["output"])
    if not output.is_relative_to(root):
        raise ValueError("output must be inside asset_root so generated assets can share the existing backgrounds")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Qwen output must be new or empty: {output}")
    directories = {key: (root / settings[key]).resolve()
                   for key in ("objects_dir", "masks_dir", "references_dir")}
    for key, directory in directories.items():
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError(f"output must not overlap input {key}")
    if model.is_relative_to(output) or output.is_relative_to(model):
        raise ValueError("output must not overlap model_dir")
    ids, names = settings["class_ids"], settings["class_names"]
    objects = discover_objects(root, settings["objects_dir"], settings["masks_dir"], ids, names,
                               mask_suffix=settings.get("mask_suffix", ""),
                               folder_categories=settings.get("folder_categories"))
    references = discover_references(root, settings["references_dir"], ids, names,
                                     folder_categories=settings.get("folder_categories"))
    if limit is not None:
        objects = objects[:_integer(limit, "limit")]
    by_class = {}
    for entry in references:
        by_class.setdefault(entry["category_id"], []).append(entry["image"])
    class_names = dict(zip(ids, names, strict=True))
    jobs = []
    for source in objects:
        category = source["category_id"]
        if category not in by_class:
            raise ValueError(f"No aerial reference crop for category_id={category}; add crops in the corresponding class folder")
        relative = (root / source["image"]).relative_to(directories["objects_dir"])
        for variant in range(settings["variants_per_object"]):
            seed_text = f"{settings['seed']}:{source['image']}:{variant}"
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big") % (2**63)
            candidates = by_class[category]
            reference = candidates[int(np.random.default_rng(seed).integers(len(candidates)))]
            target = relative.parent / f"{relative.stem}__qwen_{variant:03}.png"
            jobs.append(dict(**source, reference=reference, class_name=class_names[category],
                             variant=variant, seed=seed, target=target.as_posix()))
    return root, model, output, jobs


def load_pipeline(settings, model):
    # Set offline flags before importing the optional Hugging Face dependencies.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DIFFUSERS_OFFLINE"):
        os.environ[key] = "1"
    if settings["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Qwen editing needs a suitable GPU; --check-inputs does not load the model")
        if settings["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16; choose a supported dtype")
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise RuntimeError("Install the optional local dependencies from requirements-qwen.txt") from exc
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        str(model), local_files_only=True, use_safetensors=True,
        torch_dtype=getattr(torch, settings["dtype"]),
    )
    if settings["offload"] == "model":
        pipeline.enable_model_cpu_offload(gpu_id=0)
    elif settings["offload"] == "sequential":
        pipeline.enable_sequential_cpu_offload(gpu_id=0)
    else:
        pipeline.to(settings["device"])
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    return pipeline


def _letterbox(image, size):
    height, width = size
    ratio = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                           Image.Resampling.LANCZOS)
    left, top = (width - resized.width) // 2, (height - resized.height) // 2
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + resized.width, top + resized.height)


def prepare_source(image, mask, size, padding):
    ys, xs = np.nonzero(mask)
    border = math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * padding)
    crop = (max(0, int(xs.min()) - border), max(0, int(ys.min()) - border),
            min(image.shape[1], int(xs.max()) + 1 + border), min(image.shape[0], int(ys.max()) + 1 + border))
    clean = np.full_like(image, 128)
    clean[mask] = image[mask]
    canvas, content = _letterbox(Image.fromarray(clean).crop(crop), size)
    return canvas, dict(crop_xyxy=crop, content_xyxy=content)


def restore_foreground(generated, original, mask, transform, size):
    if not isinstance(generated, Image.Image) or generated.size != (size[1], size[0]):
        raise ValueError("Qwen output dimensions differ from work_size; refusing an unsafe coordinate remap")
    x1, y1, x2, y2 = transform["crop_xyxy"]
    crop = generated.convert("RGB").crop(transform["content_xyxy"]).resize(
        (x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    result = original.copy()
    region = result[y1:y2, x1:x2]
    selected = mask[y1:y2, x1:x2]
    region[selected] = np.asarray(crop)[selected]
    return result


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        image.save(handle, format="PNG")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _read_rgb(path):
    return cv2.cvtColor(OnlineCopyPasteDataset._read_image(path), cv2.COLOR_BGR2RGB)


def _review(source, reference, generated, exported, mask, transform, size):
    x1, y1, x2, y2 = transform["crop_xyxy"]
    mask_canvas = Image.fromarray(mask[y1:y2, x1:x2].astype(np.uint8) * 255).resize(
        (transform["content_xyxy"][2] - transform["content_xyxy"][0],
         transform["content_xyxy"][3] - transform["content_xyxy"][1]), Image.Resampling.NEAREST)
    work_mask = Image.new("L", (size[1], size[0]))
    work_mask.paste(mask_canvas, transform["content_xyxy"][:2])
    outlined = np.asarray(generated.convert("RGB")).copy()
    contours, _ = cv2.findContours(np.asarray(work_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 1)
    clean = np.full_like(exported, 128)
    clean[mask] = exported[mask]
    final, _ = _letterbox(Image.fromarray(clean).crop((x1, y1, x2, y2)), size)
    panels = [source, reference, Image.fromarray(outlined), final]
    result = Image.new("RGB", (size[1] * 4, size[0] + 24), "white")
    draw = ImageDraw.Draw(result)
    for index, (label, panel) in enumerate(zip(["source", "aerial reference", "raw edit + original mask", "exported foreground"], panels)):
        result.paste(panel, (index * size[1], 24))
        draw.text((index * size[1] + 5, 5), label, fill="black")
    return result


def adapt_objects(config, *, limit=None, check_inputs=False, pipeline=None):
    settings = read_settings(config)
    root, model, output, jobs = prepare_jobs(settings, limit)
    if check_inputs:
        for job in jobs:
            image = _read_rgb(root / job["image"])
            OnlineCopyPasteDataset._read_mask(root / job["mask"], image.shape[:2])
            _read_rgb(root / job["reference"])
        return dict(status="inputs_checked_model_not_loaded", sources=len({job["image"] for job in jobs}),
                    generations=len(jobs), model_dir=str(model), output=str(output))
    if pipeline is None:
        with startup_stage("Loading local Qwen-Image-Edit-2511; CPU offload may be slow"):
            pipeline = load_pipeline(settings, model)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    current_source = None
    for index, job in enumerate(jobs):
        if job["image"] != current_source:
            image = _read_rgb(root / job["image"])
            mask = OnlineCopyPasteDataset._read_mask(root / job["mask"], image.shape[:2])
            source, transform = prepare_source(image, mask, settings["work_size"], settings["crop_padding"])
            current_source = job["image"]
        reference_pixels = _read_rgb(root / job["reference"])
        reference, _ = _letterbox(Image.fromarray(reference_pixels), settings["work_size"])
        prompt = settings["prompt"].replace("{class_name}", job["class_name"]).replace("{category_id}", str(job["category_id"]))
        report(f"Qwen {index + 1}/{len(jobs)}: {job['image']} <- {job['reference']} (seed={job['seed']})")
        with torch.inference_mode():
            result = pipeline(image=[source.copy(), reference], prompt=prompt,
                              negative_prompt=settings["negative_prompt"],
                              height=settings["work_size"][0], width=settings["work_size"][1],
                              generator=torch.Generator(device="cpu").manual_seed(job["seed"]),
                              num_inference_steps=settings["num_inference_steps"],
                              true_cfg_scale=settings["true_cfg_scale"], guidance_scale=settings["guidance_scale"],
                              num_images_per_prompt=1, output_type="pil")
        if not hasattr(result, "images") or len(result.images) != 1:
            raise ValueError("Qwen must return exactly one edited image per job")
        generated = result.images[0]
        exported = restore_foreground(generated, image, mask, transform, settings["work_size"])
        target = Path(job["target"])
        for folder, content in [("object", Image.fromarray(exported)),
                                ("mask", Image.fromarray(mask.astype(np.uint8) * 255)),
                                ("raw", generated),
                                ("review", _review(source, reference, generated, exported, mask, transform, settings["work_size"]))]:
            _write_image(output / folder / target, content)
        record = dict(**job, prompt=prompt, transform=transform,
                      source_size=[image.shape[0], image.shape[1]],
                      reference_size=list(reference_pixels.shape[:2]),
                      output_image=(Path("object") / target).as_posix(),
                      output_mask=(Path("mask") / target).as_posix(),
                      raw_image=(Path("raw") / target).as_posix(),
                      review_image=(Path("review") / target).as_posix(),
                      mask_geometry_preserved=True, semantic_shape_preservation_verified=False,
                      review_required=True)
        _write_json(output / "metadata" / target.with_suffix(".json"), record)
        records.append(record)
    relative = output.relative_to(root)
    training_settings = dict(asset_root=str(root), objects_dir=(relative / "object").as_posix(),
                             masks_dir=(relative / "mask").as_posix(), mask_suffix="")
    if settings.get("folder_categories") is not None:
        training_settings["folder_categories"] = settings["folder_categories"]
    _write_json(output / "copy_paste_settings.json", {"copy_paste": training_settings})
    summary = dict(config=settings, model_dir=str(model), output=str(output), count=len(records),
                   model_index_sha256=hashlib.sha256((model / "model_index.json").read_bytes()).hexdigest(),
                   review_required=True, semantic_shape_preservation_verified=False,
                   samples=[(Path("metadata") / Path(job["target"]).with_suffix(".json")).as_posix() for job in jobs])
    _write_json(output / "manifest.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen_edit.json")
    parser.add_argument("--model-dir", help="Complete local Diffusers snapshot, never a remote model ID")
    parser.add_argument("--asset-root")
    parser.add_argument("--references-dir", help="Class-labelled aerial crop folders under asset-root")
    parser.add_argument("--output", help="New/empty output folder inside asset-root")
    parser.add_argument("--limit", type=int, help="Process only the first N object/mask pairs")
    parser.add_argument("--variants-per-object", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--offload", choices=("none", "model", "sequential"))
    parser.add_argument("--check-inputs", action="store_true", help="Validate files without importing/loading Qwen or creating output")
    args = parser.parse_args(argv)
    settings = json.loads(local_path(args.config, must_exist=True).read_text(encoding="utf-8"))
    for key in ("model_dir", "asset_root", "references_dir", "output", "variants_per_object",
                "num_inference_steps", "seed", "offload"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    result = adapt_objects(settings, limit=args.limit, check_inputs=args.check_inputs)
    print(json.dumps({key: result[key] for key in ("output", "count", "status", "generations") if key in result}))
    return result


if __name__ == "__main__":
    main()
