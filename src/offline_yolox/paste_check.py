"""Paste one object on a background and compare with local Harmonizer correction."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from .bootstrap import ROOT, local_path
from .copy_paste import OnlineCopyPasteDataset, bbox_from_mask, transform_object
from .patch_data import crop_canvas, dimensions
from .startup import report, startup_stage


@contextmanager
def color_filters_without_flash_attention():
    """Kornia imports LightGlue, but Harmonizer only needs its color filters."""
    missing = object()
    previous = sys.modules.get("flash_attn", missing)
    sys.modules["flash_attn"] = None
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = previous


def _finite(value, name, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (minimum is not None and value < minimum) or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {name}: {value}")
    return float(value)


def load_harmonizer(repository, checkpoint=None, device="auto"):
    repository = local_path(repository, must_exist=True)
    manifest = json.loads((ROOT / "configs/harmonizer-source.json").read_text())
    for relative, digest in manifest["files"].items():
        if not relative.endswith(".py"):
            continue
        path = repository / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Missing or unexpected Harmonizer source: {path}. Prepare the pinned source with scripts/prepare_harmonizer.py")
    checkpoint = local_path(checkpoint, must_exist=True) if checkpoint else repository / "pretrained/harmonizer.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local Harmonizer checkpoint missing: {checkpoint}")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    alias = "_copypaste_harmonizer_" + hashlib.sha256(str(repository).encode()).hexdigest()[:12]
    if alias not in sys.modules:
        package = repository / "src/model"
        spec = importlib.util.spec_from_file_location(alias, package / "__init__.py",
                                                      submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        try:
            with color_filters_without_flash_attention():
                spec.loader.exec_module(module)
        except Exception:
            for name in list(sys.modules):
                if name == alias or name.startswith(alias + "."):
                    del sys.modules[name]
            raise
    model = sys.modules[alias].Harmonizer()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, dict(device=device, checkpoint=str(checkpoint),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       repository=str(repository), source_revision=manifest["revision"])


def paste_object(background, image, mask, *, patch_size=(640, 640), crop_xy=None,
                 object_size=80, position=None, angle=0, feather=0.6, allow_partial=False):
    """Keep native background pixels, transform foreground and return exact support."""
    if not isinstance(allow_partial, bool):
        raise ValueError("allow_partial must be boolean")
    if isinstance(object_size, bool) or not isinstance(object_size, int) or object_size < 2:
        raise ValueError("object_size must be an integer >=2")
    angle = _finite(angle, "angle")
    feather = _finite(feather, "feather", minimum=0)
    for name, value in [("crop_xy", crop_xy), ("position", position)]:
        if value is not None and (not isinstance(value, (list, tuple)) or len(value) != 2
                                  or any(isinstance(v, bool) or not isinstance(v, int) for v in value)):
            raise ValueError(f"{name} must be two integers X Y")
    if patch_size is None:
        if crop_xy is not None:
            raise ValueError("crop_xy cannot be used with full-background mode")
        canvas, crop_xy = background.copy(), (0, 0)
    else:
        patch_size = dimensions(patch_size, "patch_size")
        crop_xy = crop_xy or (max(0, (background.shape[1] - patch_size[1]) // 2),
                              max(0, (background.shape[0] - patch_size[0]) // 2))
        if not (0 <= crop_xy[0] < background.shape[1] and 0 <= crop_xy[1] < background.shape[0]):
            raise ValueError("crop_xy must start inside the background")
        canvas = crop_canvas(background, *crop_xy, patch_size)
    base = canvas.copy()
    obj, support = transform_object(image, mask, object_size, angle)
    height, width = canvas.shape[:2]
    oh, ow = support.shape
    x, y = position if position is not None else ((width - ow) // 2, (height - oh) // 2)
    if not allow_partial and (x < 0 or y < 0 or x + ow > width or y + oh > height):
        raise ValueError("Object does not fit the canvas; reduce size/change position or explicitly use --allow-partial")
    left, top, right, bottom = max(0, x), max(0, y), min(width, x + ow), min(height, y + oh)
    if right <= left or bottom <= top:
        raise ValueError("Object is outside the canvas")
    sx, sy = left - x, top - y
    cropped = support[sy:sy + bottom - top, sx:sx + right - left]
    full_mask = np.zeros((height, width), dtype=bool)
    full_mask[top:bottom, left:right] = cropped
    bounds = bbox_from_mask(full_mask)
    if bounds is None:
        raise ValueError("Placed object has no visible foreground")
    alpha = support.astype(np.float32)
    if feather:
        alpha *= cv2.GaussianBlur(alpha, (0, 0), feather, borderType=cv2.BORDER_CONSTANT)
    full_alpha = np.zeros((height, width), dtype=np.float32)
    full_alpha[top:bottom, left:right] = alpha[sy:sy + bottom - top, sx:sx + right - left]
    region = canvas[top:bottom, left:right]
    weights = full_alpha[top:bottom, left:right, None]
    foreground = obj[sy:sy + bottom - top, sx:sx + right - left]
    region[:] = np.clip(np.rint(foreground * weights + region * (1 - weights)), 0, 255).astype(np.uint8)
    return base, canvas, full_mask, full_alpha, dict(background_crop_xy=list(crop_xy),
        canvas_size=[height, width], object_position_xy=[x, y], object_size=object_size,
        rotation_degrees=angle, feather=feather, bbox_xyxy=bounds,
        transformed_size=[oh, ow], visible_fraction=float(full_mask.sum() / support.sum()))


def harmonize_image(model, image, mask, alpha, *, device="cpu", strength=1.0):
    strength = _finite(strength, "strength", minimum=0, maximum=1)
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        arguments = model.predict_arguments(tensor, mask_tensor)
        restored = model.restore_image(tensor, mask_tensor, arguments)[-1]
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    if restored.shape != tensor.shape or not torch.isfinite(restored).all():
        raise ValueError("Harmonizer returned invalid dimensions or nonfinite pixels")
    parameters = [float(arg.detach().cpu().reshape(-1)[0]) for arg in arguments]
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("Harmonizer returned nonfinite filter parameters")
    adjusted = restored[0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255
    result = image.copy()
    mixed = image.astype(np.float32) * (1 - strength * alpha[:, :, None]) + adjusted * (strength * alpha[:, :, None])
    result[mask] = np.clip(np.rint(mixed[mask]), 0, 255).astype(np.uint8)
    return result, dict(strength=strength, seconds=elapsed, filter_parameters=parameters)


def _write_png(path, image):
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Cannot encode: {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def _comparison(before, after):
    height, width = before.shape[:2]
    canvas = Image.new("RGB", (width * 2, height + 28), "white")
    canvas.paste(Image.fromarray(before[:, :, ::-1]), (0, 28))
    canvas.paste(Image.fromarray(after[:, :, ::-1]), (width, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "BEFORE: plain paste", fill="black")
    draw.text((width + 8, 7), "AFTER: Harmonizer", fill="black")
    return np.asarray(canvas)[:, :, ::-1].copy()


def save_paste_check(background, object_image, object_mask, output, *, repository="models/Harmonizer",
                     checkpoint=None, device="auto", strength=1.0, plain_only=False, model=None, **placement):
    strength = _finite(strength, "strength", minimum=0, maximum=1)
    output = local_path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be new or empty: {output}")
    paths = [local_path(path, must_exist=True) for path in (background, object_image, object_mask)]
    bg, obj = [OnlineCopyPasteDataset._read_image(path) for path in paths[:2]]
    mask = OnlineCopyPasteDataset._read_mask(paths[2], obj.shape[:2])
    base, pasted, visible, alpha, metadata = paste_object(bg, obj, mask, **placement)
    model_info, correction = None, None
    after = None
    if not plain_only:
        started = perf_counter()
        if model is None:
            with startup_stage("Loading local Harmonizer"):
                model, model_info = load_harmonizer(repository, checkpoint, device)
        else:
            model_info = dict(device=device if device != "auto" else "cpu", backend="test_or_custom_model")
        model_info["load_seconds"] = perf_counter() - started
        report(f"Harmonizing {pasted.shape[1]}x{pasted.shape[0]} on {model_info['device']}")
        after, correction = harmonize_image(model, pasted, visible, alpha,
                                            device=model_info["device"], strength=strength)
    output.mkdir(parents=True, exist_ok=True)
    images = dict(background=base, before=pasted, mask=visible.astype(np.uint8) * 255,
                  alpha=np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8))
    if after is not None:
        images.update(after=after, comparison=_comparison(pasted, after))
    boxed = (after if after is not None else pasted).copy()
    x1, y1, x2, y2 = metadata["bbox_xyxy"]
    cv2.rectangle(boxed, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 1)
    images["boxes"] = boxed
    for name, pixels in images.items():
        _write_png(output / (name + ".png"), pixels)
    difference = np.abs(after.astype(np.int16) - pasted.astype(np.int16)) if after is not None else None
    summary = dict(**metadata, background=str(paths[0]), object=str(paths[1]), source_mask=str(paths[2]),
                   harmonizer_applied=after is not None, model=model_info, correction=correction,
                   foreground_mean_absolute_change=float(difference[visible].mean()) if difference is not None else None,
                   background_pixels_preserved=bool(np.array_equal((after if after is not None else pasted)[~visible], base[~visible])),
                   files={name: name + ".png" for name in images}, bbox_format="xyxy_half_open",
                   note="Ground-truth support box, not detector prediction. Harmonizer adjusts color/illumination; aerial texture transfer is not included.")
    with (output / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", required=True)
    parser.add_argument("--object", required=True, dest="object_image")
    parser.add_argument("--mask", required=True, dest="object_mask")
    parser.add_argument("--output", default="outputs/paste_check")
    parser.add_argument("--harmonizer-repo", default="models/Harmonizer", dest="repository")
    parser.add_argument("--checkpoint", help="Local fine-tuned/official Harmonizer state dict")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--plain-only", action="store_true", help="Paste without loading Harmonizer; no AFTER result is produced")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--patch-size", type=int, nargs=2, metavar=("H", "W"), default=[640, 640])
    group.add_argument("--full-background", action="store_true")
    parser.add_argument("--crop-xy", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--object-size", type=int, default=80, help="Foreground long side before rotation, in pixels")
    parser.add_argument("--position", type=int, nargs=2, metavar=("X", "Y"), help="Top-left in the resulting canvas; default: center")
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--feather", type=float, default=0.6)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = vars(parser.parse_args(argv))
    threads = args.pop("cpu_threads")
    if threads <= 0:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(threads)
    if args.pop("full_background"):
        args["patch_size"] = None
    result = save_paste_check(**args)
    print(json.dumps(dict(output=str(local_path(args["output"])), harmonizer_applied=result["harmonizer_applied"])))
    return result


if __name__ == "__main__":
    main()
