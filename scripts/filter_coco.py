#!/usr/bin/env python3
"""Create a COCO JSON excluding DRMONE images and their annotations.

Only read source images; never decrypt, rename or delete them. Preserve COCO
IDs, categories and all remaining metadata. Missing/inaccessible files are
errors, so an incorrect image root cannot silently remove a whole dataset.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    if not isinstance(value, (str, Path)) or not str(value) or "://" in str(value):
        raise ValueError("Expected a nonempty local filesystem path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def validation_paths(config_file):
    config = json.loads(local_path(config_file).read_text(encoding="utf-8"))
    data_dir = local_path(config["data_dir"])
    annotations = data_dir / "annotations" / config.get("val_ann", "instances_val2017.json")
    image_root = data_dir / config.get("val_name", "val2017")
    return annotations.resolve(), image_root.resolve()


def filter_coco(input_file, image_root, output_file, *, exclude_ids=(), check_readable=False):
    source, image_root, output = map(local_path, (input_file, image_root, output_file))
    if output == source or output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output JSON; existing files are preserved: {output}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(data.get(key), list)
                                         for key in ("images", "annotations", "categories")):
        raise ValueError("Expected COCO images, annotations and categories lists")
    ids = [image["id"] for image in data["images"]]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("COCO image IDs must be unique integers")
    known_ids = set(ids)
    if any(annotation.get("image_id") not in known_ids for annotation in data["annotations"]):
        raise ValueError("Source JSON already contains annotations referring to missing image IDs")
    excluded = set(exclude_ids)
    if excluded - known_ids:
        raise ValueError(f"Requested image IDs are absent from this JSON: {sorted(excluded - known_ids)}")
    if check_readable:
        import cv2

    kept, removed = [], []
    for image in data["images"]:
        image_id = image["id"]
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name or "://" in file_name:
            raise ValueError(f"Image {image_id} must have a nonempty local file_name")
        reason = "excluded_id" if image_id in excluded else None
        if reason is None:
            # Match official COCODataset: absolute file_name overrides the root.
            path = image_root / file_name
            try:
                with path.open("rb") as stream:
                    header = stream.read(512)
                    if b"DRMONE" in header[:64] and b"This Document is encry" in header:
                        reason = "DRMONE"
                    elif check_readable:
                        try:
                            # Use the same path-based decoder as official YOLOX.
                            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
                        except cv2.error:
                            decoded = None
                        if decoded is None:
                            reason = "decode_failed"
            except OSError as error:
                raise OSError(
                    f"Cannot read image id={image_id}: {path}. "
                    f"{type(error).__name__}: {error}. "
                    "Check image root/access; no output JSON has been written."
                ) from error
        if reason:
            removed.append({"id": image_id, "file_name": file_name, "reason": reason})
        else:
            kept.append(image)
    if not kept:
        raise ValueError("No images would remain; no output JSON was written")
    kept_ids = {image["id"] for image in kept}
    annotations = [annotation for annotation in data["annotations"] if annotation["image_id"] in kept_ids]
    result = {**data, "images": kept, "annotations": annotations}
    summary = {
        "input": str(source), "output": str(output), "image_root": str(image_root),
        "images_before": len(ids), "images_after": len(kept),
        "annotations_before": len(data["annotations"]), "annotations_after": len(annotations),
        "removed": removed, "ids_renumbered": False,
        "checked_decoding": check_readable,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".coco-filter-", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # Exclusive creation protects the input and any preexisting destination.
        with output.open("xb") as destination, temporary.open("rb") as completed:
            try:
                shutil.copyfileobj(completed, destination)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", help="Use the validation JSON/image paths from this training config")
    inputs.add_argument("--input", help="Source COCO JSON; requires --image-root")
    parser.add_argument("--image-root", help="Folder prefixed to relative file_name entries")
    parser.add_argument("--output", required=True, help="New COCO JSON path; never overwrite")
    parser.add_argument("--exclude-ids", type=int, nargs="*", default=[], help="Additional image IDs to omit")
    parser.add_argument("--check-readable", action="store_true", help="Also decode non-DRM images and omit decoding failures")
    args = parser.parse_args(argv)
    if args.config and args.image_root:
        parser.error("--image-root is only used with --input")
    if args.input and not args.image_root:
        parser.error("--input requires --image-root")
    try:
        source, root = validation_paths(args.config) if args.config else (args.input, args.image_root)
        summary = filter_coco(source, root, args.output, exclude_ids=args.exclude_ids,
                              check_readable=args.check_readable)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
