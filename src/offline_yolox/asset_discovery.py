"""Discover local object/mask pairs and backgrounds without manifest files.

Pairing uses the relative directory and filename stem. Image extensions may
differ, so ``objects/car/a.jpg`` pairs with ``masks/car/a.png``. Discovery only
reads directory entries; image and mask contents are checked by the dataset.
"""

from collections.abc import Mapping
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def _root_path(asset_root):
    if not isinstance(asset_root, (str, Path)) or not str(asset_root) or "://" in str(asset_root):
        raise ValueError("asset_root must be a nonempty local directory path")
    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"asset_root directory does not exist: {root}")
    return root


def _directory(root, value, name):
    if not isinstance(value, (str, Path)) or not str(value) or "://" in str(value):
        raise ValueError(f"{name} must be a nonempty local directory path")
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{name} escapes asset_root: {value}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


def _images(root, directory):
    """Follow contained directories, rejecting escaping links and link cycles."""
    images = []

    def visit(path, ancestors):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Asset path escapes asset_root: {path}")
        if resolved in ancestors:
            raise ValueError(f"Directory symlink cycle in asset folder: {path}")
        ancestors = ancestors | {resolved}
        for entry in sorted(path.iterdir(), key=lambda item: item.name):
            target = entry.resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Asset path escapes asset_root: {entry}")
            if entry.is_dir():
                visit(entry, ancestors)
            elif entry.suffix.lower() in IMAGE_SUFFIXES:
                if not entry.is_file():
                    raise FileNotFoundError(f"Asset image is not a readable file: {entry}")
                images.append(entry)

    visit(directory, set())
    return sorted(images, key=lambda path: path.relative_to(directory).as_posix())


def _taxonomy(class_ids, class_names, folder_categories):
    ids, names = list(class_ids), list(class_names)
    if (not ids or len(ids) != len(names) or len(set(ids)) != len(ids)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)):
        raise ValueError("class_ids and class_names must describe a nonempty, unique COCO taxonomy")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("class_names must contain unique nonempty class names for folder discovery")
    if folder_categories is None:
        folder_categories = {}
    if not isinstance(folder_categories, Mapping):
        raise ValueError("folder_categories must map first-level object folder names to COCO category IDs")
    for folder, category in folder_categories.items():
        if (not isinstance(folder, str) or not folder or folder in {".", ".."}
                or "/" in folder or "\\" in folder):
            raise ValueError("folder_categories keys must be first-level object folder names")
        if isinstance(category, bool) or not isinstance(category, int) or category not in ids:
            raise ValueError(f"folder_categories[{folder!r}] must be a category ID from {ids}")
    return ids, dict(zip(names, ids)), dict(folder_categories)


def _category(relative, class_ids, names_to_ids, folder_categories):
    if len(class_ids) == 1:
        return class_ids[0]
    folder = relative.parts[0] if len(relative.parts) > 1 else None
    if folder in folder_categories:
        return folder_categories[folder]
    matches = set()
    if folder in names_to_ids:
        matches.add(names_to_ids[folder])
    if folder is not None:
        try:
            numeric_id = int(folder)
        except ValueError:
            pass
        else:
            if numeric_id in class_ids:
                matches.add(numeric_id)
    if len(matches) == 1:
        return matches.pop()
    detail = "ambiguous" if matches else "unresolved"
    raise ValueError(
        f"Object class is {detail} for {relative.as_posix()!r}. For multiple classes, put objects "
        "in first-level folders named after a validation COCO class or category ID, or set "
        "copy_paste.folder_categories to map each folder name to its COCO category ID."
    )


def discover_objects(asset_root, objects_dir, masks_dir, class_ids, class_names, *,
                     mask_suffix="", folder_categories=None):
    """Return object entries paired with masks using relative path and stem.

    A single-class taxonomy assigns its category to every object. For multiple
    classes, the first folder below ``objects_dir`` selects a class by its COCO
    name or category ID; ``folder_categories`` can explicitly override that map.
    ``mask_suffix='_mask'`` pairs ``a.jpg`` with ``a_mask.png``. Extra mask files
    are ignored, while a missing or ambiguous mask for any object is an error.
    """
    root = _root_path(asset_root)
    objects_root = _directory(root, objects_dir, "objects_dir")
    masks_root = _directory(root, masks_dir, "masks_dir")
    if objects_root.is_relative_to(masks_root) or masks_root.is_relative_to(objects_root):
        raise ValueError("objects_dir and masks_dir must be separate, non-overlapping directories")
    if not isinstance(mask_suffix, str) or "/" in mask_suffix or "\\" in mask_suffix:
        raise ValueError("mask_suffix must be a string without directory separators")
    ids, names_to_ids, categories = _taxonomy(class_ids, class_names, folder_categories)
    images = _images(root, objects_root)
    if not images:
        raise ValueError(f"No supported object images found in objects_dir: {objects_root}")
    masks = {}
    for mask in _images(root, masks_root):
        relative = mask.relative_to(masks_root)
        masks.setdefault((relative.parent, relative.stem), []).append(mask)

    result, seen = [], set()
    for image in images:
        relative = image.relative_to(objects_root)
        key = (relative.parent, relative.stem)
        if key in seen:
            raise ValueError(
                f"Duplicate object filename stem {relative.with_suffix('').as_posix()!r} in "
                "objects_dir; different image extensions cannot share the same relative stem"
            )
        seen.add(key)
        mask_key = (relative.parent, relative.stem + mask_suffix)
        matches = masks.get(mask_key, [])
        if not matches:
            expected = masks_root / relative.parent / (relative.stem + mask_suffix)
            raise FileNotFoundError(
                f"No matching mask for object {relative.as_posix()!r}; expected {expected} "
                "with a supported image extension. Match relative subfolders and filename "
                "stems, or set copy_paste.mask_suffix."
            )
        if len(matches) != 1:
            choices = ", ".join(path.relative_to(root).as_posix() for path in matches)
            raise ValueError(f"Ambiguous masks for object {relative.as_posix()!r}: {choices}")
        result.append({
            "image": image.relative_to(root).as_posix(),
            "mask": matches[0].relative_to(root).as_posix(),
            "category_id": _category(relative, ids, names_to_ids, categories),
        })
    return result


def discover_backgrounds(asset_root, backgrounds_dir):
    """Return all supported background images as asset-root-relative paths."""
    root = _root_path(asset_root)
    directory = _directory(root, backgrounds_dir, "backgrounds_dir")
    images = _images(root, directory)
    if not images:
        raise ValueError(f"No supported background images found in backgrounds_dir: {directory}")
    return [image.relative_to(root).as_posix() for image in images]
