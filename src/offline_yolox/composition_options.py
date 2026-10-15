"""Shared, explicit composition overrides for training, preview and COCO export."""
import argparse
import copy


def add_composition_arguments(parser):
    group = parser.add_argument_group("online composition")
    group.add_argument("--objects-per-image", type=int,
                       help="Request this fixed number of objects per composite")
    sizes = group.add_mutually_exclusive_group()
    sizes.add_argument("--object-size", type=int, metavar="PIXELS",
                       help="Fixed foreground long-side size before rotation (pixels, >=2)")
    sizes.add_argument("--object-size-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                       help="Uniform inclusive foreground long-side range before rotation")
    group.add_argument("--unique-objects", action=argparse.BooleanOptionalAction, default=None,
                       help="Select each object/mask source at most once per composite")
    group.add_argument("--require-object-count", action=argparse.BooleanOptionalAction, default=None,
                       help="Retry or fail instead of accepting fewer objects than requested")


def composition_config(config, args):
    """Apply overrides to a copy so checkpoints record the effective settings."""
    values = copy.deepcopy(config)
    names = ("objects_per_image", "object_size", "object_size_range",
             "unique_objects", "require_object_count")
    if not any(getattr(args, name, None) is not None for name in names):
        return values
    if values.get("mode", "original") != "online":
        raise ValueError("Composition options require an online configuration")
    if not isinstance(values.get("copy_paste"), dict):
        raise ValueError("Online mode requires a copy_paste configuration object")
    settings = values["copy_paste"]
    count = getattr(args, "objects_per_image", None)
    if count is not None:
        if count < 0:
            raise ValueError("--objects-per-image must be nonnegative")
        settings["object_count"] = [count, count]
    size = getattr(args, "object_size", None)
    size_range = getattr(args, "object_size_range", None)
    if size is not None:
        if size < 2:
            raise ValueError("--object-size must be at least 2 pixels")
        settings.pop("object_long_side_range", None)
        settings["object_long_side_pixels"] = [size]
        settings["object_size_weights"] = [1]
    elif size_range is not None:
        if len(size_range) != 2 or size_range[0] < 2 or size_range[0] > size_range[1]:
            raise ValueError("--object-size-range requires 2 <= MIN <= MAX")
        settings.pop("object_long_side_pixels", None)
        settings.pop("object_size_weights", None)
        settings["object_long_side_range"] = list(size_range)
    for name in ("unique_objects", "require_object_count"):
        value = getattr(args, name, None)
        if value is not None:
            settings[name] = value
    return values
