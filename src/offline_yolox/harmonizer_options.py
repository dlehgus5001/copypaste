"""Explicit Harmonizer controls, separate from detector checkpoint arguments."""
import copy

from .modes import HARMONIZED_MODES


def add_harmonizer_arguments(parser):
    group = parser.add_argument_group("synthetic foreground Harmonizer")
    group.add_argument("--harmonizer-repo", help="Local Harmonizer source/weight folder")
    group.add_argument("--harmonizer-checkpoint", help="Local Harmonizer weights (not YOLOX weights)")
    group.add_argument("--harmonizer-device", choices=("auto", "cpu", "cuda"))
    group.add_argument("--harmonizer-strength", type=float, help="Correction strength, 0 to 1")


def harmonizer_config(values, args):
    overrides = {key: getattr(args, option, None) for key, option in (
        ("repository", "harmonizer_repo"), ("checkpoint", "harmonizer_checkpoint"),
        ("device", "harmonizer_device"), ("strength", "harmonizer_strength"))}
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if not overrides:
        return values
    if values.get("mode") not in HARMONIZED_MODES:
        raise ValueError("Harmonizer options require online_harmonized or online_patch_harmonized mode")
    result = copy.deepcopy(values)
    if not isinstance(result.get("harmonizer", {}), dict):
        raise ValueError("harmonizer must be a configuration object")
    result["harmonizer"] = {**result.get("harmonizer", {}), **overrides}
    return result
