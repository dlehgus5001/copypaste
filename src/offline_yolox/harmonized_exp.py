"""Additive online variants: same YOLOX detector/evaluation, corrected synthetic input."""
from .harmonized_data import HarmonizedOnlineDataset, HarmonizedOnlinePatchDataset, HarmonizerTransform
from .online_exp import OnlineExp
from .patch_exp import OnlinePatchExp


class _HarmonizedExperiment:
    def __init__(self, config):
        self.harmonizer = HarmonizerTransform(config.get("harmonizer", {}))
        self.harmonizer.validate_workers(int(config.get("num_workers", 0)))
        base = dict(config)
        mode = config["mode"]
        base.setdefault("output_dir", f"outputs/{mode}")
        base.setdefault("exp_name", f"{mode}_{config.get('preset', 's')}")
        super().__init__(base)

    def get_dataset(self, *args, **kwargs):
        self.dataset_options = {**getattr(self, "dataset_options", {}), "harmonizer": self.harmonizer}
        return super().get_dataset(*args, **kwargs)


class HarmonizedOnlineExp(_HarmonizedExperiment, OnlineExp):
    dataset_class = HarmonizedOnlineDataset


class HarmonizedOnlinePatchExp(_HarmonizedExperiment, OnlinePatchExp):
    dataset_class = HarmonizedOnlinePatchDataset
