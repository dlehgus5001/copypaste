"""Additive patch workflows sharing the official detector and training lifecycle."""
import copy

import torch

from .experiment import OfflineExp, seed_worker
from .online_exp import OnlineExp
from .patch_data import CocoPatchDataset, OnlinePatchDataset, patch_settings


class PatchEvaluation:
    def random_resize(self, data_loader, epoch, rank, is_distributed):
        return self.input_size

    def get_eval_dataset(self, **kwargs):
        from yolox.data import ValTransform
        return self._check_dataset(CocoPatchDataset(
            data_dir=self.data_dir, json_file=self.val_ann, name=self.val_name,
            input_size=self.test_size, settings=self.patch, training=False,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        ))


class OriginalPatchExp(PatchEvaluation, OfflineExp):
    def __init__(self, config):
        self.patch = patch_settings(config)
        base = dict(config)
        base.setdefault("output_dir", "outputs/original_patch")
        base.setdefault("exp_name", f"original_patch_{config.get('preset', 's')}")
        # Dataset supplies already-cropped images. Do not merge several patches
        # or rescale them with the full-image training augmentation defaults.
        base.update(mosaic_prob=0.0, mixup_prob=0.0, enable_mixup=False, multiscale_range=0)
        super().__init__(base)
        self._dataset_template = self.get_dataset()

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import TrainTransform
        if cache:
            raise ValueError("Patch mode does not enable image caching")
        if hasattr(self, "_dataset_template"):
            return copy.copy(self._dataset_template)
        return CocoPatchDataset(
            data_dir=self.data_dir, json_file=self.train_ann, name=self.train_name,
            input_size=self.input_size, settings=self.patch,
            preproc=TrainTransform(max_labels=self.patch["max_labels"],
                                   flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
        )

    # Reuse the simple no-Mosaic loader, including finite/infinite samplers.
    def get_data_loader(self, batch_size, is_distributed=False, no_aug=False,
                        cache_img=None, finite=False, epoch=0):
        from yolox.data import DataLoader, InfiniteSampler, YoloBatchSampler
        if is_distributed:
            raise ValueError("This workflow supports single GPU only")
        if cache_img is not None:
            raise ValueError("Patch workflow does not enable image caching")
        dataset = self.get_dataset()
        generator = torch.Generator().manual_seed(self.seed + epoch)
        sampler = (torch.utils.data.RandomSampler(dataset, generator=generator) if finite
                   else InfiniteSampler(len(dataset), seed=self.seed))
        batches = YoloBatchSampler(sampler, batch_size=batch_size, drop_last=False, mosaic=False)
        self.dataset = dataset
        return DataLoader(dataset, batch_sampler=batches, num_workers=self.data_num_workers,
                          pin_memory=not finite, worker_init_fn=seed_worker, generator=generator)


class OnlinePatchExp(PatchEvaluation, OnlineExp):
    dataset_class = OnlinePatchDataset

    def __init__(self, config):
        self.patch = patch_settings(config)
        self.dataset_options = {"patch": self.patch}
        base = dict(config)
        base.setdefault("output_dir", "outputs/online_patch")
        base.setdefault("exp_name", f"online_patch_{config.get('preset', 's')}")
        base.update(multiscale_range=0)
        super().__init__(base)
