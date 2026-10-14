"""Configuration of official model, optimizer, augmentation and COCO data APIs."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .bootstrap import local_path
from yolox.exp import Exp, get_exp


PRESETS = ("nano", "tiny", "s", "m", "l", "x")
EXP_FIELDS = {
    "num_classes", "input_size", "test_size", "max_epoch", "basic_lr_per_img",
    "warmup_epochs", "no_aug_epochs", "warmup_lr", "min_lr_ratio", "scheduler",
    "ema", "momentum", "weight_decay", "mosaic_prob", "mixup_prob", "hsv_prob",
    "flip_prob", "degrees", "translate", "mosaic_scale", "mixup_scale", "shear",
    "enable_mixup", "multiscale_range", "random_size", "test_conf", "nmsthre",
    "eval_interval", "print_interval", "save_history_ckpt", "seed", "exp_name",
    "train_ann", "val_ann",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Derive Python and NumPy worker seeds from PyTorch's seeded generator."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_config(path):
    with local_path(path, must_exist=True).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    return config


class OfflineExp(Exp):
    def __init__(self, config):
        preset = config.get("preset", "s")
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}")
        template = get_exp(exp_name=f"yolox-{preset}")
        self.__dict__.update(vars(template))
        self._preset_class = type(template)
        self.config = dict(config)
        self.preset = preset
        for name in EXP_FIELDS:
            if name in config:
                value = config[name]
                if name in {"input_size", "test_size", "mosaic_scale", "mixup_scale", "random_size"}:
                    value = tuple(value)
                setattr(self, name, value)
        self.seed = int(config.get("seed", 42))
        self.data_num_workers = int(config.get("num_workers", 0))
        self.data_dir = str(local_path(config["data_dir"], must_exist=True))
        self.output_dir = str(local_path(config.get("output_dir", "outputs/original")))
        self.train_name = config.get("train_name", "train2017")
        self.val_name = config.get("val_name", "val2017")
        self.exp_name = config.get("exp_name", f"original_{preset}")
        self.batch_size = int(config.get("batch_size", 8))
        if self.num_classes <= 0 or self.max_epoch <= 0 or self.batch_size <= 0:
            raise ValueError("num_classes, max_epoch and batch_size must be positive")
        if self.data_num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.eval_interval <= 0 or self.print_interval <= 0 or self.basic_lr_per_img <= 0:
            raise ValueError("eval_interval, print_interval and basic_lr_per_img must be positive")
        for size in (self.input_size, self.test_size):
            if len(size) != 2 or any(int(v) != v or v < 64 or v % 32 for v in size):
                raise ValueError("input_size and test_size must contain two multiples of 32, at least 64")
        if self.warmup_epochs < 0 or self.no_aug_epochs < 0:
            raise ValueError("warmup_epochs and no_aug_epochs must be nonnegative")
        if self.warmup_epochs + self.no_aug_epochs >= self.max_epoch and self.scheduler == "yoloxwarmcos":
            raise ValueError("YOLOX warm cosine needs warmup_epochs + no_aug_epochs < max_epoch")
        self.class_ids, self.class_names = self._validate_taxonomy()

    def _validate_taxonomy(self):
        signatures = []
        for annotation_name in (self.train_ann, self.val_ann):
            annotation_path = local_path(Path(self.data_dir) / "annotations" / annotation_name, must_exist=True)
            with annotation_path.open(encoding="utf-8") as handle:
                categories = json.load(handle).get("categories", [])
            signature = sorted((category["id"], category["name"]) for category in categories)
            ids = [item[0] for item in signature]
            if len(signature) != self.num_classes or len(set(ids)) != len(ids):
                raise ValueError(f"num_classes={self.num_classes} differs from COCO categories, or category IDs repeat: {ids}")
            if any(not isinstance(category_id, int) or not isinstance(name, str) or not name
                   for category_id, name in signature):
                raise ValueError("COCO category IDs must be integers and names nonempty strings")
            signatures.append(signature)
        if signatures[0] != signatures[1]:
            raise ValueError("Train and validation COCO category IDs and names must agree")
        return [item[0] for item in signatures[0]], [item[1] for item in signatures[0]]

    def get_model(self):
        # Includes the original nano depthwise backbone/head, with original initialization.
        return self._preset_class.get_model(self)

    def _check_dataset(self, dataset):
        if len(dataset) == 0:
            raise ValueError("COCO dataset contains no images")
        if len(dataset.class_ids) != self.num_classes:
            raise ValueError(f"num_classes={self.num_classes} differs from COCO categories={dataset.class_ids}")
        # Defensive compatibility for older pycocotools that require metadata in loadRes.
        dataset.coco.dataset.setdefault("info", {})
        return dataset

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return self._check_dataset(COCODataset(
            data_dir=self.data_dir, json_file=self.train_ann, name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache, cache_type=cache_type,
        ))

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return self._check_dataset(COCODataset(
            data_dir=self.data_dir, json_file=self.val_ann, name=self.val_name,
            img_size=self.test_size, preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        ))

    def get_data_loader(self, batch_size, is_distributed=False, no_aug=False, cache_img=None,
                        finite=False, epoch=0):
        from yolox.data import DataLoader, InfiniteSampler, MosaicDetection, TrainTransform, YoloBatchSampler
        if is_distributed:
            raise ValueError("This workflow supports single GPU only")
        if cache_img is not None:
            raise ValueError("Offline workflow does not enable image caching")
        dataset = self.get_dataset()
        dataset = MosaicDetection(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate, mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale, shear=self.shear, enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob,
        )
        generator = torch.Generator().manual_seed(self.seed + epoch)
        if finite:
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        else:
            sampler = InfiniteSampler(len(dataset), seed=self.seed)
        batches = YoloBatchSampler(sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        self.dataset = dataset
        return DataLoader(dataset, batch_sampler=batches, num_workers=self.data_num_workers,
                          pin_memory=not finite, worker_init_fn=seed_worker, generator=generator)

    def get_eval_loader(self, batch_size, is_distributed=False, **kwargs):
        if is_distributed:
            raise ValueError("This workflow supports single GPU only")
        return torch.utils.data.DataLoader(
            self.get_eval_dataset(**kwargs), batch_size=batch_size, shuffle=False,
            num_workers=self.data_num_workers, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed), pin_memory=False,
        )


def create_exp(config):
    values = read_config(config) if isinstance(config, (str, os.PathLike)) else config
    mode = values.get("mode", "original")
    if mode == "online":
        from .online_exp import OnlineExp
        return OnlineExp(values)
    if mode != "original":
        raise ValueError("mode must be 'original' or 'online'")
    return OfflineExp(values)
