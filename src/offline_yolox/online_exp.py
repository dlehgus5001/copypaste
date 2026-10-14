"""Online data only; the official model, losses, SGD and COCO evaluation are shared."""
import copy

import torch

from .copy_paste import OnlineCopyPasteDataset
from .experiment import OfflineExp, seed_worker


class OnlineExp(OfflineExp):
    def __init__(self, config):
        # Only the held-out COCO validation taxonomy is required for online data.
        # Reusing base validation avoids requiring an unrelated COCO training split.
        base = dict(config)
        base["train_ann"] = config.get("val_ann", "instances_val2017.json")
        base["train_name"] = config.get("val_name", "val2017")
        base.setdefault("output_dir", "outputs/online")
        base.setdefault("exp_name", f"online_{config.get('preset', 's')}")
        super().__init__(base)
        self.config = dict(config)
        if not isinstance(config.get("copy_paste"), dict):
            raise ValueError("Online mode requires a copy_paste configuration object")
        self.copy_paste = dict(config["copy_paste"])
        # Fail early on incorrect assets/taxonomy rather than inside worker processes.
        self._dataset_template = self.get_dataset()

    def get_dataset(self, cache=False, cache_type="ram", epoch=0):
        from yolox.data import TrainTransform
        if cache:
            raise ValueError("Online synthetic samples are generated in memory without image caching")
        if hasattr(self, "_dataset_template"):
            # Reuse validated path metadata, not image pixels or generated samples.
            # Large object banks should not be decoded in full again every epoch.
            dataset = copy.copy(self._dataset_template)
            dataset.epoch = int(epoch)
            dataset._rng, dataset._stream_key, dataset._calls = None, None, 0
            return dataset
        return OnlineCopyPasteDataset(
            config=self.copy_paste, input_size=self.input_size,
            class_ids=self.class_ids, class_names=self.class_names,
            seed=self.seed, epoch=epoch,
            preproc=TrainTransform(max_labels=self.copy_paste.get("max_labels", 120),
                                   flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
        )

    def get_data_loader(self, batch_size, is_distributed=False, no_aug=False, cache_img=None,
                        finite=False, epoch=0):
        from yolox.data import DataLoader, InfiniteSampler, YoloBatchSampler
        if is_distributed:
            raise ValueError("This workflow supports single GPU only")
        if cache_img is not None:
            raise ValueError("Online workflow does not cache generated images")
        dataset = self.get_dataset(epoch=epoch)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        sampler = (torch.utils.data.RandomSampler(dataset, generator=generator) if finite
                   else InfiniteSampler(len(dataset), seed=self.seed + epoch))
        batches = YoloBatchSampler(sampler, batch_size=batch_size, drop_last=False, mosaic=False)
        self.dataset = dataset
        return DataLoader(dataset, batch_sampler=batches, num_workers=self.data_num_workers,
                          pin_memory=not finite, worker_init_fn=seed_worker, generator=generator)
