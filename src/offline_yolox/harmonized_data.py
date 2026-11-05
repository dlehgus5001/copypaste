"""Frozen local Harmonizer applied to synthetic foreground before YOLOX transforms."""
import copy
import os
import random
from time import perf_counter

import cv2
import numpy as np
import torch

from .copy_paste import OnlineCopyPasteDataset
from .patch_data import OnlinePatchDataset
from .paste_check import _finite, harmonize_image, load_harmonizer
from .startup import report, startup_stage


class HarmonizerTransform:
    """One lazy model per process, shared by an experiment's epoch datasets.

    CUDA runs only in the main process. Explicit CPU execution may run inside
    workers, where each worker owns a fresh model. No model enters YOLOX's
    optimizer, EMA or checkpoint; only the generated image is changed.
    """
    def __init__(self, settings):
        if not isinstance(settings, dict):
            raise ValueError("harmonizer must be a configuration object")
        unknown = set(settings) - {"repository", "checkpoint", "device", "strength"}
        if unknown:
            raise ValueError(f"Unknown harmonizer settings: {sorted(unknown)}")
        self.settings = copy.deepcopy(settings)
        self.repository = settings.get("repository", "models/Harmonizer")
        self.checkpoint = settings.get("checkpoint")
        for key, value in (("repository", self.repository), ("checkpoint", self.checkpoint)):
            if (key == "repository" or value is not None) and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"harmonizer.{key} must be a nonempty local path")
        self.device = settings.get("device", "auto")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("harmonizer.device must be auto, cpu or cuda")
        self.strength = _finite(settings.get("strength", 1.0), "harmonizer.strength", minimum=0, maximum=1)
        self._model, self._model_info, self._pid = None, None, None

    def validate_workers(self, workers):
        if workers > 0 and self.device != "cpu" and self.strength > 0:
            raise ValueError("GPU/auto Harmonizer requires num_workers=0. Set num_workers to 0, "
                             "or explicitly set harmonizer.device to cpu for CPU workers.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_model=None, _model_info=None, _pid=None)
        return state

    def _load(self):
        pid = os.getpid()
        if self._pid != pid:
            self._model, self._model_info, self._pid = None, None, pid
        if self._model is not None:
            return
        worker = torch.utils.data.get_worker_info()
        if worker is not None and self.device != "cpu":
            raise RuntimeError("CUDA/auto Harmonizer cannot run inside a DataLoader worker; use num_workers=0")
        # Upstream constructors initialize random tensors before loading weights.
        # Do not let this change detection initialization/augmentation/resume RNG.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        started = perf_counter()
        try:
            with torch.random.fork_rng(devices=[]), startup_stage(f"Loading frozen training Harmonizer on {self.device}"):
                model, info = load_harmonizer(self.repository, self.checkpoint, self.device)
                model.float().eval().requires_grad_(False)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self._model = model
        self._model_info = {**info, "load_seconds": perf_counter() - started}
        report("Harmonizer ready: foreground-only correction before TrainTransform; weights frozen")

    def __call__(self, sample, blend_sigma=0.6):
        image = sample["image"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Harmonizer training input must be uint8 BGR")
        union = np.zeros(image.shape[:2], dtype=bool)
        for mask in sample["masks"]:
            if mask.shape != union.shape or mask.dtype != np.bool_:
                raise ValueError("Harmonizer masks must be boolean and match the synthetic image")
            union |= mask
        result = {**sample, "metadata": dict(sample["metadata"])}
        if not union.any() or self.strength == 0:
            result["metadata"]["harmonizer"] = dict(applied=False,
                reason="empty_mask" if not union.any() else "strength_zero", strength=self.strength)
            return result
        self._load()
        alpha = union.astype(np.float32)
        if blend_sigma:
            alpha *= cv2.GaussianBlur(alpha, (0, 0), blend_sigma, borderType=cv2.BORDER_CONSTANT)
        adjusted, correction = harmonize_image(self._model, image, union, alpha,
                                               device=self._model_info["device"], strength=self.strength)
        result["image"] = adjusted
        result["metadata"]["harmonizer"] = dict(applied=True, model=self._model_info,
            correction=correction, foreground_pixels=int(union.sum()),
            mask_semantics="union_of_final_visible_foreground", background_pixels_preserved=True)
        return result


class _HarmonizedDataset:
    def __init__(self, *args, harmonizer, **kwargs):
        self.harmonizer = harmonizer
        super().__init__(*args, **kwargs)

    def sample_raw(self):
        return self.harmonizer(super().sample_raw(), blend_sigma=self.blend_sigma)


class HarmonizedOnlineDataset(_HarmonizedDataset, OnlineCopyPasteDataset):
    pass


class HarmonizedOnlinePatchDataset(_HarmonizedDataset, OnlinePatchDataset):
    pass
