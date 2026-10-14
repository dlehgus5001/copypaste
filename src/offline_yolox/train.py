"""Official CUDA Trainer and an explicitly separate CPU diagnostic harness."""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .bootstrap import local_path
from .experiment import create_exp, seed_everything


def rng_state():
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": {"algorithm": np_state[0], "keys": torch.tensor(np_state[1].astype(np.int64)),
                  "position": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    ns = state["numpy"]
    np.random.set_state((ns["algorithm"], ns["keys"].cpu().numpy().astype(np.uint32),
                         ns["position"], ns["has_gauss"], ns["cached_gaussian"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def load_checkpoint(path, device="cpu"):
    path = local_path(path, must_exist=True)
    if not path.is_file():
        raise ValueError(f"Checkpoint is not a local file: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a YOLOX checkpoint containing a model state dictionary")
    return checkpoint


def initialize_from_checkpoint(model, path, *, device="cpu"):
    """Use official shape-matching fine tuning for a local checkpoint.

    A COCO pretrained classification head can differ from the configured classes;
    compatible backbone/head tensors are restored while unmatched tensors retain
    their original initialization. Epoch-boundary resume remains strict instead.
    """
    from yolox.utils import load_ckpt
    checkpoint = load_checkpoint(path, device)
    weights = checkpoint["model"]
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint model must contain a tensor state dictionary")
    if any(not isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Checkpoint model state dictionary must contain only tensors")
    target_state = model.state_dict()
    compatible = {
        name: value for name, value in weights.items()
        if name in target_state and value.shape == target_state[name].shape
    }
    if not compatible:
        raise ValueError("Local checkpoint has no compatible model tensor names and shapes")
    # Delegate loading to the original implementation; it logs skipped layers.
    return load_ckpt(model, weights)


def _validate_resume(checkpoint, exp, workflow):
    if checkpoint.get("workflow") != workflow:
        raise ValueError(f"Resume requires a {workflow} checkpoint")
    previous = checkpoint.get("config", {})
    # Output location can change; data, architecture, schedule and augmentation cannot.
    current = dict(exp.config)
    for name in ("output_dir", "exp_name"):
        previous = {k: v for k, v in previous.items() if k != name}
        current.pop(name, None)
    if previous != current:
        raise ValueError("Resume configuration differs from checkpoint; use --checkpoint for fine tuning")


def run_cpu_diagnostic(exp, *, resume=None, checkpoint=None, stop_after_epoch=None, evaluate=True):
    """Finite full epochs with actual official detection losses and optimizer updates.

    This exercises the official model/data/SGD/scheduler/EMA on CPU; upstream Trainer
    uses CUDA streams and is deliberately not presented as a CPU implementation.
    """
    from yolox.utils import ModelEMA
    if resume and checkpoint:
        raise ValueError("Choose resume or checkpoint")
    seed_everything(exp.seed)
    model = exp.get_model().to("cpu").train()
    optimizer = exp.get_optimizer(exp.batch_size)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ema = ModelEMA(model, decay=0.9998) if exp.ema else None
    start_epoch = 0
    best_ap = 0.0
    if checkpoint:
        model = initialize_from_checkpoint(model, checkpoint)
        if ema:
            ema.ema.load_state_dict(model.state_dict())
    if resume:
        saved = load_checkpoint(resume)
        _validate_resume(saved, exp, "cpu_diagnostic")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["start_epoch"])
        best_ap = float(saved.get("best_ap", 0.0))
        if ema:
            ema.ema.load_state_dict(saved["ema_model"], strict=True)
            ema.updates = int(saved["ema_updates"])
        restore_rng(saved["rng"])
    if start_epoch >= exp.max_epoch:
        raise ValueError("Checkpoint already completed all configured epochs")
    output = Path(exp.output_dir) / exp.exp_name
    output.mkdir(parents=True, exist_ok=True)
    final_epoch = exp.max_epoch if stop_after_epoch is None else int(stop_after_epoch)
    if not start_epoch < final_epoch <= exp.max_epoch:
        raise ValueError("stop_after_epoch must exceed completed epochs and not exceed max_epoch")
    reports = []
    scheduler = None
    for epoch in range(start_epoch, final_epoch):
        # Match native Trainer's no-mosaic / L1 transition (including its boundary).
        no_aug = epoch + 1 >= exp.max_epoch - exp.no_aug_epochs
        loader = exp.get_data_loader(exp.batch_size, no_aug=no_aug, finite=True, epoch=epoch)
        iterations = len(loader)
        if iterations == 0:
            raise ValueError("Training loader contains no batches")
        if scheduler is None:
            scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * exp.batch_size, iterations)
        model.train()
        model.head.use_l1 = no_aug
        losses = []
        seen_images = 0
        for batch_index, (images, targets, _, _) in enumerate(loader):
            images, targets = exp.preprocess(images.float(), targets.float(), exp.input_size)
            outputs = model(images, targets)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite detection loss at epoch={epoch + 1}, batch={batch_index}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("Nonfinite model gradient")
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            lr = scheduler.update_lr(epoch * iterations + batch_index + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses.append(float(loss.detach()))
            seen_images += len(images)
        if len(losses) != iterations or seen_images != len(loader.dataset):
            raise RuntimeError("Diagnostic epoch did not consume exactly the finite dataset")
        metrics = None
        if evaluate and ((epoch + 1) % exp.eval_interval == 0 or epoch + 1 == final_epoch):
            from .evaluate import evaluate_model
            metrics = evaluate_model(exp, ema.ema if ema else model, device="cpu", evaluator="standard")
            best_ap = max(best_ap, metrics["ap50_95"])
        state = {
            "format_version": 1, "workflow": "cpu_diagnostic", "start_epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(), "best_ap": best_ap,
            "config": exp.config, "input_size": list(exp.input_size),
            "ema_model": ema.ema.state_dict() if ema else None,
            "ema_updates": ema.updates if ema else 0,
        }
        torch.save(state, output / "latest_ckpt.pth")
        torch.save(state, output / f"epoch_{epoch + 1}_ckpt.pth")
        report = {"epoch": epoch + 1, "iterations": iterations, "images": seen_images,
                  "mean_loss": sum(losses) / len(losses), "losses": losses, "evaluation": metrics}
        reports.append(report)
        print(json.dumps({"workflow": "cpu_diagnostic", **report}, ensure_ascii=False), flush=True)
    report = {"workflow": "cpu_diagnostic", "native_cuda_trainer": False,
              "device": "cpu", "start_epoch": start_epoch, "completed_epochs": final_epoch,
              "epochs": reports, "checkpoint": str(output / "latest_ckpt.pth")}
    (output / "diagnostic_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_cuda_trainer(exp, args):
    from yolox.core import Trainer
    from yolox.utils import save_checkpoint

    class LocalCheckpointTrainer(Trainer):
        """Original Trainer lifecycle with local, safe and optimizer-consistent snapshots."""
        def resume_train(self, model):
            self._restored = None
            self.start_epoch = 0
            if self.args.ckpt:
                if self.args.resume:
                    saved = load_checkpoint(self.args.ckpt, self.device)
                    model.load_state_dict(saved["model"], strict=True)
                    _validate_resume(saved, self.exp, "native_cuda_trainer")
                    if bool(saved.get("amp_training", False)) != self.amp_training:
                        raise ValueError("Resume must preserve the checkpoint's FP16/FP32 setting")
                    self.optimizer.load_state_dict(saved["optimizer"])
                    self.start_epoch = int(saved["start_epoch"])
                    if self.start_epoch >= self.max_epoch:
                        raise ValueError("Checkpoint already completed configured epochs")
                    self.best_ap = float(saved.get("best_ap", 0.0))
                    self.scaler.load_state_dict(saved["scaler"])
                    restore_rng(saved["rng"])
                    self._restored = saved
                else:
                    model = initialize_from_checkpoint(model, self.args.ckpt, device=self.device)
            return model

        def before_train(self):
            super().before_train()
            if self._restored:
                self.input_size = tuple(self._restored["input_size"])
                if self.use_model_ema:
                    self.ema_model.ema.load_state_dict(self._restored["ema_model"])
                    self.ema_model.updates = int(self._restored["ema_updates"])

        def train_in_epoch(self):
            last_epoch = self.max_epoch if self.args.stop_after_epoch is None else self.args.stop_after_epoch
            if not self.start_epoch < last_epoch <= self.max_epoch:
                raise ValueError("stop_after_epoch must exceed completed epochs and not exceed max_epoch")
            for self.epoch in range(self.start_epoch, last_epoch):
                self.before_epoch()
                self.train_in_iter()
                self.after_epoch()

        def after_epoch(self):
            super().after_epoch()
            # Native Trainer saves latest before evaluation; persist the updated best AP
            # and final epoch-boundary RNG too, so latest is a complete resume snapshot.
            self.save_ckpt("latest")

        def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
            if self.rank != 0:
                return
            state = {
                "format_version": 1, "workflow": "native_cuda_trainer",
                "amp_training": bool(self.amp_training),
                "start_epoch": self.epoch if ckpt_name == "last_mosaic_epoch" else self.epoch + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "best_ap": float(self.best_ap),
                "curr_ap": float(ap) if ap is not None else None,
                "scaler": self.scaler.state_dict(), "rng": rng_state(),
                "config": self.exp.config, "input_size": list(self.input_size),
                "ema_model": self.ema_model.ema.state_dict() if self.use_model_ema else None,
                "ema_updates": self.ema_model.updates if self.use_model_ema else 0,
            }
            save_checkpoint(state, update_best_ckpt, self.file_name, ckpt_name)

        def after_train(self):
            super().after_train()
            if hasattr(self, "tblogger"):
                self.tblogger.close()

    return LocalCheckpointTrainer(exp, args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/original.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint", help="Local YOLOX checkpoint for fine tuning")
    parser.add_argument("--resume", help="Local checkpoint for an epoch-boundary resume")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop at an epoch boundary, preserving configured total schedule")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint are mutually exclusive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GPU training requires a CUDA GPU; --device cpu explicitly runs the diagnostic harness.")
    if args.device == "cpu" and args.fp16:
        parser.error("--fp16 requires CUDA")
    exp = create_exp(args.config)
    seed_everything(exp.seed)
    if args.device == "cpu":
        if args.cpu_threads <= 0:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        return run_cpu_diagnostic(exp, resume=args.resume, checkpoint=args.checkpoint,
                                  stop_after_epoch=args.stop_after_epoch)
    native_args = SimpleNamespace(
        fp16=args.fp16, batch_size=exp.batch_size, experiment_name=exp.exp_name,
        resume=bool(args.resume), ckpt=args.resume or args.checkpoint, start_epoch=None,
        stop_after_epoch=args.stop_after_epoch,
        cache=None, occupy=False, logger="tensorboard", exp_file=None,
    )
    build_cuda_trainer(exp, native_args).train()


if __name__ == "__main__":
    main()
