#!/usr/bin/env python3
"""Validate real single-GPU epochs, AMP, local checkpoint resume and evaluation.

Run under scripts/no_network.py to also prove no IPv4/IPv6 network access.
Missing hardware is a blocking result (exit 2), not a passing test run.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", default="configs/original.json")
    parser.add_argument("--online-config", default="configs/online.json")
    parser.add_argument("--output", default="artifacts/gpu_validation.json")
    args = parser.parse_args()
    report = {"status": "running", "checks": [], "unverified": []}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    try:
        import torch
        import torchvision
        report["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                             "torchvision": torchvision.__version__, "torch_cuda": torch.version.cuda,
                             "cudnn": torch.backends.cudnn.version(), "gpu_available": torch.cuda.is_available()}
        if not torch.cuda.is_available():
            report["status"] = "blocked"
            report["unverified"] = ["CUDA tensor and torchvision CUDA NMS", "YOLOX GPU forward/backward FP32/AMP",
                                    "Native Trainer epochs in both modes", "Native checkpoint resume and validation"]
            report["reason"] = "No accessible NVIDIA GPU in this execution environment"
            save()
            print(report["reason"], file=sys.stderr)
            return 2
        if sys.version_info[:2] != (3, 12) or torch.__version__ != "2.8.0+cu129" or torchvision.__version__ != "0.23.0+cu129":
            raise RuntimeError("This validation profile requires Python3.12, torch2.8.0+cu129, torchvision0.23.0+cu129")
        report["runtime"]["gpus"] = [{"name": torch.cuda.get_device_name(i),
                                      "capability": list(torch.cuda.get_device_capability(i))}
                                     for i in range(torch.cuda.device_count())]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party/YOLOX")])
        environment["OMP_NUM_THREADS"] = "2"
        environment["MAX_JOBS"] = "2"
        environment["TORCH_EXTENSIONS_DIR"] = str(ROOT / "artifacts/torch_extensions")

        def run(name, command):
            log_path = output.parent / f"gpu_{name}.log"
            with log_path.open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            report["checks"].append({"name": name, "exit_code": result.returncode,
                                     "status": "passed" if result.returncode == 0 else "failed",
                                     "log": str(log_path.relative_to(ROOT))})
            save()
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect {log_path}")

        run("cuda_ops", [sys.executable, "-m", "pytest", "tests/test_gpu.py", "-q"])
        for mode, input_config in [("original", args.original_config), ("online", args.online_config)]:
            config = json.loads((ROOT / input_config).read_text())
            config.update(max_epoch=2, warmup_epochs=0, no_aug_epochs=0, eval_interval=1,
                          print_interval=1, exp_name=mode, output_dir=f"artifacts/gpu_runs/{mode}/fp32")
            config_path = output.parent / f"gpu_{mode}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            checkpoint = ROOT / config["output_dir"] / mode / "latest_ckpt.pth"
            base = [sys.executable, "-m", "offline_yolox.train", "--config", str(config_path), "--device", "cuda"]
            run(f"{mode}_fp32_epoch1", base + ["--stop-after-epoch", "1"])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["start_epoch"] != 1 or saved["workflow"] != "native_cuda_trainer":
                raise RuntimeError("Native epoch1 checkpoint did not match the requested run")
            epoch1_updates = saved["ema_updates"]
            if epoch1_updates <= 0 or saved["amp_training"]:
                raise RuntimeError("FP32 epoch1 EMA/precision state is invalid")
            run(f"{mode}_fp32_resume_epoch2", base + ["--resume", str(checkpoint)])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["start_epoch"] != 2 or saved["ema_updates"] != 2 * epoch1_updates:
                raise RuntimeError("Native resume did not finish epoch2")
            resume_log = (output.parent / f"gpu_{mode}_fp32_resume_epoch2.log").read_text()
            if "start train epoch1" in resume_log or "start train epoch2" not in resume_log:
                raise RuntimeError("Resume did not execute only the remaining epoch")
            evaluation_path = output.parent / f"gpu_{mode}_evaluation.json"
            run(f"{mode}_evaluate", [sys.executable, "-m", "offline_yolox.evaluate", "--config", str(config_path),
                                    "--checkpoint", str(checkpoint), "--device", "cuda", "--evaluator", "fast",
                                    "--output", str(evaluation_path)])
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation["images"] <= 0 or len(evaluation["stats"]) != 12:
                raise RuntimeError("COCO evaluation did not execute")
            config["output_dir"] = f"artifacts/gpu_runs/{mode}/amp"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            amp_checkpoint = ROOT / config["output_dir"] / mode / "latest_ckpt.pth"
            run(f"{mode}_amp_epoch1", base + ["--fp16", "--stop-after-epoch", "1"])
            run(f"{mode}_amp_resume_epoch2", base + ["--fp16", "--resume", str(amp_checkpoint)])
            saved = torch.load(amp_checkpoint, map_location="cpu", weights_only=True)
            if saved["start_epoch"] != 2 or not saved["scaler"] or not saved["amp_training"]:
                raise RuntimeError("AMP scaler/resume state missing")
        report["status"] = "passed"
        report["scope"] = "Single GPU; selected local data only. Check configs to distinguish sample and real data."
        save()
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["reason"] = f"{type(error).__name__}: {error}"
        save()
        print(report["reason"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
