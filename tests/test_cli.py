"""Exercise the public Python CLI without a sourced shell environment."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run.py"


@pytest.fixture
def invoke(tmp_path):
    working = tmp_path / "unrelated working directory"
    working.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # A Python invocation must use its own interpreter even if an old shell
    # session still points the shell wrappers at another environment.
    environment["YOLOX_PYTHON"] = str(tmp_path / "missing-python")

    def run(*arguments, expected=0, without_site=False):
        command = [sys.executable]
        if without_site:
            command.append("-S")
        completed = subprocess.run(
            [*command, str(RUNNER), *map(str, arguments)],
            cwd=working,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == expected, completed.stdout + completed.stderr
        return completed

    return run


def test_top_level_help_needs_no_installed_dependencies(invoke):
    completed = invoke("--help", without_site=True)
    for command in ("check-env", "build", "sample-data", "train", "preview", "evaluate", "validate-gpu"):
        assert command in completed.stdout
    assert "Verified: IPv4/IPv6 sockets denied" not in completed.stderr


def test_train_help_and_mode_errors(invoke, tmp_path):
    help_result = invoke("train", "--help")
    assert "--mode" in help_result.stdout
    assert "--checkpoint" in help_result.stdout
    assert "Verified: IPv4/IPv6 sockets denied" not in help_result.stderr

    invalid = invoke("train", "--mode", "invalid", expected=2)
    assert "invalid choice" in invalid.stderr

    mismatched = tmp_path / "original.json"
    mismatched.write_text(json.dumps({"mode": "original", "data_dir": "/missing-data"}))
    result = invoke(
        "train", "--mode", "online", "--config", mismatched, "--device", "cpu", expected=2
    )
    assert "mode" in result.stderr.lower()
    assert "config" in result.stderr.lower()
    assert "FileNotFoundError" not in result.stderr


def test_python_cli_runs_both_workflows_from_another_directory(invoke, tmp_path):
    data = tmp_path / "sample data"
    sample = invoke(
        "sample-data", "--output", data, "--train-images", 2, "--val-images", 1
    )
    assert "Verified: IPv4/IPv6 sockets denied" in sample.stderr
    manifest = json.loads((data / "sample_manifest.json").read_text())
    assert manifest["splits"]["train2017"]["images"] == 2
    assert manifest["splits"]["val2017"]["images"] == 1

    for mode in ("original", "online"):
        config = json.loads((ROOT / "configs" / f"{mode}.json").read_text())
        config.update(
            data_dir=str(data),
            output_dir=str(tmp_path / mode),
            exp_name="cli_smoke",
            max_epoch=1,
            scheduler="cos",
            num_workers=0,
            mosaic_prob=0.0,
            enable_mixup=False,
            hsv_prob=0.0,
            flip_prob=0.0,
        )
        if mode == "online":
            config["copy_paste"].update(asset_root=str(data), epoch_length=2)
        config_path = tmp_path / f"{mode}.json"
        config_path.write_text(json.dumps(config))
        # Keep the existing config-only invocation working and exercise the
        # explicit mode selector on the second workflow.
        mode_arguments = ("--mode", "online") if mode == "online" else ()
        invoke("train", *mode_arguments, "--config", config_path, "--device", "cpu")
        run_dir = tmp_path / mode / "cli_smoke"
        report = json.loads((run_dir / "diagnostic_report.json").read_text())
        assert report["workflow"] == "cpu_diagnostic"
        assert report["native_cuda_trainer"] is False
        assert report["completed_epochs"] == 1
        epoch = report["epochs"][0]
        assert epoch["images"] == 2
        assert epoch["iterations"] == 1
        assert math.isfinite(epoch["mean_loss"]) and epoch["mean_loss"] > 0
        checkpoint = run_dir / "latest_ckpt.pth"
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0

        evaluation_path = tmp_path / mode / "evaluation.json"
        invoke(
            "evaluate", "--config", config_path, "--device", "cpu",
            "--checkpoint", checkpoint, "--output", evaluation_path,
        )
        evaluation = json.loads(evaluation_path.read_text())
        assert evaluation["device"] == "cpu"
        assert evaluation["images"] == 1
        assert evaluation["checkpoint"] == str(checkpoint)
        assert math.isfinite(evaluation["ap50_95"])

        if mode == "online":
            preview = tmp_path / "preview"
            invoke("preview", "--config", config_path, "--output", preview, "--count", 1)
            preview_report = json.loads((preview / "manifest.json").read_text())
            assert preview_report["count"] == 1
            assert preview_report["training_writes_synthetic_images"] is False
            item = json.loads((preview / preview_report["samples"][0]).read_text())
            for filename in (item["image"], item["preview"], *item["masks"]):
                assert (preview / filename).stat().st_size > 0


def test_check_env_uses_current_interpreter(invoke, tmp_path):
    output = tmp_path / "environment report"
    invoke("check-env", "--output", output)
    report = json.loads((output / "environment.json").read_text())
    assert report["python_executable"] == sys.executable
    assert report["status"] == "checked profile satisfied"
    assert report["runtime_checks"]["torch_numpy"] == "passed"
