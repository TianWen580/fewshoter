import os
import subprocess
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def run_help(module_name: str) -> int:
    command = [sys.executable, "-m", module_name, "--help"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_PARENT)
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    return completed.returncode


def test_train_help_exits_zero() -> None:
    assert run_help("fewshoter.cli.train") == 0


def test_inference_help_exits_zero() -> None:
    assert run_help("fewshoter.cli.inference") == 0


def test_evaluate_help_exits_zero() -> None:
    assert run_help("fewshoter.cli.evaluate") == 0


def test_api_server_help_exits_zero() -> None:
    assert run_help("fewshoter.cli.api_server") == 0
