import subprocess
from shutil import which


def run_help(command_name: str) -> int:
    assert which(command_name) is not None
    command = [command_name, "--help"]
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode


def test_train_help_exits_zero() -> None:
    assert run_help("fewshoter-train") == 0


def test_inference_help_exits_zero() -> None:
    assert run_help("fewshoter-inference") == 0


def test_evaluate_help_exits_zero() -> None:
    assert run_help("fewshoter-evaluate") == 0


def test_api_server_help_exits_zero() -> None:
    assert run_help("fewshoter-api-server") == 0
