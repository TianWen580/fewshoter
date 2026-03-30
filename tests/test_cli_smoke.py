import subprocess
from shutil import which


def run_help(command_name: str) -> subprocess.CompletedProcess[str]:
    assert which(command_name) is not None
    command = [command_name, "--help"]
    return subprocess.run(command, capture_output=True, text=True)


def test_train_help_exits_zero() -> None:
    completed = run_help("fewshoter-train")
    assert completed.returncode == 0
    assert "Process support set for CLIP few-shot classification" in completed.stdout


def test_inference_help_exits_zero() -> None:
    assert run_help("fewshoter-inference").returncode == 0


def test_evaluate_help_exits_zero() -> None:
    completed = run_help("fewshoter-evaluate")
    assert completed.returncode == 0
    assert "Evaluate CLIP few-shot classification performance" in completed.stdout


def test_api_server_help_exits_zero() -> None:
    assert run_help("fewshoter-api-server").returncode == 0
