import subprocess
import sys
import pytest

def _run_cli(*args):
    """Run eoles-dispatch CLI and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "eoles_dispatch"] + list(args),
        capture_output=True, text=True, timeout=10,
    )

def test_cli_help_exits_zero():
    r = _run_cli("--help")
    assert r.returncode == 0
    assert "eoles-dispatch" in r.stdout.lower() or "usage" in r.stdout.lower()

def test_cli_version():
    r = _run_cli("--version")
    assert r.returncode == 0
    assert "0." in r.stdout  # version starts with 0.x

def test_cli_list_empty(tmp_path):
    r = _run_cli("list", "--project-dir", str(tmp_path))
    assert r.returncode == 0
    assert "No runs found" in r.stdout

def test_cli_collect_start_equals_end():
    """--start == --end should error with helpful message."""
    r = _run_cli("collect", "--start", "2021", "--end", "2021")
    assert r.returncode != 0
    assert "must be greater" in r.stderr

def test_cli_create_missing_scenario(tmp_path):
    r = _run_cli("create", "foo", "--scenario", "nonexistent", "--year", "2020",
                  "--no-download", "--project-dir", str(tmp_path))
    assert r.returncode != 0
