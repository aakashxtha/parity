"""CLI smoke tests."""
from __future__ import annotations

import subprocess
import sys
import time


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "parity.cli", *args],
        capture_output=True,
        text=True,
    )


def run_entry(*args: str) -> subprocess.CompletedProcess:
    """Run via the installed entry-point script."""
    return subprocess.run(
        ["parity", *args],
        capture_output=True,
        text=True,
    )


def test_help_exits_zero() -> None:
    result = run_entry("--help")
    assert result.returncode == 0, result.stderr


def test_help_is_fast() -> None:
    start = time.monotonic()
    result = run_entry("--help")
    elapsed = time.monotonic() - start
    assert result.returncode == 0, result.stderr
    assert elapsed < 2.0, f"--help took {elapsed:.2f}s (limit 2s)"


def test_check_missing_file_exits_2() -> None:
    result = run_entry("check", "nonexistent.pdb")
    assert result.returncode == 2
    assert "error" in result.stderr.lower()


def test_diff_missing_files_exits_2() -> None:
    result = run_entry("diff", "a.pdb", "b.pdb")
    assert result.returncode == 2
    assert "error" in result.stderr.lower()


def test_check_real_file_exits_0() -> None:
    import tempfile, os
    pdb = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       2.000   1.000   3.000  1.00  0.00           C\n"
        "END\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb)
        path = f.name
    try:
        result = run_entry("check", path)
        assert result.returncode == 0, result.stderr
    finally:
        os.unlink(path)


def test_diff_self_exits_0() -> None:
    import tempfile, os
    pdb = (
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       3.000   3.000   3.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       2.000   1.000   3.000  1.00  0.00           C\n"
        "END\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb)
        path = f.name
    try:
        result = run_entry("diff", path, path)
        assert result.returncode == 0, result.stderr
        assert "equivalent" in result.stdout
    finally:
        os.unlink(path)


def test_version() -> None:
    result = run_entry("--version")
    assert result.returncode == 0
    assert "0.1.0" in result.stdout
