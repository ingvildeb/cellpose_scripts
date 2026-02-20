"""Fail if any Python file contains `test_mode = True`.

Usage:
- Run `python tools/check_no_test_mode_true.py` before committing.
- Intended as a guardrail to avoid committing development test-mode toggles.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\btest_mode\s*=\s*True\b")

# Optional: skip common noisy dirs
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
            ".pytest_cache", "build", "dist", "tools"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def main() -> int:
    bad_files: list[str] = []
    for p in Path(".").rglob("*.py"):
        if should_skip(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if PATTERN.search(text):
            bad_files.append(str(p))

    if bad_files:
        print("ERROR: Found test_mode enabled (True) in the following file(s):")
        print("\n".join(bad_files))
        print("\nDisable test_mode before committing.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
