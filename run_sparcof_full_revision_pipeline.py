#!/usr/bin/env python3
"""Backward-compatible entry point; prefer `sparcof` after installation."""

from pathlib import Path
import sys

try:
    from sparcof.cli import main
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from sparcof.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
