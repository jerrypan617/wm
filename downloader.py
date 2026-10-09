"""Backward-compatible entry point; use scripts/download_model.py instead."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parent / "scripts" / "download_model.py"),
        run_name="__main__",
    )
