"""Thin CLI wrapper for :mod:`onnxocr.qwen35_2b`.

Run from the repository root, for example:

    python examples/qwen35_2b_onnx.py download --variant q4
    python examples/qwen35_2b_onnx.py verify
    python examples/qwen35_2b_onnx.py run-python --prompt "Hello"
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onnxocr.qwen35_2b import main


if __name__ == "__main__":
    raise SystemExit(main())
