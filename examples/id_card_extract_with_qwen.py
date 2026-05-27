"""Run OnnxOCR first, then use local Qwen3.5-2B ONNX for ID-card field extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onnxocr.qwen35_2b import (
    ExtractionTemplate,
    QwenScenarioExtractor,
    TARGET_DIR,
    _safe_print,
)
from onnxocr.onnx_paddleocr import ONNXPaddleOcr


DEFAULT_IMAGE = Path("onnxocr/test_images/8f113149-ff64-4c9f-8dc0-e34100365aa4.jpg")
DEFAULT_OUTPUT = Path("result_img/id_card_front_qwen_extract.json")
FIELDS = ["姓名", "性别", "民族", "出生", "住址", "公民身份号码"]

ID_CARD_FRONT_TEMPLATE = ExtractionTemplate(
    name="id_card_front",
    description="中国居民身份证正面 OCR",
    fields=FIELDS,
    rules=[
        "只根据 OCR 文本抽取，不要编造。",
        "缺失或不确定的字段填空字符串。",
        '"出生" 统一为 YYYY-MM-DD，无法标准化则保留 OCR 原文。',
        '"公民身份号码" 只保留 18 位身份证号码，末位 X 保留大写。',
    ],
)


class QwenTextExtractor:
    """ID-card extraction wrapper built on the package-level scenario extractor."""

    def __init__(self, model_dir: Path, max_new_tokens: int = 256) -> None:
        self.extractor = QwenScenarioExtractor(model_dir=model_dir, max_new_tokens=max_new_tokens)

    def extract_json(self, ocr_text: str) -> tuple[dict[str, Any], str]:
        return self.extractor.extract(ocr_text, template=ID_CARD_FRONT_TEMPLATE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract Chinese ID-card front fields with OnnxOCR + Qwen3.5-2B ONNX.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Image path to process.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON file.")
    parser.add_argument("--qwen-dir", default=str(TARGET_DIR), help="Qwen3.5-2B ONNX directory.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum Qwen tokens for extraction.")
    parser.add_argument("--show-sensitive", action="store_true", help="Print extracted ID-card fields to stdout.")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    output_path = Path(args.output)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    ocr_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
    qwen = QwenTextExtractor(Path(args.qwen_dir), max_new_tokens=args.max_new_tokens)

    record = extract_from_image(image_path, ocr_model, qwen)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Image: {image_path.resolve()}")
    print(f"Result: {output_path.resolve()}")
    if args.show_sensitive:
        _safe_print(json.dumps(record["fields"], ensure_ascii=False, indent=2))
    else:
        print("Sensitive fields were written to the local JSON file. Use --show-sensitive to print them.")
    return 0


def extract_from_image(image_path: Path, ocr_model: ONNXPaddleOcr, qwen: QwenTextExtractor) -> dict[str, Any]:
    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    ocr_result = ocr_model.ocr(image)
    ocr_items = normalize_ocr_result(ocr_result)
    ocr_text = "\n".join(item["text"] for item in ocr_items if item["text"])
    fields, qwen_raw = qwen.extract_json(ocr_text)

    return {
        "file_name": image_path.name,
        "file_path": str(image_path),
        "fields": fields,
        "ocr_text": ocr_text,
        "ocr_items": ocr_items,
        "qwen_raw": qwen_raw,
    }


def normalize_ocr_result(ocr_result: Any) -> list[dict[str, Any]]:
    rows = ocr_result[0] if ocr_result and isinstance(ocr_result, list) else []
    items = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        box, rec = row[0], row[1]
        text = rec[0] if rec and len(rec) > 0 else ""
        score = float(rec[1]) if rec and len(rec) > 1 else 0.0
        items.append({"text": str(text), "score": score, "box": box})
    return sorted(items, key=ocr_sort_key)


def ocr_sort_key(item: dict[str, Any]) -> tuple[float, float]:
    box = item.get("box") or [[0, 0]]
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return (sum(ys) / max(len(ys), 1), sum(xs) / max(len(xs), 1))


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    raise SystemExit(main())
