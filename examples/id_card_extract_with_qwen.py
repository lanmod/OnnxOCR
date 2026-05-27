"""Run OnnxOCR first, then use local Qwen3.5-2B ONNX for ID-card field extraction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qwen35_2b_onnx import (
    TARGET_DIR,
    _build_qwen_input_ids,
    _generate_qwen_tokens,
    _safe_print,
    _version_tuple,
    verify_files,
)
from onnxocr.onnx_paddleocr import ONNXPaddleOcr


DEFAULT_IMAGE = Path("onnxocr/test_images/8f113149-ff64-4c9f-8dc0-e34100365aa4.jpg")
DEFAULT_OUTPUT = Path("result_img/id_card_front_qwen_extract.json")
FIELDS = ["\u59d3\u540d", "\u6027\u522b", "\u6c11\u65cf", "\u51fa\u751f", "\u4f4f\u5740", "\u516c\u6c11\u8eab\u4efd\u53f7\u7801"]


class QwenTextExtractor:
    """Small pure-Python wrapper around the local Qwen3.5-2B ONNX text decoder."""

    def __init__(self, model_dir: Path, max_new_tokens: int = 256) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if verify_files(model_dir, "q4") != 0:
            raise RuntimeError(f"Qwen model files are not ready: {model_dir}")
        if _version_tuple(ort.__version__) < (1, 26, 0):
            raise RuntimeError(f"onnxruntime {ort.__version__} is too old. Please install onnxruntime>=1.26.0.")

        self.max_new_tokens = max_new_tokens
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        onnx_dir = model_dir / "onnx"
        providers = ["CPUExecutionProvider"]
        self.embed_session = ort.InferenceSession(str(onnx_dir / "embed_tokens_q4.onnx"), providers=providers)
        self.decoder_session = ort.InferenceSession(str(onnx_dir / "decoder_model_merged_q4.onnx"), providers=providers)

    def extract_json(self, ocr_text: str) -> tuple[dict[str, Any], str]:
        prompt = build_prompt(ocr_text)
        input_ids = _build_qwen_input_ids(self.tokenizer, prompt)
        inputs_embeds = self.embed_session.run(None, {"input_ids": np.asarray([input_ids], dtype=np.int64)})[0]
        generated_ids, _ = _generate_qwen_tokens(
            decoder_session=self.decoder_session,
            embed_session=self.embed_session,
            inputs_embeds=inputs_embeds,
            input_length=len(input_ids),
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=32,
            eos_token_id=self.tokenizer.token_to_id("<|im_end|>"),
            tokenizer=self.tokenizer,
            stop_at_sentence=False,
            np=np,
        )
        raw = self.tokenizer.decode(generated_ids).strip()
        return parse_json_object(raw), raw


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
    qwen_fields, qwen_raw = qwen.extract_json(ocr_text)
    fields = merge_fields(qwen_fields, fallback_extract(ocr_text))

    return {
        "file_name": image_path.name,
        "file_path": str(image_path),
        "fields": fields,
        "ocr_text": ocr_text,
        "ocr_items": ocr_items,
        "qwen_raw": qwen_raw,
    }


def build_prompt(ocr_text: str) -> str:
    return f"""你是身份证 OCR 信息抽取器。下面是 OnnxOCR 对中国居民身份证正面图片的全量识别文本。

请只输出一个 JSON 对象，不要解释，不要 Markdown，不要代码块。字段固定如下：
{{
  "姓名": "",
  "性别": "",
  "民族": "",
  "出生": "",
  "住址": "",
  "公民身份号码": ""
}}

规则：
1. 只根据 OCR 文本抽取，不要编造。
2. 缺失或不确定的字段填空字符串。
3. "出生" 统一为 YYYY-MM-DD，无法标准化则保留 OCR 原文。
4. "公民身份号码" 只保留 18 位身份证号码，末位 X 保留大写。

OCR 文本：
{ocr_text}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            value = json.loads(repair_json(cleaned))
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def fallback_extract(ocr_text: str) -> dict[str, str]:
    compact = re.sub(r"\s+", "", ocr_text)
    fields = {key: "" for key in FIELDS}

    id_match = re.search(r"\d{17}[\dXx]", compact)
    if id_match:
        fields["公民身份号码"] = id_match.group(0).upper()
        id_no = fields["公民身份号码"]
        fields["出生"] = f"{id_no[6:10]}-{id_no[10:12]}-{id_no[12:14]}"

    patterns = {
        "姓名": r"姓名([\u4e00-\u9fa5·]{2,8})",
        "性别": r"性别([男女])",
        "民族": r"民族([\u4e00-\u9fa5]{1,6})",
        "住址": r"住址(.+?)(?:公民身份号码|身份号码|公民身份|$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, compact)
        if match:
            fields[key] = match.group(1)

    birth_match = re.search(r"出生(\d{4})年?(\d{1,2})月?(\d{1,2})日?", compact)
    if birth_match:
        year, month, day = birth_match.groups()
        fields["出生"] = f"{year}-{int(month):02d}-{int(day):02d}"
    return fields


def merge_fields(qwen_fields: dict[str, Any], fallback_fields: dict[str, str]) -> dict[str, str]:
    merged = {}
    for key in FIELDS:
        value = str(qwen_fields.get(key, "") or "").strip()
        fallback_value = fallback_fields.get(key, "")
        if key == "住址" and fallback_value and len(fallback_value) > len(value) + 4:
            value = fallback_value
        merged[key] = value or fallback_value

    id_no = re.sub(r"[^0-9Xx]", "", merged["公民身份号码"]).upper()
    if re.fullmatch(r"\d{17}[\dX]", id_no):
        merged["公民身份号码"] = id_no
    return merged


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
