"""HTTP API service for OnnxOCR.

This module is intentionally separate from ``webui.py``:

- ``webui.py`` serves the browser UI.
- ``app-service.py`` exposes stable JSON APIs for deployment and integration.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from onnxocr.api_utils import ModelRegistry, decode_base64_image, format_ocr_results
from onnxocr.visualization import (
    draw_layout_analysis,
    draw_plate_recognition,
    draw_table_recognition,
    image_to_base64,
)


def create_app(model_registry: ModelRegistry | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("ONNXOCR_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    app.config["OUTPUT_DIR"] = os.getenv("ONNXOCR_OUTPUT_DIR", "result_img")
    models = model_registry or ModelRegistry(use_gpu=_env_bool("ONNXOCR_USE_GPU", default=False))

    @app.get("/")
    def index():
        return jsonify(
            {
                "name": "OnnxOCR API Service",
                "status": "ok",
                "endpoints": ["/health", "/ocr", "/plate", "/table", "/layout", "/layout_markdown"],
            }
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/ocr")
    def ocr():
        try:
            image = _read_image_from_request()
            start_time = time.time()
            result = models.get_ocr_model().ocr(image)
            return jsonify(
                {
                    "success": True,
                    "processing_time": time.time() - start_time,
                    "results": format_ocr_results(result),
                }
            )
        except ValueError as exc:
            return _bad_request(str(exc))
        except Exception as exc:  # pragma: no cover - protects production API responses
            return _server_error("OCR failed", exc)

    @app.post("/plate")
    def plate():
        data = _json_data()
        try:
            image = _read_image_from_request(data)
            min_score = float(data.get("min_score", request.form.get("min_score", 0.4)))
            start_time = time.time()
            result = models.get_plate_model().ocr(image, plate_min_score=min_score)
            response: dict[str, Any] = {
                "success": True,
                "processing_time": time.time() - start_time,
                "results": result,
            }
            if _truthy(data.get("visualize", request.form.get("visualize"))):
                response["visualization"] = image_to_base64(draw_plate_recognition(image, result))
            return jsonify(response)
        except ValueError as exc:
            return _bad_request(str(exc))
        except Exception as exc:  # pragma: no cover
            return _server_error("License plate recognition failed", exc)

    @app.post("/table")
    def table():
        data = _json_data()
        try:
            image = _read_image_from_request(data)
            start_time = time.time()
            result = models.get_table_model().ocr(image)
            result["success"] = True
            result["processing_time"] = time.time() - start_time
            if _truthy(data.get("visualize", request.form.get("visualize"))):
                result["visualization"] = image_to_base64(
                    draw_table_recognition(image, result, show_logic=_truthy(data.get("show_logic")))
                )
            return jsonify(result)
        except ValueError as exc:
            return _bad_request(str(exc))
        except Exception as exc:  # pragma: no cover
            return _server_error("Table recognition failed", exc)

    @app.post("/layout")
    def layout():
        data = _json_data()
        try:
            image = _read_image_from_request(data)
            model_type = data.get("model_type", request.form.get("model_type", "pp_doclayoutv2"))
            conf_thresh = float(data.get("conf_thresh", request.form.get("conf_thresh", 0.4)))
            iou_thresh = float(data.get("iou_thresh", request.form.get("iou_thresh", 0.5)))
            start_time = time.time()
            result = models.get_layout_model(model_type, conf_thresh, iou_thresh).ocr(image)
            result["success"] = True
            result["processing_time"] = time.time() - start_time
            if _truthy(data.get("visualize", request.form.get("visualize"))):
                result["visualization"] = image_to_base64(draw_layout_analysis(image, result))
            return jsonify(result)
        except ValueError as exc:
            return _bad_request(str(exc))
        except Exception as exc:  # pragma: no cover
            return _server_error("Layout analysis failed", exc)

    @app.post("/layout_markdown")
    def layout_markdown():
        data = _json_data()
        try:
            model_type = data.get("model_type", request.form.get("model_type", "pp_doclayoutv2"))
            conf_thresh = float(data.get("conf_thresh", request.form.get("conf_thresh", 0.4)))
            iou_thresh = float(data.get("iou_thresh", request.form.get("iou_thresh", 0.5)))
            output_dir = Path(app.config["OUTPUT_DIR"])
            output_dir.mkdir(parents=True, exist_ok=True)

            if request.files:
                source_path = _save_uploaded_file()
                output_md_path = output_dir / f"{source_path.stem}.md"
                start_time = time.time()
                result = models.get_layout_markdown_converter(model_type, conf_thresh, iou_thresh).convert_file(
                    str(source_path),
                    output_md_path=str(output_md_path),
                )
            else:
                image = _read_image_from_request(data)
                filename = secure_filename(str(data.get("filename", "layout_markdown.md"))) or "layout_markdown.md"
                if not filename.lower().endswith(".md"):
                    filename = f"{filename}.md"
                output_md_path = output_dir / filename
                start_time = time.time()
                result = models.get_layout_markdown_converter(model_type, conf_thresh, iou_thresh).convert_images(
                    [image],
                    output_md_path=str(output_md_path),
                    source_name=Path(filename).stem,
                )

            result["success"] = True
            result["processing_time"] = result.get("processing_time", time.time() - start_time)
            return jsonify(result)
        except ValueError as exc:
            return _bad_request(str(exc))
        except Exception as exc:  # pragma: no cover
            return _server_error("Layout markdown failed", exc)

    return app


def _json_data() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


def _read_image_from_request(data: dict[str, Any] | None = None):
    data = data or _json_data()
    if "image" in data:
        return decode_base64_image(str(data["image"]))
    file = request.files.get("image") or request.files.get("file")
    if file:
        image_base64 = base64.b64encode(file.read()).decode("ascii")
        return decode_base64_image(image_base64)
    raise ValueError("Image is required. Send JSON {'image': '<base64>'} or multipart file field 'image'.")


def _save_uploaded_file() -> Path:
    file = request.files.get("file") or request.files.get("image")
    if not file:
        raise ValueError("File is required.")
    suffix = Path(file.filename or "document").suffix or ".bin"
    filename = secure_filename(file.filename or f"document{suffix}") or f"document{suffix}"
    upload_dir = Path(tempfile.mkdtemp(prefix="onnxocr_upload_"))
    path = upload_dir / filename
    file.save(path)
    return path


def _bad_request(message: str):
    return jsonify({"success": False, "error": message}), 400


def _server_error(message: str, exc: Exception):
    return jsonify({"success": False, "error": f"{message}: {exc}"}), 500


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else _truthy(value)


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("ONNXOCR_PORT", "5005"))
    debug = _env_bool("ONNXOCR_DEBUG", default=False)
    app.run(host="0.0.0.0", port=port, debug=debug)
