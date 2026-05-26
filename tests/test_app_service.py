from __future__ import annotations

import importlib.util
import base64
from pathlib import Path

import cv2
import numpy as np


def load_app_service():
    module_path = Path(__file__).resolve().parents[1] / "app-service.py"
    spec = importlib.util.spec_from_file_location("app_service", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_app_service_health_endpoint():
    module = load_app_service()
    app = module.create_app()

    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_app_service_ocr_requires_image():
    module = load_app_service()
    app = module.create_app()

    response = app.test_client().post("/ocr", json={})

    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert "Image is required" in response.get_json()["error"]


def test_app_service_ocr_accepts_base64_image():
    class FakeOcrModel:
        def ocr(self, image):
            assert image.shape[:2] == (8, 8)
            return [[[[[0, 0], [4, 0], [4, 4], [0, 4]], ("hello", 0.99)]]]

    class FakeRegistry:
        def get_ocr_model(self):
            return FakeOcrModel()

    module = load_app_service()
    app = module.create_app(FakeRegistry())
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    image_base64 = base64.b64encode(buffer).decode("ascii")

    response = app.test_client().post("/ocr", json={"image": image_base64})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["results"][0]["text"] == "hello"
    assert payload["results"][0]["confidence"] == 0.99
