from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import src.api as api


def _png_bytes():
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_loaded_predictor(monkeypatch):
    predictor = type(
        "Predictor",
        (),
        {
            "model_version": "model-v1",
            "preprocessing_version": "preprocess-v1",
        },
    )()
    api.app.state.predictor = predictor

    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "model-v1",
        "preprocessing_version": "preprocess-v1",
    }


def test_predict_validates_upload_and_returns_prediction(monkeypatch):
    api.app.state.predictor = object()
    monkeypatch.setattr(
        api,
        "predict_image",
        lambda predictor, image, top_k: {
            "predicted_label": "forest",
            "confidence": 0.9,
            "top_k": [{"label": "forest", "confidence": 0.9}],
        },
    )

    client = TestClient(api.app)
    response = client.post(
        "/predict?top_k=1",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_label"] == "forest"


def test_predict_rejects_invalid_upload_metadata():
    api.app.state.predictor = object()
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        files={"file": ("scene.jpg", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "extension_content_type_mismatch"
