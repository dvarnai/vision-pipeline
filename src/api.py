import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.inference.prediction import load_inference_bundle, predict_image
from src.inference.validation import (
    DEFAULT_MAX_IMAGE_BYTES,
    InvalidInputError,
    validate_image_bytes,
)


def _max_image_bytes() -> int:
    return int(os.getenv("MAX_IMAGE_BYTES", str(DEFAULT_MAX_IMAGE_BYTES)))


app = FastAPI(title="Vision Pipeline Intel Inference", version="0.1.0")


@app.on_event("startup")
def load_model():
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if checkpoint_path is None:
        raise RuntimeError("CHECKPOINT_PATH environment variable is required")

    app.state.predictor = load_inference_bundle(
        checkpoint_path,
        config_override=os.getenv("CONFIG_MODULE"),
        device=os.getenv("DEVICE", "auto"),
    )


@app.get("/health")
def health():
    predictor = getattr(app.state, "predictor", None)
    return {
        "status": "ok" if predictor is not None else "not_ready",
        "model_loaded": predictor is not None,
        "model_version": getattr(predictor, "model_version", None),
        "preprocessing_version": getattr(predictor, "preprocessing_version", None),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(1, ge=1, le=10),
):
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_not_loaded", "message": "model is not loaded"},
        )

    image_bytes = await file.read()
    try:
        image = validate_image_bytes(
            image_bytes,
            filename=file.filename,
            content_type=file.content_type,
            max_bytes=_max_image_bytes(),
        )
        return predict_image(predictor, image, top_k=top_k)
    except InvalidInputError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
