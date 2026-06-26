"""
main.py — FastAPI application for RetinaTriage DR staging.

Endpoints:
    GET  /health   → {"status": "ok"}
    POST /predict  → DR stage, confidence, probabilities, Grad-CAM heatmap
"""

from contextlib import asynccontextmanager
from io import BytesIO

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from gradcam import generate_cam
from model import CLASS_NAMES, get_model, preprocess

# ---------------------------------------------------------------------------
# Globals set at startup
# ---------------------------------------------------------------------------
_model = None
_untrained = True


# ---------------------------------------------------------------------------
# Lifespan — load model once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _untrained
    _model, _untrained = get_model()
    print(f"🚀  RetinaTriage backend ready (device={next(_model.parameters()).device})")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RetinaTriage API",
    description="Diabetic retinopathy staging with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept a fundus image upload and return the DR prediction + Grad-CAM."""
    # --- Read & validate ------------------------------------------------
    try:
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file as an image. "
            "Please upload a valid image (PNG, JPEG, etc.).",
        )

    # --- Inference ------------------------------------------------------
    try:
        input_tensor = preprocess(pil_image)

        with torch.no_grad():
            logits = _model(input_tensor)  # (1, 5)
            probs = F.softmax(logits, dim=1).squeeze(0)  # (5,)

        pred_class = int(probs.argmax().item())
        confidence = round(float(probs[pred_class].item()), 4)

        prob_dict = {
            name: round(float(probs[i].item()), 4)
            for i, name in enumerate(CLASS_NAMES)
        }

        # --- Grad-CAM ---------------------------------------------------
        heatmap_b64 = generate_cam(_model, input_tensor, pred_class, pil_image)

        return {
            "stage": pred_class,
            "stage_label": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "probabilities": prob_dict,
            "heatmap": heatmap_b64,
            "untrained_warning": _untrained,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
