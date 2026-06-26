"""
gradcam.py — Grad-CAM heatmap generation for DR predictions.

Produces a base64-encoded PNG overlay highlighting the regions that most
influenced the model's classification decision.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn


def generate_cam(
    model: nn.Module,
    input_tensor,
    target_class: int,
    original_pil: Image.Image,
) -> str:
    """Generate a Grad-CAM heatmap overlay and return it as a base64 PNG.

    Args:
        model:          The ResNet50 model (eval mode, on correct device).
        input_tensor:   Preprocessed tensor of shape ``(1, 3, 224, 224)``.
        target_class:   Predicted (or target) class index (0-4).
        original_pil:   The original uploaded PIL image (any size).

    Returns:
        A base64-encoded PNG string of the heatmap overlay, sized to match
        the original image dimensions.
    """
    original_size = original_pil.size  # (W, H)

    # Target the last bottleneck block of layer4
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (1, H, W)
        grayscale_cam = grayscale_cam[0, :]  # (H, W)  — single image

    # Prepare the original image as a 224x224 float32 RGB array in [0, 1]
    rgb_img = original_pil.convert("RGB").resize((224, 224))
    rgb_img = np.array(rgb_img, dtype=np.float32) / 255.0

    # Create the overlay (returns BGR uint8 from OpenCV internals)
    overlay_bgr = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    # Convert BGR → RGB
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Resize overlay back to the original image dimensions
    overlay_rgb = cv2.resize(
        overlay_rgb, original_size, interpolation=cv2.INTER_LINEAR
    )

    # Encode as PNG → base64
    pil_overlay = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_overlay.save(buf, format="PNG")
    buf.seek(0)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64_str
