"""
model.py — ResNet50 backbone for 5-class diabetic retinopathy staging.

Classes: 0-No DR, 1-Mild, 2-Moderate, 3-Severe, 4-Proliferative DR
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "model.pth"

# ---------------------------------------------------------------------------
# Preprocessing — THIS MUST MATCH the validation transform in training/train.py
# ---------------------------------------------------------------------------
_preprocess_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet channel means
            std=[0.229, 0.224, 0.225],  # ImageNet channel stds
        ),
    ]
)


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a model-ready tensor.

    Returns:
        Tensor of shape (1, 3, 224, 224) on ``DEVICE``.
    """
    # THIS MUST MATCH the validation transform in training/train.py
    tensor = _preprocess_transforms(pil_image)  # (3, 224, 224)
    return tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_model() -> tuple[nn.Module, bool]:
    """Load (or initialise) a ResNet50 for DR staging.

    Returns:
        model:      ``nn.Module`` in eval mode, placed on ``DEVICE``.
        untrained:  ``True`` when no trained weights were found — predictions
                    will be random and are only useful for pipeline testing.
    """
    # Build a ResNet50 with the final FC replaced for 5-class output
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    untrained = True

    if _WEIGHTS_PATH.is_file():
        state_dict = torch.load(_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        untrained = False
        print(f"✅  Loaded trained weights from {_WEIGHTS_PATH}")
    else:
        print()
        print("=" * 72)
        print(
            "⚠️  No trained weights found — predictions are random/untrained, "
            "pipeline-test mode only."
        )
        print(f"    Expected path: {_WEIGHTS_PATH}")
        print("=" * 72)
        print()

    model = model.to(DEVICE)
    model.eval()
    return model, untrained
