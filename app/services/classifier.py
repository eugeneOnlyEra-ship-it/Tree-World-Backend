"""
EfficientNet-B0 Soil Classifier
--------------------------------
Fine-tuned on 6 soil classes. The model is loaded once at startup
and shared across requests via FastAPI's dependency injection.

Soil classes (must match training label order):
  0: clay
  1: laterite
  2: loam
  3: peat
  4: sandy
  5: silt
"""

import io
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

logger = logging.getLogger(__name__)

SOIL_CLASSES = ["clay", "laterite", "loam", "peat", "sandy", "silt"]

# ImageNet normalisation (EfficientNet was pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes: int = len(SOIL_CLASSES)) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classifier head.
    Matches the architecture used during fine-tuning.
    """
    model = models.efficientnet_b0(weights=None)

    # Replace final classifier to match our number of soil classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class SoilClassifier:
    """
    Wraps the fine-tuned EfficientNet-B0 model for inference.
    Instantiated once at app startup and injected via FastAPI dependency.
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load(checkpoint_path)

    def _load(self, checkpoint_path: str) -> nn.Module:
        path = Path(checkpoint_path)
        model = build_model()

        if path.exists():
            logger.info(f"Loading soil model weights from {path}")
            state = torch.load(path, map_location=self.device)
            # Support both raw state_dict and checkpoint dicts
            state_dict = state.get("model_state_dict", state)
            model.load_state_dict(state_dict)
        else:
            logger.warning(
                f"Checkpoint not found at {path}. "
                "Running with random weights — fine-tune the model first."
            )

        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on raw image bytes.

        Returns:
            {
                "soil_type": "laterite",
                "confidence": 0.91,
                "probabilities": {"clay": 0.03, "laterite": 0.91, ...}
            }
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)                   # (1, num_classes)
            probs  = torch.softmax(logits, dim=1)[0]      # (num_classes,)

        probs_np = probs.cpu().numpy()
        top_idx  = int(np.argmax(probs_np))

        return {
            "soil_type":     SOIL_CLASSES[top_idx],
            "confidence":    float(probs_np[top_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(SOIL_CLASSES, probs_np)},
        }


# ── Soil type metadata ────────────────────────────────────────────────────────

SOIL_METADATA = {
    "clay": {
        "characteristics": "Heavy, dense soil with high water retention. Rich in minerals but prone to waterlogging.",
        "ph_range": "6.0 – 7.0 (neutral to slightly acidic)",
        "drainage": "Poor — needs amendment for most trees",
    },
    "laterite": {
        "characteristics": "Iron-rich reddish tropical soil. Hardens on exposure. Common across Sub-Saharan Africa.",
        "ph_range": "5.0 – 6.5 (moderately acidic)",
        "drainage": "Moderate to well-drained",
    },
    "loam": {
        "characteristics": "Ideal balanced soil — mix of sand, silt, and clay. Excellent for most trees.",
        "ph_range": "6.0 – 7.0 (neutral)",
        "drainage": "Well-drained with good moisture retention",
    },
    "peat": {
        "characteristics": "Dark, organic-rich soil. Highly acidic and retains moisture. Good carbon store.",
        "ph_range": "3.5 – 5.5 (very acidic)",
        "drainage": "Poor — waterlogged conditions common",
    },
    "sandy": {
        "characteristics": "Light, coarse-grained soil with low nutrient retention. Drains rapidly.",
        "ph_range": "5.5 – 7.0 (variable)",
        "drainage": "Very well-drained, prone to drought stress",
    },
    "silt": {
        "characteristics": "Fine-grained, fertile soil with good moisture retention. Common near river valleys.",
        "ph_range": "6.0 – 7.0 (near neutral)",
        "drainage": "Moderate — can compact under rain",
    },
}
