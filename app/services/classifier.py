"""
EfficientNet-B0 Soil Classifier — loads classes dynamically from class_info.json
"""
import io, json, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_class_info(checkpoint_path: str) -> list:
    info_path = Path(checkpoint_path).parent / "class_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)["classes"]
    return []

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

class SoilClassifier:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soil_classes = []
        self.model = self._load(checkpoint_path)

    def _load(self, checkpoint_path: str) -> nn.Module:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found at {path}. Using random weights.")
            self.soil_classes = ["black_soil", "cinder_soil", "laterite_soil", "peat_soil", "yellow_soil"]
            model = build_model(len(self.soil_classes))
            model.to(self.device); model.eval()
            return model

        logger.info(f"Loading soil model weights from {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.soil_classes = load_class_info(checkpoint_path) or state.get("classes", [])
        if not self.soil_classes:
            raise RuntimeError("Could not determine class list from checkpoint or class_info.json")
        logger.info(f"Soil classes ({len(self.soil_classes)}): {self.soil_classes}")
        model = build_model(len(self.soil_classes))
        model.load_state_dict(state.get("model_state_dict", state))
        model.to(self.device); model.eval()
        return model

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        top_idx = int(np.argmax(probs))
        return {
            "soil_type":     self.soil_classes[top_idx],
            "confidence":    float(probs[top_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(self.soil_classes, probs)},
        }

SOIL_METADATA = {
    "black_soil":    {"characteristics": "Dark, fertile soil rich in clay minerals. High moisture retention.", "ph_range": "6.5 – 8.0 (neutral to mildly alkaline)", "drainage": "Poor to moderate — can crack when dry"},
    "cinder_soil":   {"characteristics": "Volcanic soil with coarse, porous texture. Low nutrients but great aeration.", "ph_range": "5.5 – 7.0 (slightly acidic to neutral)", "drainage": "Excellent — drains very rapidly"},
    "laterite_soil": {"characteristics": "Iron-rich reddish tropical soil. Hardens on exposure. Common across Sub-Saharan Africa.", "ph_range": "5.0 – 6.5 (moderately acidic)", "drainage": "Moderate to well-drained"},
    "peat_soil":     {"characteristics": "Dark, organic-rich soil. Highly acidic and retains moisture. Good carbon store.", "ph_range": "3.5 – 5.5 (very acidic)", "drainage": "Poor — waterlogged conditions common"},
    "yellow_soil":   {"characteristics": "Iron oxide-rich soil with yellowish colour. Moderate fertility, common in tropical regions.", "ph_range": "5.0 – 6.5 (acidic)", "drainage": "Moderate"},
}
