"""
Tests for the /api/analyze endpoint.
Run with: pytest tests/ -v
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from app.dependencies import get_classifier, get_groq_client


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_image(color=(139, 90, 43)) -> bytes:
    """Create a tiny RGB JPEG in memory."""
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MOCK_RECOMMENDATIONS = {
    "trees": [
        {
            "name": "African Mahogany",
            "scientific_name": "Khaya senegalensis",
            "description": "Thrives in laterite soils with good drainage. Sequesters significant carbon.",
            "tags": ["carbon-sequestering", "native", "drought-resistant"],
            "carbon_sequestration": "High — ~18 kg CO₂/year",
            "growth_rate": "Moderate (1–2 m/year)",
        },
        {
            "name": "Moringa",
            "scientific_name": "Moringa oleifera",
            "description": "Fast-growing multipurpose tree. Excellent for degraded laterite soils.",
            "tags": ["fast-growing", "drought-resistant", "nitrogen-fixing"],
            "carbon_sequestration": "Moderate — ~10 kg CO₂/year",
            "growth_rate": "Fast (3–5 m/year)",
        },
        {
            "name": "Mango",
            "scientific_name": "Mangifera indica",
            "description": "Provides shade, food security, and steady carbon sequestration.",
            "tags": ["carbon-sequestering", "food-forest"],
            "carbon_sequestration": "Moderate — ~12 kg CO₂/year",
            "growth_rate": "Moderate (1–2 m/year)",
        },
        {
            "name": "Msandarusi",
            "scientific_name": "Brachystegia spiciformis",
            "description": "Iconic miombo woodland tree. Essential for local ecosystem restoration.",
            "tags": ["native", "carbon-sequestering", "biodiversity"],
            "carbon_sequestration": "High — ~20 kg CO₂/year",
            "growth_rate": "Slow–Moderate (0.5–1 m/year)",
        },
    ],
    "planting_guide": {
        "best_season": "Start of rainy season (November–December)",
        "spacing": "5–8 m between trees depending on species",
        "soil_preparation": "Break up hardened laterite crust; add organic compost",
        "water_needs": "Weekly watering first 3 months; rain-fed thereafter",
        "care_notes": "Mulch around base; protect from goats in first year",
        "climate_impact": "4 trees can sequester ~60 kg CO₂/year collectively",
    },
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_classifier():
    clf = MagicMock()
    clf.predict.return_value = {
        "soil_type": "laterite",
        "confidence": 0.88,
        "probabilities": {
            "clay": 0.03, "laterite": 0.88, "loam": 0.04,
            "peat": 0.01, "sandy": 0.02, "silt": 0.02,
        },
    }
    return clf


@pytest.fixture
def mock_groq():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = json.dumps(MOCK_RECOMMENDATIONS)
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create = MagicMock(return_value=response)
    return client


@pytest.fixture
def client(mock_classifier, mock_groq):
    app.dependency_overrides[get_classifier] = lambda: mock_classifier
    app.dependency_overrides[get_groq_client] = lambda: mock_groq
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_success(client):
    resp = client.post(
        "/api/analyze",
        files={"image": ("soil.jpg", make_fake_image(), "image/jpeg")},
        data={"region": "Malawi, Central Africa", "climate_zone": "tropical"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["soil_analysis"]["soil_type"] == "Laterite"
    assert body["soil_analysis"]["confidence"] == pytest.approx(0.88)
    assert len(body["trees"]) == 4
    assert body["trees"][0]["name"] == "African Mahogany"
    assert body["planting_guide"]["best_season"] != ""


def test_analyze_no_image(client):
    resp = client.post(
        "/api/analyze",
        data={"region": "Kenya"},
    )
    assert resp.status_code == 422  # Unprocessable Entity


def test_analyze_no_region(client):
    resp = client.post(
        "/api/analyze",
        files={"image": ("soil.jpg", make_fake_image(), "image/jpeg")},
    )
    assert resp.status_code == 422


def test_analyze_unsupported_image_type(client):
    resp = client.post(
        "/api/analyze",
        files={"image": ("soil.gif", b"GIF89a", "image/gif")},
        data={"region": "Nigeria"},
    )
    assert resp.status_code == 415


def test_analyze_climate_zone_optional(client):
    resp = client.post(
        "/api/analyze",
        files={"image": ("soil.jpg", make_fake_image(), "image/jpeg")},
        data={"region": "Tanzania"},
        # No climate_zone
    )
    assert resp.status_code == 200
    assert resp.json()["climate_zone"] is None
