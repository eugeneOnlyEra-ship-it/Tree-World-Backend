from pydantic import BaseModel, Field
from typing import Optional


# ── Request ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    region: str = Field(..., description="User's country or region, e.g. 'Malawi, Central Africa'")
    climate_zone: Optional[str] = Field(None, description="Optional climate zone hint")


# ── Response ─────────────────────────────────────────────────────────────────

class TreeRecommendation(BaseModel):
    name: str
    scientific_name: str
    description: str
    tags: list[str]
    carbon_sequestration: str      # e.g. "High — ~22 kg CO₂/year"
    growth_rate: str               # e.g. "Fast (2–3 m/year)"


class PlantingGuide(BaseModel):
    best_season: str
    spacing: str
    soil_preparation: str
    water_needs: str
    care_notes: str
    climate_impact: str


class SoilAnalysis(BaseModel):
    soil_type: str                 # e.g. "Red Laterite"
    confidence: float              # model confidence 0–1
    characteristics: str           # short human-readable description
    ph_range: str                  # e.g. "5.5 – 6.5 (slightly acidic)"
    drainage: str                  # e.g. "Well-drained"


class AnalyzeResponse(BaseModel):
    soil_analysis: SoilAnalysis
    trees: list[TreeRecommendation]
    planting_guide: PlantingGuide
    region: str
    climate_zone: Optional[str]
