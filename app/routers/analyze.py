"""
/api/analyze  — main endpoint
Accepts a multipart form: image file + region + optional climate zone.
Returns soil classification + tree recommendations.
"""

import logging
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from groq import AsyncGroq

from app.config import get_settings
from app.schemas.analyze import AnalyzeResponse, SoilAnalysis
from app.services.classifier import SoilClassifier, SOIL_METADATA
from app.services.recommender import get_tree_recommendations
from app.dependencies import get_classifier, get_groq_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analyze"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze soil image and return tree recommendations",
)
async def analyze(
    image: UploadFile = File(..., description="Close-up photo of soil"),
    region: str = Form(..., description="Country or region name"),
    climate_zone: str | None = Form(None, description="Optional climate zone"),
    classifier: SoilClassifier = Depends(get_classifier),
    groq_client: AsyncGroq = Depends(get_groq_client),
):
    # ── Validate image ────────────────────────────────────────────────────────
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{image.content_type}'. Use JPEG, PNG, or WEBP.",
        )

    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image must be under 10 MB.",
        )

    # ── Classify soil ─────────────────────────────────────────────────────────
    try:
        prediction = classifier.predict(image_bytes)
    except Exception as exc:
        logger.exception("Soil classification failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Soil classification error: {exc}",
        )

    soil_type = prediction["soil_type"]
    soil_meta = SOIL_METADATA.get(soil_type, {})

    soil_analysis = SoilAnalysis(
        soil_type=soil_type.title(),
        confidence=prediction["confidence"],
        characteristics=soil_meta.get("characteristics", ""),
        ph_range=soil_meta.get("ph_range", ""),
        drainage=soil_meta.get("drainage", ""),
    )

    # ── Get tree recommendations from Claude ──────────────────────────────────
    try:
        rec_data = await get_tree_recommendations(
            client=groq_client,
            soil_type=soil_type,
            soil_meta=soil_meta,
            region=region,
            climate_zone=climate_zone,
        )
    except Exception as exc:
        logger.exception("Tree recommendation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI recommendation error: {exc}",
        )

    # ── Build response ────────────────────────────────────────────────────────
    from app.schemas.analyze import TreeRecommendation, PlantingGuide

    trees = [TreeRecommendation(**t) for t in rec_data.get("trees", [])]
    guide_raw = rec_data.get("planting_guide", {})
    planting_guide = PlantingGuide(
        best_season=guide_raw.get("best_season", ""),
        spacing=guide_raw.get("spacing", ""),
        soil_preparation=guide_raw.get("soil_preparation", ""),
        water_needs=guide_raw.get("water_needs", ""),
        care_notes=guide_raw.get("care_notes", ""),
        climate_impact=guide_raw.get("climate_impact", ""),
    )

    return AnalyzeResponse(
        soil_analysis=soil_analysis,
        trees=trees,
        planting_guide=planting_guide,
        region=region,
        climate_zone=climate_zone,
    )


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "Tree World API"}
