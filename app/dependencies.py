"""
Shared FastAPI dependencies.
Classifier and Groq client are instantiated once and reused.
"""

from functools import lru_cache
from groq import AsyncGroq
from app.config import get_settings
from app.services.classifier import SoilClassifier


@lru_cache
def get_classifier() -> SoilClassifier:
    settings = get_settings()
    return SoilClassifier(checkpoint_path=settings.model_checkpoint_path)


@lru_cache
def get_groq_client() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(api_key=settings.groq_api_key)
