"""
Groq Tree Recommendation Service
----------------------------------
Uses Groq's API (OpenAI-compatible) to get structured tree
recommendations based on classified soil type + region.

Recommended models:
  - llama-3.3-70b-versatile  (best quality, default)
  - llama-3.1-8b-instant     (fastest, lowest latency)
  - mixtral-8x7b-32768       (good balance)
"""

import json
import logging
from groq import AsyncGroq

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert agroforestry and reforestation scientist specialising in 
climate change mitigation through tree planting. Your role is to recommend the best tree species 
for a given soil type and geographic region, with a strong focus on:

- Native or well-adapted species that support local biodiversity
- High carbon sequestration potential
- Climate resilience and drought tolerance
- Practical planting feasibility for smallholder farmers and local communities
- Long-term ecosystem restoration

Always respond with valid JSON only. No markdown, no backticks, no prose outside the JSON."""

RECOMMENDATION_TEMPLATE = """Soil type: {soil_type}
Soil characteristics: {characteristics}
Soil pH range: {ph_range}
Drainage: {drainage}
Region: {region}
Climate zone: {climate_zone}

Recommend exactly 4 trees best suited for reforestation and climate change mitigation 
in this soil and region. Respond ONLY with this JSON structure:

{{
  "trees": [
    {{
      "name": "Common Name",
      "scientific_name": "Scientific name",
      "description": "2 sentences: why this tree suits this specific soil and its climate impact.",
      "tags": ["carbon-sequestering", "native", "drought-resistant", "fast-growing"],
      "carbon_sequestration": "e.g. High — ~22 kg CO2/year at maturity",
      "growth_rate": "e.g. Fast (2-3 m/year)"
    }}
  ],
  "planting_guide": {{
    "best_season": "When to plant given the region's rainfall patterns",
    "spacing": "Recommended spacing between trees",
    "soil_preparation": "How to prepare this specific soil type before planting",
    "water_needs": "Watering guidance for the first 1-2 years",
    "care_notes": "Key first-year care tips for this region",
    "climate_impact": "Expected combined carbon/ecosystem benefit of planting these trees"
  }}
}}"""


async def get_tree_recommendations(
    client: AsyncGroq,
    soil_type: str,
    soil_meta: dict,
    region: str,
    climate_zone: str | None,
) -> dict:
    """
    Call Groq to get tree recommendations for the given soil + region.
    Returns parsed JSON dict with 'trees' and 'planting_guide' keys.
    """
    prompt = RECOMMENDATION_TEMPLATE.format(
        soil_type=soil_type,
        characteristics=soil_meta.get("characteristics", ""),
        ph_range=soil_meta.get("ph_range", "unknown"),
        drainage=soil_meta.get("drainage", "unknown"),
        region=region,
        climate_zone=climate_zone or "not specified",
    )

    logger.info(f"Requesting tree recommendations via Groq ({GROQ_MODEL}) for {soil_type} in {region}")

    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,       # low temp = more consistent JSON output
        max_tokens=1024,
        response_format={"type": "json_object"},  # forces valid JSON output
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences just in case
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)
