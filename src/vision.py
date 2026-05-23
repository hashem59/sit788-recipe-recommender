"""Azure AI Vision 4.0: photo of fridge -> ingredient list + caption."""
from __future__ import annotations
from dataclasses import dataclass
from azure.ai.vision.imageanalysis.models import VisualFeatures
from .config import vision_client

# Tag whitelist — coarse but sufficient for a fridge photo demo.
# Anything matching these substrings is treated as a likely ingredient.
FOOD_TOKENS = {
    "fruit", "vegetable", "meat", "cheese", "milk", "egg", "bread", "butter",
    "yogurt", "tomato", "onion", "carrot", "potato", "pepper", "lettuce",
    "cucumber", "garlic", "ginger", "lemon", "lime", "apple", "banana",
    "orange", "berry", "grape", "broccoli", "mushroom", "spinach", "kale",
    "chicken", "beef", "pork", "fish", "salmon", "shrimp", "rice", "pasta",
    "noodle", "bean", "lentil", "corn", "avocado", "herb", "spice", "sauce",
    "juice", "drink", "snack", "fridge", "food", "produce", "dairy",
}

# Generic tags we never want to surface as "ingredients".
GENERIC_TAGS = {"food", "fridge", "produce", "dairy", "indoor", "snack"}


@dataclass
class VisionResult:
    caption: str
    ingredients: list[str]   # de-duplicated, lowercased
    raw_tags: list[tuple[str, float]]  # (name, confidence)


def _is_food(tag_name: str) -> bool:
    n = tag_name.lower()
    return any(tok in n for tok in FOOD_TOKENS)


def analyze_image(image_bytes: bytes, min_confidence: float = 0.55) -> VisionResult:
    """Run Azure AI Vision over raw image bytes; return ingredients + caption.

    Note: the Caption feature is region-locked (not available in australiaeast).
    We request only Tags and synthesize a short caption from the top-N tags.
    """
    client = vision_client()
    result = client.analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.TAGS],
    )
    raw_tags = [
        (t.name, t.confidence) for t in (result.tags.list if result.tags else [])
    ]
    top = ", ".join(name for name, _ in raw_tags[:5])
    caption = f"Image contains: {top}" if top else ""
    ingredients = sorted({
        name.lower() for name, conf in raw_tags
        if conf >= min_confidence
        and _is_food(name)
        and name.lower() not in GENERIC_TAGS
    })
    return VisionResult(caption=caption, ingredients=ingredients, raw_tags=raw_tags)
