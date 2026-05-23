"""Central config: loads .env once and exposes Azure clients."""
from __future__ import annotations
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing env var: {key}. Check .env.")
    return val


class Settings:
    vision_endpoint = _require("AZURE_VISION_ENDPOINT")
    vision_key = _require("AZURE_VISION_KEY")
    speech_key = _require("AZURE_SPEECH_KEY")
    speech_region = _require("AZURE_SPEECH_REGION")
    openai_endpoint = _require("AZURE_OPENAI_ENDPOINT")
    openai_key = _require("AZURE_OPENAI_KEY")
    openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    chat_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini")
    embed_deployment = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")


@lru_cache
def openai_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=Settings.openai_endpoint,
        api_key=Settings.openai_key,
        api_version=Settings.openai_api_version,
    )


@lru_cache
def vision_client():
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    return ImageAnalysisClient(
        endpoint=Settings.vision_endpoint,
        credential=AzureKeyCredential(Settings.vision_key),
    )
