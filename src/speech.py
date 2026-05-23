"""Azure Speech: WAV bytes -> transcript string."""
from __future__ import annotations
import tempfile
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from .config import Settings


def transcribe_wav(audio_bytes: bytes, language: str = "en-AU") -> str:
    """Single-shot recognition. Returns empty string on no-match/timeout."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        cfg = speechsdk.SpeechConfig(
            subscription=Settings.speech_key, region=Settings.speech_region
        )
        cfg.speech_recognition_language = language
        audio = speechsdk.audio.AudioConfig(filename=str(tmp_path))
        recogniser = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)
        result = recogniser.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)
