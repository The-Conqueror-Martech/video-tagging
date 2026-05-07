"""Speech-to-text transcription using OpenAI Whisper.

Provides local transcription without requiring API keys.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded Whisper model cache
_whisper_model: Any = None


def _get_whisper_model(model_name: str = "base"):
    """Load Whisper model (cached).

    Args:
        model_name: Whisper model size. Options: tiny, base, small, medium, large.
                   Default is "base" for balance of speed and accuracy.

    Returns:
        Loaded Whisper model.
    """
    global _whisper_model

    if _whisper_model is None:
        import whisper

        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
        logger.info(f"Whisper model loaded: {model_name}")

    return _whisper_model


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
) -> str:
    """Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.).
        model_name: Whisper model size (default: "base").
        language: Optional language code (e.g., "en"). If None, auto-detects.

    Returns:
        Transcribed text as string. Returns empty string if transcription fails
        or audio contains no speech.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return ""

    try:
        model = _get_whisper_model(model_name)

        logger.info(f"Transcribing audio: {audio_path}")

        # Transcribe with optional language hint
        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(str(audio_path), **options)

        transcript = result.get("text", "").strip()

        if transcript:
            logger.info(f"Transcription complete: {len(transcript)} characters")
            logger.debug(f"Transcript preview: {transcript[:200]}...")
        else:
            logger.info("No speech detected in audio")

        return transcript

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""


def transcribe_video(
    video_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
) -> str:
    """Extract audio from video and transcribe to text.

    Convenience function that handles audio extraction and cleanup.

    Args:
        video_path: Path to video file.
        model_name: Whisper model size (default: "base").
        language: Optional language code (e.g., "en"). If None, auto-detects.

    Returns:
        Transcribed text as string. Returns empty string if transcription fails.
    """
    from videotagger.audio_extract import extract_audio

    video_path = Path(video_path)
    audio_path = None

    try:
        # Extract audio to temp file
        logger.info(f"Extracting audio from video: {video_path}")
        audio_path = extract_audio(video_path)

        # Transcribe
        transcript = transcribe_audio(audio_path, model_name=model_name, language=language)

        return transcript

    except Exception as e:
        logger.error(f"Video transcription failed: {e}")
        return ""

    finally:
        # Clean up temp audio file
        if audio_path and audio_path.exists():
            audio_path.unlink()
            logger.debug(f"Cleaned up temp audio: {audio_path}")
