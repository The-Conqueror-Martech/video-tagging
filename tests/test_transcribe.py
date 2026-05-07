"""Tests for speech-to-text transcription."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTranscribeAudio:
    """Tests for audio transcription."""

    def test_returns_empty_for_missing_file(self) -> None:
        """Test that missing audio file returns empty string."""
        from videotagger.transcribe import transcribe_audio

        result = transcribe_audio("/nonexistent/path/audio.wav")

        assert result == ""

    def test_transcribes_audio_with_whisper(self) -> None:
        """Test that Whisper model is called correctly."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world, this is a test."}

        with patch("videotagger.transcribe._get_whisper_model", return_value=mock_model):
            with patch("pathlib.Path.exists", return_value=True):
                from videotagger.transcribe import transcribe_audio

                result = transcribe_audio("/fake/path/audio.wav")

                assert result == "Hello world, this is a test."
                mock_model.transcribe.assert_called_once()

    def test_returns_empty_for_no_speech(self) -> None:
        """Test that empty transcript is returned when no speech detected."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}

        with patch("videotagger.transcribe._get_whisper_model", return_value=mock_model):
            with patch("pathlib.Path.exists", return_value=True):
                from videotagger.transcribe import transcribe_audio

                result = transcribe_audio("/fake/path/silent.wav")

                assert result == ""


class TestTranscribeVideo:
    """Tests for video transcription convenience function."""

    def test_extracts_audio_and_transcribes(self) -> None:
        """Test that audio is extracted from video before transcription."""
        mock_audio_path = Path("/tmp/extracted_audio.wav")

        with patch("videotagger.audio_extract.extract_audio", return_value=mock_audio_path) as mock_extract:
            with patch("videotagger.transcribe.transcribe_audio", return_value="Test transcript") as mock_transcribe:
                with patch.object(Path, "exists", return_value=True):
                    with patch.object(Path, "unlink"):
                        from videotagger.transcribe import transcribe_video

                        result = transcribe_video("/fake/video.mp4")

                        assert result == "Test transcript"
                        mock_extract.assert_called_once()
                        mock_transcribe.assert_called_once()
