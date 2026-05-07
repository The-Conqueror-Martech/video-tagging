"""CLI entry point for VideoTagger."""

import json
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from videotagger.config import Settings, mask_credential
from videotagger.exceptions import LLMError, VideoProcessingError
from videotagger.logging_config import setup_logging
from videotagger.pipeline import process_video, process_video_with_context

# Global debug flag
DEBUG = False


def validate_config() -> int:
    """Validate configuration and display status.

    Returns:
        Exit code: 0 for success, 1 for validation errors.
    """
    # Load .env file
    load_dotenv()

    try:
        settings = Settings()
    except ValidationError as e:
        print("Configuration validation failed!\n")
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            print(f"  - {field}: {msg}")
        print("\nPlease check your .env file or environment variables.")
        return 1

    print("Configuration validated successfully!\n")

    # Display masked credentials
    print("Synology NAS:")
    print(f"  Host: {settings.synology.host}")
    print(f"  User: {settings.synology.user}")
    print(f"  Password: {mask_credential(settings.synology.password)}")
    print(f"  Video Path: {settings.synology.video_path}")

    print("\nAirtable:")
    print(f"  API Key: {mask_credential(settings.airtable.api_key)}")
    print(f"  Base ID: {settings.airtable.base_id}")
    print(f"  Table ID: {settings.airtable.table_id}")

    print("\nRunPod S3:")
    print(f"  Endpoint: {settings.runpod_s3.endpoint}")
    print(f"  Bucket: {settings.runpod_s3.bucket}")
    print(f"  Access Key: {mask_credential(settings.runpod_s3.access_key)}")
    print(f"  Secret Key: {mask_credential(settings.runpod_s3.secret_key)}")

    print("\nRunPod SSH:")
    print(f"  Host: {settings.runpod_ssh.host}")
    print(f"  User: {settings.runpod_ssh.user}")
    print(f"  Key Path: {settings.runpod_ssh.key_path}")
    print(f"  Pod ID: {settings.runpod_ssh.pod_id}")

    return 0


def process_video_command(video_path: str, debug: bool = False) -> int:
    """Process a video file and output tags as JSON.

    Args:
        video_path: Path to the video file.
        debug: Enable debug logging.

    Returns:
        Exit code: 0 for success, 1 for errors.
    """
    load_dotenv()
    setup_logging(debug=debug)

    try:
        print(f"Processing video: {video_path}")
        tags = process_video(video_path)
        print("\nExtracted tags:")
        print(json.dumps(tags, indent=2, ensure_ascii=False))
        return 0

    except VideoProcessingError as e:
        print(f"Video processing error: {e}")
        if debug and e.video_path:
            print(f"  Video path: {e.video_path}")
        return 1

    except LLMError as e:
        print(f"LLM error: {e}")
        if debug and e.original_error:
            print(f"  Original error: {type(e.original_error).__name__}: {e.original_error}")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return 1


def run_tui() -> int:
    """Run the TUI application.

    Returns:
        Exit code: 0 for success.
    """
    load_dotenv()

    from videotagger.tui.app import run_tui as start_tui

    start_tui()
    return 0


def analyze_audio_command(video_path: str, debug: bool = False) -> int:
    """Analyze audio from a local video file.

    Args:
        video_path: Path to the video file.
        debug: Enable debug logging.

    Returns:
        Exit code: 0 for success, 1 for errors.
    """
    import json
    from pathlib import Path

    setup_logging(debug=debug)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: File not found: {video_path}")
        return 1

    try:
        from videotagger.audio_analysis import analyze_video_audio

        print(f"Analyzing audio: {video_path.name}")
        print("Loading models (first run may take a moment)...\n")

        result = analyze_video_audio(video_path)

        print("Audio Analysis Results:")
        print("-" * 40)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

        # Summary
        print("-" * 40)
        if result.voice_detected:
            print(f"Voice: Detected ({len(result.voice_segments)} segments)")
            if result.prosody:
                print(f"Style: {result.prosody.voiceover_style}")
                print(f"  Tempo: {result.prosody.tempo_bpm:.0f} BPM")
                print(f"  Pitch: {result.prosody.mean_pitch_hz:.0f} Hz (±{result.prosody.pitch_variation_hz:.0f})")
                print(f"  Energy: {result.prosody.energy_level:.4f}")
        else:
            print("Voice: Not detected")
        print(f"Genre: {result.music_genre} ({result.music_genre_confidence:.0%})")
        print(f"Time: {result.processing_time_ms:.0f}ms")

        return 0

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("\nInstall audio dependencies with:")
        print("  pip install -e '.[audio]'")
        return 1

    except Exception as e:
        print(f"Error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return 1


def process_with_context_command(
    video_path: str,
    art_id: str | None = None,
    no_transcript: bool = False,
    debug: bool = False,
) -> int:
    """Process a video with Visual Hook analysis and optional Airtable context.

    This uses the enhanced pipeline with:
    - Weighted frame extraction (dense first 1.5s, sparse after)
    - Dynamic prompts from Airtable context (if art_id provided)
    - Speech-to-text transcription for Copy Structure analysis
    - Visual Hook output (action, subject, environment)

    Args:
        video_path: Path to the video file.
        art_id: Optional Art ID to fetch Airtable context (e.g., "a1433").
        no_transcript: Skip transcription (faster, no Copy Structure).
        debug: Enable debug logging.

    Returns:
        Exit code: 0 for success, 1 for errors.
    """
    from pathlib import Path

    load_dotenv()
    setup_logging(debug=debug)

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"Error: File not found: {video_path}")
        return 1

    try:
        print(f"Processing video: {video_path_obj.name}")
        if art_id:
            print(f"Fetching Airtable context for: {art_id}")
        if not no_transcript:
            print("Transcription enabled (first run downloads Whisper model ~140MB)")
        print()

        tags = process_video_with_context(
            video_path=video_path,
            art_id=art_id,
            include_transcript=not no_transcript,
        )

        print("\nExtracted tags:")
        print(json.dumps(tags, indent=2, ensure_ascii=False))

        # Summary
        print("\n" + "-" * 40)
        print("Summary:")
        if "visual_hook" in tags:
            vh = tags["visual_hook"]
            print(f"  Visual Hook: {vh.get('action', 'unknown')} | {vh.get('subject', 'unknown')} | {vh.get('environment', 'unknown')}")
        if "copy_structure" in tags:
            cs = tags["copy_structure"]
            print(f"  Copy Structure: {cs.get('framework', 'unknown')}")
        if "_metadata" in tags:
            meta = tags["_metadata"]
            print(f"  Frames: {meta.get('hook_frames', 0)} hook + {meta.get('context_frames', 0)} context")
            print(f"  Transcript: {'Yes' if meta.get('has_transcript') else 'No'}")
            print(f"  Airtable Context: {'Yes' if meta.get('has_airtable_context') else 'No'}")

        return 0

    except VideoProcessingError as e:
        print(f"Video processing error: {e}")
        if debug and e.video_path:
            print(f"  Video path: {e.video_path}")
        return 1

    except LLMError as e:
        print(f"LLM error: {e}")
        if debug and e.original_error:
            print(f"  Original error: {type(e.original_error).__name__}: {e.original_error}")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return 1


def main() -> None:
    """Main CLI entry point."""
    # Check for flags
    debug = "--debug" in sys.argv or "-d" in sys.argv
    no_transcript = "--no-transcript" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--debug", "-d", "--no-transcript")]

    # Default to TUI if no command given
    if len(args) < 1:
        sys.exit(run_tui())

    command = args[0]

    if command == "tui":
        sys.exit(run_tui())
    elif command == "validate-config":
        sys.exit(validate_config())
    elif command == "process":
        if len(args) < 2:
            print("Usage: python -m videotagger process <video_path> [--debug]")
            sys.exit(1)
        sys.exit(process_video_command(args[1], debug=debug))
    elif command == "analyze":
        # New command: analyze with Visual Hook + Copy Structure
        if len(args) < 2:
            print("Usage: python -m videotagger analyze <video_path> [art_id] [--no-transcript] [--debug]")
            print("\nExamples:")
            print("  python -m videotagger analyze video.mp4")
            print("  python -m videotagger analyze video.mp4 a1433")
            print("  python -m videotagger analyze video.mp4 --no-transcript")
            sys.exit(1)
        video_path = args[1]
        # Check if second arg is art_id (starts with 'a' followed by digits)
        art_id = None
        if len(args) >= 3 and args[2].startswith("a") and args[2][1:].isdigit():
            art_id = args[2]
        sys.exit(process_with_context_command(video_path, art_id=art_id, no_transcript=no_transcript, debug=debug))
    elif command == "audio":
        if len(args) < 2:
            print("Usage: python -m videotagger audio <video_path> [--debug]")
            sys.exit(1)
        sys.exit(analyze_audio_command(args[1], debug=debug))
    elif command in ["--help", "-h"]:
        print("Usage: python -m videotagger [command] [args] [--debug]")
        print("\nCommands:")
        print("  tui                       Launch interactive TUI (default)")
        print("  validate-config           Validate configuration and display status")
        print("  analyze <video> [art_id]  NEW: Visual Hook + Copy Structure analysis")
        print("  process <video_path>      Legacy: Process a video (basic tagging)")
        print("  audio <video_path>        Analyze audio only (local, no GPU needed)")
        print("\nOptions:")
        print("  --debug, -d               Enable debug logging")
        print("  --no-transcript           Skip audio transcription (faster)")
        print("\nExamples:")
        print("  python -m videotagger analyze my_video.mp4")
        print("  python -m videotagger analyze my_video.mp4 a1433  # with Airtable context")
        print("  python -m videotagger analyze my_video.mp4 --no-transcript --debug")
        sys.exit(0)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: tui, validate-config, analyze, process, audio")
        print("Run with --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
