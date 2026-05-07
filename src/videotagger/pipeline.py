"""Video processing pipeline."""

import logging
from pathlib import Path
from typing import Any

from videotagger.airtable import ArtContext, get_art_context
from videotagger.config import LLMConfig, get_settings
from videotagger.exceptions import RecordNotFoundError
from videotagger.llm import analyze_frames
from videotagger.prompt_builder import build_dynamic_prompt
from videotagger.video import extract_frames_as_base64, extract_weighted_frames_as_base64

logger = logging.getLogger(__name__)


def process_video(
    video_path: str | Path,
    config: LLMConfig | None = None,
) -> dict[str, Any]:
    """Process a video file and extract tags using vision-language model.

    Args:
        video_path: Path to the video file.
        config: Optional LLMConfig. If None, loads from Settings.

    Returns:
        Dictionary with extracted tags:
        - setting: str
        - branded_items: list
        - cta: list
        - key_text: list
        - content_type: str
        - visual_hook: dict

    Raises:
        VideoProcessingError: If frame extraction fails.
        LLMError: If LLM analysis fails.
    """
    if config is None:
        config = get_settings().llm

    # Extract frames as base64
    frames = extract_frames_as_base64(
        video_path,
        num_frames=config.frame_count,
        max_size=config.frame_max_size,
    )

    # Analyze with LLM
    tags = analyze_frames(frames, config)

    return tags


def process_video_with_context(
    video_path: str | Path,
    art_id: str | None = None,
    include_transcript: bool = True,
    config: LLMConfig | None = None,
    endpoint_override: str | None = None,
    auto_detect_pod: bool = True,
) -> dict[str, Any]:
    """Process a video with Airtable context and optional transcript analysis.

    This is the enhanced pipeline that:
    1. Auto-detects running RunPod with vLLM (if auto_detect_pod=True)
    2. Fetches Airtable ART Grid context (if art_id provided)
    3. Extracts weighted frames (dense for first 1.5s, sparse after)
    4. Optionally transcribes audio for Copy Structure analysis
    5. Builds dynamic prompt based on context
    6. Analyzes with LLM and returns structured tags

    Args:
        video_path: Path to the video file.
        art_id: Optional Art ID to fetch context from Airtable.
        include_transcript: Whether to transcribe audio for Copy Structure. Default True.
        config: Optional LLMConfig. If None, loads from Settings.
        endpoint_override: Optional endpoint URL override for LLM.
        auto_detect_pod: Auto-detect running RunPod vLLM endpoint. Default True.

    Returns:
        Dictionary with extracted tags including:
        - visual_hook: dict with action, subject, environment
        - copy_structure: dict with framework and breakdown (if transcript enabled)
        - setting, branded_items, cta, key_text, content_type, etc.

    Raises:
        VideoProcessingError: If frame extraction fails.
        LLMError: If LLM analysis fails.
    """
    if config is None:
        config = get_settings().llm

    video_path = Path(video_path)
    logger.info(f"Processing video with context: {video_path}")

    # Auto-detect running RunPod if no endpoint override provided
    if auto_detect_pod and endpoint_override is None:
        try:
            from videotagger.runpod_api import find_running_vllm_pod

            pod = find_running_vllm_pod()
            if pod:
                endpoint_override = pod.get_vllm_endpoint()
                logger.info(f"Auto-detected RunPod: {pod.name} -> {endpoint_override}")
            else:
                logger.warning("No running RunPod found, using config endpoint")
        except Exception as e:
            logger.warning(f"Failed to auto-detect RunPod: {e}, using config endpoint")

    # Step 1: Fetch Airtable context if art_id provided
    art_context: ArtContext | None = None
    if art_id:
        try:
            art_context = get_art_context(art_id)
            logger.info(f"Fetched context for {art_id}: template={art_context.testing_concept}")
        except RecordNotFoundError:
            logger.warning(f"No Airtable record found for {art_id}, proceeding without context")
        except Exception as e:
            logger.warning(f"Failed to fetch Airtable context: {e}, proceeding without context")

    # Step 2: Extract weighted frames (dense first 3s = 80% of frames, sparse after)
    # First 3 seconds: 12 frames at 0.25s intervals (0, 0.25, 0.5, ..., 2.75)
    # Remaining: 1-2 frames for context
    logger.info("Extracting weighted frames (dense 0.25s intervals for first 3 seconds)...")
    weighted_frames = extract_weighted_frames_as_base64(
        video_path,
        hook_interval=0.25,
        hook_duration=3.0,
        context_interval=4.0,  # Sparse context frames after hook
        max_size=config.frame_max_size,
    )

    # Combine frames: hook frames first, then context frames
    all_frames = weighted_frames.all_frames
    logger.info(
        f"Extracted {len(weighted_frames.hook_frames)} hook frames + "
        f"{len(weighted_frames.context_frames)} context frames"
    )

    # Step 3: Optionally transcribe audio
    transcript: str | None = None
    if include_transcript:
        try:
            from videotagger.transcribe import transcribe_video

            logger.info("Transcribing audio...")
            transcript = transcribe_video(video_path)
            if transcript:
                logger.info(f"Transcription complete: {len(transcript)} chars")
            else:
                logger.info("No speech detected in video")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}, proceeding without transcript")

    # Step 4: Build dynamic prompt
    system_prompt, user_prompt = build_dynamic_prompt(
        art_context=art_context,
        transcript=transcript,
    )

    # Step 5: Analyze with LLM
    logger.info("Analyzing frames with LLM...")
    tags = analyze_frames(
        all_frames,
        config=config,
        endpoint_override=endpoint_override,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Add metadata about processing
    tags["_metadata"] = {
        "hook_frames": len(weighted_frames.hook_frames),
        "context_frames": len(weighted_frames.context_frames),
        "has_transcript": bool(transcript),
        "has_airtable_context": art_context.has_context() if art_context else False,
        "art_id": art_id,
    }

    # Add Airtable context to output if available
    if art_context and art_context.has_context():
        tags["airtable_context"] = {
            "product": art_context.product,
            "template": art_context.testing_concept,
            "visual_category": art_context.visual_category,
            "copy_category": art_context.copy_category,
            "perspective": art_context.perspective,
            "angle": art_context.angle,
            "copy_hook": art_context.copy_hook,
            "pitch": art_context.pitch,
        }
        # Remove None values
        tags["airtable_context"] = {
            k: v for k, v in tags["airtable_context"].items() if v is not None
        }

    # Rename branded_items to items
    if "branded_items" in tags:
        tags["items"] = tags.pop("branded_items")

    # Add full transcript to copy_structure if available
    if transcript:
        if "copy_structure" not in tags:
            tags["copy_structure"] = {}
        tags["copy_structure"]["full_transcript"] = transcript

    logger.info(f"Processing complete. Tags: {list(tags.keys())}")
    return tags
