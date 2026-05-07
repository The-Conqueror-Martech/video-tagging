"""LLM client for video analysis using vision-language models."""

import json
import logging
from typing import Any

from openai import OpenAI

from videotagger.config import LLMConfig, get_settings
from videotagger.exceptions import LLMError

# Configure logger
logger = logging.getLogger(__name__)

# System prompt for video tagging
SYSTEM_PROMPT = (
    "You are a video content tagger. Your job is to extract metadata from video frames. "
    "You must output ONLY valid JSON text. Do not use Markdown code blocks. "
    "Do not explain your answer."
)

# User prompt template
USER_PROMPT = """Analyze this video for tagging. Output ONLY valid JSON.

Required fields:
1. "setting": Physical location (e.g., "Bedroom", "Gym").
2. "branded_items": List objects with visible branding. Format: [{"name": "Brand", "type": "product/app/franchise"}]
3. "copyright_markers": {"trademarked_characters": [], "brand_names": []}
4. "cta": List any URLs, codes, or action instructions visible on screen.
5. "key_text": Extract 3-5 short phrases (2-4 words) that represent the main value propositions, product features, or tangible items shown. Focus on what makes the content unique or valuable - include both physical objects and benefit statements.
6. "content_type": "promotional", "tutorial", "vlog", "review", or "entertainment".
7. "copyright_risk": "High" (trademarked IP), "Medium" (brand mentions), or "Low" (none).

Rules:
- For "trademarked_characters", identify any fictional characters visible or mentioned by name - from video games, movies, TV shows, comics, or other franchises. Look for character names in text overlays and visual appearances.
- For "cta", capture explicit calls to action, website URLs, or promotional codes.
- For "key_text", balance concrete nouns with benefit-driven phrases. Keep phrases concise and machine-learning friendly."""


def get_llm_client(config: LLMConfig | None = None, endpoint_override: str | None = None) -> OpenAI:
    """Get configured OpenAI client for vLLM.

    Args:
        config: Optional LLMConfig. If None, loads from Settings.
        endpoint_override: Optional endpoint URL to override config (for dynamic RunPod URLs).

    Returns:
        Configured OpenAI client.
    """
    if config is None:
        config = get_settings().llm

    base_url = endpoint_override or config.endpoint

    return OpenAI(
        base_url=base_url,
        api_key=config.api_key,
    )


def build_vision_messages(
    frames_base64: list[str],
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build messages array for vision model.

    Args:
        frames_base64: List of base64-encoded frame images.
        system_prompt: Optional custom system prompt. Defaults to SYSTEM_PROMPT.
        user_prompt: Optional custom user prompt. Defaults to USER_PROMPT.

    Returns:
        Messages array for OpenAI chat completion.
    """
    # Use provided prompts or fall back to defaults
    sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    usr_prompt = user_prompt if user_prompt is not None else USER_PROMPT

    # Build content array with text and images
    content: list[dict[str, Any]] = [{"type": "text", "text": usr_prompt}]

    for frame_b64 in frames_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                },
            }
        )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]


def parse_tags_response(response_text: str, strict: bool = False) -> dict[str, Any]:
    """Parse and validate the LLM response as JSON.

    Args:
        response_text: Raw text response from the LLM.
        strict: If True, require visual_hook field. Default False for backward compat.

    Returns:
        Parsed tags dictionary.

    Raises:
        LLMError: If response is not valid JSON or missing required fields.
    """
    # Clean up response (remove potential markdown code blocks)
    text = response_text.strip()

    # Handle "Thinking" models that output reasoning before JSON
    # Look for </think> tag and extract content after it
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Also try to find JSON object if there's text before it
    if not text.startswith("{"):
        # Try to find the first { and last }
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx : end_idx + 1]

    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're code block markers
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Failed to parse LLM response as JSON: {e}", e) from e

    # Validate required fields (accept either branded_items or items)
    required_fields = [
        "setting",
        "cta",
        "key_text",
        "content_type",
        "copyright_risk",
    ]
    missing = [f for f in required_fields if f not in data]

    # Check for items or branded_items
    if "items" not in data and "branded_items" not in data:
        missing.append("items")

    if missing:
        raise LLMError(f"LLM response missing required fields: {missing}")

    # Normalize: rename branded_items to items if present
    if "branded_items" in data and "items" not in data:
        data["items"] = data.pop("branded_items")

    # Ensure list fields are lists
    for field in ["items", "cta", "key_text"]:
        if not isinstance(data[field], list):
            data[field] = [data[field]] if data[field] else []

    # Handle copyright_markers - ensure it exists and is a dict
    if "copyright_markers" not in data:
        data["copyright_markers"] = {"trademarked_characters": [], "brand_names": []}
    elif not isinstance(data["copyright_markers"], dict):
        data["copyright_markers"] = {"trademarked_characters": [], "brand_names": []}
    else:
        # Ensure sub-fields are lists
        for subfield in ["trademarked_characters", "brand_names"]:
            if subfield not in data["copyright_markers"]:
                data["copyright_markers"][subfield] = []
            elif not isinstance(data["copyright_markers"][subfield], list):
                data["copyright_markers"][subfield] = []

    # Remove copyright_markers and copyright_risk if risk is not High
    risk = data.get("copyright_risk", "").lower()
    if risk != "high":
        data.pop("copyright_markers", None)
        data.pop("copyright_risk", None)

    # Handle visual_hook - new field for Visual Hook analysis
    if "visual_hook" not in data:
        data["visual_hook"] = {
            "action": "unknown",
            "subject": "unknown",
            "environment": "unknown",
        }
    elif isinstance(data["visual_hook"], dict):
        # Ensure all sub-fields exist
        for subfield in ["action", "subject", "environment"]:
            if subfield not in data["visual_hook"]:
                data["visual_hook"][subfield] = "unknown"
    else:
        # Invalid format, reset to default
        data["visual_hook"] = {
            "action": "unknown",
            "subject": "unknown",
            "environment": "unknown",
        }

    # Handle copy_structure - new field for transcript analysis
    if "copy_structure" in data:
        if isinstance(data["copy_structure"], dict):
            # Ensure framework exists
            if "framework" not in data["copy_structure"]:
                data["copy_structure"]["framework"] = "unknown"
            if "breakdown" not in data["copy_structure"]:
                data["copy_structure"]["breakdown"] = {}
            elif not isinstance(data["copy_structure"]["breakdown"], dict):
                data["copy_structure"]["breakdown"] = {}
        else:
            # Invalid format, reset to default
            data["copy_structure"] = {
                "framework": "unknown",
                "breakdown": {},
            }
    # Note: copy_structure is optional - only present when transcript was analyzed

    return data


def analyze_frames(
    frames_base64: list[str],
    config: LLMConfig | None = None,
    endpoint_override: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Analyze video frames using vision-language model.

    Args:
        frames_base64: List of base64-encoded frame images.
        config: Optional LLMConfig. If None, loads from Settings.
        endpoint_override: Optional endpoint URL to override config (for dynamic RunPod URLs).
        system_prompt: Optional custom system prompt for dynamic prompting.
        user_prompt: Optional custom user prompt for dynamic prompting.

    Returns:
        Dictionary with extracted tags.

    Raises:
        LLMError: If API call fails or response parsing fails.
    """
    if config is None:
        config = get_settings().llm

    endpoint = endpoint_override or config.endpoint
    logger.info(f"LLM endpoint: {endpoint}")
    logger.info(f"LLM model: {config.model}")
    logger.info(f"Number of frames: {len(frames_base64)}")

    client = get_llm_client(config, endpoint_override=endpoint_override)
    messages = build_vision_messages(
        frames_base64,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    logger.debug(f"Sending request to LLM with {len(messages)} messages")

    try:
        logger.info("Making LLM API call...")
        # Use higher max_tokens for "Thinking" models that output reasoning
        max_tokens = 4096 if "Thinking" in config.model else 2048

        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        logger.info("LLM API call successful")
        logger.debug(f"Response object: {response}")

        if not response.choices:
            logger.error("LLM returned empty response (no choices)")
            raise LLMError("LLM returned empty response")

        response_text = response.choices[0].message.content or ""
        logger.info(f"LLM response length: {len(response_text)} chars")
        logger.debug(f"LLM response text: {response_text[:500]}...")

        result = parse_tags_response(response_text)
        logger.info(f"Successfully parsed tags: {list(result.keys())}")
        return result

    except LLMError:
        raise
    except Exception as e:
        logger.exception(f"LLM API call failed: {e}")
        raise LLMError(f"LLM API call failed: {e}", e) from e
