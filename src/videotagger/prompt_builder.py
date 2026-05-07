"""Dynamic prompt builder for video tagging.

Constructs LLM prompts based on Airtable ART Grid context and optional transcript.
"""

from videotagger.airtable import ArtContext

# Base system prompt
BASE_SYSTEM_PROMPT = (
    "You are a video content tagger specializing in analyzing TikTok-style marketing videos. "
    "Your job is to extract structured metadata from video frames with special focus on the "
    "first 3 seconds (the Visual Hook). "
    "You must output ONLY valid JSON text. Do not use Markdown code blocks. "
    "Do not explain your answer. Be concise - minimize reasoning and output JSON directly."
)

# System prompt addition when context is available
CONTEXT_SYSTEM_ADDITION = """
You are analyzing a video with the following known context:
{context}

Use this context to guide your analysis and ensure tags are relevant to the video's category and purpose."""

# Base user prompt for visual analysis
BASE_USER_PROMPT = """Analyze this video for tagging. The first frames are from the "Visual Hook" (first 1.5 seconds) which is the most important part. Output ONLY valid JSON.

Required fields:

1. "visual_hook": Analysis of the PHYSICAL ACTION in the first 3 seconds. Object with:
   - "action": DETAILED SEQUENCE of physical actions in the hook (NOT text overlays). Describe the FULL MOTION from start to end. Examples: "Medal Enter Frame, Flip, Reveal Back", "Hand Grab Medal, Rotate, Show Front", "Box Open, Medal Lift, Display", "Person Walk In, Hold Medal Up". Use commas to separate each movement in the sequence. Be specific about the motion!
   - "subject": Main physical subject and camera perspective (e.g., "Hands + Medal - POV", "Medal Close-up", "Neck + Medal - 3rd Person", "Person Speaking", "Product on Table")
   - "environment": Physical setting/location (e.g., "Wooden Table", "Living Room", "Gym", "Outdoor", "Bathroom Mirror", "Studio")

2. "setting": Physical location (e.g., "Bedroom", "Gym", "Kitchen", "Office").

3. "items": List objects/products visible in the video. Format: [{{"name": "Item Name", "type": "product/app/franchise/object"}}]

4. "copyright_markers": {{"trademarked_characters": [], "brand_names": []}}

5. "cta": List any URLs, codes, or action instructions visible on screen.

6. "key_text": Extract 3-5 short phrases (2-4 words) that represent the main value propositions, product features, or tangible items shown.

7. "content_type": "promotional", "tutorial", "vlog", "review", or "entertainment".

8. "copyright_risk": "High" (trademarked IP), "Medium" (brand mentions), or "Low" (none).

Rules:
- ALL tags must be in ENGLISH regardless of video language.
- Focus extra attention on the first frames (Visual Hook).
- For "action", describe the FULL SEQUENCE of physical movements (e.g., "Medal Enter, Flip, Rotate, Reveal"). Watch how the object MOVES frame by frame!
- For "subject", describe the PHYSICAL OBJECT/PERSON shown and camera perspective.
- For "environment", describe the PHYSICAL LOCATION where the scene takes place.
- IMPORTANT: Text overlays are NOT actions. Look at what the hands/objects/people are physically doing frame by frame."""

# Addition when transcript is available
TRANSCRIPT_ADDITION = """

9. "copy_structure": Analysis of the script/voiceover structure. Object with:
   - "framework": The copywriting framework used. One of: "PAS", "BAB", "AIDA", "PPPP", "OCR", or "unknown"
   - "breakdown": Object with the framework stages and what content maps to each

Framework definitions:
- PAS: Problem, Agitate, Solution
- BAB: Before, After, Bridge
- AIDA: Attention, Interest, Desire, Action
- PPPP: Picture, Promise, Prove, Push
- OCR: One-time-offer, Call-to-action, Result

TRANSCRIPT OF VIDEO AUDIO:
\"\"\"
{transcript}
\"\"\"

Analyze the transcript above and classify which copywriting framework is used. Include a breakdown showing which parts of the transcript map to each stage of the framework."""

# Template-specific guidance
TEMPLATE_GUIDANCE = {
    "Hand medal": "Focus on PHYSICAL hand movements with medal - holding, flipping, rotating, presenting the medal.",
    "Medal Name Distance": "Look for medal being shown with name/distance visible. Note physical medal handling.",
    "Earn Medal": "Focus on medal reveal/presentation moments - hands receiving or displaying the medal.",
    "Unveil": "Look for PHYSICAL reveal moments - hands opening box, flipping medal, unboxing actions.",
    "Podcast": "Focus on face-to-face speaking, interview setup, and conversational framing.",
}


def build_dynamic_prompt(
    art_context: ArtContext | None = None,
    transcript: str | None = None,
) -> tuple[str, str]:
    """Build dynamic system and user prompts based on context.

    Args:
        art_context: Optional ArtContext from Airtable with video metadata.
        transcript: Optional transcript from speech-to-text for Copy Structure analysis.

    Returns:
        Tuple of (system_prompt, user_prompt) strings.
    """
    # Build system prompt
    system_prompt = BASE_SYSTEM_PROMPT

    if art_context and art_context.has_context():
        context_str = art_context.to_prompt_context()
        system_prompt += CONTEXT_SYSTEM_ADDITION.format(context=context_str)

    # Build user prompt
    user_prompt = BASE_USER_PROMPT

    # Add template-specific guidance if available
    if art_context and art_context.testing_concept:
        template = art_context.testing_concept
        if template in TEMPLATE_GUIDANCE:
            user_prompt += f"\n\nTEMPLATE GUIDANCE ({template}): {TEMPLATE_GUIDANCE[template]}"

    # Add transcript analysis if available
    if transcript and transcript.strip():
        user_prompt += TRANSCRIPT_ADDITION.format(transcript=transcript.strip())

    return system_prompt, user_prompt


def get_static_prompts() -> tuple[str, str]:
    """Get the static prompts for backward compatibility.

    Returns:
        Tuple of (system_prompt, user_prompt) without any dynamic context.
    """
    return build_dynamic_prompt(art_context=None, transcript=None)
