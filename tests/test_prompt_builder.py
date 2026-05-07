"""Tests for dynamic prompt builder."""

import pytest

from videotagger.airtable import ArtContext
from videotagger.prompt_builder import build_dynamic_prompt, get_static_prompts


class TestBuildDynamicPrompt:
    """Tests for building dynamic prompts."""

    def test_returns_base_prompts_without_context(self) -> None:
        """Test that base prompts are returned when no context provided."""
        system_prompt, user_prompt = build_dynamic_prompt()

        assert "video content tagger" in system_prompt
        assert "visual_hook" in user_prompt
        assert "action" in user_prompt
        assert "subject" in user_prompt
        assert "environment" in user_prompt

    def test_includes_airtable_context(self) -> None:
        """Test that Airtable context is injected into system prompt."""
        context = ArtContext(
            product="Test Product",
            testing_concept="Hand medal",
            visual_category="Action",
        )

        system_prompt, user_prompt = build_dynamic_prompt(art_context=context)

        assert "Test Product" in system_prompt
        assert "Hand medal" in system_prompt
        assert "Action" in system_prompt

    def test_includes_template_guidance(self) -> None:
        """Test that template-specific guidance is added for known templates."""
        context = ArtContext(testing_concept="Hand medal")

        system_prompt, user_prompt = build_dynamic_prompt(art_context=context)

        assert "hand movements" in user_prompt.lower() or "medal presentation" in user_prompt.lower()

    def test_includes_transcript_for_copy_structure(self) -> None:
        """Test that transcript is included for Copy Structure analysis."""
        transcript = "Are you struggling to stay motivated? This medal will change everything!"

        system_prompt, user_prompt = build_dynamic_prompt(transcript=transcript)

        assert "copy_structure" in user_prompt
        assert "PAS" in user_prompt
        assert "AIDA" in user_prompt
        assert transcript in user_prompt

    def test_no_copy_structure_without_transcript(self) -> None:
        """Test that copy_structure is not requested without transcript."""
        system_prompt, user_prompt = build_dynamic_prompt(transcript=None)

        # copy_structure should not be in the numbered requirements
        assert "9. \"copy_structure\"" not in user_prompt

    def test_static_prompts_backward_compatible(self) -> None:
        """Test that get_static_prompts returns valid prompts."""
        system_prompt, user_prompt = get_static_prompts()

        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
        assert "visual_hook" in user_prompt
