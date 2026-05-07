"""Airtable integration for VideoTagger.

Provides functions to find records by Art ID and update TagsKG column.
"""

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pyairtable import Api, Table
from pyairtable.api.types import RecordDict

from videotagger.config import AirtableConfig, get_settings
from videotagger.exceptions import (
    AirtableAPIError,
    ArtIdExtractionError,
    RecordNotFoundError,
)

logger = logging.getLogger(__name__)

# Regex pattern to extract Art ID from filename
# Matches 'a' followed by digits at end of filename before extension
ART_ID_PATTERN = re.compile(r"(a\d+)\.mp4$", re.IGNORECASE)


def extract_art_id(filename: str) -> str:
    """Extract Art ID from video filename.

    Args:
        filename: Video filename like "V - something a1433.mp4"

    Returns:
        Art ID string like "a1433"

    Raises:
        ArtIdExtractionError: If Art ID cannot be found in filename.
    """
    match = ART_ID_PATTERN.search(filename)
    if not match:
        raise ArtIdExtractionError(filename)
    return match.group(1).lower()


def get_airtable_table(config: AirtableConfig | None = None) -> Table:
    """Get configured Airtable Table instance.

    Args:
        config: Optional AirtableConfig. If None, loads from Settings.

    Returns:
        Configured pyairtable Table instance.
    """
    if config is None:
        config = get_settings().airtable

    api = Api(config.api_key)
    return api.table(config.base_id, config.table_id)


@lru_cache
def get_airtable_client() -> Table:
    """Get cached Airtable Table instance.

    Returns:
        Configured pyairtable Table instance.
    """
    return get_airtable_table()


def normalize_art_id(art_id: str) -> str:
    """Normalize Art ID to just the numeric part.

    Airtable stores Art ID as just the number (e.g., "7028"),
    but filenames use "a7028" format.

    Args:
        art_id: Art ID with or without 'a' prefix (e.g., "a7028" or "7028")

    Returns:
        Numeric Art ID without prefix (e.g., "7028")
    """
    art_id = art_id.strip().lower()
    if art_id.startswith("a"):
        return art_id[1:]
    return art_id


def find_by_art_id(art_id: str, table: Table | None = None) -> RecordDict:
    """Find an Airtable record by Art ID.

    Args:
        art_id: The Art ID to search for (e.g., "a1433" or "1433" - both work)
        table: Optional Table instance. If None, uses default client.

    Returns:
        Record dict with 'id', 'fields', and 'createdTime'.

    Raises:
        RecordNotFoundError: If no record found with the given Art ID.
        AirtableAPIError: If the API call fails.
    """
    if table is None:
        table = get_airtable_client()

    # Normalize to just the numeric ID
    numeric_id = normalize_art_id(art_id)

    try:
        # Use formula to find exact match on Art ID column
        formula = f"{{Art ID}} = '{numeric_id}'"
        record = table.first(formula=formula)

        if record is None:
            raise RecordNotFoundError(art_id)

        return record

    except RecordNotFoundError:
        raise
    except Exception as e:
        raise AirtableAPIError(f"Failed to find record: {e}", e) from e


def update_tags(art_id: str, tags: dict[str, Any], table: Table | None = None) -> RecordDict:
    """Update TagsKG column for a record identified by Art ID.

    Args:
        art_id: The Art ID of the record to update.
        tags: Dictionary of tags to store as JSON string.
        table: Optional Table instance. If None, uses default client.

    Returns:
        Updated record dict.

    Raises:
        RecordNotFoundError: If no record found with the given Art ID.
        AirtableAPIError: If the API call fails.
    """
    if table is None:
        table = get_airtable_client()

    # Find the record first
    record = find_by_art_id(art_id, table)

    try:
        # Serialize tags to JSON string
        tags_json = json.dumps(tags, ensure_ascii=False, indent=2)

        # Update the TagsKG field
        updated_record = table.update(record["id"], {"TagsKG": tags_json})
        return updated_record

    except Exception as e:
        raise AirtableAPIError(f"Failed to update record: {e}", e) from e


@dataclass
class ArtContext:
    """Context from Airtable ART Grid for dynamic prompt generation."""

    product: str | None = None
    testing_concept: str | None = None
    visual_category: str | None = None
    copy_category: str | None = None
    perspective: str | None = None
    angle: str | None = None
    copy_hook: str | None = None
    pitch: list[str] | None = None

    def to_prompt_context(self) -> str:
        """Convert context to a string for prompt injection.

        Returns:
            Formatted string describing the video context for LLM.
        """
        parts = []

        if self.product:
            parts.append(f"Product: {self.product}")
        if self.testing_concept:
            parts.append(f"Testing Concept/Template: {self.testing_concept}")
        if self.visual_category:
            parts.append(f"Visual Category: {self.visual_category}")
        if self.copy_category:
            parts.append(f"Copy Category: {self.copy_category}")
        if self.perspective:
            parts.append(f"Perspective: {self.perspective}")
        if self.angle:
            parts.append(f"Angle: {self.angle}")
        if self.copy_hook:
            parts.append(f"Copy Hook: {self.copy_hook}")
        if self.pitch:
            parts.append(f"Pitch: {', '.join(self.pitch)}")

        if not parts:
            return "No additional context available."

        return "\n".join(parts)

    def has_context(self) -> bool:
        """Check if any context fields are populated."""
        return any([
            self.product,
            self.testing_concept,
            self.visual_category,
            self.copy_category,
            self.perspective,
            self.angle,
            self.copy_hook,
            self.pitch,
        ])


# Cache for art context to avoid redundant API calls
_art_context_cache: dict[str, ArtContext] = {}


def get_art_context(art_id: str, table: Table | None = None) -> ArtContext:
    """Fetch ART Grid context for a video by Art ID.

    Args:
        art_id: The Art ID to look up (e.g., "a1433").
        table: Optional Table instance. If None, uses default client.

    Returns:
        ArtContext dataclass with available fields populated.

    Raises:
        RecordNotFoundError: If no record found with the given Art ID.
        AirtableAPIError: If the API call fails.
    """
    # Check cache first
    if art_id in _art_context_cache:
        logger.debug(f"Using cached context for {art_id}")
        return _art_context_cache[art_id]

    # Fetch record
    record = find_by_art_id(art_id, table)
    fields = record.get("fields", {})

    # Extract fields with safe defaults
    # Handle Pitch as multiple select (returns list)
    pitch_value = fields.get("Pitch")
    if isinstance(pitch_value, str):
        pitch = [pitch_value]
    elif isinstance(pitch_value, list):
        pitch = pitch_value
    else:
        pitch = None

    context = ArtContext(
        product=fields.get("Product"),
        testing_concept=fields.get("Testing Concept (Template)"),
        visual_category=fields.get("Visual Category"),
        copy_category=fields.get("Copy Category"),
        perspective=fields.get("Perspective"),
        angle=fields.get("Angle"),
        copy_hook=fields.get("Copy Hook"),
        pitch=pitch,
    )

    # Cache the result
    _art_context_cache[art_id] = context
    logger.info(f"Fetched context for {art_id}: {context.testing_concept or 'no template'}")

    return context


def clear_context_cache() -> None:
    """Clear the art context cache. Useful for batch processing resets."""
    global _art_context_cache
    _art_context_cache = {}
    logger.debug("Art context cache cleared")
