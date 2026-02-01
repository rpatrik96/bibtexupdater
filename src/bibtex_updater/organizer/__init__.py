"""Zotero Paper Organizer - AI-powered automatic paper classification and organization.

This module provides tools for automatically classifying research papers by topic
and organizing them into Zotero collections using AI backends (Claude, OpenAI, or
local embeddings).
"""

from bibtex_updater.organizer.config import ClassifierConfig, OrganizerConfig

__all__ = [
    "ClassifierConfig",
    "OrganizerConfig",
]
