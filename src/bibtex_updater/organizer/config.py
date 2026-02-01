"""Configuration dataclasses for the Zotero Paper Organizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassifierConfig:
    """Configuration for the AI classifier backend.

    Attributes:
        backend: The classifier backend to use ("claude", "openai", "embedding")
        model: Model name/ID for the backend. Defaults vary by backend:
            - claude: "claude-sonnet-4-20250514"
            - openai: "gpt-4o-mini"
            - embedding: "all-MiniLM-L6-v2"
        api_key: API key for Claude/OpenAI backends. Can be None if using
            environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY)
        temperature: Sampling temperature for LLM backends (0.0-1.0)
        max_topics: Maximum number of topics to suggest per paper
        confidence_threshold: Minimum confidence score for accepting a classification
    """

    backend: str = "claude"
    model: str | None = None
    api_key: str | None = None
    temperature: float = 0.3
    max_topics: int = 3
    confidence_threshold: float = 0.7

    def get_model(self) -> str:
        """Get the model name, using backend-specific defaults if not set."""
        if self.model:
            return self.model
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-4o-mini",
            "embedding": "all-MiniLM-L6-v2",
        }
        return defaults.get(self.backend, "claude-sonnet-4-20250514")


@dataclass
class OrganizerConfig:
    """Configuration for the Zotero Paper Organizer.

    Attributes:
        classifier: ClassifierConfig for the AI backend
        library_id: Zotero library ID (user ID for personal libraries)
        api_key: Zotero API key with write permissions
        library_type: "user" or "group"
        taxonomy_file: Optional path to a YAML taxonomy file
        create_collections: Whether to create new collections for novel topics
        dry_run: Preview changes without modifying Zotero
        verbose: Enable verbose logging
        cache_path: Path to the classification cache file
        processed_tag: Tag to add after processing an item
        error_tag: Tag to add when processing fails
    """

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    library_id: str = ""
    api_key: str = ""
    library_type: str = "user"
    taxonomy_file: str | None = None
    create_collections: bool = True
    dry_run: bool = False
    verbose: bool = False
    cache_path: str | None = ".cache.organizer.json"
    processed_tag: str = "organized"
    error_tag: str = "organize-error"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrganizerConfig:
        """Create config from a dictionary (e.g., loaded from YAML)."""
        classifier_data = data.pop("classifier", {})
        classifier = ClassifierConfig(**classifier_data)
        return cls(classifier=classifier, **data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for serialization."""
        return {
            "classifier": {
                "backend": self.classifier.backend,
                "model": self.classifier.model,
                "temperature": self.classifier.temperature,
                "max_topics": self.classifier.max_topics,
                "confidence_threshold": self.classifier.confidence_threshold,
            },
            "library_id": self.library_id,
            "library_type": self.library_type,
            "taxonomy_file": self.taxonomy_file,
            "create_collections": self.create_collections,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
            "cache_path": self.cache_path,
            "processed_tag": self.processed_tag,
            "error_tag": self.error_tag,
        }
