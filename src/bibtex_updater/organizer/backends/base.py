"""Abstract base class and data structures for classifier backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopicPrediction:
    """A single topic prediction for a paper.

    Attributes:
        topic_id: Hierarchical topic ID (e.g., "ml/transformers")
        topic_name: Human-readable topic name (e.g., "Transformers")
        confidence: Confidence score from 0.0 to 1.0
        is_existing: Whether this maps to an existing Zotero collection
        collection_key: Zotero collection key if is_existing is True
        is_new: Whether this is a suggested new topic
        parent_topic: Parent topic ID for hierarchical topics
    """

    topic_id: str
    topic_name: str
    confidence: float
    is_existing: bool = False
    collection_key: str | None = None
    is_new: bool = False
    parent_topic: str | None = None

    def __post_init__(self) -> None:
        """Validate confidence score."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ClassificationResult:
    """Result of classifying a paper.

    Attributes:
        primary_topic: The most likely topic for the paper
        secondary_topics: Additional relevant topics
        suggested_new_topics: Topics not in existing collections
        reasoning: Explanation of the classification (for debugging)
        raw_response: Raw response from the backend (for debugging)
        tokens_used: Number of tokens used (for cost tracking)
        cached: Whether this result came from cache
    """

    primary_topic: TopicPrediction | None = None
    secondary_topics: list[TopicPrediction] = field(default_factory=list)
    suggested_new_topics: list[TopicPrediction] = field(default_factory=list)
    reasoning: str = ""
    raw_response: dict[str, Any] | None = None
    tokens_used: int = 0
    cached: bool = False

    @property
    def all_topics(self) -> list[TopicPrediction]:
        """Get all topics (primary + secondary), excluding suggested new ones."""
        topics = []
        if self.primary_topic:
            topics.append(self.primary_topic)
        topics.extend(self.secondary_topics)
        return topics

    @property
    def all_topic_ids(self) -> list[str]:
        """Get all topic IDs."""
        return [t.topic_id for t in self.all_topics]

    @property
    def has_classification(self) -> bool:
        """Check if any classification was made."""
        return self.primary_topic is not None

    @property
    def max_confidence(self) -> float:
        """Get the maximum confidence score across all topics."""
        if not self.has_classification:
            return 0.0
        confidences = [t.confidence for t in self.all_topics]
        return max(confidences) if confidences else 0.0


class AbstractClassifier(ABC):
    """Abstract base class for paper classifiers.

    Subclasses must implement:
    - classify(): Classify a paper given its title and abstract
    - estimate_cost(): Estimate API cost for a batch of papers
    """

    def __init__(self, config: Any) -> None:
        """Initialize the classifier.

        Args:
            config: ClassifierConfig instance
        """
        self.config = config

    @abstractmethod
    def classify(
        self,
        title: str,
        abstract: str | None,
        existing_topics: list[dict[str, Any]],
        taxonomy: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        """Classify a paper into topics.

        Args:
            title: Paper title
            abstract: Paper abstract (may be None or empty)
            existing_topics: List of existing Zotero collections with structure:
                [{"key": "ABC123", "name": "Machine Learning", "parent_key": None}, ...]
            taxonomy: Optional taxonomy dict with topic hierarchy and keywords

        Returns:
            ClassificationResult with predicted topics
        """
        ...

    @abstractmethod
    def estimate_cost(self, num_papers: int, avg_abstract_length: int = 500) -> float:
        """Estimate the cost of classifying a batch of papers.

        Args:
            num_papers: Number of papers to classify
            avg_abstract_length: Average abstract length in characters

        Returns:
            Estimated cost in USD
        """
        ...

    def _truncate_abstract(self, abstract: str | None, max_chars: int = 2000) -> str:
        """Truncate abstract to fit within context limits.

        Args:
            abstract: Paper abstract
            max_chars: Maximum characters to keep

        Returns:
            Truncated abstract or empty string if None
        """
        if not abstract:
            return ""
        if len(abstract) <= max_chars:
            return abstract
        # Truncate at word boundary
        truncated = abstract[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _format_existing_topics(self, topics: list[dict[str, Any]]) -> str:
        """Format existing topics for the prompt.

        Args:
            topics: List of topic dicts with "key" and "name" fields

        Returns:
            Formatted string listing topics
        """
        if not topics:
            return "No existing collections."
        lines = []
        for t in topics:
            name = t.get("name", "Unknown")
            key = t.get("key", "")
            parent = t.get("parent_key")
            if parent:
                lines.append(f"- {name} (key: {key}, parent: {parent})")
            else:
                lines.append(f"- {name} (key: {key})")
        return "\n".join(lines)

    def _parse_topic_from_response(
        self,
        topic_data: dict[str, Any],
        existing_topics: list[dict[str, Any]],
    ) -> TopicPrediction:
        """Parse a topic prediction from the response.

        Args:
            topic_data: Dict with topic_id, topic_name, confidence
            existing_topics: List of existing collections for mapping

        Returns:
            TopicPrediction instance
        """
        topic_id = topic_data.get("topic_id", "")
        topic_name = topic_data.get("topic_name", topic_id)
        confidence = float(topic_data.get("confidence", 0.5))
        is_new = topic_data.get("is_new", False)
        parent = topic_data.get("parent_topic")

        # Check if this maps to an existing collection
        collection_key = None
        is_existing = False
        for t in existing_topics:
            if t.get("key") == topic_id or t.get("name", "").lower() == topic_name.lower():
                collection_key = t.get("key")
                is_existing = True
                break

        return TopicPrediction(
            topic_id=topic_id,
            topic_name=topic_name,
            confidence=confidence,
            is_existing=is_existing,
            collection_key=collection_key,
            is_new=is_new,
            parent_topic=parent,
        )
