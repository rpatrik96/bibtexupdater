"""Local embedding-based classifier using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Any

from bibtex_updater.organizer.backends.base import (
    AbstractClassifier,
    ClassificationResult,
    TopicPrediction,
)
from bibtex_updater.organizer.config import ClassifierConfig

logger = logging.getLogger(__name__)


class EmbeddingClassifier(AbstractClassifier):
    """Classifier using local sentence-transformer embeddings.

    This classifier computes embeddings for paper titles/abstracts and
    compares them to existing collection names/descriptions using
    cosine similarity. It requires no API calls and is free to use.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        """Initialize the embedding classifier.

        Args:
            config: ClassifierConfig with model name

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        super().__init__(config)
        self.model_name = config.get_model()
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError as err:
            raise ImportError(
                "sentence-transformers required for embedding backend. "
                "Install with: pip install sentence-transformers"
            ) from err

    def classify(
        self,
        title: str,
        abstract: str | None,
        existing_topics: list[dict[str, Any]],
        taxonomy: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        """Classify a paper using embedding similarity.

        Args:
            title: Paper title
            abstract: Paper abstract (may be None)
            existing_topics: List of existing Zotero collections
            taxonomy: Optional taxonomy (used for enhanced topic descriptions)

        Returns:
            ClassificationResult with predicted topics
        """
        if not existing_topics:
            return ClassificationResult(reasoning="No existing topics to match against")

        # Build paper text
        paper_text = self._build_paper_text(title, abstract)

        # Build topic texts with optional taxonomy keywords
        topic_texts = self._build_topic_texts(existing_topics, taxonomy)

        # Compute embeddings
        try:
            paper_embedding = self._model.encode([paper_text], convert_to_tensor=True)
            topic_embeddings = self._model.encode([t["text"] for t in topic_texts], convert_to_tensor=True)

            # Compute cosine similarities
            from sentence_transformers import util

            similarities = util.cos_sim(paper_embedding, topic_embeddings)[0]

            # Convert to list for sorting
            scored_topics = []
            for i, topic_info in enumerate(topic_texts):
                score = float(similarities[i])
                scored_topics.append(
                    {
                        "topic": topic_info["topic"],
                        "score": score,
                    }
                )

            # Sort by score descending
            scored_topics.sort(key=lambda x: x["score"], reverse=True)

            # Build result
            return self._build_result(scored_topics, existing_topics)

        except Exception as e:
            logger.error(f"Embedding classification error: {e}")
            return ClassificationResult(reasoning=f"Error: {e}")

    def _build_paper_text(self, title: str, abstract: str | None) -> str:
        """Build text representation of a paper.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            Combined text for embedding
        """
        parts = [title]
        if abstract:
            # Use truncated abstract
            truncated = self._truncate_abstract(abstract, max_chars=1500)
            parts.append(truncated)
        return " ".join(parts)

    def _build_topic_texts(
        self,
        existing_topics: list[dict[str, Any]],
        taxonomy: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build text representations for topics.

        Args:
            existing_topics: List of existing Zotero collections
            taxonomy: Optional taxonomy with keywords

        Returns:
            List of dicts with "topic" and "text" keys
        """
        topic_texts = []
        taxonomy_topics = taxonomy.get("topics", {}) if taxonomy else {}

        for topic in existing_topics:
            name = topic.get("name", "")

            # Start with topic name
            text_parts = [name]

            # Add taxonomy keywords if available
            taxonomy_entry = self._find_taxonomy_entry(name, taxonomy_topics)
            if taxonomy_entry:
                keywords = taxonomy_entry.get("keywords", [])
                if keywords:
                    text_parts.extend(keywords)

            topic_texts.append(
                {
                    "topic": topic,
                    "text": " ".join(text_parts),
                }
            )

        return topic_texts

    def _find_taxonomy_entry(
        self,
        topic_name: str,
        taxonomy_topics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Find a taxonomy entry matching the topic name.

        Args:
            topic_name: Name of the topic to find
            taxonomy_topics: Taxonomy topics dict

        Returns:
            Taxonomy entry dict or None
        """
        topic_name_lower = topic_name.lower()

        def search_topics(topics: dict) -> dict | None:
            for topic_id, topic_data in topics.items():
                name = topic_data.get("name", topic_id).lower()
                if name == topic_name_lower or topic_id.lower() == topic_name_lower:
                    return topic_data
                # Search children
                children = topic_data.get("children", {})
                if children:
                    result = search_topics(children)
                    if result:
                        return result
            return None

        return search_topics(taxonomy_topics)

    def _build_result(
        self,
        scored_topics: list[dict[str, Any]],
        existing_topics: list[dict[str, Any]],
    ) -> ClassificationResult:
        """Build classification result from scored topics.

        Args:
            scored_topics: List of topics with scores, sorted descending
            existing_topics: Original topic list

        Returns:
            ClassificationResult
        """
        threshold = self.config.confidence_threshold
        max_topics = self.config.max_topics

        # Filter by threshold and limit
        qualified_topics = [t for t in scored_topics if t["score"] >= threshold][:max_topics]

        if not qualified_topics:
            # Return top topic even if below threshold, with note
            if scored_topics:
                top = scored_topics[0]
                topic = top["topic"]
                primary = TopicPrediction(
                    topic_id=topic.get("key", ""),
                    topic_name=topic.get("name", ""),
                    confidence=top["score"],
                    is_existing=True,
                    collection_key=topic.get("key"),
                )
                return ClassificationResult(
                    primary_topic=primary,
                    reasoning=f"Best match below threshold ({top['score']:.2f} < {threshold})",
                )
            return ClassificationResult(reasoning="No topics scored")

        # Build primary topic
        top = qualified_topics[0]
        topic = top["topic"]
        primary = TopicPrediction(
            topic_id=topic.get("key", ""),
            topic_name=topic.get("name", ""),
            confidence=top["score"],
            is_existing=True,
            collection_key=topic.get("key"),
        )

        # Build secondary topics
        secondary = []
        for item in qualified_topics[1:]:
            topic = item["topic"]
            secondary.append(
                TopicPrediction(
                    topic_id=topic.get("key", ""),
                    topic_name=topic.get("name", ""),
                    confidence=item["score"],
                    is_existing=True,
                    collection_key=topic.get("key"),
                )
            )

        # Build reasoning
        scores_str = ", ".join(f"{t['topic'].get('name', '')}: {t['score']:.2f}" for t in qualified_topics)
        reasoning = f"Embedding similarity scores: {scores_str}"

        return ClassificationResult(
            primary_topic=primary,
            secondary_topics=secondary,
            reasoning=reasoning,
        )

    def estimate_cost(self, num_papers: int, avg_abstract_length: int = 500) -> float:
        """Estimate cost (always 0 for local embeddings).

        Args:
            num_papers: Number of papers to classify
            avg_abstract_length: Average abstract length (unused)

        Returns:
            0.0 (local model, no API cost)
        """
        return 0.0
