"""Taxonomy handling for hierarchical topic classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Topic:
    """A topic in the taxonomy hierarchy.

    Attributes:
        id: Unique topic identifier (e.g., "ml/transformers")
        name: Human-readable name
        keywords: Keywords associated with this topic
        children: Child topics
        parent_id: Parent topic ID (None for root topics)
        collection_key: Mapped Zotero collection key (if any)
    """

    id: str
    name: str
    keywords: list[str] = field(default_factory=list)
    children: dict[str, Topic] = field(default_factory=dict)
    parent_id: str | None = None
    collection_key: str | None = None


class Taxonomy:
    """Manages a hierarchical taxonomy of research topics.

    The taxonomy can be loaded from a YAML file with structure:
    ```yaml
    topics:
      machine-learning:
        name: "Machine Learning"
        keywords: ["ML", "deep learning"]
        children:
          transformers:
            name: "Transformers"
            keywords: ["attention", "BERT", "GPT"]

    collection_mappings:
      "machine-learning/transformers": "ABC123"  # Zotero key
    ```
    """

    def __init__(self) -> None:
        """Initialize an empty taxonomy."""
        self.topics: dict[str, Topic] = {}
        self.collection_mappings: dict[str, str] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Taxonomy:
        """Load taxonomy from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Taxonomy instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        try:
            import yaml
        except ImportError as err:
            raise ImportError("PyYAML required for taxonomy support. Install with: pip install pyyaml") from err

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Invalid taxonomy format: expected dict")

        taxonomy = cls()
        taxonomy._load_from_dict(data)
        return taxonomy

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Taxonomy:
        """Load taxonomy from a dictionary.

        Args:
            data: Taxonomy data dict

        Returns:
            Taxonomy instance
        """
        taxonomy = cls()
        taxonomy._load_from_dict(data)
        return taxonomy

    def _load_from_dict(self, data: dict[str, Any]) -> None:
        """Load taxonomy from dictionary data.

        Args:
            data: Taxonomy data dict
        """
        # Load collection mappings
        self.collection_mappings = data.get("collection_mappings", {})

        # Load topics recursively
        topics_data = data.get("topics", {})
        for topic_id, topic_data in topics_data.items():
            topic = self._parse_topic(topic_id, topic_data, parent_id=None)
            self.topics[topic_id] = topic

    def _parse_topic(
        self,
        topic_id: str,
        data: dict[str, Any],
        parent_id: str | None,
    ) -> Topic:
        """Parse a topic from dictionary data.

        Args:
            topic_id: Topic identifier
            data: Topic data dict
            parent_id: Parent topic ID

        Returns:
            Topic instance
        """
        full_id = f"{parent_id}/{topic_id}" if parent_id else topic_id

        topic = Topic(
            id=full_id,
            name=data.get("name", topic_id),
            keywords=data.get("keywords", []),
            parent_id=parent_id,
            collection_key=self.collection_mappings.get(full_id),
        )

        # Parse children
        children_data = data.get("children", {})
        for child_id, child_data in children_data.items():
            child = self._parse_topic(child_id, child_data, parent_id=full_id)
            topic.children[child_id] = child

        return topic

    def get_topic(self, topic_id: str) -> Topic | None:
        """Get a topic by ID.

        Args:
            topic_id: Topic ID (can be hierarchical like "ml/transformers")

        Returns:
            Topic if found, None otherwise
        """
        parts = topic_id.split("/")

        # Start at root
        current = self.topics.get(parts[0])
        if not current:
            return None

        # Navigate down hierarchy
        for part in parts[1:]:
            current = current.children.get(part)
            if not current:
                return None

        return current

    def get_all_topics_flat(self) -> list[Topic]:
        """Get all topics as a flat list.

        Returns:
            List of all topics (including nested ones)
        """
        result = []

        def collect(topic: Topic) -> None:
            result.append(topic)
            for child in topic.children.values():
                collect(child)

        for topic in self.topics.values():
            collect(topic)

        return result

    def get_topic_path(self, topic_id: str) -> list[str]:
        """Get the path of topic names from root to topic.

        Args:
            topic_id: Topic ID

        Returns:
            List of topic names in path order
        """
        parts = topic_id.split("/")
        path = []

        current_id = ""
        for part in parts:
            current_id = f"{current_id}/{part}" if current_id else part
            topic = self.get_topic(current_id)
            if topic:
                path.append(topic.name)
            else:
                path.append(part)

        return path

    def find_topics_by_keyword(self, keyword: str) -> list[Topic]:
        """Find topics that have a matching keyword.

        Args:
            keyword: Keyword to search for (case-insensitive)

        Returns:
            List of matching topics
        """
        keyword_lower = keyword.lower()
        matches = []

        for topic in self.get_all_topics_flat():
            for kw in topic.keywords:
                if keyword_lower in kw.lower():
                    matches.append(topic)
                    break

        return matches

    def to_dict(self) -> dict[str, Any]:
        """Convert taxonomy to dictionary (for serialization or classifier input).

        Returns:
            Dict representation of taxonomy
        """

        def topic_to_dict(topic: Topic) -> dict[str, Any]:
            result: dict[str, Any] = {"name": topic.name}
            if topic.keywords:
                result["keywords"] = topic.keywords
            if topic.children:
                result["children"] = {child_id: topic_to_dict(child) for child_id, child in topic.children.items()}
            return result

        topics_dict = {}
        for topic_id, topic in self.topics.items():
            # Use just the last part of ID as key at root level
            key = topic_id.split("/")[-1]
            topics_dict[key] = topic_to_dict(topic)

        result: dict[str, Any] = {"topics": topics_dict}
        if self.collection_mappings:
            result["collection_mappings"] = self.collection_mappings

        return result

    def save_yaml(self, path: str | Path) -> None:
        """Save taxonomy to a YAML file.

        Args:
            path: Path to save to
        """
        try:
            import yaml
        except ImportError as err:
            raise ImportError("PyYAML required. Install with: pip install pyyaml") from err

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Saved taxonomy to {path}")


def create_default_taxonomy() -> Taxonomy:
    """Create a default ML research taxonomy.

    Returns:
        Taxonomy with common ML research topics
    """
    data = {
        "topics": {
            "machine-learning": {
                "name": "Machine Learning",
                "keywords": ["ML", "deep learning", "neural network"],
                "children": {
                    "transformers": {
                        "name": "Transformers",
                        "keywords": ["attention", "BERT", "GPT", "LLM"],
                    },
                    "reinforcement-learning": {
                        "name": "Reinforcement Learning",
                        "keywords": ["RL", "policy", "reward", "agent"],
                    },
                    "generative-models": {
                        "name": "Generative Models",
                        "keywords": ["GAN", "VAE", "diffusion", "generative"],
                    },
                    "representation-learning": {
                        "name": "Representation Learning",
                        "keywords": [
                            "embedding",
                            "feature learning",
                            "self-supervised",
                        ],
                    },
                },
            },
            "computer-vision": {
                "name": "Computer Vision",
                "keywords": ["CV", "image", "video", "visual"],
                "children": {
                    "object-detection": {
                        "name": "Object Detection",
                        "keywords": ["YOLO", "detection", "bounding box"],
                    },
                    "segmentation": {
                        "name": "Segmentation",
                        "keywords": ["semantic", "instance", "mask"],
                    },
                },
            },
            "natural-language-processing": {
                "name": "Natural Language Processing",
                "keywords": ["NLP", "text", "language", "linguistic"],
                "children": {
                    "question-answering": {
                        "name": "Question Answering",
                        "keywords": ["QA", "reading comprehension"],
                    },
                    "machine-translation": {
                        "name": "Machine Translation",
                        "keywords": ["MT", "translation", "multilingual"],
                    },
                },
            },
            "optimization": {
                "name": "Optimization",
                "keywords": ["optimizer", "gradient", "convergence"],
            },
            "theory": {
                "name": "Theory",
                "keywords": ["theoretical", "analysis", "bounds", "complexity"],
            },
        }
    }

    return Taxonomy.from_dict(data)
