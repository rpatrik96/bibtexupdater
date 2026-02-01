"""Classifier backends for the Zotero Paper Organizer.

This package provides different AI backends for paper classification:
- claude: Anthropic Claude API
- openai: OpenAI API
- embedding: Local sentence-transformers embeddings
"""

from bibtex_updater.organizer.backends.base import (
    AbstractClassifier,
    ClassificationResult,
    TopicPrediction,
)

__all__ = [
    "AbstractClassifier",
    "ClassificationResult",
    "TopicPrediction",
]
