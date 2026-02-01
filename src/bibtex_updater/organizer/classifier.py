"""Classifier factory for creating backend instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bibtex_updater.organizer.backends.base import AbstractClassifier
from bibtex_updater.organizer.config import ClassifierConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Supported backend names
BACKENDS = ("claude", "openai", "embedding")


class ClassifierFactory:
    """Factory for creating classifier backend instances.

    Supported backends:
    - "claude": Anthropic Claude API (requires ANTHROPIC_API_KEY)
    - "openai": OpenAI API (requires OPENAI_API_KEY)
    - "embedding": Local sentence-transformers (requires sentence-transformers package)
    """

    @staticmethod
    def create(config: ClassifierConfig) -> AbstractClassifier:
        """Create a classifier instance from config.

        Args:
            config: ClassifierConfig specifying backend and settings

        Returns:
            AbstractClassifier instance

        Raises:
            ValueError: If backend is not supported
            ImportError: If required dependencies are not installed
        """
        backend = config.backend.lower()

        if backend == "claude":
            from bibtex_updater.organizer.backends.claude_backend import ClaudeClassifier

            return ClaudeClassifier(config)

        elif backend == "openai":
            from bibtex_updater.organizer.backends.openai_backend import OpenAIClassifier

            return OpenAIClassifier(config)

        elif backend == "embedding":
            from bibtex_updater.organizer.backends.embedding_backend import EmbeddingClassifier

            return EmbeddingClassifier(config)

        else:
            raise ValueError(
                f"Unsupported classifier backend: {backend}. " f"Supported backends: {', '.join(BACKENDS)}"
            )

    @staticmethod
    def available_backends() -> list[str]:
        """Get list of available backends.

        Returns:
            List of backend names that can be instantiated
        """
        available = []

        # Check Claude (just needs httpx which is a core dep)
        try:
            import httpx  # noqa: F401

            available.append("claude")
        except ImportError:
            pass

        # Check OpenAI (just needs httpx which is a core dep)
        try:
            import httpx  # noqa: F401

            available.append("openai")
        except ImportError:
            pass

        # Check embedding
        try:
            import sentence_transformers  # noqa: F401

            available.append("embedding")
        except ImportError:
            pass

        return available

    @staticmethod
    def check_backend(backend: str) -> tuple[bool, str]:
        """Check if a backend can be used.

        Args:
            backend: Backend name to check

        Returns:
            Tuple of (available, message)
        """
        backend = backend.lower()

        if backend not in BACKENDS:
            return False, f"Unknown backend: {backend}"

        if backend == "claude":
            try:
                import os

                import httpx  # noqa: F401

                if not os.environ.get("ANTHROPIC_API_KEY"):
                    return False, "ANTHROPIC_API_KEY environment variable not set"
                return True, "Claude backend available"
            except ImportError:
                return False, "httpx not installed"

        elif backend == "openai":
            try:
                import os

                import httpx  # noqa: F401

                if not os.environ.get("OPENAI_API_KEY"):
                    return False, "OPENAI_API_KEY environment variable not set"
                return True, "OpenAI backend available"
            except ImportError:
                return False, "httpx not installed"

        elif backend == "embedding":
            try:
                import sentence_transformers  # noqa: F401

                return True, "Embedding backend available"
            except ImportError:
                return False, "sentence-transformers not installed"

        return False, f"Unknown backend: {backend}"
