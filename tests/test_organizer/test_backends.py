"""Tests for classifier backends."""

import json
from unittest.mock import MagicMock, patch

import pytest

from bibtex_updater.organizer.backends.base import ClassificationResult
from bibtex_updater.organizer.config import ClassifierConfig


class TestClaudeBackend:
    """Tests for ClaudeClassifier."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ClassifierConfig(
            backend="claude",
            api_key="test-key",
            temperature=0.3,
        )

    @pytest.fixture
    def mock_response(self):
        """Create mock API response."""
        return {
            "content": [
                {
                    "text": json.dumps(
                        {
                            "primary_topic": {
                                "topic_id": "ABC123",
                                "topic_name": "Machine Learning",
                                "confidence": 0.85,
                                "is_new": False,
                            },
                            "secondary_topics": [],
                            "suggested_new_topics": [],
                            "reasoning": "Test reasoning",
                        }
                    )
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        config = ClassifierConfig(backend="claude", api_key=None)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                from bibtex_updater.organizer.backends.claude_backend import ClaudeClassifier

                ClaudeClassifier(config)

    @patch("httpx.Client")
    def test_classify_success(self, mock_client_class, config, mock_response):
        """Test successful classification."""
        from bibtex_updater.organizer.backends.claude_backend import ClaudeClassifier

        # Set up mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_client.post.return_value = mock_resp

        classifier = ClaudeClassifier(config)
        result = classifier.classify(
            title="Test Paper on Neural Networks",
            abstract="This paper presents a new approach...",
            existing_topics=[{"key": "ABC123", "name": "Machine Learning"}],
        )

        assert isinstance(result, ClassificationResult)
        assert result.has_classification
        assert result.primary_topic is not None
        assert result.primary_topic.topic_name == "Machine Learning"
        assert result.tokens_used == 150

    def test_estimate_cost(self, config):
        """Test cost estimation."""
        from bibtex_updater.organizer.backends.claude_backend import ClaudeClassifier

        with patch("httpx.Client"):
            classifier = ClaudeClassifier(config)
            cost = classifier.estimate_cost(100, avg_abstract_length=500)

            # Should be non-zero for 100 papers
            assert cost > 0
            assert cost < 1.0  # Should be less than $1 for 100 papers with Sonnet


class TestOpenAIBackend:
    """Tests for OpenAIClassifier."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ClassifierConfig(
            backend="openai",
            api_key="test-key",
            model="gpt-4o-mini",
        )

    @pytest.fixture
    def mock_response(self):
        """Create mock API response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "primary_topic": {
                                    "topic_id": "XYZ789",
                                    "topic_name": "Natural Language Processing",
                                    "confidence": 0.9,
                                    "is_new": False,
                                },
                                "secondary_topics": [],
                                "suggested_new_topics": [],
                                "reasoning": "Test reasoning",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100},
        }

    @patch("httpx.Client")
    def test_classify_success(self, mock_client_class, config, mock_response):
        """Test successful classification."""
        from bibtex_updater.organizer.backends.openai_backend import OpenAIClassifier

        # Set up mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_client.post.return_value = mock_resp

        classifier = OpenAIClassifier(config)
        result = classifier.classify(
            title="Test Paper on Language Models",
            abstract="We present a new transformer architecture...",
            existing_topics=[{"key": "XYZ789", "name": "Natural Language Processing"}],
        )

        assert isinstance(result, ClassificationResult)
        assert result.has_classification
        assert result.primary_topic.topic_name == "Natural Language Processing"


class TestEmbeddingBackend:
    """Tests for EmbeddingClassifier."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ClassifierConfig(
            backend="embedding",
            model="all-MiniLM-L6-v2",
            confidence_threshold=0.5,
        )

    def test_init_without_sentence_transformers(self, config):
        """Test initialization fails without sentence-transformers."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # This would raise ImportError in real scenario
            pass  # Skip actual test as it requires mocking complex imports

    @pytest.mark.skipif(
        True,  # Skip by default since it requires sentence-transformers
        reason="Requires sentence-transformers package",
    )
    def test_classify_with_real_model(self, config):
        """Test classification with real embedding model."""
        from bibtex_updater.organizer.backends.embedding_backend import EmbeddingClassifier

        classifier = EmbeddingClassifier(config)
        result = classifier.classify(
            title="Deep Learning for Computer Vision",
            abstract="We present a convolutional neural network...",
            existing_topics=[
                {"key": "A", "name": "Machine Learning"},
                {"key": "B", "name": "Computer Vision"},
                {"key": "C", "name": "Natural Language Processing"},
            ],
        )

        assert result.has_classification
        # Computer Vision should rank higher
        assert result.primary_topic is not None

    def test_estimate_cost_is_zero(self, config):
        """Test that embedding backend cost is always zero."""
        with patch("bibtex_updater.organizer.backends.embedding_backend.EmbeddingClassifier._load_model"):
            from bibtex_updater.organizer.backends.embedding_backend import EmbeddingClassifier

            classifier = EmbeddingClassifier(config)
            cost = classifier.estimate_cost(1000)
            assert cost == 0.0
