"""Tests for classifier base classes and data structures."""

from bibtex_updater.organizer.backends.base import (
    ClassificationResult,
    TopicPrediction,
)


class TestTopicPrediction:
    """Tests for TopicPrediction dataclass."""

    def test_basic_creation(self):
        """Test basic TopicPrediction creation."""
        topic = TopicPrediction(
            topic_id="ml/transformers",
            topic_name="Transformers",
            confidence=0.85,
        )
        assert topic.topic_id == "ml/transformers"
        assert topic.topic_name == "Transformers"
        assert topic.confidence == 0.85
        assert topic.is_existing is False
        assert topic.is_new is False

    def test_confidence_clamping(self):
        """Test that confidence is clamped to 0-1 range."""
        topic_high = TopicPrediction(
            topic_id="test",
            topic_name="Test",
            confidence=1.5,
        )
        assert topic_high.confidence == 1.0

        topic_low = TopicPrediction(
            topic_id="test",
            topic_name="Test",
            confidence=-0.5,
        )
        assert topic_low.confidence == 0.0

    def test_existing_topic(self):
        """Test topic marked as existing."""
        topic = TopicPrediction(
            topic_id="ABC123",
            topic_name="Machine Learning",
            confidence=0.9,
            is_existing=True,
            collection_key="ABC123",
        )
        assert topic.is_existing is True
        assert topic.collection_key == "ABC123"


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_empty_result(self):
        """Test empty classification result."""
        result = ClassificationResult()
        assert result.primary_topic is None
        assert result.secondary_topics == []
        assert result.has_classification is False
        assert result.max_confidence == 0.0
        assert result.all_topics == []

    def test_with_primary_topic(self):
        """Test result with primary topic."""
        primary = TopicPrediction(
            topic_id="ml",
            topic_name="Machine Learning",
            confidence=0.9,
        )
        result = ClassificationResult(primary_topic=primary)

        assert result.has_classification is True
        assert result.max_confidence == 0.9
        assert len(result.all_topics) == 1
        assert result.all_topic_ids == ["ml"]

    def test_with_secondary_topics(self):
        """Test result with primary and secondary topics."""
        primary = TopicPrediction(
            topic_id="ml",
            topic_name="Machine Learning",
            confidence=0.9,
        )
        secondary = [
            TopicPrediction(
                topic_id="nlp",
                topic_name="NLP",
                confidence=0.7,
            ),
            TopicPrediction(
                topic_id="cv",
                topic_name="Computer Vision",
                confidence=0.5,
            ),
        ]
        result = ClassificationResult(
            primary_topic=primary,
            secondary_topics=secondary,
        )

        assert len(result.all_topics) == 3
        assert result.all_topic_ids == ["ml", "nlp", "cv"]
        assert result.max_confidence == 0.9

    def test_suggested_new_topics_not_in_all(self):
        """Test that suggested_new_topics are separate from all_topics."""
        primary = TopicPrediction(
            topic_id="ml",
            topic_name="Machine Learning",
            confidence=0.9,
        )
        suggested = [
            TopicPrediction(
                topic_id="new/topic",
                topic_name="New Topic",
                confidence=0.8,
                is_new=True,
            ),
        ]
        result = ClassificationResult(
            primary_topic=primary,
            suggested_new_topics=suggested,
        )

        # suggested_new_topics should not be in all_topics
        assert len(result.all_topics) == 1
        assert "new/topic" not in result.all_topic_ids
        assert len(result.suggested_new_topics) == 1
