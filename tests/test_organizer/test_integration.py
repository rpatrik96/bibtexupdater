"""Integration tests for the Zotero Paper Organizer."""

from unittest.mock import MagicMock, patch

import pytest

from bibtex_updater.organizer.config import ClassifierConfig, OrganizerConfig
from bibtex_updater.organizer.main import OrganizeResult, ZoteroPaperOrganizer
from bibtex_updater.organizer.taxonomy import Taxonomy, create_default_taxonomy


class TestOrganizerConfig:
    """Tests for configuration classes."""

    def test_classifier_config_defaults(self):
        """Test ClassifierConfig default values."""
        config = ClassifierConfig()

        assert config.backend == "claude"
        assert config.temperature == 0.3
        assert config.max_topics == 3
        assert config.confidence_threshold == 0.7

    def test_classifier_config_get_model(self):
        """Test model name resolution."""
        config = ClassifierConfig(backend="claude")
        assert "claude" in config.get_model()

        config = ClassifierConfig(backend="openai")
        assert "gpt" in config.get_model()

        config = ClassifierConfig(backend="embedding")
        assert "MiniLM" in config.get_model()

        config = ClassifierConfig(backend="claude", model="custom-model")
        assert config.get_model() == "custom-model"

    def test_organizer_config_from_dict(self):
        """Test loading config from dict."""
        data = {
            "classifier": {
                "backend": "openai",
                "model": "gpt-4o",
            },
            "library_id": "12345",
            "api_key": "test-key",
            "taxonomy_file": "taxonomy.yaml",
        }

        config = OrganizerConfig.from_dict(data)

        assert config.classifier.backend == "openai"
        assert config.classifier.model == "gpt-4o"
        assert config.library_id == "12345"
        assert config.taxonomy_file == "taxonomy.yaml"


class TestTaxonomy:
    """Tests for Taxonomy class."""

    def test_create_default_taxonomy(self):
        """Test creating default ML taxonomy."""
        taxonomy = create_default_taxonomy()

        assert len(taxonomy.topics) > 0
        assert "machine-learning" in taxonomy.topics

        ml_topic = taxonomy.topics["machine-learning"]
        assert ml_topic.name == "Machine Learning"
        assert len(ml_topic.children) > 0
        assert "transformers" in ml_topic.children

    def test_taxonomy_from_dict(self):
        """Test loading taxonomy from dict."""
        data = {
            "topics": {
                "test-topic": {
                    "name": "Test Topic",
                    "keywords": ["test", "example"],
                    "children": {
                        "sub-topic": {
                            "name": "Sub Topic",
                            "keywords": ["sub"],
                        }
                    },
                }
            },
            "collection_mappings": {
                "test-topic": "ABC123",
            },
        }

        taxonomy = Taxonomy.from_dict(data)

        assert "test-topic" in taxonomy.topics
        topic = taxonomy.topics["test-topic"]
        assert topic.name == "Test Topic"
        assert "test" in topic.keywords
        assert "sub-topic" in topic.children

    def test_taxonomy_get_topic(self):
        """Test getting topic by ID."""
        taxonomy = create_default_taxonomy()

        # Root level
        topic = taxonomy.get_topic("machine-learning")
        assert topic is not None
        assert topic.name == "Machine Learning"

        # Nested
        topic = taxonomy.get_topic("machine-learning/transformers")
        assert topic is not None
        assert topic.name == "Transformers"

        # Non-existent
        topic = taxonomy.get_topic("nonexistent")
        assert topic is None

    def test_taxonomy_get_all_flat(self):
        """Test getting all topics as flat list."""
        taxonomy = create_default_taxonomy()
        all_topics = taxonomy.get_all_topics_flat()

        # Should include nested topics
        names = [t.name for t in all_topics]
        assert "Machine Learning" in names
        assert "Transformers" in names

    def test_taxonomy_to_dict(self):
        """Test converting taxonomy to dict."""
        taxonomy = create_default_taxonomy()
        data = taxonomy.to_dict()

        assert "topics" in data
        assert isinstance(data["topics"], dict)


class TestZoteroPaperOrganizer:
    """Tests for ZoteroPaperOrganizer."""

    @pytest.fixture
    def mock_zotero(self):
        """Create mock Zotero client."""
        zot = MagicMock()
        zot.collections.return_value = [
            {"data": {"key": "ML", "name": "Machine Learning", "parentCollection": False, "version": 1}},
            {"data": {"key": "NLP", "name": "NLP", "parentCollection": False, "version": 2}},
        ]
        zot.items.return_value = [
            {
                "data": {
                    "key": "ITEM1",
                    "title": "Attention Is All You Need",
                    "abstractNote": "We propose a new architecture based on attention mechanisms...",
                    "collections": [],
                    "tags": [],
                    "version": 1,
                }
            }
        ]
        return zot

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OrganizerConfig(
            classifier=ClassifierConfig(backend="claude", api_key="test-key"),
            library_id="12345",
            api_key="zotero-key",
            dry_run=True,
        )

    @patch("pyzotero.zotero.Zotero")
    @patch("bibtex_updater.organizer.classifier.ClassifierFactory.create")
    def test_organizer_init(self, mock_factory, mock_zotero_class, config):
        """Test organizer initialization."""
        mock_classifier = MagicMock()
        mock_factory.return_value = mock_classifier

        organizer = ZoteroPaperOrganizer(config)

        assert organizer.config == config
        mock_zotero_class.assert_called_once_with("12345", "user", "zotero-key")

    @patch("pyzotero.zotero.Zotero")
    @patch("bibtex_updater.organizer.classifier.ClassifierFactory.create")
    def test_fetch_items(self, mock_factory, mock_zotero_class, config, mock_zotero):
        """Test fetching items."""
        mock_zotero_class.return_value = mock_zotero
        mock_factory.return_value = MagicMock()

        organizer = ZoteroPaperOrganizer(config)
        items = organizer.fetch_items_to_process(tag="organize", limit=10)

        assert len(items) == 1
        mock_zotero.items.assert_called_once()

    @patch("pyzotero.zotero.Zotero")
    @patch("bibtex_updater.organizer.classifier.ClassifierFactory.create")
    def test_process_item_dry_run(self, mock_factory, mock_zotero_class, config, mock_zotero):
        """Test processing item in dry run mode."""
        from bibtex_updater.organizer.backends.base import ClassificationResult, TopicPrediction

        mock_zotero_class.return_value = mock_zotero

        # Mock classifier to return a classification
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = ClassificationResult(
            primary_topic=TopicPrediction(
                topic_id="ML",
                topic_name="Machine Learning",
                confidence=0.9,
                is_existing=True,
            ),
            reasoning="Test",
        )
        mock_factory.return_value = mock_classifier

        organizer = ZoteroPaperOrganizer(config)
        item = mock_zotero.items.return_value[0]
        result = organizer.process_item(item)

        assert isinstance(result, OrganizeResult)
        assert result.action == "dry_run"
        assert "Machine Learning" in result.topics
        # Should not modify Zotero in dry run
        mock_zotero.update_item.assert_not_called()

    @patch("pyzotero.zotero.Zotero")
    @patch("bibtex_updater.organizer.classifier.ClassifierFactory.create")
    def test_process_item_low_confidence(self, mock_factory, mock_zotero_class, config, mock_zotero):
        """Test that low confidence results are skipped."""
        from bibtex_updater.organizer.backends.base import ClassificationResult, TopicPrediction

        mock_zotero_class.return_value = mock_zotero

        # Mock classifier to return low confidence
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = ClassificationResult(
            primary_topic=TopicPrediction(
                topic_id="ML",
                topic_name="Machine Learning",
                confidence=0.3,  # Below threshold
            ),
        )
        mock_factory.return_value = mock_classifier

        organizer = ZoteroPaperOrganizer(config)
        item = mock_zotero.items.return_value[0]
        result = organizer.process_item(item)

        assert result.action == "skipped"
        assert "Low confidence" in result.message

    @patch("pyzotero.zotero.Zotero")
    @patch("bibtex_updater.organizer.classifier.ClassifierFactory.create")
    def test_estimate_cost(self, mock_factory, mock_zotero_class, config):
        """Test cost estimation."""
        mock_classifier = MagicMock()
        mock_classifier.estimate_cost.return_value = 0.05
        mock_factory.return_value = mock_classifier

        organizer = ZoteroPaperOrganizer(config)
        cost = organizer.estimate_cost(100)

        assert cost == 0.05
        mock_classifier.estimate_cost.assert_called_once_with(100, 500)
