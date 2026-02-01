"""Tests for CollectionManager."""

from unittest.mock import MagicMock

import pytest

from bibtex_updater.organizer.collection_manager import CollectionManager, CollectionMapping


class TestCollectionManager:
    """Tests for CollectionManager."""

    @pytest.fixture
    def mock_zotero(self):
        """Create mock Zotero client."""
        zot = MagicMock()
        zot.collections.return_value = [
            {
                "data": {
                    "key": "ABC123",
                    "name": "Machine Learning",
                    "parentCollection": False,
                    "version": 1,
                }
            },
            {
                "data": {
                    "key": "DEF456",
                    "name": "Transformers",
                    "parentCollection": "ABC123",
                    "version": 2,
                }
            },
            {
                "data": {
                    "key": "GHI789",
                    "name": "Computer Vision",
                    "parentCollection": False,
                    "version": 3,
                }
            },
        ]
        return zot

    @pytest.fixture
    def manager(self, mock_zotero):
        """Create CollectionManager with mock client."""
        return CollectionManager(mock_zotero, create_collections=True)

    def test_get_all_collections(self, manager, mock_zotero):
        """Test fetching all collections."""
        collections = manager.get_all_collections()

        assert len(collections) == 3
        assert "ABC123" in collections
        assert collections["ABC123"]["name"] == "Machine Learning"
        assert collections["DEF456"]["parent_key"] == "ABC123"
        mock_zotero.collections.assert_called_once()

    def test_get_all_collections_cached(self, manager, mock_zotero):
        """Test that collections are cached."""
        manager.get_all_collections()
        manager.get_all_collections()

        # Should only call API once due to caching
        mock_zotero.collections.assert_called_once()

    def test_get_all_collections_refresh(self, manager, mock_zotero):
        """Test cache refresh."""
        manager.get_all_collections()
        manager.get_all_collections(refresh=True)

        # Should call API twice when refresh=True
        assert mock_zotero.collections.call_count == 2

    def test_find_collection_by_name(self, manager):
        """Test finding collection by name."""
        key = manager.find_collection_by_name("Machine Learning")
        assert key == "ABC123"

    def test_find_collection_by_name_case_insensitive(self, manager):
        """Test case-insensitive name matching."""
        key = manager.find_collection_by_name("machine learning")
        assert key == "ABC123"

    def test_find_collection_by_name_not_found(self, manager):
        """Test finding non-existent collection."""
        key = manager.find_collection_by_name("Nonexistent")
        assert key is None

    def test_find_collection_by_name_with_parent(self, manager):
        """Test finding collection with parent filter."""
        key = manager.find_collection_by_name("Transformers", parent_key="ABC123")
        assert key == "DEF456"

        # Should not find with wrong parent
        key = manager.find_collection_by_name("Transformers", parent_key="GHI789")
        assert key is None

    def test_create_collection(self, manager, mock_zotero):
        """Test creating a new collection."""
        mock_zotero.create_collections.return_value = {
            "successful": {"0": {"key": "NEW123"}},
            "failed": {},
        }

        key = manager.create_collection("New Topic")

        assert key == "NEW123"
        mock_zotero.create_collections.assert_called_once_with([{"name": "New Topic"}])

    def test_create_collection_with_parent(self, manager, mock_zotero):
        """Test creating collection with parent."""
        mock_zotero.create_collections.return_value = {
            "successful": {"0": {"key": "NEW456"}},
            "failed": {},
        }

        key = manager.create_collection("Sub Topic", parent_key="ABC123")

        assert key == "NEW456"
        mock_zotero.create_collections.assert_called_once_with([{"name": "Sub Topic", "parentCollection": "ABC123"}])

    def test_create_collection_disabled(self, mock_zotero):
        """Test that collection creation can be disabled."""
        manager = CollectionManager(mock_zotero, create_collections=False)

        with pytest.raises(PermissionError, match="disabled"):
            manager.create_collection("New Topic")

    def test_add_item_to_collection(self, manager, mock_zotero):
        """Test adding item to collection."""
        mock_zotero.item.return_value = {
            "data": {
                "key": "ITEM001",
                "version": 5,
                "collections": ["ABC123"],
            }
        }

        result = manager.add_item_to_collection("ITEM001", "DEF456")

        assert result is True
        mock_zotero.update_item.assert_called_once()
        call_args = mock_zotero.update_item.call_args[0][0]
        assert "DEF456" in call_args["collections"]
        assert "ABC123" in call_args["collections"]

    def test_add_item_already_in_collection(self, manager, mock_zotero):
        """Test adding item that's already in collection."""
        mock_zotero.item.return_value = {
            "data": {
                "key": "ITEM001",
                "version": 5,
                "collections": ["ABC123"],
            }
        }

        result = manager.add_item_to_collection("ITEM001", "ABC123")

        assert result is True
        # Should not call update_item since already in collection
        mock_zotero.update_item.assert_not_called()

    def test_map_topic_to_existing_collection(self, manager):
        """Test mapping topic to existing collection."""
        mapping = manager.map_topic_to_collection(
            topic_id="ml",
            topic_name="Machine Learning",
        )

        assert isinstance(mapping, CollectionMapping)
        assert mapping.collection_key == "ABC123"
        assert mapping.created is False

    def test_map_topic_creates_new_collection(self, manager, mock_zotero):
        """Test mapping topic creates new collection."""
        mock_zotero.create_collections.return_value = {
            "successful": {"0": {"key": "NEWCOL"}},
            "failed": {},
        }

        mapping = manager.map_topic_to_collection(
            topic_id="rl",
            topic_name="Reinforcement Learning",
        )

        assert mapping.collection_key == "NEWCOL"
        assert mapping.created is True

    def test_map_topic_by_key(self, manager):
        """Test mapping when topic_id is a collection key."""
        mapping = manager.map_topic_to_collection(
            topic_id="ABC123",
            topic_name="Any Name",
        )

        assert mapping.collection_key == "ABC123"
        assert mapping.created is False

    def test_create_collection_path(self, manager, mock_zotero):
        """Test creating hierarchical collection path."""
        # First level exists
        mock_zotero.create_collections.return_value = {
            "successful": {"0": {"key": "LEVEL2"}},
            "failed": {},
        }

        key = manager.create_collection_path("Machine Learning/Deep Learning")

        # Should create "Deep Learning" under "Machine Learning"
        assert key == "LEVEL2"
        call_args = mock_zotero.create_collections.call_args[0][0]
        assert call_args[0]["name"] == "Deep Learning"
        assert call_args[0]["parentCollection"] == "ABC123"
