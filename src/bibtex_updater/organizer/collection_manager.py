"""Zotero collection management for the Paper Organizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyzotero import zotero

logger = logging.getLogger(__name__)


@dataclass
class CollectionMapping:
    """Result of mapping a topic to a Zotero collection.

    Attributes:
        topic_id: The topic ID that was mapped
        topic_name: Human-readable topic name
        collection_key: Zotero collection key
        created: Whether the collection was newly created
        parent_key: Parent collection key (if hierarchical)
    """

    topic_id: str
    topic_name: str
    collection_key: str
    created: bool = False
    parent_key: str | None = None


class CollectionManager:
    """Manages Zotero collections for paper organization.

    Provides methods to:
    - List existing collections
    - Find collections by name
    - Create new collections (with hierarchy support)
    - Add items to collections
    - Map topics to collections
    """

    def __init__(
        self,
        zot: zotero.Zotero,
        create_collections: bool = True,
    ) -> None:
        """Initialize the collection manager.

        Args:
            zot: pyzotero Zotero instance
            create_collections: Whether to allow creating new collections
        """
        self.zot = zot
        self.create_collections = create_collections
        self._collections_cache: dict[str, dict[str, Any]] | None = None

    def get_all_collections(self, refresh: bool = False) -> dict[str, dict[str, Any]]:
        """Get all collections in the library.

        Args:
            refresh: Force refresh from API (bypass cache)

        Returns:
            Dict mapping collection key to collection data:
            {
                "ABC123": {
                    "key": "ABC123",
                    "name": "Machine Learning",
                    "parent_key": None,
                    "version": 123,
                }
            }
        """
        if self._collections_cache is not None and not refresh:
            return self._collections_cache

        collections = self.zot.collections()
        result = {}

        for coll in collections:
            data = coll.get("data", coll)
            key = data.get("key", "")
            result[key] = {
                "key": key,
                "name": data.get("name", ""),
                "parent_key": data.get("parentCollection") or None,
                "version": data.get("version", 0),
            }

        self._collections_cache = result
        logger.debug(f"Loaded {len(result)} collections")
        return result

    def get_collections_list(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Get collections as a list (for classifier input).

        Args:
            refresh: Force refresh from API

        Returns:
            List of collection dicts
        """
        collections = self.get_all_collections(refresh)
        return list(collections.values())

    def find_collection_by_name(
        self,
        name: str,
        parent_key: str | None = None,
    ) -> str | None:
        """Find a collection by name.

        Args:
            name: Collection name to find (case-insensitive)
            parent_key: Optional parent collection key to filter by

        Returns:
            Collection key if found, None otherwise
        """
        collections = self.get_all_collections()
        name_lower = name.lower()

        for key, coll in collections.items():
            if coll["name"].lower() == name_lower:
                # If parent_key specified, must match
                if parent_key is not None and coll["parent_key"] != parent_key:
                    continue
                return key

        return None

    def create_collection(
        self,
        name: str,
        parent_key: str | None = None,
    ) -> str:
        """Create a new collection.

        Args:
            name: Collection name
            parent_key: Optional parent collection key

        Returns:
            New collection key

        Raises:
            RuntimeError: If collection creation fails
            PermissionError: If create_collections is False
        """
        if not self.create_collections:
            raise PermissionError("Collection creation is disabled")

        payload = {"name": name}
        if parent_key:
            payload["parentCollection"] = parent_key

        try:
            result = self.zot.create_collections([payload])

            # Parse result - pyzotero returns dict with 'successful', 'failed', etc.
            if isinstance(result, dict):
                successful = result.get("successful", {})
                if successful:
                    # Get first successful creation
                    first_key = list(successful.keys())[0]
                    new_key = successful[first_key].get("key")
                    if new_key:
                        # Invalidate cache
                        self._collections_cache = None
                        logger.info(f"Created collection '{name}' with key {new_key}")
                        return new_key

                failed = result.get("failed", {})
                if failed:
                    first_error = list(failed.values())[0]
                    raise RuntimeError(f"Failed to create collection: {first_error}")

            raise RuntimeError(f"Unexpected response from create_collections: {result}")

        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise

    def create_collection_path(self, path: str) -> str:
        """Create a hierarchical collection path.

        Creates any missing collections in the path.

        Args:
            path: Slash-separated path like "ML/Transformers/BERT"

        Returns:
            Collection key of the deepest (last) collection
        """
        parts = [p.strip() for p in path.split("/") if p.strip()]
        if not parts:
            raise ValueError("Empty collection path")

        parent_key = None
        for part in parts:
            # Check if this level exists
            existing_key = self.find_collection_by_name(part, parent_key)
            if existing_key:
                parent_key = existing_key
            else:
                # Create it
                parent_key = self.create_collection(part, parent_key)

        return parent_key  # type: ignore[return-value]

    def add_item_to_collection(
        self,
        item_key: str,
        collection_key: str,
    ) -> bool:
        """Add an item to a collection.

        Args:
            item_key: Zotero item key
            collection_key: Collection key to add to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current item to check existing collections
            item = self.zot.item(item_key)
            data = item.get("data", item)

            current_collections = data.get("collections", [])
            if collection_key in current_collections:
                logger.debug(f"Item {item_key} already in collection {collection_key}")
                return True

            # Add to collections
            new_collections = current_collections + [collection_key]
            update_payload = {
                "key": data["key"],
                "version": data["version"],
                "collections": new_collections,
            }

            self.zot.update_item(update_payload)
            logger.debug(f"Added item {item_key} to collection {collection_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to add item {item_key} to collection {collection_key}: {e}")
            return False

    def map_topic_to_collection(
        self,
        topic_id: str,
        topic_name: str,
        parent_topic_id: str | None = None,
    ) -> CollectionMapping:
        """Map a topic to a Zotero collection.

        First tries to find an existing collection matching the topic name.
        If not found and create_collections is True, creates a new collection.

        Args:
            topic_id: Topic ID (may be hierarchical like "ml/transformers")
            topic_name: Human-readable topic name
            parent_topic_id: Optional parent topic ID (for hierarchy)

        Returns:
            CollectionMapping with the collection key

        Raises:
            ValueError: If collection not found and creation disabled
        """
        # First, try to find by topic_id (might be a collection key)
        collections = self.get_all_collections()
        if topic_id in collections:
            return CollectionMapping(
                topic_id=topic_id,
                topic_name=topic_name,
                collection_key=topic_id,
                created=False,
            )

        # Try to find by name
        existing_key = self.find_collection_by_name(topic_name)
        if existing_key:
            return CollectionMapping(
                topic_id=topic_id,
                topic_name=topic_name,
                collection_key=existing_key,
                created=False,
            )

        # Need to create
        if not self.create_collections:
            raise ValueError(f"Collection '{topic_name}' not found and creation disabled")

        # Determine parent collection
        parent_key = None
        if parent_topic_id:
            # Try to find parent by ID or name
            if parent_topic_id in collections:
                parent_key = parent_topic_id
            else:
                parent_key = self.find_collection_by_name(parent_topic_id)

        # Create the collection
        new_key = self.create_collection(topic_name, parent_key)

        return CollectionMapping(
            topic_id=topic_id,
            topic_name=topic_name,
            collection_key=new_key,
            created=True,
            parent_key=parent_key,
        )

    def get_item_collections(self, item_key: str) -> list[str]:
        """Get the collection keys an item belongs to.

        Args:
            item_key: Zotero item key

        Returns:
            List of collection keys
        """
        try:
            item = self.zot.item(item_key)
            data = item.get("data", item)
            return data.get("collections", [])
        except Exception as e:
            logger.error(f"Failed to get collections for item {item_key}: {e}")
            return []

    def is_item_in_collection(self, item_key: str, collection_key: str) -> bool:
        """Check if an item is in a specific collection.

        Args:
            item_key: Zotero item key
            collection_key: Collection key to check

        Returns:
            True if item is in collection
        """
        return collection_key in self.get_item_collections(item_key)
