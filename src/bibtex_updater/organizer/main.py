"""Main orchestrator for the Zotero Paper Organizer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bibtex_updater.organizer.cache import ClassificationCache, ClassificationCacheEntry
from bibtex_updater.organizer.classifier import ClassifierFactory
from bibtex_updater.organizer.collection_manager import CollectionManager
from bibtex_updater.organizer.config import OrganizerConfig
from bibtex_updater.organizer.taxonomy import Taxonomy

if TYPE_CHECKING:
    from pyzotero import zotero

    from bibtex_updater.organizer.backends.base import AbstractClassifier

logger = logging.getLogger(__name__)


@dataclass
class OrganizeResult:
    """Result of organizing a single paper.

    Attributes:
        item_key: Zotero item key
        title: Paper title
        action: Action taken ("organized", "skipped", "error", "dry_run")
        message: Human-readable description
        topics: List of topic names the paper was assigned to
        collections_added: Collection keys the paper was added to
        collections_created: Collection keys that were created
        confidence: Classification confidence score
        cached: Whether result came from cache
    """

    item_key: str
    title: str
    action: str
    message: str
    topics: list[str] = field(default_factory=list)
    collections_added: list[str] = field(default_factory=list)
    collections_created: list[str] = field(default_factory=list)
    confidence: float = 0.0
    cached: bool = False


class ZoteroPaperOrganizer:
    """Orchestrates AI-powered paper organization in Zotero.

    This class coordinates:
    - Fetching items from Zotero
    - Classifying papers using AI backends
    - Mapping topics to collections
    - Adding papers to collections
    - Caching results
    """

    def __init__(self, config: OrganizerConfig) -> None:
        """Initialize the organizer.

        Args:
            config: OrganizerConfig with all settings

        Raises:
            ImportError: If pyzotero is not installed
            ValueError: If required config is missing
        """
        try:
            from pyzotero import zotero as pyzotero
        except ImportError as err:
            raise ImportError("pyzotero required. Install with: pip install pyzotero") from err

        if not config.library_id:
            raise ValueError("library_id required in config")
        if not config.api_key:
            raise ValueError("api_key required in config")

        self.config = config
        self.zot: zotero.Zotero = pyzotero.Zotero(
            config.library_id,
            config.library_type,
            config.api_key,
        )

        # Set up components
        self.classifier: AbstractClassifier = ClassifierFactory.create(config.classifier)
        self.collection_manager = CollectionManager(
            self.zot,
            create_collections=config.create_collections,
        )
        self.cache = ClassificationCache(config.cache_path)

        # Load taxonomy if specified
        self.taxonomy: Taxonomy | None = None
        if config.taxonomy_file:
            try:
                self.taxonomy = Taxonomy.from_yaml(config.taxonomy_file)
                logger.info(f"Loaded taxonomy from {config.taxonomy_file}")
            except Exception as e:
                logger.warning(f"Failed to load taxonomy: {e}")

        # Set up logging
        level = logging.DEBUG if config.verbose else logging.INFO
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    def fetch_items_to_process(
        self,
        tag: str | None = None,
        collection_key: str | None = None,
        item_keys: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch items to process from Zotero.

        Args:
            tag: Only fetch items with this tag
            collection_key: Only fetch items from this collection
            item_keys: Specific item keys to process
            limit: Maximum items to fetch
            offset: Skip first N items

        Returns:
            List of Zotero items
        """
        if item_keys:
            # Fetch specific items
            items = []
            for key in item_keys:
                try:
                    item = self.zot.item(key)
                    items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to fetch item {key}: {e}")
            return items

        # Build query params
        params: dict[str, Any] = {
            "limit": limit,
            "start": offset,
            "itemType": "-attachment && -note",  # Exclude attachments and notes
        }

        if tag:
            params["tag"] = tag

        if collection_key:
            items = self.zot.collection_items(collection_key, **params)
        else:
            items = self.zot.items(**params)

        logger.info(f"Fetched {len(items)} items")
        return items

    def process_item(self, item: dict[str, Any]) -> OrganizeResult:
        """Process a single Zotero item.

        Args:
            item: Zotero item dict

        Returns:
            OrganizeResult with processing outcome
        """
        data = item.get("data", item)
        item_key = data.get("key", "")
        title = data.get("title", "")[:80]
        abstract = data.get("abstractNote", "")

        # Check cache
        cached_entry = self.cache.get(item_key)
        if cached_entry:
            logger.debug(f"Cache hit for {item_key}")
            return self._apply_cached_result(item, cached_entry)

        # Get existing collections for classifier
        existing_topics = self.collection_manager.get_collections_list()

        # Classify the paper
        taxonomy_dict = self.taxonomy.to_dict() if self.taxonomy else None
        result = self.classifier.classify(
            title=title,
            abstract=abstract,
            existing_topics=existing_topics,
            taxonomy=taxonomy_dict,
        )

        if not result.has_classification:
            return OrganizeResult(
                item_key=item_key,
                title=title,
                action="skipped",
                message=f"No classification: {result.reasoning}",
            )

        # Check confidence threshold
        if result.max_confidence < self.config.classifier.confidence_threshold:
            return OrganizeResult(
                item_key=item_key,
                title=title,
                action="skipped",
                message=f"Low confidence ({result.max_confidence:.2f})",
                confidence=result.max_confidence,
            )

        # Process topics
        topics_to_assign = result.all_topics
        if self.config.create_collections:
            topics_to_assign.extend(result.suggested_new_topics)

        # Dry run mode
        if self.config.dry_run:
            topic_names = [t.topic_name for t in topics_to_assign]
            return OrganizeResult(
                item_key=item_key,
                title=title,
                action="dry_run",
                message=f"Would assign to: {', '.join(topic_names)}",
                topics=topic_names,
                confidence=result.max_confidence,
            )

        # Map topics to collections and add item
        return self._assign_to_collections(item, topics_to_assign, result)

    def _apply_cached_result(
        self,
        item: dict[str, Any],
        cached: ClassificationCacheEntry,
    ) -> OrganizeResult:
        """Apply a cached classification result.

        Args:
            item: Zotero item dict
            cached: Cached classification entry

        Returns:
            OrganizeResult
        """
        data = item.get("data", item)
        item_key = data.get("key", "")
        title = data.get("title", "")[:80]

        if self.config.dry_run:
            return OrganizeResult(
                item_key=item_key,
                title=title,
                action="dry_run",
                message=f"Would assign to: {', '.join(cached.topics)} (cached)",
                topics=cached.topics,
                confidence=cached.confidence,
                cached=True,
            )

        # Add to collections
        collections_added = []
        for topic_name in cached.topics:
            key = self.collection_manager.find_collection_by_name(topic_name)
            if key:
                if self.collection_manager.add_item_to_collection(item_key, key):
                    collections_added.append(key)

        # Add processed tag
        self._add_tag(item_key, self.config.processed_tag)

        return OrganizeResult(
            item_key=item_key,
            title=title,
            action="organized",
            message=f"Assigned to {len(collections_added)} collection(s) (cached)",
            topics=cached.topics,
            collections_added=collections_added,
            confidence=cached.confidence,
            cached=True,
        )

    def _assign_to_collections(
        self,
        item: dict[str, Any],
        topics: list,
        classification_result: Any,
    ) -> OrganizeResult:
        """Assign an item to collections based on topics.

        Args:
            item: Zotero item dict
            topics: List of TopicPrediction objects
            classification_result: Full classification result

        Returns:
            OrganizeResult
        """
        data = item.get("data", item)
        item_key = data.get("key", "")
        title = data.get("title", "")[:80]

        collections_added = []
        collections_created = []
        topic_names = []

        for topic in topics:
            try:
                # Map topic to collection
                mapping = self.collection_manager.map_topic_to_collection(
                    topic_id=topic.topic_id,
                    topic_name=topic.topic_name,
                    parent_topic_id=topic.parent_topic,
                )

                if mapping.created:
                    collections_created.append(mapping.collection_key)

                # Add item to collection
                if self.collection_manager.add_item_to_collection(item_key, mapping.collection_key):
                    collections_added.append(mapping.collection_key)
                    topic_names.append(topic.topic_name)

            except Exception as e:
                logger.warning(f"Failed to assign topic {topic.topic_name}: {e}")

        # Cache the result
        self.cache.set(
            ClassificationCacheEntry(
                item_key=item_key,
                title=title,
                topics=topic_names,
                confidence=classification_result.max_confidence,
                timestamp=time.time(),
                backend=self.config.classifier.backend,
            )
        )

        # Add processed tag
        self._add_tag(item_key, self.config.processed_tag)

        return OrganizeResult(
            item_key=item_key,
            title=title,
            action="organized",
            message=f"Assigned to {len(collections_added)} collection(s)",
            topics=topic_names,
            collections_added=collections_added,
            collections_created=collections_created,
            confidence=classification_result.max_confidence,
        )

    def _add_tag(self, item_key: str, tag_name: str) -> None:
        """Add a tag to an item.

        Args:
            item_key: Zotero item key
            tag_name: Tag to add
        """
        try:
            item = self.zot.item(item_key)
            data = item.get("data", item)
            tags = data.get("tags", [])

            if not any(t.get("tag") == tag_name for t in tags):
                tags.append({"tag": tag_name})
                self.zot.update_item(
                    {
                        "key": data["key"],
                        "version": data["version"],
                        "tags": tags,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to add tag to {item_key}: {e}")

    def run(
        self,
        tag: str | None = None,
        collection_key: str | None = None,
        item_keys: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[OrganizeResult]:
        """Run the organizer on items.

        Args:
            tag: Only process items with this tag
            collection_key: Only process items from this collection
            item_keys: Specific item keys to process
            limit: Maximum items to process
            offset: Skip first N items

        Returns:
            List of OrganizeResult objects
        """
        items = self.fetch_items_to_process(
            tag=tag,
            collection_key=collection_key,
            item_keys=item_keys,
            limit=limit,
            offset=offset,
        )

        results = []
        for i, item in enumerate(items, 1):
            data = item.get("data", item)
            title = data.get("title", "")[:50]
            logger.info(f"[{i}/{len(items)}] Processing: {title}...")

            try:
                result = self.process_item(item)
                results.append(result)

                # Log result
                if result.action == "organized":
                    logger.info(f"  + Organized: {', '.join(result.topics)}")
                elif result.action == "dry_run":
                    logger.info(f"  ~ Would assign: {', '.join(result.topics)}")
                elif result.action == "skipped":
                    logger.info(f"  - Skipped: {result.message}")
                elif result.action == "error":
                    logger.error(f"  ! Error: {result.message}")

            except Exception as e:
                logger.exception(f"Error processing item: {e}")
                results.append(
                    OrganizeResult(
                        item_key=data.get("key", ""),
                        title=title,
                        action="error",
                        message=str(e),
                    )
                )
                # Add error tag
                if not self.config.dry_run:
                    self._add_tag(data.get("key", ""), self.config.error_tag)

        return results

    def estimate_cost(self, num_papers: int, avg_abstract_length: int = 500) -> float:
        """Estimate the cost of organizing papers.

        Args:
            num_papers: Number of papers to process
            avg_abstract_length: Average abstract length in characters

        Returns:
            Estimated cost in USD
        """
        return self.classifier.estimate_cost(num_papers, avg_abstract_length)
