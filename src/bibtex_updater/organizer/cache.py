"""Classification cache for the Zotero Paper Organizer.

Reuses the DiskCache pattern from bibtex_updater.utils for thread-safe
JSON-based caching of classification results.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ClassificationCacheEntry:
    """Cached classification result for a paper.

    Attributes:
        item_key: Zotero item key
        title: Paper title (for verification)
        topics: List of assigned topic IDs
        confidence: Overall classification confidence
        timestamp: Unix timestamp when cached
        backend: Classifier backend used
        ttl_days: Cache validity period in days
    """

    item_key: str
    title: str
    topics: list[str]
    confidence: float
    timestamp: float
    backend: str
    ttl_days: int = 30


class ClassificationCache:
    """Thread-safe on-disk cache for classification results.

    This cache stores classification results to avoid re-classifying papers
    that have already been processed. It supports TTL-based expiration.
    """

    def __init__(self, path: str | None, ttl_days: int = 30) -> None:
        """Initialize the classification cache.

        Args:
            path: Path to the cache file. If None, caching is disabled.
            ttl_days: Time-to-live in days for cache entries.
        """
        self.path = path
        self.ttl_days = ttl_days
        self.lock = threading.Lock()
        self.data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def _save(self) -> None:
        """Save cache to disk atomically."""
        if not self.path:
            return
        tmp = tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".json", prefix=".tmp_class_cache_"
        )
        try:
            json.dump(self.data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, self.path)

    def get(self, item_key: str) -> ClassificationCacheEntry | None:
        """Get cached classification result if valid.

        Args:
            item_key: Zotero item key

        Returns:
            ClassificationCacheEntry if a valid cached result exists, None otherwise.
        """
        if not self.path:
            return None
        with self.lock:
            if item_key not in self.data:
                return None
            entry_data = self.data[item_key]
            # Check TTL
            age_days = (time.time() - entry_data["timestamp"]) / 86400
            if age_days > entry_data.get("ttl_days", self.ttl_days):
                return None  # Expired
            return ClassificationCacheEntry(**entry_data)

    def set(self, entry: ClassificationCacheEntry) -> None:
        """Store classification result.

        Args:
            entry: The ClassificationCacheEntry to store.
        """
        if not self.path:
            return
        with self.lock:
            self.data[entry.item_key] = {
                "item_key": entry.item_key,
                "title": entry.title,
                "topics": entry.topics,
                "confidence": entry.confidence,
                "timestamp": entry.timestamp,
                "backend": entry.backend,
                "ttl_days": entry.ttl_days,
            }
            self._save()

    def invalidate(self, item_key: str) -> None:
        """Remove a cached entry.

        Args:
            item_key: Zotero item key to invalidate.
        """
        if not self.path:
            return
        with self.lock:
            if item_key in self.data:
                del self.data[item_key]
                self._save()

    def clear(self) -> None:
        """Clear all cached entries."""
        if not self.path:
            return
        with self.lock:
            self.data = {}
            self._save()

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        if not self.path:
            return 0
        removed = 0
        with self.lock:
            now = time.time()
            keys_to_remove = []
            for key, entry_data in self.data.items():
                age_days = (now - entry_data["timestamp"]) / 86400
                if age_days > entry_data.get("ttl_days", self.ttl_days):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.data[key]
                removed += 1
            if removed > 0:
                self._save()
        return removed
