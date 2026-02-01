"""Zotero sync module for bibtex-update command.

This module provides functionality to sync preprint-to-published updates
from the bibtex-update command to a Zotero library. It matches bib entries
to Zotero items and applies the same updates.

Usage:
    bibtex-update input.bib -o output.bib --zotero

The sync is performed after bib processing completes, reusing the
PublishedRecord results (no duplicate API calls).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz.fuzz import token_sort_ratio

from bibtex_updater.utils import (
    PublishedRecord,
    authors_last_names,
    doi_normalize,
    extract_arxiv_id_from_text,
    jaccard_similarity,
    normalize_title_for_match,
)
from bibtex_updater.zotero import (
    is_zotero_preprint,
    published_record_to_zotero_update,
)


@dataclass
class ZoteroSyncResult:
    """Result of syncing a single bib entry to Zotero."""

    bib_key: str
    zotero_item_key: str | None
    action: str  # "updated", "no_match", "already_published", "error", "would_update"
    match_method: str | None  # "arxiv_id", "doi", "title_author"
    message: str | None = None


class ZoteroSyncer:
    """Syncs preprint-to-published updates to a Zotero library.

    This class matches bib entries that were upgraded by bibtex-update to
    corresponding Zotero items and applies the same metadata updates.

    Matching strategy (in priority order):
    1. arXiv ID - Match original preprint's arXiv ID with Zotero item's arXiv ID
    2. DOI - Match bib entry's original preprint DOI with Zotero item DOI
    3. Title+Author - Fuzzy match using token_sort_ratio + jaccard_similarity
    """

    # Matching thresholds (same as bibtex_updater)
    TITLE_THRESHOLD = 90  # token_sort_ratio score (0-100)
    COMBINED_THRESHOLD = 0.9  # 0.7 * title + 0.3 * author

    def __init__(
        self,
        library_id: str,
        api_key: str,
        library_type: str = "user",
        collection_key: str | None = None,
        dry_run: bool = False,
        logger: logging.Logger | None = None,
    ):
        """Initialize ZoteroSyncer.

        Args:
            library_id: Zotero library ID (user ID or group ID)
            api_key: Zotero API key with write permissions
            library_type: "user" or "group"
            collection_key: Optional collection key to limit sync scope
            dry_run: If True, preview changes without applying
            logger: Logger instance (creates one if not provided)
        """
        self.library_id = library_id
        self.api_key = api_key
        self.library_type = library_type
        self.collection_key = collection_key
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)

        # Lazy-loaded Zotero client and cached preprints
        self._zot = None
        self._preprints_cache: list[dict[str, Any]] | None = None

    @property
    def zot(self):
        """Lazy-load pyzotero client."""
        if self._zot is None:
            try:
                from pyzotero import zotero

                self._zot = zotero.Zotero(self.library_id, self.library_type, self.api_key)
            except ImportError as e:
                raise ImportError("pyzotero not installed. Run: pip install pyzotero") from e
        return self._zot

    def fetch_preprints(self, limit: int = 500) -> list[dict[str, Any]]:
        """Fetch preprint items from Zotero library.

        Args:
            limit: Maximum items to fetch

        Returns:
            List of Zotero items that are detected as preprints
        """
        if self._preprints_cache is not None:
            return self._preprints_cache

        params = {"limit": limit, "itemType": "journalArticle || preprint"}

        if self.collection_key:
            items = self.zot.collection_items(self.collection_key, **params)
        else:
            items = self.zot.items(**params)

        preprints = []
        for item in items:
            is_preprint, arxiv_id = is_zotero_preprint(item)
            if is_preprint:
                # Store arxiv_id in item for faster matching
                item["_arxiv_id"] = arxiv_id
                preprints.append(item)

        self.logger.info(f"Found {len(preprints)} preprint(s) in Zotero out of {len(items)} items")
        self._preprints_cache = preprints
        return preprints

    def _extract_arxiv_id_from_entry(self, entry: dict[str, Any]) -> str | None:
        """Extract arXiv ID from a bib entry."""
        # Check eprint field first
        if entry.get("eprint"):
            arxiv_id = extract_arxiv_id_from_text(entry["eprint"])
            if arxiv_id:
                return arxiv_id

        # Check URL
        if entry.get("url"):
            arxiv_id = extract_arxiv_id_from_text(entry["url"])
            if arxiv_id:
                return arxiv_id

        # Check journal field (often contains "arXiv:XXXX.XXXXX")
        if entry.get("journal"):
            arxiv_id = extract_arxiv_id_from_text(entry["journal"])
            if arxiv_id:
                return arxiv_id

        # Check note field
        if entry.get("note"):
            arxiv_id = extract_arxiv_id_from_text(entry["note"])
            if arxiv_id:
                return arxiv_id

        return None

    def _extract_zotero_arxiv_id(self, item: dict[str, Any]) -> str | None:
        """Extract arXiv ID from a Zotero item."""
        # Use cached value if available
        if item.get("_arxiv_id"):
            return item["_arxiv_id"]

        data = item.get("data", item)

        # Check URL
        url = data.get("url", "")
        if url:
            arxiv_id = extract_arxiv_id_from_text(url)
            if arxiv_id:
                return arxiv_id

        # Check extra field
        extra = data.get("extra", "")
        if extra:
            arxiv_id = extract_arxiv_id_from_text(extra)
            if arxiv_id:
                return arxiv_id

        return None

    def _extract_zotero_doi(self, item: dict[str, Any]) -> str | None:
        """Extract DOI from a Zotero item."""
        data = item.get("data", item)
        doi = data.get("DOI", "")
        return doi_normalize(doi) if doi else None

    def _compute_title_author_score(
        self, bib_entry: dict[str, Any], zotero_item: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Compute title and author similarity scores.

        Returns:
            Tuple of (title_score, author_score, combined_score)
        """
        data = zotero_item.get("data", zotero_item)

        # Title matching
        bib_title = normalize_title_for_match(bib_entry.get("title", ""))
        zot_title = normalize_title_for_match(data.get("title", ""))
        title_score = token_sort_ratio(bib_title, zot_title) / 100.0 if bib_title and zot_title else 0.0

        # Author matching (Jaccard on last names)
        bib_authors = authors_last_names(bib_entry.get("author", ""), limit=5)

        # Extract Zotero authors
        creators = data.get("creators", [])
        zot_authors = []
        for c in creators:
            if c.get("creatorType") == "author":
                last = c.get("lastName", "").strip().lower()
                if last:
                    zot_authors.append(last)

        author_score = jaccard_similarity(bib_authors, zot_authors)

        # Combined score (same as bibtex_updater matching)
        combined = 0.7 * title_score + 0.3 * author_score

        return title_score, author_score, combined

    def find_match(
        self,
        bib_entry: dict[str, Any],
        arxiv_id: str | None = None,
        preprint_doi: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Find matching Zotero item for a bib entry.

        Args:
            bib_entry: Original bib entry (before upgrade)
            arxiv_id: arXiv ID of the original preprint (if known)
            preprint_doi: DOI of the original preprint (if known)

        Returns:
            Tuple of (zotero_item, match_method) or (None, None) if no match
        """
        preprints = self.fetch_preprints()

        # Extract identifiers from bib entry if not provided
        if not arxiv_id:
            arxiv_id = self._extract_arxiv_id_from_entry(bib_entry)
        if not preprint_doi:
            preprint_doi = doi_normalize(bib_entry.get("doi"))

        # Strategy 1: Match by arXiv ID
        if arxiv_id:
            for item in preprints:
                zot_arxiv = self._extract_zotero_arxiv_id(item)
                if zot_arxiv and self._normalize_arxiv_id(zot_arxiv) == self._normalize_arxiv_id(arxiv_id):
                    return item, "arxiv_id"

        # Strategy 2: Match by DOI
        if preprint_doi:
            for item in preprints:
                zot_doi = self._extract_zotero_doi(item)
                if zot_doi and zot_doi == preprint_doi:
                    return item, "doi"

        # Strategy 3: Match by title + author (fuzzy)
        best_match = None
        best_score = 0.0
        for item in preprints:
            title_score, author_score, combined = self._compute_title_author_score(bib_entry, item)

            # Require title score >= 90% AND combined >= 0.9
            if title_score >= (self.TITLE_THRESHOLD / 100.0) and combined >= self.COMBINED_THRESHOLD:
                if combined > best_score:
                    best_score = combined
                    best_match = item

        if best_match:
            return best_match, "title_author"

        return None, None

    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize arXiv ID for comparison (remove version suffix)."""
        # Remove version suffix like v1, v2, etc.
        return re.sub(r"v\d+$", "", arxiv_id.lower().strip())

    def sync_update(
        self,
        zotero_item: dict[str, Any],
        record: PublishedRecord,
        bib_key: str,
        match_method: str,
    ) -> ZoteroSyncResult:
        """Apply update to a single Zotero item.

        Args:
            zotero_item: The Zotero item to update
            record: PublishedRecord with the published metadata
            bib_key: The bib entry key (for reporting)
            match_method: How the match was found

        Returns:
            ZoteroSyncResult with the outcome
        """
        data = zotero_item.get("data", zotero_item)
        item_key = data.get("key", "")

        try:
            # Prepare update payload
            update_payload = published_record_to_zotero_update(record, zotero_item)

            if self.dry_run:
                return ZoteroSyncResult(
                    bib_key=bib_key,
                    zotero_item_key=item_key,
                    action="would_update",
                    match_method=match_method,
                    message=f"Would update to {record.journal}",
                )

            # Apply update
            self._update_item_with_retry(update_payload)

            # Add tag to mark as processed
            self._add_tag(item_key, "preprint-upgraded")

            return ZoteroSyncResult(
                bib_key=bib_key,
                zotero_item_key=item_key,
                action="updated",
                match_method=match_method,
                message=f"Updated to {record.journal}",
            )

        except Exception as e:
            self.logger.exception(f"Error updating Zotero item {item_key}")
            return ZoteroSyncResult(
                bib_key=bib_key,
                zotero_item_key=item_key,
                action="error",
                match_method=match_method,
                message=str(e),
            )

    def _update_item_with_retry(self, update_payload: dict[str, Any]) -> None:
        """Update item with retry on version conflict."""
        try:
            from pyzotero import zotero as pyzotero_module

            try:
                self.zot.update_item(update_payload)
            except pyzotero_module.PreConditionFailed:
                # Refresh version and retry once
                key = update_payload["key"]
                fresh = self.zot.item(key)
                update_payload["version"] = fresh["data"]["version"]
                self.zot.update_item(update_payload)
        except ImportError as e:
            raise ImportError("pyzotero not installed. Run: pip install pyzotero") from e

    def _add_tag(self, item_key: str, tag_name: str) -> None:
        """Add a tag to an item."""
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
            self.logger.warning(f"Failed to add tag to {item_key}: {e}")

    def sync_batch(
        self,
        updates: list[tuple[dict[str, Any], str | None, PublishedRecord]],
    ) -> list[ZoteroSyncResult]:
        """Sync a batch of bib updates to Zotero.

        Args:
            updates: List of (original_bib_entry, arxiv_id, record) tuples
                    from entries that were upgraded by bibtex-update

        Returns:
            List of ZoteroSyncResult for each entry
        """
        results: list[ZoteroSyncResult] = []

        for i, (bib_entry, arxiv_id, record) in enumerate(updates, 1):
            bib_key = bib_entry.get("ID", f"entry_{i}")
            preprint_doi = doi_normalize(bib_entry.get("doi"))

            self.logger.debug(f"[{i}/{len(updates)}] Finding Zotero match for {bib_key}...")

            # Find matching Zotero item
            zotero_item, match_method = self.find_match(bib_entry, arxiv_id, preprint_doi)

            if not zotero_item:
                results.append(
                    ZoteroSyncResult(
                        bib_key=bib_key,
                        zotero_item_key=None,
                        action="no_match",
                        match_method=None,
                        message="No matching Zotero item found",
                    )
                )
                continue

            # Check if already published (not a preprint anymore)
            is_preprint, _ = is_zotero_preprint(zotero_item)
            if not is_preprint:
                item_key = zotero_item.get("data", zotero_item).get("key", "")
                results.append(
                    ZoteroSyncResult(
                        bib_key=bib_key,
                        zotero_item_key=item_key,
                        action="already_published",
                        match_method=match_method,
                        message="Zotero item is already published (not a preprint)",
                    )
                )
                continue

            # Apply update
            result = self.sync_update(zotero_item, record, bib_key, match_method)
            results.append(result)

            if result.action in ("updated", "would_update"):
                self.logger.info(f"  [{match_method}] {bib_key} -> Zotero:{result.zotero_item_key}")

        return results


def print_zotero_sync_summary(results: list[ZoteroSyncResult], logger: logging.Logger) -> None:
    """Print summary of Zotero sync results."""
    updated = [r for r in results if r.action in ("updated", "would_update")]
    no_match = [r for r in results if r.action == "no_match"]
    already_published = [r for r in results if r.action == "already_published"]
    errors = [r for r in results if r.action == "error"]

    logger.info("")
    logger.info("=" * 50)
    logger.info("ZOTERO SYNC SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Matched & updated:  {len(updated)}")
    logger.info(f"  No Zotero match:    {len(no_match)}")
    logger.info(f"  Already published:  {len(already_published)}")
    logger.info(f"  Errors:             {len(errors)}")

    if updated:
        logger.info("")
        logger.info("--- Updated ---")
        for r in updated:
            logger.info(f"  {r.bib_key} -> {r.zotero_item_key} ({r.match_method})")

    if errors:
        logger.info("")
        logger.info("--- Errors ---")
        for r in errors:
            logger.info(f"  {r.bib_key}: {r.message}")
