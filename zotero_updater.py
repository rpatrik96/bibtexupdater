#!/usr/bin/env python3
"""
zotero_updater.py — Update Zotero preprints using bibtex_updater resolution pipeline.

This script:
1. Fetches arXiv preprints from your Zotero library
2. Uses the bibtex_updater resolution pipeline to find published versions
3. Updates Zotero items with published metadata (preserving notes, tags, attachments)

Usage:
    python zotero_updater.py --dry-run              # Preview changes
    python zotero_updater.py                        # Apply updates
    python zotero_updater.py --tag "to-update"      # Only process items with specific tag
    python zotero_updater.py --collection ABCD1234  # Only process specific collection

Requirements:
    pip install pyzotero

Environment variables:
    ZOTERO_LIBRARY_ID - Your Zotero user ID (find at zotero.org/settings/keys)
    ZOTERO_API_KEY    - API key with write permissions
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Import from bibtex_updater (same directory)
from bibtex_updater import (
    Detector,
    DiskCache,
    HttpClient,
    PublishedRecord,
    RateLimiter,
    Resolver,
    doi_normalize,
    doi_url,
    extract_arxiv_id_from_text,
    safe_lower,
)

try:
    from pyzotero import zotero
except ImportError:
    print("Error: pyzotero not installed. Run: pip install pyzotero", file=sys.stderr)
    sys.exit(1)


# ------------- Zotero <-> BibTeX Bridge -------------


def zotero_to_bibtex_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Zotero item to bibtex-like dict for the Detector/Resolver."""
    data = item.get("data", item)

    # Extract authors
    creators = data.get("creators", [])
    author_parts = []
    for c in creators:
        if c.get("creatorType") == "author":
            if c.get("lastName") and c.get("firstName"):
                author_parts.append(f"{c['lastName']}, {c['firstName']}")
            elif c.get("name"):
                author_parts.append(c["name"])

    entry = {
        "ENTRYTYPE": "article",
        "ID": data.get("key", ""),
        "title": data.get("title", ""),
        "author": " and ".join(author_parts),
        "year": data.get("date", "")[:4] if data.get("date") else "",
        "journal": data.get("publicationTitle", ""),
        "doi": data.get("DOI", ""),
        "url": data.get("url", ""),
        "volume": data.get("volume", ""),
        "issue": data.get("issue", ""),
        "pages": data.get("pages", ""),
        "abstract": data.get("abstractNote", ""),
        # Store extra fields for arXiv detection
        "extra": data.get("extra", ""),
        "archiveID": data.get("archiveID", ""),
    }

    # Check extra field for arXiv ID (common pattern: "arXiv:2001.01234")
    extra = data.get("extra", "")
    if extra:
        arxiv_match = re.search(r"arXiv[:\s]+(\d{4}\.\d{4,5})", extra, re.IGNORECASE)
        if arxiv_match:
            entry["eprint"] = arxiv_match.group(1)
            entry["archiveprefix"] = "arxiv"

    return entry


def is_zotero_preprint(item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if a Zotero item is a preprint.
    Returns (is_preprint, arxiv_id).
    """
    data = item.get("data", item)

    journal = safe_lower(data.get("publicationTitle", ""))
    url = data.get("url", "")
    extra = data.get("extra", "")
    doi = data.get("DOI", "")

    arxiv_id = None

    # Check URL
    if url:
        arxiv_id = extract_arxiv_id_from_text(url)

    # Check extra field
    if not arxiv_id and extra:
        arxiv_id = extract_arxiv_id_from_text(extra)

    # Check journal field
    if "arxiv" in journal or "biorxiv" in journal or "medrxiv" in journal:
        if not arxiv_id:
            arxiv_id = extract_arxiv_id_from_text(journal)
        return True, arxiv_id

    # Check DOI pattern
    if doi:
        d = doi_normalize(doi)
        if d and (d.startswith("10.48550/arxiv") or d.startswith("10.1101")):
            return True, arxiv_id

    # Check URL
    if url and any(host in safe_lower(url) for host in ["arxiv.org", "biorxiv.org", "medrxiv.org"]):
        return True, arxiv_id

    return bool(arxiv_id), arxiv_id


def published_record_to_zotero_update(rec: PublishedRecord, original_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PublishedRecord to Zotero update payload.
    Preserves: tags, collections, notes, attachments (not touched here).
    Updates: title, authors, journal, DOI, URL, volume, issue, pages, date.
    """
    data = original_item.get("data", original_item)

    update = {
        "key": data["key"],
        "version": data["version"],
        "itemType": "journalArticle",  # Upgrade to proper journal article
    }

    # Update metadata
    if rec.title:
        update["title"] = rec.title

    if rec.journal:
        update["publicationTitle"] = rec.journal

    if rec.doi:
        update["DOI"] = rec.doi
        update["url"] = doi_url(rec.doi)
    elif rec.url:
        update["url"] = rec.url

    if rec.year:
        update["date"] = str(rec.year)

    if rec.volume:
        update["volume"] = str(rec.volume)

    if rec.number:
        update["issue"] = str(rec.number)

    if rec.pages:
        update["pages"] = str(rec.pages)

    # Update authors
    if rec.authors:
        creators = []
        for a in rec.authors:
            given = a.get("given", "").strip()
            family = a.get("family", "").strip()
            if given or family:
                creators.append(
                    {
                        "creatorType": "author",
                        "firstName": given,
                        "lastName": family,
                    }
                )
        if creators:
            update["creators"] = creators

    # Preserve arXiv reference in extra field
    original_url = data.get("url", "")
    arxiv_id = extract_arxiv_id_from_text(original_url)
    if arxiv_id:
        extra = data.get("extra", "")
        note = f"arXiv:{arxiv_id}"
        if note.lower() not in extra.lower():
            update["extra"] = (extra + "\n" + note).strip() if extra else note

    return update


# ------------- Main Logic -------------


@dataclass
class UpdateResult:
    item_key: str
    title: str
    action: str  # "updated", "not_found", "skipped", "error"
    message: str
    old_journal: Optional[str] = None
    new_journal: Optional[str] = None
    doi: Optional[str] = None
    method: Optional[str] = None
    confidence: Optional[float] = None


class ZoteroPrePrintUpdater:
    def __init__(
        self,
        library_id: str,
        api_key: str,
        library_type: str = "user",
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.zot = zotero.Zotero(library_id, library_type, api_key)
        self.dry_run = dry_run
        self.verbose = verbose

        # Set up logging
        self.logger = logging.getLogger("zotero_updater")
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

        # Set up HTTP client and resolver (reuse bibtex_updater infrastructure)
        rate_limiter = RateLimiter(req_per_min=45)
        cache = DiskCache(".cache.zotero_updater.json")
        user_agent = "zotero-preprint-updater/1.0 (mailto:you@example.com)"
        self.http = HttpClient(
            timeout=20.0,
            user_agent=user_agent,
            rate_limiter=rate_limiter,
            cache=cache,
            verbose=verbose,
        )
        self.resolver = Resolver(http=self.http, logger=self.logger)
        self.detector = Detector()

    def fetch_preprints(
        self,
        collection_key: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch items that look like preprints from Zotero."""
        params = {"limit": limit, "itemType": "journalArticle || preprint"}

        if collection_key:
            items = self.zot.collection_items(collection_key, **params)
        elif tag:
            params["tag"] = tag
            items = self.zot.items(**params)
        else:
            items = self.zot.items(**params)

        preprints = []
        for item in items:
            is_preprint, arxiv_id = is_zotero_preprint(item)
            if is_preprint:
                preprints.append(item)

        self.logger.info(f"Found {len(preprints)} preprint(s) out of {len(items)} items")
        return preprints

    def process_item(self, item: Dict[str, Any]) -> UpdateResult:
        """Process a single Zotero item."""
        data = item.get("data", item)
        key = data.get("key", "")
        title = data.get("title", "")[:60]
        old_journal = data.get("publicationTitle", "")

        try:
            # Convert to bibtex-like entry
            entry = zotero_to_bibtex_entry(item)

            # Detect preprint
            detection = self.detector.detect(entry)
            if not detection.is_preprint:
                return UpdateResult(
                    item_key=key,
                    title=title,
                    action="skipped",
                    message="Not detected as preprint",
                    old_journal=old_journal,
                )

            # Resolve to published version
            rec = self.resolver.resolve(entry, detection)

            if not rec:
                return UpdateResult(
                    item_key=key,
                    title=title,
                    action="not_found",
                    message="No published version found",
                    old_journal=old_journal,
                )

            # Prepare update
            update_payload = published_record_to_zotero_update(rec, item)

            if self.dry_run:
                return UpdateResult(
                    item_key=key,
                    title=title,
                    action="would_update",
                    message=f"Would update to {rec.journal}",
                    old_journal=old_journal,
                    new_journal=rec.journal,
                    doi=rec.doi,
                    method=rec.method,
                    confidence=rec.confidence,
                )

            # Apply update
            self.zot.update_item(update_payload)

            # Add tag to mark as processed
            self._add_tag(data["key"], "preprint-upgraded")

            return UpdateResult(
                item_key=key,
                title=title,
                action="updated",
                message=f"Updated to {rec.journal}",
                old_journal=old_journal,
                new_journal=rec.journal,
                doi=rec.doi,
                method=rec.method,
                confidence=rec.confidence,
            )

        except Exception as e:
            self.logger.exception(f"Error processing {key}")
            return UpdateResult(
                item_key=key,
                title=title,
                action="error",
                message=str(e),
                old_journal=old_journal,
            )

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

    def run(
        self,
        collection_key: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[UpdateResult]:
        """Run the update process."""
        preprints = self.fetch_preprints(collection_key=collection_key, tag=tag, limit=limit)

        results = []
        for i, item in enumerate(preprints, 1):
            data = item.get("data", item)
            self.logger.info(f"[{i}/{len(preprints)}] Processing: {data.get('title', '')[:50]}...")
            result = self.process_item(item)
            results.append(result)

            # Log result
            if result.action in ("updated", "would_update"):
                self.logger.info(f"  ✓ {result.action}: {result.old_journal} → {result.new_journal} ({result.method})")
            elif result.action == "not_found":
                self.logger.info("  ✗ No published version found")
            elif result.action == "error":
                self.logger.error(f"  ! Error: {result.message}")

        return results


def print_summary(results: List[UpdateResult]) -> None:
    """Print summary of results."""
    updated = [r for r in results if r.action in ("updated", "would_update")]
    not_found = [r for r in results if r.action == "not_found"]
    errors = [r for r in results if r.action == "error"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed:  {len(results)}")
    print(f"Updated:          {len(updated)}")
    print(f"Not found:        {len(not_found)}")
    print(f"Errors:           {len(errors)}")

    if updated:
        print("\n--- Updated/Would Update ---")
        for r in updated:
            print(f"  [{r.item_key}] {r.title}")
            print(f"    {r.old_journal} → {r.new_journal}")
            print(f"    DOI: {r.doi or 'N/A'}, Method: {r.method}, Conf: {r.confidence:.2f}" if r.confidence else "")

    if not_found:
        print("\n--- Not Found (may still be preprint-only) ---")
        for r in not_found:
            print(f"  [{r.item_key}] {r.title}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Update Zotero preprints to published versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--collection", help="Only process items in this collection (key)")
    parser.add_argument("--tag", help="Only process items with this tag")
    parser.add_argument("--limit", type=int, default=100, help="Max items to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--library-id", help="Zotero library ID (or set ZOTERO_LIBRARY_ID)")
    parser.add_argument("--api-key", help="Zotero API key (or set ZOTERO_API_KEY)")
    parser.add_argument("--library-type", default="user", choices=["user", "group"], help="Library type")

    args = parser.parse_args(argv)

    # Get credentials
    library_id = args.library_id or os.environ.get("ZOTERO_LIBRARY_ID")
    api_key = args.api_key or os.environ.get("ZOTERO_API_KEY")

    if not library_id or not api_key:
        print("Error: ZOTERO_LIBRARY_ID and ZOTERO_API_KEY required", file=sys.stderr)
        print("  Set environment variables or use --library-id and --api-key", file=sys.stderr)
        print("  Get your library ID and create an API key at: https://www.zotero.org/settings/keys", file=sys.stderr)
        return 1

    updater = ZoteroPrePrintUpdater(
        library_id=library_id,
        api_key=api_key,
        library_type=args.library_type,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    results = updater.run(
        collection_key=args.collection,
        tag=args.tag,
        limit=args.limit,
    )

    print_summary(results)

    errors = [r for r in results if r.action == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
