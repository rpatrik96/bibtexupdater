#!/usr/bin/env python3
"""
replace_preprints.py — Upgrade BibTeX preprints to published journal articles.

This CLI scans one or more .bib files for preprint-like entries (arXiv/bioRxiv/medRxiv)
and replaces them with published metadata when reliably found via:
  1) arXiv API (extracts a DOI from an arXiv id),
  2) Crossref /works/{doi} relations (is-preprint-of / has-preprint),
  3) DBLP bibliographic search (title + first author),
  4) Semantic Scholar Graph API (safe alternative to Google Scholar scraping),
  5) Crossref bibliographic search as a final fallback.

Notes
-----
* Direct Google Scholar scraping is not implemented to respect ToS. Semantic Scholar is used
  as a Scholar-like source where Crossref/DBLP do not suffice.
* Only upgrades to credible journal articles are applied; never downgrades.

Examples
--------
$ python replace_preprints.py input.bib -o output.bib --report report.jsonl
$ python replace_preprints.py a.bib b.bib --in-place --dedupe --keep-preprint-note
$ python replace_preprints.py input.bib --dry-run --verbose

Minimal sample (preprint) that can be upgraded when a journal DOI is discoverable:
@article{smith2020example,
  title={An Example Result},
  author={Smith, John and Doe, Jane},
  journal={arXiv preprint arXiv:2001.01234},
  year={2020},
  url={https://arxiv.org/abs/2001.01234}
}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz.fuzz import token_sort_ratio

# Shared utilities
from bibtex_updater.utils import (
    # Constants
    ARXIV_API,
    CROSSREF_API,
    DBLP_API_SEARCH,
    PREPRINT_HOSTS,
    S2_API,
    # Async HTTP infrastructure
    AsyncHttpClient,
    DiskCache,
    HttpClient,
    # Data classes
    PublishedRecord,
    # HTTP infrastructure
    RateLimiter,
    # Resolution caching
    ResolutionCache,
    ResolutionCacheEntry,
    authors_last_names,
    # API converters
    crossref_message_to_record,
    dblp_hit_to_record,
    # DOI/arXiv
    doi_normalize,
    doi_url,
    extract_arxiv_id_from_text,
    first_author_surname,
    # Matching
    jaccard_similarity,
    last_name_from_person,
    latex_to_plain,
    normalize_title_for_match,
    safe_lower,
    # Author handling
    split_authors_bibtex,
    strip_diacritics,
)

# External library: bibtexparser
try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.bwriter import BibTexWriter
except Exception:  # pragma: no cover
    print(
        "Error: This tool requires the 'bibtexparser' package. Install via 'pip install bibtexparser'.", file=sys.stderr
    )
    raise


# ------------- IO Helpers -------------
class BibLoader:
    def __init__(self) -> None:
        self.parser = BibTexParser(common_strings=True)
        self.parser.customization = None

    def load_file(self, path: str) -> bibtexparser.bibdatabase.BibDatabase:
        with open(path, encoding="utf-8") as f:
            return bibtexparser.load(f, parser=self.parser)

    def loads(self, text: str) -> bibtexparser.bibdatabase.BibDatabase:
        return bibtexparser.loads(text, parser=self.parser)


class BibWriter:
    def __init__(self) -> None:
        self.writer = BibTexWriter()
        self.writer.indent = "  "
        self.writer.order_entries_by = None
        self.writer.comma_first = False

    def dumps(self, db: bibtexparser.bibdatabase.BibDatabase) -> str:
        return bibtexparser.dumps(db, writer=self.writer)

    def dump_to_file(self, db: bibtexparser.bibdatabase.BibDatabase, path: str) -> None:
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".bib", prefix=".tmp_bib_")
        try:
            tmp.write(self.dumps(db))
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, path)


# ------------- Google Scholar Client (optional) -------------
class ScholarlyClient:
    """Google Scholar client via scholarly package (opt-in, reliability-focused)."""

    def __init__(self, proxy: str = "none", delay: float = 5.0, logger: logging.Logger | None = None):
        self.delay = delay
        self.logger = logger or logging.getLogger(__name__)
        self._last_request = 0.0
        self._scholarly = None
        self._setup(proxy)

    def _setup(self, proxy: str) -> None:
        try:
            from scholarly import ProxyGenerator, scholarly

            self._scholarly = scholarly
            if proxy == "tor":
                pg = ProxyGenerator()
                pg.Tor_Internal()
                scholarly.use_proxy(pg)
                self.logger.debug("Scholarly: using Tor proxy")
            elif proxy == "free":
                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)
                self.logger.debug("Scholarly: using free proxies")
            else:
                self.logger.debug("Scholarly: no proxy configured")
        except ImportError:
            self.logger.warning("scholarly package not installed; Google Scholar fallback disabled")
            self._scholarly = None

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def search(self, title: str, first_author: str) -> dict[str, Any] | None:
        """Search Google Scholar and return filled publication or None."""
        if not self._scholarly:
            return None
        try:
            self._rate_limit()
            query = f"{title} {first_author}"
            self.logger.debug("Scholarly search: %s", query[:80])
            search_results = self._scholarly.search_pubs(query)
            pub = next(search_results, None)
            if pub:
                self._rate_limit()
                filled = self._scholarly.fill(pub)
                return filled
        except StopIteration:
            self.logger.debug("Scholarly: no results found")
        except Exception as e:
            self.logger.warning("Scholarly search failed: %s", e)
        return None


# ------------- Detection -------------
@dataclass
class PreprintDetection:
    is_preprint: bool
    reason: str = ""
    arxiv_id: str | None = None
    doi: str | None = None


# ------------- Field Checking -------------
@dataclass
class FieldRequirement:
    """Defines required/recommended/optional fields for a BibTeX entry type."""

    required: frozenset
    recommended: frozenset
    optional: frozenset = field(default_factory=frozenset)


@dataclass
class MissingFieldReport:
    """Report of missing fields for a single entry."""

    entry_key: str
    entry_type: str
    missing_required: list[str]
    missing_recommended: list[str]
    filled_fields: dict[str, tuple[str, str]] = field(default_factory=dict)  # field -> (value, source)
    errors: list[str] = field(default_factory=list)


@dataclass
class FieldCheckResult:
    """Result of field checking/filling for a single entry."""

    original: dict[str, Any]
    updated: dict[str, Any]
    report: MissingFieldReport
    changed: bool
    action: str  # "complete", "filled", "partial", "unfillable"


class FieldRequirementRegistry:
    """Registry of required/recommended fields per BibTeX entry type."""

    ENTRY_REQUIREMENTS: dict[str, FieldRequirement] = {
        "article": FieldRequirement(
            required=frozenset({"author", "title", "journal", "year"}),
            recommended=frozenset({"volume", "number", "pages", "doi", "url"}),
            optional=frozenset({"month", "note", "abstract", "keywords", "publisher"}),
        ),
        "inproceedings": FieldRequirement(
            required=frozenset({"author", "title", "booktitle", "year"}),
            recommended=frozenset({"pages", "doi", "url", "publisher", "address"}),
            optional=frozenset({"editor", "volume", "number", "series", "organization", "month"}),
        ),
        "incollection": FieldRequirement(
            required=frozenset({"author", "title", "booktitle", "publisher", "year"}),
            recommended=frozenset({"editor", "pages", "doi", "url", "address"}),
            optional=frozenset({"volume", "number", "series", "chapter", "edition", "month"}),
        ),
        "book": FieldRequirement(
            required=frozenset({"title", "publisher", "year"}),  # author or editor required
            recommended=frozenset({"author", "editor", "volume", "number", "series", "address", "edition", "isbn"}),
            optional=frozenset({"month", "note", "doi", "url"}),
        ),
        "phdthesis": FieldRequirement(
            required=frozenset({"author", "title", "school", "year"}),
            recommended=frozenset({"address", "month", "url", "doi"}),
            optional=frozenset({"note", "type"}),
        ),
        "mastersthesis": FieldRequirement(
            required=frozenset({"author", "title", "school", "year"}),
            recommended=frozenset({"address", "month", "url"}),
            optional=frozenset({"note", "type"}),
        ),
        "techreport": FieldRequirement(
            required=frozenset({"author", "title", "institution", "year"}),
            recommended=frozenset({"number", "address", "month", "url"}),
            optional=frozenset({"note", "type"}),
        ),
        "unpublished": FieldRequirement(
            required=frozenset({"author", "title", "note"}),
            recommended=frozenset({"year", "month", "url"}),
            optional=frozenset(),
        ),
        "misc": FieldRequirement(
            required=frozenset({"title"}),
            recommended=frozenset({"author", "year", "url", "howpublished"}),
            optional=frozenset({"note", "month"}),
        ),
    }

    # Default for unknown entry types
    _DEFAULT_REQUIREMENT = FieldRequirement(
        required=frozenset({"title"}),
        recommended=frozenset({"author", "year"}),
        optional=frozenset(),
    )

    @classmethod
    def get_requirements(cls, entry_type: str) -> FieldRequirement:
        """Get field requirements for an entry type."""
        return cls.ENTRY_REQUIREMENTS.get(entry_type.lower(), cls._DEFAULT_REQUIREMENT)

    @classmethod
    def get_all_entry_types(cls) -> list[str]:
        """Get all registered entry types."""
        return list(cls.ENTRY_REQUIREMENTS.keys())


class FieldChecker:
    """Checks BibTeX entries for missing required/recommended fields."""

    def __init__(self, registry: FieldRequirementRegistry | None = None):
        self.registry = registry or FieldRequirementRegistry()

    def _field_present(self, entry: dict[str, Any], field_name: str) -> bool:
        """Check if a field is present and non-empty."""
        value = entry.get(field_name, "")
        if isinstance(value, str):
            return bool(value.strip())
        return bool(value)

    def _check_venue(self, entry: dict[str, Any], entry_type: str) -> bool:
        """Check if venue field is present based on entry type."""
        etype = entry_type.lower()
        if etype == "article":
            return self._field_present(entry, "journal")
        elif etype in ("inproceedings", "incollection"):
            return self._field_present(entry, "booktitle")
        return True  # Other types don't require venue

    def check_entry(self, entry: dict[str, Any]) -> MissingFieldReport:
        """Check an entry for missing fields."""
        entry_key = entry.get("ID", "unknown")
        entry_type = entry.get("ENTRYTYPE", "misc").lower()
        requirements = self.registry.get_requirements(entry_type)

        missing_required = []
        missing_recommended = []

        # Check required fields
        for field_name in requirements.required:
            if not self._field_present(entry, field_name):
                # Special case: book can have author OR editor
                if entry_type == "book" and field_name in ("author", "editor"):
                    if not (self._field_present(entry, "author") or self._field_present(entry, "editor")):
                        if "author" not in missing_required:
                            missing_required.append("author")
                else:
                    missing_required.append(field_name)

        # Check recommended fields
        for field_name in requirements.recommended:
            if not self._field_present(entry, field_name):
                # Skip author/editor for book if one is already present
                if entry_type == "book" and field_name in ("author", "editor"):
                    if self._field_present(entry, "author") or self._field_present(entry, "editor"):
                        continue
                missing_recommended.append(field_name)

        return MissingFieldReport(
            entry_key=entry_key,
            entry_type=entry_type,
            missing_required=sorted(missing_required),
            missing_recommended=sorted(missing_recommended),
            filled_fields={},
            errors=[],
        )

    def has_missing_fields(self, report: MissingFieldReport) -> bool:
        """Check if any fields are missing."""
        return bool(report.missing_required or report.missing_recommended)

    def has_missing_required(self, report: MissingFieldReport) -> bool:
        """Check if any required fields are missing."""
        return bool(report.missing_required)


class FieldFiller:
    """Attempts to fill missing fields using external APIs (Crossref, DBLP)."""

    # Minimum match score for accepting a search result
    MATCH_THRESHOLD = 0.85

    def __init__(self, resolver: Resolver, logger: logging.Logger):
        self.resolver = resolver
        self.logger = logger

    def fill_entry(
        self, entry: dict[str, Any], report: MissingFieldReport, fill_mode: str = "recommended"
    ) -> tuple[dict[str, Any], MissingFieldReport]:
        """
        Attempt to fill missing fields in an entry.

        Args:
            entry: The BibTeX entry dict
            report: MissingFieldReport from FieldChecker
            fill_mode: "required" | "recommended" | "all"

        Returns:
            Tuple of (updated_entry, updated_report)
        """
        fields_to_fill = self._get_fields_to_fill(report, fill_mode)

        if not fields_to_fill:
            return entry, report

        updated = dict(entry)
        filled_fields: dict[str, tuple[str, str]] = {}
        errors: list[str] = []

        # Strategy 1: If DOI present, use Crossref for authoritative data
        doi = doi_normalize(entry.get("doi"))
        if doi:
            filled, remaining = self._fill_from_doi(doi, fields_to_fill)
            for field_name, (value, source) in filled.items():
                updated[field_name] = value
                filled_fields[field_name] = (value, source)
            fields_to_fill = remaining

        # Strategy 2: Search by title + author (fuzzy matching)
        if fields_to_fill and (entry.get("title") or entry.get("author")):
            filled, remaining, search_errors = self._fill_from_search(entry, fields_to_fill)
            for field_name, (value, source) in filled.items():
                updated[field_name] = value
                filled_fields[field_name] = (value, source)
            fields_to_fill = remaining
            errors.extend(search_errors)

        # Update report with filled fields
        updated_report = MissingFieldReport(
            entry_key=report.entry_key,
            entry_type=report.entry_type,
            missing_required=[f for f in report.missing_required if f not in filled_fields],
            missing_recommended=[f for f in report.missing_recommended if f not in filled_fields],
            filled_fields=filled_fields,
            errors=report.errors + errors,
        )

        return updated, updated_report

    def _get_fields_to_fill(self, report: MissingFieldReport, fill_mode: str) -> list[str]:
        """Determine which fields to attempt filling based on mode."""
        if fill_mode == "required":
            return list(report.missing_required)
        elif fill_mode == "recommended":
            return list(report.missing_required) + list(report.missing_recommended)
        else:  # "all"
            return list(report.missing_required) + list(report.missing_recommended)

    def _fill_from_doi(self, doi: str, fields_to_fill: list[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
        """Fill fields using DOI lookup via Crossref."""
        filled: dict[str, tuple[str, str]] = {}

        msg = self.resolver.crossref_get(doi)
        if not msg:
            return filled, fields_to_fill

        rec = self.resolver._message_to_record(msg)
        if not rec:
            return filled, fields_to_fill

        source = "Crossref(DOI)"
        filled = self._extract_fields_from_record(rec, fields_to_fill, source)
        remaining = [f for f in fields_to_fill if f not in filled]

        return filled, remaining

    def _fill_from_search(
        self, entry: dict[str, Any], fields_to_fill: list[str]
    ) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
        """Fill fields using search APIs."""
        filled: dict[str, tuple[str, str]] = {}
        errors: list[str] = []

        title = normalize_title_for_match(entry.get("title", ""))
        first_author = first_author_surname(entry)

        if not title:
            return filled, fields_to_fill, ["No title for search"]

        query = f"{title} {first_author}".strip()

        # Try Crossref search first (no filter_type to get all results)
        items = self.resolver.crossref_search(query, rows=5, filter_type=None)
        if items:
            best_match = self._find_best_crossref_match(entry, items)
            if best_match:
                rec = self.resolver._message_to_record(best_match)
                if rec:
                    filled = self._extract_fields_from_record(rec, fields_to_fill, "Crossref(search)")
                    remaining = [f for f in fields_to_fill if f not in filled]
                    return filled, remaining, errors

        # Try DBLP as fallback
        hits = self.resolver.dblp_search(query, h=5)
        if hits:
            best_hit = self._find_best_dblp_match(entry, hits)
            if best_hit:
                rec = self.resolver._dblp_hit_to_record(best_hit)
                if rec:
                    filled = self._extract_fields_from_record(rec, fields_to_fill, "DBLP(search)")
                    remaining = [f for f in fields_to_fill if f not in filled]
                    return filled, remaining, errors

        if not filled:
            errors.append("No matching record found in APIs")

        return filled, fields_to_fill, errors

    def _find_best_crossref_match(self, entry: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find best matching Crossref item using title/author similarity."""
        title_a = normalize_title_for_match(entry.get("title", ""))
        authors_a = authors_last_names(entry.get("author", ""), limit=3)

        best_score = 0.0
        best_item = None

        for item in items:
            rec = self.resolver._message_to_record(item)
            if not rec:
                continue

            title_b = normalize_title_for_match(rec.title or "")
            title_score = token_sort_ratio(title_a, title_b)

            authors_b = [a.get("family", "").lower() for a in rec.authors[:3]]
            auth_score = jaccard_similarity(authors_a, authors_b)

            combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score

            if combined > best_score and combined >= self.MATCH_THRESHOLD:
                best_score = combined
                best_item = item

        return best_item

    def _find_best_dblp_match(self, entry: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find best matching DBLP hit using title/author similarity."""
        title_a = normalize_title_for_match(entry.get("title", ""))
        authors_a = authors_last_names(entry.get("author", ""), limit=3)

        best_score = 0.0
        best_hit = None

        for hit in hits:
            info = hit.get("info", {})
            title_raw = info.get("title", "")
            title_b = normalize_title_for_match(re.sub(r"<[^>]*>", "", title_raw))
            title_score = token_sort_ratio(title_a, title_b)

            # Extract author names from DBLP format
            authors_field = info.get("authors", {}).get("author", [])
            if isinstance(authors_field, dict):
                authors_field = [authors_field]
            authors_b = []
            for a in authors_field[:3]:
                if isinstance(a, dict):
                    name = a.get("text") or a.get("name") or ""
                else:
                    name = str(a)
                if name:
                    parts = name.split()
                    if parts:
                        authors_b.append(parts[-1].lower())

            auth_score = jaccard_similarity(authors_a, authors_b)
            combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score

            if combined > best_score and combined >= self.MATCH_THRESHOLD:
                best_score = combined
                best_hit = hit

        return best_hit

    def _extract_fields_from_record(
        self, rec: PublishedRecord, fields: list[str], source: str
    ) -> dict[str, tuple[str, str]]:
        """Extract requested fields from PublishedRecord."""
        filled: dict[str, tuple[str, str]] = {}

        field_mapping: dict[str, str | None] = {
            "year": str(rec.year) if rec.year else None,
            "journal": rec.journal,
            "booktitle": rec.journal,  # Use journal for booktitle if available
            "volume": rec.volume,
            "number": rec.number,
            "pages": rec.pages,
            "doi": rec.doi,
            "url": rec.url or (doi_url(rec.doi) if rec.doi else None),
            "publisher": rec.publisher,
            "title": rec.title,
            "author": self.resolver._authors_to_bibtex_string(rec) if rec.authors else None,
        }

        for field_name in fields:
            value = field_mapping.get(field_name)
            if value and value.strip():
                filled[field_name] = (value.strip(), source)

        return filled


class MissingFieldProcessor:
    """Orchestrates field checking and filling with reporting."""

    def __init__(
        self,
        checker: FieldChecker,
        filler: FieldFiller | None = None,
        fill_mode: str = "recommended",
        fill_enabled: bool = True,
    ):
        self.checker = checker
        self.filler = filler
        self.fill_mode = fill_mode
        self.fill_enabled = fill_enabled and filler is not None

    def process_entry(self, entry: dict[str, Any]) -> FieldCheckResult:
        """Process a single entry: check and optionally fill missing fields."""
        report = self.checker.check_entry(entry)

        has_missing = self.checker.has_missing_fields(report)

        if not has_missing:
            return FieldCheckResult(
                original=entry,
                updated=entry,
                report=report,
                changed=False,
                action="complete",
            )

        # Attempt to fill if enabled and filler is available
        if self.fill_enabled and self.filler:
            updated_entry, updated_report = self.filler.fill_entry(entry, report, self.fill_mode)
            changed = entry != updated_entry

            # Determine action based on results
            if updated_report.missing_required:
                action = "partial" if updated_report.filled_fields else "unfillable"
            elif updated_report.filled_fields:
                action = "filled"
            else:
                action = "complete" if not updated_report.missing_recommended else "partial"

            return FieldCheckResult(
                original=entry,
                updated=updated_entry,
                report=updated_report,
                changed=changed,
                action=action,
            )
        else:
            # Check only mode - no filling
            action = "unfillable" if report.missing_required else "partial"
            return FieldCheckResult(
                original=entry,
                updated=entry,
                report=report,
                changed=False,
                action=action,
            )

    def generate_summary(self, results: list[FieldCheckResult]) -> dict[str, Any]:
        """Generate a summary of field check results."""
        complete = sum(1 for r in results if r.action == "complete")
        filled = sum(1 for r in results if r.action == "filled")
        partial = sum(1 for r in results if r.action == "partial")
        unfillable = sum(1 for r in results if r.action == "unfillable")

        # Field statistics
        field_stats: dict[str, dict[str, int]] = {}
        for result in results:
            for field_name in result.report.missing_required + result.report.missing_recommended:
                if field_name not in field_stats:
                    field_stats[field_name] = {"total_missing": 0, "filled": 0, "still_missing": 0}
                field_stats[field_name]["total_missing"] += 1
                if field_name in result.report.filled_fields:
                    field_stats[field_name]["filled"] += 1
                else:
                    field_stats[field_name]["still_missing"] += 1

        return {
            "total": len(results),
            "complete": complete,
            "filled": filled,
            "partial": partial,
            "unfillable": unfillable,
            "field_statistics": field_stats,
        }

    def generate_json_report(self, results: list[FieldCheckResult]) -> dict[str, Any]:
        """Generate a detailed JSON report."""
        import datetime

        entries = []
        for result in results:
            entry_data = {
                "key": result.report.entry_key,
                "type": result.report.entry_type,
                "action": result.action,
                "missing_required": result.report.missing_required,
                "missing_recommended": result.report.missing_recommended,
                "filled_fields": {
                    field: {"value": value, "source": source}
                    for field, (value, source) in result.report.filled_fields.items()
                },
                "errors": result.report.errors,
            }
            entries.append(entry_data)

        summary = self.generate_summary(results)
        summary["timestamp"] = datetime.datetime.now().isoformat()

        return {
            "summary": summary,
            "entries": entries,
        }


class Detector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _has_preprint_host(value: str) -> bool:
        v = safe_lower(value)
        return any(host in v for host in PREPRINT_HOSTS)

    def detect(self, entry: dict[str, Any]) -> PreprintDetection:
        etype = safe_lower(entry.get("ENTRYTYPE"))
        journal = safe_lower(entry.get("journal"))
        howpub = safe_lower(entry.get("howpublished"))
        note = entry.get("note") or ""
        url = entry.get("url") or ""
        eprint = entry.get("eprint") or ""
        archive_prefix = safe_lower(entry.get("archiveprefix") or entry.get("archivePrefix"))
        doi = doi_normalize(entry.get("doi"))

        if etype == "article" and doi and not (doi.startswith("10.1101") or doi.startswith("10.48550/arxiv")):
            j = safe_lower(journal)
            if not self._has_preprint_host(j):
                return PreprintDetection(False)

        if eprint and (archive_prefix == "arxiv" or extract_arxiv_id_from_text(eprint)):
            arx = extract_arxiv_id_from_text(eprint) or eprint.strip()
            return PreprintDetection(True, reason="eprint arXiv", arxiv_id=arx, doi=doi)
        if url and extract_arxiv_id_from_text(url):
            return PreprintDetection(True, reason="url arXiv", arxiv_id=extract_arxiv_id_from_text(url), doi=doi)
        if note and extract_arxiv_id_from_text(note):
            return PreprintDetection(True, reason="note arXiv", arxiv_id=extract_arxiv_id_from_text(note), doi=doi)

        if journal and self._has_preprint_host(journal):
            return PreprintDetection(
                True,
                reason="journal contains preprint host",
                doi=doi,
                arxiv_id=extract_arxiv_id_from_text(url or note or eprint),
            )
        if howpub and self._has_preprint_host(howpub):
            return PreprintDetection(
                True,
                reason="howpublished contains preprint host",
                doi=doi,
                arxiv_id=extract_arxiv_id_from_text(url or note or eprint),
            )

        if doi and (doi.startswith("10.48550/arxiv") or doi.startswith("10.1101")):
            return PreprintDetection(
                True, reason="preprint DOI pattern", doi=doi, arxiv_id=extract_arxiv_id_from_text(url or note or eprint)
            )

        if etype in {"unpublished", "misc"}:
            return PreprintDetection(
                True,
                reason="entrytype preprint-ish",
                doi=doi,
                arxiv_id=extract_arxiv_id_from_text(url or note or eprint),
            )
        if etype == "article" and ("preprint" in safe_lower(note)):
            return PreprintDetection(
                True,
                reason="note mentions preprint",
                doi=doi,
                arxiv_id=extract_arxiv_id_from_text(url or note or eprint),
            )

        return PreprintDetection(False)


# ------------- Resolver & Matching -------------
class Resolver:
    # Accepted publication types for upgrades (includes ML conference papers)
    ACCEPTED_TYPES = {
        "journal-article",
        "proceedings-article",  # Conference papers (NeurIPS, ICML, AISTATS, etc.)
        "book-chapter",
    }

    # Semantic Scholar publication type mappings
    ACCEPTED_S2_TYPES = {
        "journalarticle",
        "conference",
    }

    def __init__(
        self,
        http: HttpClient,
        logger: logging.Logger,
        scholarly_client: ScholarlyClient | None = None,
        resolution_cache: ResolutionCache | None = None,
    ) -> None:
        self.http = http
        self.logger = logger
        self.scholarly_client = scholarly_client
        self.resolution_cache = resolution_cache

    # --- arXiv ---
    def arxiv_candidate_doi(self, arxiv_id: str) -> str | None:
        params = {"id_list": arxiv_id}
        try:
            resp = self.http._request("GET", ARXIV_API, params=params, accept="application/atom+xml", service="arxiv")
            xml = resp.text
        except Exception as e:
            self.logger.debug("arXiv lookup failed for %s: %s", arxiv_id, e)
            return None
        m = re.search(r"<arxiv:doi>([^<]+)</arxiv:doi>", xml, flags=re.IGNORECASE)
        if m:
            return doi_normalize(m.group(1))
        m = re.search(r"<doi>([^<]+)</doi>", xml, flags=re.IGNORECASE)
        if m:
            return doi_normalize(m.group(1))
        return None

    def arxiv_batch_doi_lookup(self, arxiv_ids: list[str]) -> dict[str, str | None]:
        """Fetch DOIs for multiple arXiv IDs in a single request (up to 100).

        Args:
            arxiv_ids: List of arXiv IDs to look up

        Returns:
            Dict mapping arXiv ID to DOI (or None if no DOI found)
        """
        if not arxiv_ids:
            return {}

        # arXiv API supports comma-separated IDs (max 100 per request)
        params = {"id_list": ",".join(arxiv_ids[:100])}
        try:
            resp = self.http._request("GET", ARXIV_API, params=params, accept="application/atom+xml", service="arxiv")
            xml = resp.text
        except Exception as e:
            self.logger.debug("arXiv batch lookup failed: %s", e)
            return {}

        results: dict[str, str | None] = {}

        # Parse XML entries - each entry has an <id> tag with the arXiv URL
        # and optionally a <arxiv:doi> or <doi> tag
        entry_pattern = re.compile(r"<entry>(.*?)</entry>", re.DOTALL | re.IGNORECASE)
        id_pattern = re.compile(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", re.IGNORECASE)
        doi_pattern = re.compile(r"<(?:arxiv:)?doi>([^<]+)</(?:arxiv:)?doi>", re.IGNORECASE)

        for entry_match in entry_pattern.finditer(xml):
            entry_xml = entry_match.group(1)

            # Extract arXiv ID from the entry
            id_match = id_pattern.search(entry_xml)
            if not id_match:
                continue

            arxiv_id = id_match.group(1)
            # Normalize arXiv ID (remove version suffix for matching)
            arxiv_id_base = re.sub(r"v\d+$", "", arxiv_id)

            # Extract DOI if present
            doi_match = doi_pattern.search(entry_xml)
            doi = doi_normalize(doi_match.group(1)) if doi_match else None

            # Map both versioned and unversioned IDs
            results[arxiv_id] = doi
            if arxiv_id != arxiv_id_base:
                results[arxiv_id_base] = doi

        # Also map original input IDs to results
        for input_id in arxiv_ids[:100]:
            input_id_base = re.sub(r"v\d+$", "", input_id)
            if input_id not in results and input_id_base in results:
                results[input_id] = results[input_id_base]

        return results

    # --- Crossref Works ---
    def crossref_get(self, doi: str) -> dict[str, Any] | None:
        doi = doi_normalize(doi) or ""
        from urllib.parse import quote

        url = f"{CROSSREF_API}/{quote(doi, safe='')}"
        # url = f"{CROSSREF_API}/{httpx.utils.quote(doi, safe='')}"
        try:
            resp = self.http._request("GET", url, accept="application/json", service="crossref")
            if resp.status_code != 200:
                return None
            data = resp.json().get("message", {})
            return data
        except Exception as e:
            self.logger.debug("Crossref works failed for %s: %s", doi, e)
            return None

    def crossref_search(
        self, query: str, rows: int = 25, filter_type: str | None = "journal-article"
    ) -> list[dict[str, Any]]:
        params = {"query.bibliographic": query, "rows": rows}
        if filter_type:
            params["filter"] = f"type:{filter_type}"
        try:
            resp = self.http._request("GET", CROSSREF_API, params=params, accept="application/json")
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
            return items
        except Exception as e:
            self.logger.debug("Crossref search failed '%s': %s", query, e)
            return []

    def crossref_batch_lookup(self, dois: list[str]) -> dict[str, PublishedRecord | None]:
        """Fetch multiple DOIs via Crossref filter query (up to 100).

        Args:
            dois: List of DOIs to look up

        Returns:
            Dict mapping DOI (lowercase) to PublishedRecord (or None if not found)
        """
        if not dois:
            return {}

        # Normalize DOIs and limit to 100
        normalized_dois = [doi_normalize(d) for d in dois[:100] if doi_normalize(d)]
        if not normalized_dois:
            return {}

        # Crossref supports filter with multiple DOIs
        filter_str = ",".join(f"doi:{doi}" for doi in normalized_dois)
        params: dict[str, Any] = {
            "filter": filter_str,
            "rows": 100,
        }

        try:
            resp = self.http._request(
                "GET",
                CROSSREF_API,
                params=params,
                accept="application/json",
                service="crossref",
            )
            if resp.status_code != 200:
                self.logger.debug("Crossref batch lookup returned status %d", resp.status_code)
                return {}
            data = resp.json()
        except Exception as e:
            self.logger.debug("Crossref batch lookup failed: %s", e)
            return {}

        results: dict[str, PublishedRecord | None] = {}

        for item in data.get("message", {}).get("items", []):
            doi = doi_normalize(item.get("DOI"))
            if doi:
                record = crossref_message_to_record(item)
                results[doi.lower()] = record

        # Mark DOIs that weren't found as None
        for doi in normalized_dois:
            if doi.lower() not in results:
                results[doi.lower()] = None

        return results

    # --- DBLP ---
    def dblp_search(self, query: str, h: int = 25) -> list[dict[str, Any]]:
        params = {"q": query, "h": h, "format": "json"}
        try:
            resp = self.http._request("GET", DBLP_API_SEARCH, params=params, accept="application/json", service="dblp")
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(hits, dict):
                hits = [hits]
            return hits
        except Exception as e:
            self.logger.debug("DBLP search failed '%s': %s", query, e)
            return []

    @staticmethod
    def _dblp_hit_to_record(hit: dict[str, Any]) -> PublishedRecord | None:
        """Convert DBLP hit to PublishedRecord. Delegates to bib_utils."""
        return dblp_hit_to_record(hit)

    # --- Semantic Scholar (safe alternative to Google Scholar scraping) ---
    def s2_from_arxiv(self, arxiv_id: str) -> PublishedRecord | None:
        fields = "externalIds,doi,title,year,authors,venue,publicationTypes,publicationVenue,url"
        url = f"{S2_API}/paper/arXiv:{arxiv_id}"
        try:
            resp = self.http._request(
                "GET", url, params={"fields": fields}, accept="application/json", service="semanticscholar"
            )
            if resp.status_code != 200:
                return None
            msg = resp.json()
        except Exception as e:
            self.logger.debug("Semantic Scholar arXiv lookup failed for %s: %s", arxiv_id, e)
            return None
        doi = doi_normalize(msg.get("doi") or (msg.get("externalIds") or {}).get("DOI"))
        pub_types = msg.get("publicationTypes") or []
        is_journal = any(pt.lower() == "journalarticle" for pt in pub_types)
        if not (doi and is_journal):
            return None
        title = msg.get("title")
        year = msg.get("year")
        venue = (msg.get("publicationVenue") or {}).get("name") or msg.get("venue")
        authors = [
            {
                "given": (a.get("name") or "").split()[:-1] and " ".join((a.get("name") or "").split()[:-1]) or "",
                "family": (a.get("name") or "").split()[-1] if (a.get("name") or "").split() else "",
            }
            for a in (msg.get("authors") or [])
        ]
        return PublishedRecord(
            doi=doi,
            url=doi_url(doi),
            title=title,
            authors=authors,
            journal=venue,
            year=year,
            type="journal-article",
            method="SemanticScholar(arXiv)",
            confidence=0.95,
        )

    def s2_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        params = {"query": query, "limit": limit, "fields": "title,authors,year,venue,publicationTypes,doi,url"}
        url = f"{S2_API}/paper/search"
        try:
            resp = self.http._request("GET", url, params=params, accept="application/json", service="semanticscholar")
            if resp.status_code != 200:
                return []
            return resp.json().get("data", []) or []
        except Exception as e:
            self.logger.debug("Semantic Scholar search failed '%s': %s", query, e)
            return []

    def s2_batch_lookup(self, paper_ids: list[str]) -> dict[str, PublishedRecord | None]:
        """Fetch papers via Semantic Scholar batch endpoint (up to 500 papers).

        Args:
            paper_ids: List of paper IDs (can be arXiv IDs like "arXiv:2301.00001",
                      DOIs like "DOI:10.1234/...", or S2 paper IDs)

        Returns:
            Dict mapping input paper ID to PublishedRecord (or None if not found)
        """
        if not paper_ids:
            return {}

        S2_BATCH_URL = f"{S2_API}/paper/batch"
        fields = "externalIds,doi,title,year,authors,venue,publicationTypes,publicationVenue,url,paperId"

        try:
            resp = self.http._request(
                "POST",
                S2_BATCH_URL,
                json_body={"ids": paper_ids[:500]},
                params={"fields": fields},
                accept="application/json",
                service="semanticscholar",
            )
            if resp.status_code != 200:
                self.logger.debug("S2 batch lookup returned status %d", resp.status_code)
                return {}
            data = resp.json()
        except Exception as e:
            self.logger.debug("S2 batch lookup failed: %s", e)
            return {}

        results: dict[str, PublishedRecord | None] = {}

        # Process results - S2 returns results in same order as input, with null for not found
        for i, paper in enumerate(data):
            if i >= len(paper_ids):
                break

            input_id = paper_ids[i]

            if paper is None:
                results[input_id] = None
                continue

            # Convert to PublishedRecord
            doi = doi_normalize(paper.get("doi") or (paper.get("externalIds") or {}).get("DOI"))
            pub_types = paper.get("publicationTypes") or []
            is_journal = any(pt.lower() == "journalarticle" for pt in pub_types)

            # Only include journal articles with DOIs
            if not (doi and is_journal):
                results[input_id] = None
                continue

            title = paper.get("title")
            year = paper.get("year")
            venue = (paper.get("publicationVenue") or {}).get("name") or paper.get("venue")
            authors = [
                {
                    "given": (a.get("name") or "").split()[:-1] and " ".join((a.get("name") or "").split()[:-1]) or "",
                    "family": (a.get("name") or "").split()[-1] if (a.get("name") or "").split() else "",
                }
                for a in (paper.get("authors") or [])
            ]

            record = PublishedRecord(
                doi=doi,
                url=doi_url(doi),
                title=title,
                authors=authors,
                journal=venue,
                year=year,
                type="journal-article",
                method="SemanticScholar(batch)",
                confidence=0.95,
            )
            results[input_id] = record

        return results

    # --- Batch Pre-resolution ---
    def batch_preload(
        self, entries: list[dict[str, Any]], detections: list[PreprintDetection]
    ) -> dict[str, dict[str, Any]]:
        """Pre-fetch data for multiple entries using batch APIs.

        This method performs batch lookups to reduce the number of API calls
        when processing multiple preprint entries. It should be called before
        individual entry processing.

        Args:
            entries: List of BibTeX entry dicts
            detections: List of PreprintDetection results corresponding to entries

        Returns:
            Dict mapping entry keys to pre-fetched data:
            {
                "entry_key": {
                    "s2_record": PublishedRecord | None,  # From S2 batch lookup
                    "arxiv_doi": str | None,              # DOI from arXiv API
                }
            }
        """
        preloaded: dict[str, dict[str, Any]] = {}

        # Collect arXiv IDs from preprint detections
        arxiv_ids: list[str] = []
        arxiv_entry_map: dict[str, str] = {}  # arxiv_id -> entry key

        for entry, det in zip(entries, detections):
            if det.is_preprint and det.arxiv_id:
                entry_key = entry.get("ID", "")
                if entry_key and det.arxiv_id not in arxiv_entry_map:
                    arxiv_ids.append(det.arxiv_id)
                    arxiv_entry_map[det.arxiv_id] = entry_key

        if not arxiv_ids:
            return preloaded

        # Batch fetch from Semantic Scholar using arXiv IDs
        s2_ids = [f"arXiv:{aid}" for aid in arxiv_ids]
        self.logger.debug("Batch preload: fetching %d entries from Semantic Scholar", len(s2_ids))
        s2_results = self.s2_batch_lookup(s2_ids)

        for s2_id, record in s2_results.items():
            # Extract arXiv ID from "arXiv:xxxx" format
            arxiv_id = s2_id.replace("arXiv:", "") if s2_id.startswith("arXiv:") else s2_id
            entry_key = arxiv_entry_map.get(arxiv_id)
            if entry_key and record:
                if entry_key not in preloaded:
                    preloaded[entry_key] = {}
                preloaded[entry_key]["s2_record"] = record

        # Batch fetch DOIs from arXiv API
        self.logger.debug("Batch preload: fetching %d DOIs from arXiv", len(arxiv_ids))
        arxiv_dois = self.arxiv_batch_doi_lookup(arxiv_ids)

        for arxiv_id, doi in arxiv_dois.items():
            entry_key = arxiv_entry_map.get(arxiv_id)
            if entry_key and doi:
                if entry_key not in preloaded:
                    preloaded[entry_key] = {}
                preloaded[entry_key]["arxiv_doi"] = doi

        self.logger.debug(
            "Batch preload complete: %d entries with S2 records, %d with arXiv DOIs",
            sum(1 for v in preloaded.values() if v.get("s2_record")),
            sum(1 for v in preloaded.values() if v.get("arxiv_doi")),
        )

        return preloaded

    # --- Shared helpers ---
    @staticmethod
    def _message_to_record(msg: dict[str, Any]) -> PublishedRecord | None:
        """Convert Crossref message to PublishedRecord. Delegates to bib_utils."""
        return crossref_message_to_record(msg)

    @staticmethod
    def _credible_journal_article(rec: PublishedRecord) -> bool:
        if rec.type != "journal-article":
            return False
        if not rec.journal:
            return False
        # Reject if journal looks like a preprint venue (ensures idempotency)
        j_lower = rec.journal.lower()
        if any(host in j_lower for host in PREPRINT_HOSTS):
            return False
        if not rec.year:
            return False
        return bool(rec.volume or rec.number or rec.pages or rec.url)

    @staticmethod
    def _authors_to_bibtex_string(rec: PublishedRecord) -> str:
        parts = []
        for a in rec.authors:
            given = a.get("given", "").strip()
            family = a.get("family", "").strip()
            if given and family:
                parts.append(f"{given} {family}")
            elif family:
                parts.append(family)
            elif given:
                parts.append(given)
        return " and ".join(parts)

    def _scholarly_to_record(self, pub: dict[str, Any]) -> PublishedRecord | None:
        """Convert scholarly publication dict to PublishedRecord."""
        if not pub:
            return None
        bib = pub.get("bib", {})
        if not bib:
            return None

        # Extract DOI from pub_url or eprint_url if possible
        doi = None
        for url_field in ["pub_url", "eprint_url"]:
            url = pub.get(url_field, "") or ""
            if "doi.org/" in url:
                doi = doi_normalize(url.split("doi.org/")[-1])
                break

        # Parse authors from "Author One and Author Two" format
        authors = []
        author_str = bib.get("author", "")
        if author_str:
            for name in author_str.split(" and "):
                name = name.strip()
                if not name:
                    continue
                parts = name.split()
                if len(parts) >= 2:
                    authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
                elif parts:
                    authors.append({"given": "", "family": parts[0]})

        venue = bib.get("venue") or bib.get("journal") or bib.get("booktitle")
        year = None
        if bib.get("pub_year"):
            try:
                year = int(bib["pub_year"])
            except (ValueError, TypeError):
                pass

        return PublishedRecord(
            doi=doi,
            url=pub.get("pub_url"),
            title=bib.get("title"),
            authors=authors,
            journal=venue,
            year=year,
            volume=bib.get("volume"),
            number=bib.get("number"),
            pages=bib.get("pages"),
            type="journal-article" if venue else "unknown",
            method="GoogleScholar(search)",
            confidence=0.0,
        )

    def resolve(self, entry: dict[str, Any], detection: PreprintDetection) -> PublishedRecord | None:
        # Check resolution cache first
        if self.resolution_cache:
            cached = self.resolution_cache.get(detection.arxiv_id, detection.doi)
            if cached:
                if cached.status == "no_match":
                    self.logger.debug(
                        "Resolution cache hit (no_match) for arxiv=%s, doi=%s",
                        detection.arxiv_id,
                        detection.doi,
                    )
                    return None
                elif cached.status == "resolved" and cached.resolved_doi:
                    self.logger.debug(
                        "Resolution cache hit for arxiv=%s, doi=%s -> %s via %s",
                        detection.arxiv_id,
                        detection.doi,
                        cached.resolved_doi,
                        cached.method,
                    )
                    # Fetch the full record from Crossref using the cached DOI
                    msg = self.crossref_get(cached.resolved_doi)
                    if msg:
                        rec = self._message_to_record(msg)
                        if rec:
                            rec.method = f"{cached.method}(cached)"
                            rec.confidence = cached.confidence
                            return rec

        # Perform resolution and cache the result
        result = self._resolve_uncached(entry, detection)

        # Cache the result
        if self.resolution_cache:
            if result:
                cache_entry = ResolutionCacheEntry(
                    arxiv_id=detection.arxiv_id,
                    preprint_doi=detection.doi,
                    resolved_doi=result.doi,
                    method=result.method,
                    confidence=result.confidence,
                    timestamp=time.time(),
                    status="resolved",
                    ttl_days=self.resolution_cache.ttl_days,
                )
                self.resolution_cache.set(cache_entry)
                self.logger.debug(
                    "Cached resolution: arxiv=%s, doi=%s -> %s via %s",
                    detection.arxiv_id,
                    detection.doi,
                    result.doi,
                    result.method,
                )
            else:
                self.resolution_cache.set_no_match(detection.arxiv_id, detection.doi)
                self.logger.debug(
                    "Cached no_match: arxiv=%s, doi=%s",
                    detection.arxiv_id,
                    detection.doi,
                )

        return result

    def _resolve_uncached(self, entry: dict[str, Any], detection: PreprintDetection) -> PublishedRecord | None:
        """Internal resolution logic without caching.

        Orchestrates a 6-stage resolution pipeline, returning early if any stage succeeds:
        1. Direct lookup: arXiv -> Semantic Scholar / Crossref
        2. Crossref relations: is-preprint-of links
        3. DBLP bibliographic search
        4. Semantic Scholar search
        5. Crossref bibliographic search
        6. Google Scholar fallback (opt-in)
        """
        # Normalize title once for stages 3-6
        title = entry.get("title") or ""
        title_norm = normalize_title_for_match(title)

        # Stage 1: Direct lookup (arXiv -> S2 -> Crossref)
        result, candidate_doi = self._stage1_direct_lookup(detection)
        if result:
            return result

        # Stage 2: Crossref relations (is-preprint-of)
        result = self._stage2_crossref_relations(detection, candidate_doi)
        if result:
            return result

        # Stage 3: DBLP bibliographic search
        result = self._stage3_dblp_search(entry, title_norm)
        if result:
            return result

        # Stage 4: Semantic Scholar search
        result = self._stage4_s2_search(entry, title_norm)
        if result:
            return result

        # Stage 5: Crossref bibliographic search
        result = self._stage5_crossref_search(entry, title_norm)
        if result:
            return result

        # Stage 6: Google Scholar fallback (opt-in only)
        result = self._stage6_scholarly_search(entry, title_norm)
        if result:
            return result

        return None

    def _stage1_direct_lookup(self, detection: PreprintDetection) -> tuple[PublishedRecord | None, str | None]:
        """Stage 1: Direct lookup via arXiv -> Semantic Scholar / Crossref.

        Attempts to find the published version using direct mappings:
        - Semantic Scholar arXiv paper lookup
        - arXiv metadata DOI -> Crossref works lookup
        - arXiv metadata DOI -> Crossref is-preprint-of relation

        Args:
            detection: PreprintDetection with arxiv_id and/or doi

        Returns:
            Tuple of (PublishedRecord if found, candidate_doi from arXiv for later stages)
        """
        candidate_doi: str | None = None

        if not detection.arxiv_id:
            return None, None

        # Semantic Scholar first (direct arXiv mapping is strong)
        s2 = self.s2_from_arxiv(detection.arxiv_id)
        if s2 and self._credible_journal_article(s2):
            self.logger.debug("Stage 1: Found via Semantic Scholar arXiv lookup")
            return s2, None

        # Get DOI from arXiv metadata
        candidate_doi = self.arxiv_candidate_doi(detection.arxiv_id)
        if not candidate_doi:
            return None, None

        # Try Crossref lookup with the candidate DOI
        msg = self.crossref_get(candidate_doi)
        if not msg:
            return None, candidate_doi

        # Check if the candidate DOI points to a journal article
        rec = self._message_to_record(msg)
        if rec and rec.type == "journal-article" and self._credible_journal_article(rec):
            rec.method = "arXiv->Crossref(works)"
            rec.confidence = 1.0
            self.logger.debug("Stage 1: Found via arXiv->Crossref(works)")
            return rec, candidate_doi

        # Check is-preprint-of relations in the candidate DOI
        rec_rel = self._check_preprint_of_relations(msg, "arXiv->Crossref(relation)")
        if rec_rel:
            return rec_rel, candidate_doi

        return None, candidate_doi

    def _check_preprint_of_relations(self, msg: dict[str, Any], method_name: str) -> PublishedRecord | None:
        """Check is-preprint-of relations in a Crossref message.

        Args:
            msg: Crossref message dict
            method_name: Method name to set on the returned record

        Returns:
            PublishedRecord if a credible journal article is found, None otherwise
        """
        rel = msg.get("relation") or {}
        pre_of = rel.get("is-preprint-of") or []
        for node in pre_of:
            if node.get("id-type") == "doi" and node.get("id"):
                pub_doi = doi_normalize(node["id"])
                pub_msg = self.crossref_get(pub_doi)
                rec = self._message_to_record(pub_msg or {})
                if rec and self._credible_journal_article(rec):
                    rec.method = method_name
                    rec.confidence = 1.0
                    self.logger.debug("Found via %s", method_name)
                    return rec
        return None

    def _stage2_crossref_relations(
        self, detection: PreprintDetection, candidate_doi: str | None
    ) -> PublishedRecord | None:
        """Stage 2: Crossref is-preprint-of relation lookup.

        Checks if the preprint DOI or candidate DOI has an is-preprint-of relation
        pointing to the published version.

        Args:
            detection: PreprintDetection with arxiv_id and/or doi
            candidate_doi: DOI from arXiv metadata (from stage 1)

        Returns:
            PublishedRecord if found, None otherwise
        """
        for d in filter(None, (detection.doi, candidate_doi)):
            msg = self.crossref_get(d)
            if not msg:
                continue

            # Check is-preprint-of relations
            rec = self._check_preprint_of_relations(msg, "Crossref(relation)")
            if rec:
                self.logger.debug("Stage 2: Found via Crossref(relation)")
                return rec

            # Check if the DOI itself is a journal article
            rec0 = self._message_to_record(msg)
            if rec0 and self._credible_journal_article(rec0):
                rec0.method = "Crossref(works)"
                rec0.confidence = 1.0
                self.logger.debug("Stage 2: Found via Crossref(works)")
                return rec0

        return None

    def _stage3_dblp_search(self, entry: dict[str, Any], title_norm: str) -> PublishedRecord | None:
        """Stage 3: DBLP bibliographic search.

        Searches DBLP by title and first author, scoring candidates by title
        and author similarity.

        Args:
            entry: BibTeX entry dict
            title_norm: Normalized title for matching

        Returns:
            PublishedRecord if a credible match is found, None otherwise
        """
        if not title_norm:
            return None

        first_author = first_author_surname(entry)
        dblp_query = f"{title_norm} {first_author}".strip()
        hits = self.dblp_search(dblp_query, h=30)

        if not hits:
            return None

        authors_ref = authors_last_names(entry.get("author", ""))
        best: tuple[float, PublishedRecord] | None = None

        for h in hits:
            rec = self._dblp_hit_to_record(h)
            if not rec:
                continue

            combined = self._compute_match_score(title_norm, rec, authors_ref)

            if combined >= 0.9 and self._credible_journal_article(rec):
                rec.method = "DBLP(search)"
                rec.confidence = combined
                if self._is_better_candidate(combined, rec, best):
                    best = (combined, rec)

        if best:
            self.logger.debug("Stage 3: Found via DBLP(search) with score %.2f", best[0])
            return best[1]

        return None

    def _compute_match_score(self, title_norm: str, rec: PublishedRecord, authors_ref: list[str]) -> float:
        """Compute combined title and author match score.

        Args:
            title_norm: Normalized reference title
            rec: Candidate PublishedRecord
            authors_ref: Reference author last names

        Returns:
            Combined score (0.0 to 1.0)
        """
        tb = normalize_title_for_match(rec.title or "")
        title_score = token_sort_ratio(title_norm, tb)  # 0..100
        blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
        auth_score = jaccard_similarity(authors_ref[:3], blns)
        return 0.7 * (title_score / 100.0) + 0.3 * auth_score

    def _stage4_s2_search(self, entry: dict[str, Any], title_norm: str) -> PublishedRecord | None:
        """Stage 4: Semantic Scholar search.

        Searches Semantic Scholar by title and first author, scoring candidates
        by title and author similarity.

        Args:
            entry: BibTeX entry dict
            title_norm: Normalized title for matching

        Returns:
            PublishedRecord if a credible match is found, None otherwise
        """
        if not title_norm:
            return None

        first_author = first_author_surname(entry)
        s2_query = f"{title_norm} {first_author}".strip()
        data = self.s2_search(s2_query, limit=25)

        if not data:
            return None

        authors_ref = authors_last_names(entry.get("author", ""))
        candidates: list[tuple[float, PublishedRecord]] = []

        for item in data:
            rec = self._s2_item_to_record(item)
            if not rec:
                continue

            combined = self._compute_match_score(title_norm, rec, authors_ref)

            if combined >= 0.9 and self._credible_journal_article(rec):
                rec.method = "SemanticScholar(search)"
                rec.confidence = combined
                candidates.append((combined, rec))

        if candidates:
            candidates.sort(
                key=lambda x: (
                    x[0],
                    int(bool(x[1].pages)) + int(bool(x[1].volume)) + int(bool(x[1].number)),
                ),
                reverse=True,
            )
            self.logger.debug(
                "Stage 4: Found via SemanticScholar(search) with score %.2f",
                candidates[0][0],
            )
            return candidates[0][1]

        return None

    def _s2_item_to_record(self, item: dict[str, Any]) -> PublishedRecord | None:
        """Convert Semantic Scholar search item to PublishedRecord.

        Args:
            item: Semantic Scholar search result item

        Returns:
            PublishedRecord if item is a valid journal article with DOI, None otherwise
        """
        doi = doi_normalize(item.get("doi"))
        pub_types = item.get("publicationTypes") or []
        is_journal = any(pt.lower() == "journalarticle" for pt in pub_types)

        if not (doi and is_journal):
            return None

        return PublishedRecord(
            doi=doi,
            url=doi_url(doi),
            title=item.get("title"),
            authors=[
                {
                    "given": n.split()[:-1] and " ".join(n.split()[:-1]) or "",
                    "family": (n.split()[-1] if n.split() else ""),
                }
                for n in [a.get("name", "") for a in (item.get("authors") or [])]
            ],
            journal=item.get("venue"),
            year=item.get("year"),
            type="journal-article",
        )

    def _stage5_crossref_search(self, entry: dict[str, Any], title_norm: str) -> PublishedRecord | None:
        """Stage 5: Crossref bibliographic search.

        Searches Crossref by title and first author as a fallback when direct
        lookups fail. Uses relaxed matching (0.85 threshold) if strict matching fails.

        Args:
            entry: BibTeX entry dict
            title_norm: Normalized title for matching

        Returns:
            PublishedRecord if a credible match is found, None otherwise
        """
        if not title_norm:
            return None

        first_author = first_author_surname(entry)
        query = f"{title_norm} {first_author}".strip()
        items = self.crossref_search(query, rows=30, filter_type="journal-article")

        if not items:
            return None

        authors_a = authors_last_names(entry.get("author", ""))
        candidates = [self._score_crossref_item(item, title_norm, authors_a) for item in items]

        # Filter passing candidates (score >= 0.9 and credible)
        passing = [(s, r) for (s, r) in candidates if (s >= 0.9 and self._credible_journal_article(r))]

        if not passing:
            return self._try_relaxed_crossref_match(candidates)

        # Sort by score, tie-break by metadata completeness
        passing.sort(
            key=lambda x: (
                x[0],
                int(bool(x[1].pages)) + int(bool(x[1].volume)) + int(bool(x[1].number)),
            ),
            reverse=True,
        )
        score, best = passing[0]
        best.method = "Crossref(search)"
        best.confidence = score
        self.logger.debug("Stage 5: Found via Crossref(search) with score %.2f", score)
        return best

    def _try_relaxed_crossref_match(self, candidates: list[tuple[float, PublishedRecord]]) -> PublishedRecord | None:
        """Try relaxed matching on Crossref candidates.

        Args:
            candidates: List of (score, record) tuples

        Returns:
            PublishedRecord if relaxed match found, None otherwise
        """
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[0] if candidates else None
        if top and top[0] >= 0.85 and self._credible_journal_article(top[1]):
            rec = top[1]
            rec.method = "Crossref(search,relaxed)"
            rec.confidence = top[0]
            self.logger.debug("Stage 5: Found via Crossref(search,relaxed) with score %.2f", top[0])
            return rec
        return None

    def _score_crossref_item(
        self, msg: dict[str, Any], title_norm: str, authors_ref: list[str]
    ) -> tuple[float, PublishedRecord]:
        """Score a Crossref search result for matching.

        Args:
            msg: Crossref message dict
            title_norm: Normalized reference title
            authors_ref: Reference author last names

        Returns:
            Tuple of (combined score, PublishedRecord)
        """
        rec = self._message_to_record(msg)
        if not rec or rec.type != "journal-article":
            return (0.0, PublishedRecord(doi=""))

        combined = self._compute_match_score(title_norm, rec, authors_ref)
        return (combined, rec)

    def _stage6_scholarly_search(self, entry: dict[str, Any], title_norm: str) -> PublishedRecord | None:
        """Stage 6: Google Scholar fallback (opt-in only).

        Uses the scholarly library to search Google Scholar. Only enabled when
        scholarly_client is configured.

        Args:
            entry: BibTeX entry dict
            title_norm: Normalized title for matching

        Returns:
            PublishedRecord if a credible match is found, None otherwise
        """
        if not self.scholarly_client or not title_norm:
            return None

        first_author = first_author_surname(entry)
        pub = self.scholarly_client.search(title_norm, first_author)

        if not pub:
            return None

        rec = self._scholarly_to_record(pub)
        if not rec or not rec.title:
            return None

        authors_ref = authors_last_names(entry.get("author", ""))
        combined = self._compute_match_score(title_norm, rec, authors_ref)

        if combined >= 0.9 and self._credible_journal_article(rec):
            rec.method = "GoogleScholar(search)"
            rec.confidence = combined
            tb = normalize_title_for_match(rec.title)
            title_score = token_sort_ratio(title_norm, tb)
            self.logger.debug("Stage 6: Google Scholar match: %.2f (title=%.0f)", combined, title_score)
            return rec

        return None

    @staticmethod
    def _is_better_candidate(
        score: float, rec: PublishedRecord, current_best: tuple[float, PublishedRecord] | None
    ) -> bool:
        """Check if a candidate is better than the current best.

        Args:
            score: Combined match score for the candidate
            rec: Candidate PublishedRecord
            current_best: Current best (score, record) tuple, or None

        Returns:
            True if candidate should replace current best
        """
        if current_best is None:
            return True

        if score > current_best[0]:
            return True

        if score == current_best[0]:
            # Tie-break by metadata completeness
            rec_completeness = int(bool(rec.pages)) + int(bool(rec.volume)) + int(bool(rec.number))
            best_completeness = (
                int(bool(current_best[1].pages)) + int(bool(current_best[1].volume)) + int(bool(current_best[1].number))
            )
            return rec_completeness > best_completeness

        return False


# ------------- Async Resolver -------------
class AsyncResolver:
    """Async resolver for bibliographic searches with parallel execution.

    This class provides async versions of the Resolver methods, allowing
    concurrent searches across multiple APIs (DBLP, Semantic Scholar, Crossref).
    """

    # Accepted publication types for upgrades (same as sync Resolver)
    ACCEPTED_TYPES = {
        "journal-article",
        "proceedings-article",
        "book-chapter",
    }

    ACCEPTED_S2_TYPES = {
        "journalarticle",
        "conference",
    }

    def __init__(
        self,
        http: AsyncHttpClient,
        logger: logging.Logger,
    ) -> None:
        """Initialize the async resolver.

        Args:
            http: AsyncHttpClient instance for making HTTP requests
            logger: Logger for debug/info messages
        """

        self.http: AsyncHttpClient = http
        self.logger = logger

    # --- Shared helpers (static, same as sync Resolver) ---
    @staticmethod
    def _message_to_record(msg: dict[str, Any]) -> PublishedRecord | None:
        """Convert Crossref message to PublishedRecord."""
        return crossref_message_to_record(msg)

    @staticmethod
    def _dblp_hit_to_record(hit: dict[str, Any]) -> PublishedRecord | None:
        """Convert DBLP hit to PublishedRecord."""
        return dblp_hit_to_record(hit)

    @staticmethod
    def _credible_journal_article(rec: PublishedRecord) -> bool:
        """Check if a record is a credible journal article."""
        if rec.type != "journal-article":
            return False
        if not rec.journal:
            return False
        j_lower = rec.journal.lower()
        if any(host in j_lower for host in PREPRINT_HOSTS):
            return False
        if not rec.year:
            return False
        return bool(rec.volume or rec.number or rec.pages or rec.url)

    @staticmethod
    def _authors_to_bibtex_string(rec: PublishedRecord) -> str:
        """Convert record authors to BibTeX author string."""
        return Resolver._authors_to_bibtex_string(rec)

    # --- Async API methods ---
    async def arxiv_candidate_doi(self, arxiv_id: str) -> str | None:
        """Fetch DOI from arXiv API for an arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., '2301.00001')

        Returns:
            DOI if found, None otherwise
        """
        params = {"id_list": arxiv_id}
        try:
            resp = await self.http.get(
                ARXIV_API,
                service="arxiv",
                params=params,
                accept="application/atom+xml",
            )
            xml = resp.text
        except Exception as e:
            self.logger.debug("arXiv lookup failed for %s: %s", arxiv_id, e)
            return None

        m = re.search(r"<arxiv:doi>([^<]+)</arxiv:doi>", xml, flags=re.IGNORECASE)
        if m:
            return doi_normalize(m.group(1))
        m = re.search(r"<doi>([^<]+)</doi>", xml, flags=re.IGNORECASE)
        if m:
            return doi_normalize(m.group(1))
        return None

    async def crossref_get(self, doi: str) -> dict[str, Any] | None:
        """Fetch Crossref works data for a DOI.

        Args:
            doi: The DOI to look up

        Returns:
            Crossref works message dict if found, None otherwise
        """
        from urllib.parse import quote

        doi = doi_normalize(doi) or ""
        url = f"{CROSSREF_API}/{quote(doi, safe='')}"
        try:
            resp = await self.http.get(url, service="crossref", accept="application/json")
            if resp.status_code != 200:
                return None
            data = resp.json().get("message", {})
            return data
        except Exception as e:
            self.logger.debug("Crossref works failed for %s: %s", doi, e)
            return None

    async def crossref_search(
        self,
        query: str,
        rows: int = 25,
        filter_type: str | None = "journal-article",
    ) -> list[dict[str, Any]]:
        """Search Crossref by bibliographic query.

        Args:
            query: Bibliographic search query
            rows: Maximum number of results
            filter_type: Filter by document type (e.g., 'journal-article')

        Returns:
            List of Crossref work items
        """
        params: dict[str, Any] = {"query.bibliographic": query, "rows": rows}
        if filter_type:
            params["filter"] = f"type:{filter_type}"
        try:
            resp = await self.http.get(
                CROSSREF_API,
                service="crossref",
                params=params,
                accept="application/json",
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
            return items
        except Exception as e:
            self.logger.debug("Crossref search failed '%s': %s", query, e)
            return []

    async def dblp_search(self, query: str, h: int = 25) -> list[dict[str, Any]]:
        """Search DBLP by bibliographic query.

        Args:
            query: Bibliographic search query
            h: Maximum number of results

        Returns:
            List of DBLP hit objects
        """
        params = {"q": query, "h": h, "format": "json"}
        try:
            resp = await self.http.get(
                DBLP_API_SEARCH,
                service="dblp",
                params=params,
                accept="application/json",
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(hits, dict):
                hits = [hits]
            return hits
        except Exception as e:
            self.logger.debug("DBLP search failed '%s': %s", query, e)
            return []

    async def s2_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search Semantic Scholar by query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of Semantic Scholar paper data
        """
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,venue,publicationTypes,doi,url",
        }
        url = f"{S2_API}/paper/search"
        try:
            resp = await self.http.get(
                url,
                service="semanticscholar",
                params=params,
                accept="application/json",
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("data", []) or []
        except Exception as e:
            self.logger.debug("Semantic Scholar search failed '%s': %s", query, e)
            return []

    async def s2_from_arxiv(self, arxiv_id: str) -> PublishedRecord | None:
        """Fetch published version info from Semantic Scholar using arXiv ID.

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            PublishedRecord if a journal article is found, None otherwise
        """
        fields = "externalIds,doi,title,year,authors,venue,publicationTypes,publicationVenue,url"
        url = f"{S2_API}/paper/arXiv:{arxiv_id}"
        try:
            resp = await self.http.get(
                url,
                service="semanticscholar",
                params={"fields": fields},
                accept="application/json",
            )
            if resp.status_code != 200:
                return None
            msg = resp.json()
        except Exception as e:
            self.logger.debug("Semantic Scholar arXiv lookup failed for %s: %s", arxiv_id, e)
            return None

        doi = doi_normalize(msg.get("doi") or (msg.get("externalIds") or {}).get("DOI"))
        pub_types = msg.get("publicationTypes") or []
        is_journal = any(pt.lower() == "journalarticle" for pt in pub_types)
        if not (doi and is_journal):
            return None

        title = msg.get("title")
        year = msg.get("year")
        venue = (msg.get("publicationVenue") or {}).get("name") or msg.get("venue")
        authors = [
            {
                "given": (a.get("name") or "").split()[:-1] and " ".join((a.get("name") or "").split()[:-1]) or "",
                "family": (a.get("name") or "").split()[-1] if (a.get("name") or "").split() else "",
            }
            for a in (msg.get("authors") or [])
        ]
        return PublishedRecord(
            doi=doi,
            url=doi_url(doi),
            title=title,
            authors=authors,
            journal=venue,
            year=year,
            type="journal-article",
            method="SemanticScholar(arXiv)",
            confidence=0.95,
        )

    # --- Parallel search ---
    async def parallel_bibliographic_search(
        self,
        title: str,
        authors: list[str],
    ) -> PublishedRecord | None:
        """Run DBLP, Semantic Scholar, and Crossref searches in parallel.

        This method executes bibliographic searches against all three APIs
        concurrently, then returns the highest confidence result that passes
        the matching threshold.

        Args:
            title: Normalized title for matching
            authors: List of author last names for matching

        Returns:
            Best matching PublishedRecord, or None if no good match found
        """
        import asyncio

        if not title:
            return None

        query = f"{title} {authors[0] if authors else ''}".strip()
        title_norm = title

        async def dblp_search_task() -> list[tuple[float, PublishedRecord]]:
            """Search DBLP and score results."""
            candidates: list[tuple[float, PublishedRecord]] = []
            try:
                hits = await self.dblp_search(query, h=30)
                for hit in hits:
                    rec = self._dblp_hit_to_record(hit)
                    if not rec:
                        continue
                    tb = normalize_title_for_match(rec.title or "")
                    title_score = token_sort_ratio(title_norm, tb)
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors[:3], blns)
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    if combined >= 0.9 and self._credible_journal_article(rec):
                        rec.method = "DBLP(search,parallel)"
                        rec.confidence = combined
                        candidates.append((combined, rec))
            except Exception as e:
                self.logger.debug("DBLP parallel search failed: %s", e)
            return candidates

        async def s2_search_task() -> list[tuple[float, PublishedRecord]]:
            """Search Semantic Scholar and score results."""
            candidates: list[tuple[float, PublishedRecord]] = []
            try:
                data = await self.s2_search(query, limit=25)
                for item in data:
                    doi = doi_normalize(item.get("doi"))
                    pub_types = item.get("publicationTypes") or []
                    is_journal = any(pt.lower() == "journalarticle" for pt in pub_types)
                    if not (doi and is_journal):
                        continue
                    rec = PublishedRecord(
                        doi=doi,
                        url=doi_url(doi),
                        title=item.get("title"),
                        authors=[
                            {
                                "given": n.split()[:-1] and " ".join(n.split()[:-1]) or "",
                                "family": (n.split()[-1] if n.split() else ""),
                            }
                            for n in [a.get("name", "") for a in (item.get("authors") or [])]
                        ],
                        journal=item.get("venue"),
                        year=item.get("year"),
                        type="journal-article",
                    )
                    tb = normalize_title_for_match(rec.title or "")
                    title_score = token_sort_ratio(title_norm, tb)
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors[:3], blns)
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    if combined >= 0.9 and self._credible_journal_article(rec):
                        rec.method = "SemanticScholar(search,parallel)"
                        rec.confidence = combined
                        candidates.append((combined, rec))
            except Exception as e:
                self.logger.debug("S2 parallel search failed: %s", e)
            return candidates

        async def crossref_search_task() -> list[tuple[float, PublishedRecord]]:
            """Search Crossref and score results."""
            candidates: list[tuple[float, PublishedRecord]] = []
            try:
                items = await self.crossref_search(query, rows=30, filter_type="journal-article")
                for item in items:
                    rec = self._message_to_record(item)
                    if not rec or rec.type != "journal-article":
                        continue
                    tb = normalize_title_for_match(rec.title or "")
                    title_score = token_sort_ratio(title_norm, tb)
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors[:3], blns)
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    if combined >= 0.9 and self._credible_journal_article(rec):
                        rec.method = "Crossref(search,parallel)"
                        rec.confidence = combined
                        candidates.append((combined, rec))
            except Exception as e:
                self.logger.debug("Crossref parallel search failed: %s", e)
            return candidates

        # Run all searches concurrently
        results = await asyncio.gather(
            dblp_search_task(),
            s2_search_task(),
            crossref_search_task(),
            return_exceptions=True,
        )

        # Collect all valid candidates from all sources
        all_candidates: list[tuple[float, PublishedRecord]] = []
        for result in results:
            if isinstance(result, list):
                all_candidates.extend(result)
            elif isinstance(result, Exception):
                self.logger.debug("Search task raised exception: %s", result)

        if not all_candidates:
            return None

        # Sort by (confidence, metadata completeness) and return best
        all_candidates.sort(
            key=lambda x: (
                x[0],
                int(bool(x[1].pages)) + int(bool(x[1].volume)) + int(bool(x[1].number)),
            ),
            reverse=True,
        )
        return all_candidates[0][1]

    async def resolve(
        self,
        entry: dict[str, Any],
        detection: PreprintDetection,
    ) -> PublishedRecord | None:
        """Async resolve a preprint entry to its published version.

        Uses a multi-stage resolution strategy:
        1. Semantic Scholar arXiv lookup (if arXiv ID present)
        2. arXiv API -> DOI -> Crossref lookup
        3. Crossref relations (is-preprint-of)
        4. Parallel bibliographic search (DBLP, S2, Crossref)
        5. Relaxed Crossref search fallback

        Args:
            entry: BibTeX entry dict
            detection: PreprintDetection result

        Returns:
            PublishedRecord if found, None otherwise
        """
        # 1) arXiv -> DOI -> Crossref
        candidate_doi: str | None = None
        if detection.arxiv_id:
            # Semantic Scholar first (direct arXiv mapping is strong)
            s2 = await self.s2_from_arxiv(detection.arxiv_id)
            if s2 and self._credible_journal_article(s2):
                return s2

            candidate_doi = await self.arxiv_candidate_doi(detection.arxiv_id)
            if candidate_doi:
                msg = await self.crossref_get(candidate_doi)
                if msg:
                    rec = self._message_to_record(msg)
                    if rec and rec.type == "journal-article" and self._credible_journal_article(rec):
                        rec.method = "arXiv->Crossref(works)"
                        rec.confidence = 1.0
                        return rec
                    # Check relations
                    rel = msg.get("relation") or {}
                    pre_of = rel.get("is-preprint-of") or []
                    for node in pre_of:
                        if node.get("id-type") == "doi" and node.get("id"):
                            pub_doi = doi_normalize(node["id"])
                            pub_msg = await self.crossref_get(pub_doi)
                            rec2 = self._message_to_record(pub_msg or {})
                            if rec2 and self._credible_journal_article(rec2):
                                rec2.method = "arXiv->Crossref(relation)"
                                rec2.confidence = 1.0
                                return rec2

        # 2) Crossref by DOI (preprint DOI or candidate)
        for d in filter(None, (detection.doi, candidate_doi)):
            msg = await self.crossref_get(d)
            if msg:
                rel = msg.get("relation") or {}
                pre_of = rel.get("is-preprint-of") or []
                for node in pre_of:
                    if node.get("id-type") == "doi" and node.get("id"):
                        pub_doi = doi_normalize(node["id"])
                        pub_msg = await self.crossref_get(pub_doi)
                        rec2 = self._message_to_record(pub_msg or {})
                        if rec2 and self._credible_journal_article(rec2):
                            rec2.method = "Crossref(relation)"
                            rec2.confidence = 1.0
                            return rec2
                rec0 = self._message_to_record(msg)
                if rec0 and self._credible_journal_article(rec0):
                    rec0.method = "Crossref(works)"
                    rec0.confidence = 1.0
                    return rec0

        # 3) Parallel bibliographic search (DBLP, S2, Crossref)
        title = entry.get("title") or ""
        title_norm = normalize_title_for_match(title)
        if title_norm:
            authors_ref = authors_last_names(entry.get("author", ""))
            rec = await self.parallel_bibliographic_search(title_norm, authors_ref)
            if rec:
                return rec

        return None


# ------------- Updater -------------
class Updater:
    PREPRINT_ONLY_FIELDS = {
        "eprint",
        "archiveprefix",
        "archivePrefix",
        "primaryClass",
        "primaryclass",
        "eprinttype",
        "eprintclass",
    }

    def __init__(self, keep_preprint_note: bool = False, rekey: bool = False) -> None:
        self.keep_preprint_note = keep_preprint_note
        self.rekey = rekey

    @staticmethod
    def _author_bibtex_from_record(rec: PublishedRecord) -> str:
        return Resolver._authors_to_bibtex_string(rec)

    @staticmethod
    def _year_from_record(rec: PublishedRecord) -> str | None:
        return str(rec.year) if rec.year else None

    @staticmethod
    def _generate_key(entry: dict[str, Any], rec: PublishedRecord) -> str:
        first_author = ""
        if rec.authors:
            fa = rec.authors[0]
            first_author = (fa.get("family") or fa.get("given") or "").split()[-1]
        elif entry.get("author"):
            first_author = last_name_from_person(split_authors_bibtex(entry["author"])[0])
        year = rec.year or entry.get("year") or "n.d."
        title = normalize_title_for_match(rec.title or entry.get("title") or "")
        title = "".join(w for w in re.split(r"\s+", title) if w)[:40]
        key = f"{first_author}{year}{title}"
        key = re.sub(r"[^A-Za-z0-9]+", "", key)
        return key or (entry.get("ID") or "key")

    def update_entry(self, entry: dict[str, Any], rec: PublishedRecord, detection: PreprintDetection) -> dict[str, Any]:
        new_entry = dict(entry)
        new_entry["ENTRYTYPE"] = "article"
        if rec.title:
            new_entry["title"] = rec.title
        if rec.authors:
            new_entry["author"] = self._author_bibtex_from_record(rec)
        if rec.journal:
            new_entry["journal"] = rec.journal
        if rec.publisher:
            new_entry["publisher"] = rec.publisher
        if rec.year:
            new_entry["year"] = str(rec.year)
        if rec.volume:
            new_entry["volume"] = str(rec.volume)
        if rec.number:
            new_entry["number"] = str(rec.number)
        if rec.pages:
            new_entry["pages"] = str(rec.pages)
        if rec.doi:
            new_entry["doi"] = rec.doi
            new_entry["url"] = doi_url(rec.doi)
        elif rec.url:
            new_entry["url"] = rec.url

        for f in list(self.PREPRINT_ONLY_FIELDS):
            if f in new_entry:
                new_entry.pop(f, None)

        if self.keep_preprint_note:
            arx = detection.arxiv_id or extract_arxiv_id_from_text(entry.get("url", "") or entry.get("note", "") or "")
            if arx:
                note = new_entry.get("note", "")
                msg = f"Also available as arXiv:{arx}"
                if "also available as arxiv:" not in safe_lower(note):
                    new_entry["note"] = (note + (" " if note else "") + msg).strip()

        if self.rekey:
            new_entry["ID"] = self._generate_key(entry, rec)
        else:
            new_entry["ID"] = entry.get("ID")

        return new_entry


# ------------- Dedupe -------------
class Dedupe:
    @staticmethod
    def _key(entry: dict[str, Any]) -> tuple[str, str]:
        doi = doi_normalize(entry.get("doi") or "")
        if doi:
            return ("doi", doi)
        title = normalize_title_for_match(entry.get("title") or "")
        auths = authors_last_names(entry.get("author", ""))[:3]
        key = title + "|" + ",".join(sorted(auths))
        return ("fuzzy", key)

    @staticmethod
    def _score(entry: dict[str, Any]) -> int:
        score = 0
        if safe_lower(entry.get("ENTRYTYPE")) == "article":
            score += 5
        for f in ("title", "author", "journal", "year", "volume", "number", "pages", "doi", "url", "publisher"):
            if entry.get(f):
                score += 1
        return score

    @staticmethod
    def merge_entries(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        best = a if Dedupe._score(a) >= Dedupe._score(b) else b
        other = b if best is a else a
        merged = dict(best)
        for k, v in other.items():
            if k in {"ID", "ENTRYTYPE"}:
                continue
            if not merged.get(k) and v:
                merged[k] = v
        merged["ID"] = best.get("ID")
        merged["ENTRYTYPE"] = best.get("ENTRYTYPE")
        return merged

    def dedupe_db(
        self, db: bibtexparser.bibdatabase.BibDatabase, logger: logging.Logger
    ) -> tuple[bibtexparser.bibdatabase.BibDatabase, list[tuple[str, list[str]]]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for e in db.entries:
            groups.setdefault(self._key(e), []).append(e)

        new_entries: list[dict[str, Any]] = []
        merged_info: list[tuple[str, list[str]]] = []

        for (_k, _), entries in groups.items():
            if len(entries) == 1:
                new_entries.append(entries[0])
                continue
            base = entries[0]
            ids = [base.get("ID")]
            for other in entries[1:]:
                ids.append(other.get("ID"))
                base = self.merge_entries(base, other)
            new_entries.append(base)
            merged_info.append((base.get("ID"), ids))
            logger.info("Merged duplicates into %s from %s", base.get("ID"), ids)

        nd = bibtexparser.bibdatabase.BibDatabase()
        nd.entries = new_entries
        return nd, merged_info


# ------------- Diff Preview -------------
def entry_to_bib(entry: dict[str, Any]) -> str:
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = [entry]
    writer = BibWriter()
    return writer.dumps(db).strip()


def diff_entries(old: dict[str, Any], new: dict[str, Any], key: str) -> str:
    a = entry_to_bib(old).splitlines(keepends=True)
    b = entry_to_bib(new).splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=f"{key} (old)", tofile=f"{key} (new)", lineterm=""))


# ------------- Processing Pipeline -------------
@dataclass
class ProcessResult:
    original: dict[str, Any]
    updated: dict[str, Any]
    changed: bool
    action: str
    method: str | None = None
    confidence: float | None = None
    message: str | None = None


# ------------- Entry Prioritization -------------
def prioritize_entries(
    entries: list[dict[str, Any]], detector: Detector
) -> list[tuple[int, dict[str, Any], PreprintDetection]]:
    """Sort entries by resolution likelihood for optimal processing order.

    Prioritizes entries based on available identifiers:
    - Priority 4: Both arXiv ID and DOI available (best case)
    - Priority 3: arXiv ID only (S2 direct match likely)
    - Priority 2: DOI only (Crossref direct match likely)
    - Priority 1: Bibliographic search only (slowest)

    Args:
        entries: List of BibTeX entry dicts
        detector: Detector instance for preprint detection

    Returns:
        List of (priority, entry, detection) tuples sorted by priority descending.
        Higher priority = more likely to resolve quickly.
    """
    prioritized: list[tuple[int, dict[str, Any], PreprintDetection]] = []

    for entry in entries:
        det = detector.detect(entry)
        if not det.is_preprint:
            continue  # Skip non-preprints

        # Assign priority based on available identifiers
        if det.arxiv_id and det.doi:
            priority = 4  # Best case: both IDs available
        elif det.arxiv_id:
            priority = 3  # arXiv ID -> S2 direct match likely
        elif det.doi:
            priority = 2  # DOI -> Crossref direct match likely
        else:
            priority = 1  # Bibliographic search only (slowest)

        prioritized.append((priority, entry, det))

    # Sort by priority descending (highest priority first)
    prioritized.sort(key=lambda x: x[0], reverse=True)
    return prioritized


def process_entry_with_preload(
    entry: dict[str, Any],
    detection: PreprintDetection,
    preload_data: dict[str, Any],
    resolver: Resolver,
    updater: Updater,
    logger: logging.Logger,
) -> ProcessResult:
    """Process single entry using pre-loaded data when available.

    Args:
        entry: BibTeX entry dict
        detection: PreprintDetection result for this entry
        preload_data: Pre-loaded API data (e.g., {'s2_record': PublishedRecord})
        resolver: Resolver instance
        updater: Updater instance
        logger: Logger instance

    Returns:
        ProcessResult with the processing outcome
    """
    # Check for pre-loaded S2 result
    if "s2_record" in preload_data:
        record = preload_data["s2_record"]
        if record and resolver._credible_journal_article(record):
            new_entry = updater.update_entry(entry, record, detection)
            changed = json.dumps(entry, sort_keys=True) != json.dumps(new_entry, sort_keys=True)
            return ProcessResult(
                original=entry,
                updated=new_entry,
                changed=changed,
                action="upgraded",
                method="batch:S2",
                confidence=record.confidence,
            )

    # Fall back to regular resolution
    record = resolver.resolve(entry, detection)
    if not record:
        return ProcessResult(
            original=entry,
            updated=entry,
            changed=False,
            action="failed",
            message="No reliable published match found",
        )

    if record.type != "journal-article":
        return ProcessResult(
            original=entry,
            updated=entry,
            changed=False,
            action="failed",
            message="Candidate not a journal-article",
        )

    if not resolver._credible_journal_article(record):
        return ProcessResult(
            original=entry,
            updated=entry,
            changed=False,
            action="failed",
            message="Candidate lacks sufficient metadata",
        )

    new_entry = updater.update_entry(entry, record, detection)
    changed = json.dumps(entry, sort_keys=True) != json.dumps(new_entry, sort_keys=True)
    return ProcessResult(
        original=entry,
        updated=new_entry,
        changed=changed,
        action="upgraded",
        method=record.method,
        confidence=record.confidence,
    )


def process_entries_optimized(
    entries: list[dict[str, Any]],
    detector: Detector,
    resolver: Resolver,
    updater: Updater,
    logger: logging.Logger,
    max_workers: int = 8,
) -> list[ProcessResult]:
    """Process entries with prioritization and batch optimization.

    Processing pipeline:
    1. Detect and prioritize all entries by resolution likelihood
    2. Batch pre-load data for high-priority entries (if resolver supports it)
    3. Process entries in priority order with concurrent execution

    Args:
        entries: List of BibTeX entry dicts
        detector: Detector instance for preprint detection
        resolver: Resolver instance for finding published versions
        updater: Updater instance for applying updates
        logger: Logger instance
        max_workers: Maximum concurrent workers (default 8)

    Returns:
        List of ProcessResult for all entries (including unchanged non-preprints)
    """
    # Build a map from entry to its original index for result ordering
    entry_to_idx = {id(e): i for i, e in enumerate(entries)}

    # Phase 1: Detect and prioritize preprints
    prioritized = prioritize_entries(entries, detector)

    if not prioritized:
        # No preprints found, return unchanged results for all entries
        return [ProcessResult(original=e, updated=e, changed=False, action="unchanged") for e in entries]

    # Extract entries and detections in priority order
    entries_sorted = [item[1] for item in prioritized]
    detections = [item[2] for item in prioritized]

    # Phase 2: Batch pre-load (if resolver supports it)
    preloaded: dict[str, dict[str, Any]] = {}
    if hasattr(resolver, "batch_preload"):
        try:
            preloaded = resolver.batch_preload(entries_sorted, detections)
            logger.debug("Batch pre-loaded data for %d entries", len(preloaded))
        except Exception as e:
            logger.warning("Batch pre-load failed: %s", e)
            preloaded = {}

    # Phase 3: Process entries with pre-loaded data
    preprint_results: dict[int, ProcessResult] = {}  # Map original index to result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_entry: dict[concurrent.futures.Future, tuple[dict[str, Any], PreprintDetection]] = {}

        for entry, det in zip(entries_sorted, detections):
            entry_key = entry.get("ID", "")
            preload_data = preloaded.get(entry_key, {})
            future = ex.submit(
                process_entry_with_preload,
                entry,
                det,
                preload_data,
                resolver,
                updater,
                logger,
            )
            future_to_entry[future] = (entry, det)

        for future in concurrent.futures.as_completed(future_to_entry):
            entry, _ = future_to_entry[future]
            original_idx = entry_to_idx[id(entry)]
            try:
                result = future.result()
                preprint_results[original_idx] = result
            except Exception as e:
                logger.error("Processing failed for %s: %s", entry.get("ID", "unknown"), e)
                preprint_results[original_idx] = ProcessResult(
                    original=entry,
                    updated=entry,
                    changed=False,
                    action="failed",
                    message=str(e),
                )

    # Build final results in original entry order
    results: list[ProcessResult] = []
    preprint_ids = {id(item[1]) for item in prioritized}

    for i, entry in enumerate(entries):
        if i in preprint_results:
            results.append(preprint_results[i])
        elif id(entry) in preprint_ids:
            # This shouldn't happen, but handle gracefully
            results.append(
                ProcessResult(
                    original=entry,
                    updated=entry,
                    changed=False,
                    action="failed",
                    message="Processing result lost",
                )
            )
        else:
            # Non-preprint entry
            results.append(ProcessResult(original=entry, updated=entry, changed=False, action="unchanged"))

    return results


def process_entry(
    entry: dict[str, Any], detector: Detector, resolver: Resolver, updater: Updater, logger: logging.Logger
) -> ProcessResult:
    """Process a single entry for preprint resolution (original non-optimized version).

    Args:
        entry: BibTeX entry dict
        detector: Detector instance for preprint detection
        resolver: Resolver instance for finding published versions
        updater: Updater instance for applying updates
        logger: Logger instance

    Returns:
        ProcessResult with the processing outcome
    """
    det = detector.detect(entry)
    if not det.is_preprint:
        return ProcessResult(original=entry, updated=entry, changed=False, action="unchanged")

    rec = resolver.resolve(entry, det)
    if not rec:
        return ProcessResult(
            original=entry, updated=entry, changed=False, action="failed", message="No reliable published match found"
        )

    if rec and rec.type != "journal-article":
        return ProcessResult(
            original=entry, updated=entry, changed=False, action="failed", message="Candidate not a journal-article"
        )

    if rec and not Resolver._credible_journal_article(rec):
        return ProcessResult(
            original=entry, updated=entry, changed=False, action="failed", message="Candidate lacks sufficient metadata"
        )

    new_entry = updater.update_entry(entry, rec, det)
    changed = json.dumps(entry, sort_keys=True) != json.dumps(new_entry, sort_keys=True)
    return ProcessResult(
        original=entry,
        updated=new_entry,
        changed=changed,
        action="upgraded",
        method=rec.method,
        confidence=rec.confidence,
    )


# ------------- CLI -------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replace_preprints.py",
        description="Replace preprint BibTeX entries with published versions when available.",
    )
    p.add_argument("inputs", nargs="+", help="Input .bib files")
    out = p.add_mutually_exclusive_group()
    out.add_argument("-o", "--output", help="Output merged .bib (when not using --in-place)")
    out.add_argument("--in-place", action="store_true", help="Edit files in place")
    p.add_argument("--keep-preprint-note", action="store_true", help="Keep a note pointing to arXiv id")
    p.add_argument("--rekey", action="store_true", help="Regenerate BibTeX keys as authorYearTitle")
    p.add_argument("--dedupe", action="store_true", help="Merge duplicates by DOI or normalized title+authors")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    p.add_argument("--report", help="Write JSONL report mapping original→updated")
    p.add_argument("--cache", default=".cache.replace_preprints.json", help="On-disk cache file for API responses")
    p.add_argument(
        "--resolution-cache",
        default=".cache.resolution.json",
        help="On-disk cache file for resolution results (default: .cache.resolution.json)",
    )
    p.add_argument(
        "--resolution-cache-ttl",
        type=int,
        default=30,
        help="TTL in days for resolution cache entries (default: 30)",
    )
    p.add_argument("--rate-limit", type=int, default=45, help="Requests per minute (default 45)")
    p.add_argument("--max-workers", type=int, default=8, help="Max concurrent workers")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Field checking options
    field_group = p.add_argument_group("field checking", "Options for missing field detection and filling")
    field_group.add_argument(
        "--check-fields",
        action="store_true",
        help="Check entries for missing required/recommended fields (report only)",
    )
    field_group.add_argument(
        "--fill-fields",
        action="store_true",
        help="Fill missing fields from external APIs (implies --check-fields)",
    )
    field_group.add_argument(
        "--field-fill-mode",
        choices=["required", "recommended", "all"],
        default="recommended",
        help="Which missing fields to fill: 'required' only, 'recommended' (default), or 'all'",
    )
    field_group.add_argument(
        "--field-report",
        metavar="FILE",
        help="Write field check report to FILE (JSON format)",
    )
    field_group.add_argument(
        "--skip-preprint-upgrade",
        action="store_true",
        help="Only check/fill fields, skip preprint-to-published resolution",
    )
    field_group.add_argument(
        "--strict-fields",
        action="store_true",
        help="Exit with error code 3 if required fields remain missing after filling",
    )

    # Google Scholar options (opt-in)
    p.add_argument(
        "--use-scholarly", action="store_true", help="Enable Google Scholar fallback (requires scholarly package)"
    )
    p.add_argument(
        "--scholarly-proxy",
        choices=["tor", "free", "none"],
        default="none",
        help="Proxy for Google Scholar requests (default: none)",
    )
    p.add_argument(
        "--scholarly-delay",
        type=float,
        default=5.0,
        help="Delay between Google Scholar requests in seconds (default: 5.0)",
    )
    return p


def init_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    return logging.getLogger("replace_preprints")


def summarize(results: list[ProcessResult], logger: logging.Logger) -> dict[str, int]:
    total = len(results)
    upgraded = sum(1 for r in results if r.action == "upgraded")
    failed = sum(1 for r in results if r.action == "failed")
    unchanged = total - upgraded - failed
    logger.info(
        "Summary: total=%d, detected_preprints=%d, upgraded=%d, unchanged=%d, failures=%d",
        total,
        upgraded + failed,
        upgraded,
        unchanged,
        failed,
    )
    return {
        "total": total,
        "detected_preprints": upgraded + failed,
        "upgraded": upgraded,
        "unchanged": unchanged,
        "failures": failed,
    }


def print_failures(results: list[ProcessResult], logger: logging.Logger) -> None:
    """Print details of preprints that could not be upgraded."""
    failures = [r for r in results if r.action == "failed"]
    if not failures:
        return

    logger.info("Failed to find published versions for %d preprint(s):", len(failures))
    for r in failures:
        key = r.original.get("ID", "unknown")
        title = latex_to_plain(r.original.get("title", ""))[:80]
        reason = r.message or "unknown reason"
        logger.info("  - [%s] %s (%s)", key, title, reason)


def summarize_field_check(field_results: list[FieldCheckResult], logger: logging.Logger) -> dict[str, Any]:
    """Summarize field check results and print to console."""
    processor = MissingFieldProcessor(FieldChecker())  # Just for summary generation
    summary = processor.generate_summary(field_results)

    logger.info(
        "Field Check Summary: total=%d, complete=%d, filled=%d, partial=%d, unfillable=%d",
        summary["total"],
        summary["complete"],
        summary["filled"],
        summary["partial"],
        summary["unfillable"],
    )

    # Print field statistics if there are any
    if summary["field_statistics"]:
        logger.info("Missing Field Statistics:")
        for field_name, stats in sorted(summary["field_statistics"].items()):
            logger.info(
                "  %s: %d entries (%d filled, %d still missing)",
                field_name,
                stats["total_missing"],
                stats["filled"],
                stats["still_missing"],
            )

    return summary


def print_field_check_details(
    field_results: list[FieldCheckResult], logger: logging.Logger, verbose: bool = False
) -> None:
    """Print detailed field check results."""
    for i, result in enumerate(field_results, 1):
        key = result.report.entry_key
        action = result.action.upper()

        if action == "COMPLETE":
            if verbose:
                logger.info("[%d/%d] %s: COMPLETE", i, len(field_results), key)
        elif action == "FILLED":
            filled_list = ", ".join(result.report.filled_fields.keys())
            sources = {src for _, src in result.report.filled_fields.values()}
            source_str = ", ".join(sources)
            logger.info(
                "[%d/%d] %s: FILLED %d fields (%s) via %s",
                i,
                len(field_results),
                key,
                len(result.report.filled_fields),
                filled_list,
                source_str,
            )
        elif action == "PARTIAL":
            filled_list = ", ".join(result.report.filled_fields.keys()) if result.report.filled_fields else "none"
            missing = result.report.missing_required + result.report.missing_recommended
            missing_str = ", ".join(missing[:5])
            if len(missing) > 5:
                missing_str += f", ... ({len(missing) - 5} more)"
            logger.info(
                "[%d/%d] %s: PARTIAL - filled: %s, missing: %s",
                i,
                len(field_results),
                key,
                filled_list,
                missing_str,
            )
        elif action == "UNFILLABLE":
            missing = result.report.missing_required
            missing_str = ", ".join(missing[:5])
            logger.info(
                "[%d/%d] %s: UNFILLABLE - missing required: %s",
                i,
                len(field_results),
                key,
                missing_str,
            )


def write_report_line(fh, res: ProcessResult, src_file: str | None = None) -> None:
    line = {
        "file": src_file,
        "key_old": res.original.get("ID"),
        "key_new": res.updated.get("ID"),
        "doi_old": doi_normalize(res.original.get("doi")),
        "doi_new": doi_normalize(res.updated.get("doi")),
        "action": res.action,
        "method": res.method,
        "confidence": res.confidence,
        "title_old": latex_to_plain(res.original.get("title") or ""),
        "title_new": res.updated.get("title"),
    }
    fh.write(json.dumps(line, ensure_ascii=False) + "\n")


# ------------- Main function helper components -------------


@dataclass
class MainComponents:
    """Container for main function components to simplify parameter passing."""

    logger: logging.Logger
    http: HttpClient
    resolver: Resolver
    detector: Detector
    updater: Updater
    loader: BibLoader
    writer: BibWriter
    field_processor: MissingFieldProcessor | None
    args: argparse.Namespace


def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> str | None:
    """Validate parsed command-line arguments.

    Args:
        args: Parsed command-line arguments.
        logger: Logger instance for error messages.

    Returns:
        Error message string if validation fails, None if validation passes.
    """
    if not args.in_place and not args.output:
        return "Specify -o OUTPUT.bib or --in-place"
    return None


def setup_http_client(args: argparse.Namespace) -> HttpClient:
    """Create and configure the HTTP client with rate limiting and caching.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured HttpClient instance.
    """
    rate_limiter = RateLimiter(args.rate_limit)
    cache = DiskCache(args.cache) if args.cache else DiskCache(None)
    user_agent = "bib-preprint-upgrader/1.1 (mailto:you@example.com)"
    return HttpClient(
        timeout=args.timeout, user_agent=user_agent, rate_limiter=rate_limiter, cache=cache, verbose=args.verbose
    )


def setup_scholarly_client(args: argparse.Namespace, logger: logging.Logger) -> ScholarlyClient | None:
    """Set up Google Scholar client if enabled.

    Args:
        args: Parsed command-line arguments.
        logger: Logger instance.

    Returns:
        ScholarlyClient instance if enabled and available, None otherwise.
    """
    if not args.use_scholarly:
        return None

    scholarly_client = ScholarlyClient(
        proxy=args.scholarly_proxy,
        delay=args.scholarly_delay,
        logger=logger,
    )
    if scholarly_client._scholarly:
        logger.info(
            "Google Scholar fallback enabled (delay=%.1fs, proxy=%s)", args.scholarly_delay, args.scholarly_proxy
        )
        return scholarly_client
    else:
        logger.warning("Google Scholar fallback requested but scholarly package not available")
        return None


def setup_field_processor(
    args: argparse.Namespace, resolver: Resolver, logger: logging.Logger
) -> MissingFieldProcessor | None:
    """Set up field checking/filling processor if enabled.

    Args:
        args: Parsed command-line arguments.
        resolver: Resolver instance for field filling.
        logger: Logger instance.

    Returns:
        MissingFieldProcessor if field checking enabled, None otherwise.
    """
    field_check_enabled = args.check_fields or args.fill_fields
    if not field_check_enabled:
        return None

    field_fill_enabled = args.fill_fields
    field_checker = FieldChecker()
    field_filler = FieldFiller(resolver=resolver, logger=logger) if field_fill_enabled else None
    field_processor = MissingFieldProcessor(
        checker=field_checker,
        filler=field_filler,
        fill_mode=args.field_fill_mode,
        fill_enabled=field_fill_enabled,
    )
    mode_str = "fill" if field_fill_enabled else "check-only"
    logger.info("Field checking enabled (mode: %s, fill_mode: %s)", mode_str, args.field_fill_mode)
    return field_processor


def load_databases(
    args: argparse.Namespace, loader: BibLoader, logger: logging.Logger
) -> list[tuple[str, bibtexparser.bibdatabase.BibDatabase]] | None:
    """Load all input BibTeX files.

    Args:
        args: Parsed command-line arguments.
        loader: BibLoader instance.
        logger: Logger instance.

    Returns:
        List of (path, database) tuples, or None if loading fails.
    """
    databases: list[tuple[str, bibtexparser.bibdatabase.BibDatabase]] = []
    try:
        for path in args.inputs:
            db = loader.load_file(path)
            databases.append((path, db))
        return databases
    except Exception as e:
        logger.error("Failed to read inputs: %s", e)
        return None


def process_entries_parallel(
    entries: list[dict[str, Any]],
    detector: Detector,
    resolver: Resolver,
    updater: Updater,
    logger: logging.Logger,
    max_workers: int,
) -> list[ProcessResult]:
    """Process entries in parallel for preprint upgrade.

    Args:
        entries: List of BibTeX entries.
        detector: Preprint detector.
        resolver: Resolver for finding published versions.
        updater: Entry updater.
        logger: Logger instance.
        max_workers: Maximum number of parallel workers.

    Returns:
        List of ProcessResult objects (order may differ from input).
    """
    results: list[ProcessResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_entry, entry, detector, resolver, updater, logger) for entry in entries]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return results


def apply_field_checking(
    new_entries: list[dict[str, Any]],
    results: list[ProcessResult],
    field_processor: MissingFieldProcessor,
) -> list[FieldCheckResult]:
    """Apply field checking and filling to entries.

    Args:
        new_entries: List of entries to check (modified in place).
        results: List of ProcessResult objects (modified in place).
        field_processor: Field processor instance.

    Returns:
        List of FieldCheckResult objects.
    """
    field_results: list[FieldCheckResult] = []
    for i, entry in enumerate(new_entries):
        field_result = field_processor.process_entry(entry)
        field_results.append(field_result)
        if field_result.changed:
            new_entries[i] = field_result.updated
            # Update the ProcessResult to reflect field changes
            if results[i].action != "upgraded":
                results[i] = ProcessResult(
                    original=results[i].original,
                    updated=field_result.updated,
                    changed=True,
                    action="field_filled",
                    method=None,
                    confidence=0.0,
                    message=None,
                )
    return field_results


def write_output_and_report(
    new_db: bibtexparser.bibdatabase.BibDatabase,
    results: list[ProcessResult],
    args: argparse.Namespace,
    writer: BibWriter,
    logger: logging.Logger,
    output_path: str,
    src_for_entry: list[str] | None = None,
    merged_info: list[tuple[str, list[str]]] | None = None,
    append_report: bool = False,
) -> int:
    """Write output file and optional report.

    Args:
        new_db: Database to write.
        results: Processing results.
        args: Parsed command-line arguments.
        writer: BibWriter instance.
        logger: Logger instance.
        output_path: Path to write the output file.
        src_for_entry: Optional list of source files for each entry.
        merged_info: Optional dedupe merged info.
        append_report: If True, append to report file; otherwise overwrite.

    Returns:
        0 on success, 1 on failure.
    """
    if args.dry_run:
        for r in results:
            if r.changed:
                print(diff_entries(r.original, r.updated, r.original.get("ID")))
        if args.dedupe and merged_info:
            print(f"# Dedupe merged groups: {merged_info}")
        return 0

    try:
        writer.dump_to_file(new_db, output_path)
    except Exception as e:
        logger.error("Failed to write %s: %s", output_path, e)
        return 1

    if args.report:
        mode = "a" if append_report else "w"
        with open(args.report, mode, encoding="utf-8") as fh:
            for idx, res in enumerate(results):
                src_file = src_for_entry[idx] if src_for_entry and idx < len(src_for_entry) else None
                write_report_line(fh, res, src_file=src_file)

    return 0


def handle_field_check_reporting(
    field_results: list[FieldCheckResult],
    field_processor: MissingFieldProcessor | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int | None:
    """Handle field check summary and reporting.

    Args:
        field_results: List of field check results.
        field_processor: Field processor instance.
        args: Parsed command-line arguments.
        logger: Logger instance.

    Returns:
        Exit code 3 if strict mode fails, None otherwise.
    """
    if not field_results:
        return None

    print_field_check_details(field_results, logger, verbose=args.verbose)
    field_summary = summarize_field_check(field_results, logger)

    if args.field_report and field_processor:
        report_data = field_processor.generate_json_report(field_results)
        with open(args.field_report, "w", encoding="utf-8") as fh:
            json.dump(report_data, fh, indent=2, ensure_ascii=False)
        logger.info("Field check report written to %s", args.field_report)

    # Strict mode: exit with error if required fields missing
    if args.strict_fields and field_summary["unfillable"] > 0:
        logger.error(
            "Strict mode: %d entries have required fields that could not be filled",
            field_summary["unfillable"],
        )
        return 3

    return None


def process_single_database_in_place(
    path: str,
    db: bibtexparser.bibdatabase.BibDatabase,
    components: MainComponents,
) -> tuple[int, list[ProcessResult], list[FieldCheckResult]]:
    """Process a single database for in-place update.

    Args:
        path: Path to the database file.
        db: The BibTeX database.
        components: MainComponents container.

    Returns:
        Tuple of (exit_code, results, field_results).
    """
    args = components.args
    results: list[ProcessResult] = []
    field_results: list[FieldCheckResult] = []

    # Step 1: Preprint upgrade (unless skipped)
    if not args.skip_preprint_upgrade:
        results = process_entries_parallel(
            db.entries,
            components.detector,
            components.resolver,
            components.updater,
            components.logger,
            args.max_workers,
        )
        new_entries = [r.updated for r in results]
    else:
        # Skip preprint upgrade, use entries as-is
        new_entries = list(db.entries)
        results = [ProcessResult(original=e, updated=e, changed=False, action="skipped") for e in db.entries]

    # Step 2: Field checking (if enabled)
    if components.field_processor:
        field_results = apply_field_checking(new_entries, results, components.field_processor)

    new_db = bibtexparser.bibdatabase.BibDatabase()
    new_db.entries = new_entries

    merged_info: list[tuple[str, list[str]]] = []
    if args.dedupe:
        new_db, merged_info = Dedupe().dedupe_db(new_db, components.logger)

    exit_code = write_output_and_report(
        new_db, results, args, components.writer, components.logger, path, merged_info=merged_info, append_report=True
    )

    return exit_code, results, field_results


def process_in_place_mode(
    databases: list[tuple[str, bibtexparser.bibdatabase.BibDatabase]],
    components: MainComponents,
) -> int:
    """Process databases in in-place mode.

    Args:
        databases: List of (path, database) tuples.
        components: MainComponents container.

    Returns:
        Exit code (0=success, 2=some failures, 3=strict field failure).
    """
    args = components.args
    overall_exit = 0
    all_field_results: list[FieldCheckResult] = []

    for path, db in databases:
        exit_code, results, field_results = process_single_database_in_place(path, db, components)

        if exit_code != 0:
            overall_exit = 1

        all_field_results.extend(field_results)

        if not args.skip_preprint_upgrade:
            summary = summarize(results, components.logger)
            print_failures(results, components.logger)
            if summary["failures"] > 0 and overall_exit == 0:
                overall_exit = 2

    # Field check summary and reporting
    if components.field_processor and all_field_results:
        strict_exit = handle_field_check_reporting(
            all_field_results, components.field_processor, args, components.logger
        )
        if strict_exit is not None:
            return strict_exit

    return overall_exit


def merge_databases(
    databases: list[tuple[str, bibtexparser.bibdatabase.BibDatabase]],
) -> tuple[bibtexparser.bibdatabase.BibDatabase, list[str]]:
    """Merge multiple databases into one.

    Args:
        databases: List of (path, database) tuples.

    Returns:
        Tuple of (merged_database, src_for_entry list).
    """
    merged_db = bibtexparser.bibdatabase.BibDatabase()
    merged_db.entries = []
    src_for_entry: list[str] = []
    for path, db in databases:
        merged_db.entries.extend(db.entries)
        src_for_entry.extend([path] * len(db.entries))
    return merged_db, src_for_entry


def order_results_by_entries(
    results: list[ProcessResult],
    entries: list[dict[str, Any]],
) -> list[ProcessResult]:
    """Order results to match entry order (parallel processing may reorder).

    Args:
        results: Unordered list of ProcessResult objects.
        entries: Original entries in desired order.

    Returns:
        Results ordered to match entries.
    """
    obj_map = {id(r.original): r for r in results}
    ordered_results: list[ProcessResult] = []
    for e in entries:
        r = obj_map.get(id(e))
        ordered_results.append(r if r else ProcessResult(original=e, updated=e, changed=False, action="unchanged"))
    return ordered_results


def process_to_output_mode(
    databases: list[tuple[str, bibtexparser.bibdatabase.BibDatabase]],
    components: MainComponents,
) -> int:
    """Process databases and write to single output file.

    Args:
        databases: List of (path, database) tuples.
        components: MainComponents container.

    Returns:
        Exit code (0=success, 1=write error, 2=some failures, 3=strict field failure).
    """
    args = components.args

    # Merge all databases
    merged_db, src_for_entry = merge_databases(databases)

    field_results: list[FieldCheckResult] = []

    # Step 1: Preprint upgrade (unless skipped)
    if not args.skip_preprint_upgrade:
        results = process_entries_parallel(
            merged_db.entries,
            components.detector,
            components.resolver,
            components.updater,
            components.logger,
            args.max_workers,
        )
        ordered_results = order_results_by_entries(results, merged_db.entries)
        new_entries = [r.updated for r in ordered_results]
    else:
        # Skip preprint upgrade, use entries as-is
        new_entries = list(merged_db.entries)
        ordered_results = [
            ProcessResult(original=e, updated=e, changed=False, action="skipped") for e in merged_db.entries
        ]

    # Step 2: Field checking (if enabled)
    if components.field_processor:
        field_results = apply_field_checking(new_entries, ordered_results, components.field_processor)

    new_db = bibtexparser.bibdatabase.BibDatabase()
    new_db.entries = new_entries

    merged_info: list[tuple[str, list[str]]] = []
    if args.dedupe:
        new_db, merged_info = Dedupe().dedupe_db(new_db, components.logger)

    exit_code = write_output_and_report(
        new_db,
        ordered_results,
        args,
        components.writer,
        components.logger,
        args.output,
        src_for_entry=src_for_entry,
        merged_info=merged_info,
    )
    if exit_code != 0:
        return exit_code

    summary = {"failures": 0}
    if not args.skip_preprint_upgrade:
        summary = summarize(ordered_results, components.logger)
        print_failures(ordered_results, components.logger)

    # Field check summary and reporting
    if components.field_processor and field_results:
        strict_exit = handle_field_check_reporting(field_results, components.field_processor, args, components.logger)
        if strict_exit is not None:
            return strict_exit

    if not args.skip_preprint_upgrade and summary["failures"] > 0:
        return 2

    return 0


def build_main_components(args: argparse.Namespace, logger: logging.Logger) -> MainComponents:
    """Build all main function components from parsed arguments.

    Args:
        args: Parsed command-line arguments.
        logger: Logger instance.

    Returns:
        MainComponents containing all initialized components.
    """
    # Build HTTP client
    http = setup_http_client(args)

    # Build resolution cache
    resolution_cache = (
        ResolutionCache(args.resolution_cache, ttl_days=args.resolution_cache_ttl) if args.resolution_cache else None
    )

    # Build scholarly client
    scholarly_client = setup_scholarly_client(args, logger)

    # Build resolver
    resolver = Resolver(http=http, logger=logger, scholarly_client=scholarly_client, resolution_cache=resolution_cache)

    # Build field processor
    field_processor = setup_field_processor(args, resolver, logger)

    return MainComponents(
        logger=logger,
        http=http,
        resolver=resolver,
        detector=Detector(),
        updater=Updater(keep_preprint_note=args.keep_preprint_note, rekey=args.rekey),
        loader=BibLoader(),
        writer=BibWriter(),
        field_processor=field_processor,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the BibTeX preprint updater.

    Parses command-line arguments, sets up components, and orchestrates
    the preprint-to-published resolution and optional field checking.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code: 0=success, 1=error, 2=some failures, 3=strict field failure.
    """
    # Parse arguments and setup logging
    args = build_arg_parser().parse_args(argv)
    logger = init_logging(args.verbose)

    # Validate arguments
    validation_error = validate_arguments(args, logger)
    if validation_error:
        logger.error(validation_error)
        return 1

    # Build all components
    components = build_main_components(args, logger)

    # Load input databases
    databases = load_databases(args, components.loader, logger)
    if databases is None:
        return 1

    # Execute appropriate processing mode
    if args.in_place:
        return process_in_place_mode(databases, components)
    else:
        return process_to_output_mode(databases, components)


# ------------- Tests (pytest style) -------------
def _make_entry(**kwargs) -> dict[str, Any]:
    e = {
        "ENTRYTYPE": "article",
        "ID": kwargs.pop("ID", "key"),
        "title": "Example",
        "author": "Doe, Jane and Smith, John",
        "year": "2020",
    }
    e.update(kwargs)
    return e


def test_detector_arxiv_url():
    d = Detector()
    e = _make_entry(url="https://arxiv.org/abs/2001.01234", journal="arXiv preprint", ID="a")
    det = d.detect(e)
    assert det.is_preprint and det.arxiv_id.startswith("2001.01234")


def test_detector_preprint_doi():
    d = Detector()
    e = _make_entry(doi="10.1101/123456", ID="b")
    det = d.detect(e)
    assert det.is_preprint and det.doi.startswith("10.1101")


def test_matcher_thresholds():
    entry = _make_entry(title="A Study of Widgets", author="Jane Doe and John Smith")
    rec = PublishedRecord(
        doi="10.1000/j.journal.1",
        title="A Study of Widgets",
        authors=[{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
        journal="Journal of Widget Studies",
        year=2021,
        type="journal-article",
    )
    title_score = token_sort_ratio(
        normalize_title_for_match(entry["title"]), normalize_title_for_match(rec.title or "")
    )
    auth_score = jaccard_similarity(authors_last_names(entry["author"]), ["doe", "smith"])
    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
    assert combined >= 0.9


def test_updater_idempotent():
    upd = Updater(keep_preprint_note=True, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(ID="k1", url="https://arxiv.org/abs/2001.01234", journal="arXiv preprint")
    rec = PublishedRecord(
        doi="10.1000/j.journal.2",
        title="A Better Title",
        authors=[{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
        journal="Journal Name",
        year=2022,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    d = Detector()
    det2 = d.detect(updated)
    assert not det2.is_preprint


def test_credible_journal_rejects_preprint_venues():
    """Ensure _credible_journal_article rejects records with preprint venue names."""
    # Records with preprint hosts in journal name should be rejected
    for preprint_journal in ["arXiv", "arxiv.org", "bioRxiv", "medRxiv", "ArXiv preprint"]:
        rec = PublishedRecord(
            doi="10.1000/valid",
            journal=preprint_journal,
            year=2023,
            volume="1",
            type="journal-article",
        )
        assert not Resolver._credible_journal_article(rec), f"Should reject journal='{preprint_journal}'"

    # Records with legitimate journal names should be accepted
    rec_ok = PublishedRecord(
        doi="10.1000/valid",
        journal="Nature",
        year=2023,
        volume="1",
        type="journal-article",
    )
    assert Resolver._credible_journal_article(rec_ok)


def test_pipeline_with_dblp_fallback():
    # Simulate DBLP fallback when Crossref is unavailable.
    entry = _make_entry(
        ID="dblp1",
        title="Learning Widgets from Data",
        author="Jane Doe and John Smith",
        url="https://arxiv.org/abs/2101.00001",
        journal="arXiv preprint",
    )
    detector = Detector()

    class FakeHTTP(HttpClient):  # pragma: no cover - behavior unimportant
        def __init__(self):  # no rate/caching in test
            pass

    http = FakeHTTP()

    class FakeResolver(Resolver):
        def __init__(self):
            self.logger = logging.getLogger("test")
            self.http = http

        def resolve(self, entry, detection):
            return PublishedRecord(
                doi="10.5555/1234567",
                title=entry["title"],
                authors=[{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
                journal="International Journal of Widgetry",
                year=2023,
                volume="10",
                pages="1-15",
                type="journal-article",
                method="DBLP(search)",
                confidence=0.95,
            )

    updater = Updater(keep_preprint_note=False, rekey=False)
    res = process_entry(entry, detector, FakeResolver(), updater, logging.getLogger("t"))
    assert res.action == "upgraded"
    assert res.updated.get("journal") == "International Journal of Widgetry"


def test_update_preserves_author_list():
    """Test that author list is correctly transferred from PublishedRecord to updated entry."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="auth_test",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        author="Doe, Jane and Smith, John",
        title="Original Preprint Title",
    )
    rec = PublishedRecord(
        doi="10.1000/j.journal.3",
        title="Published Version Title",
        authors=[{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
        journal="Journal of Testing",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    # Authors should be converted to bibtex format: "Given Family and Given Family"
    assert updated["author"] == "Jane Doe and John Smith"
    # Verify the last names match using the project's comparison functions
    original_last_names = set(authors_last_names(entry["author"]))
    updated_last_names = set(authors_last_names(updated["author"]))
    assert (
        original_last_names == updated_last_names
    ), f"Author last names changed: {original_last_names} -> {updated_last_names}"


def test_update_preserves_title():
    """Test that title is correctly transferred from PublishedRecord to updated entry."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="title_test",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        title="A Study of Machine Learning",
    )
    rec = PublishedRecord(
        doi="10.1000/j.journal.4",
        title="A Study of Machine Learning",  # Same title in published version
        authors=[{"given": "Jane", "family": "Doe"}],
        journal="Journal of ML",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    assert updated["title"] == rec.title
    # Verify normalized titles match
    assert normalize_title_for_match(updated["title"]) == normalize_title_for_match(rec.title)


def test_update_author_list_multiple_authors():
    """Test that multiple authors are correctly preserved."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="multi_auth",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        author="First, Alice and Second, Bob and Third, Charlie",
    )
    rec = PublishedRecord(
        doi="10.1000/j.journal.5",
        title="Multi Author Paper",
        authors=[
            {"given": "Alice", "family": "First"},
            {"given": "Bob", "family": "Second"},
            {"given": "Charlie", "family": "Third"},
        ],
        journal="Collaboration Journal",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    assert updated["author"] == "Alice First and Bob Second and Charlie Third"
    # All three authors preserved
    updated_last_names = authors_last_names(updated["author"], limit=10)
    assert updated_last_names == ["first", "second", "third"]


def test_update_title_with_special_characters():
    """Test that titles with special LaTeX characters are handled correctly."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="special_title",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        title="{Deep Learning for Schrödinger Equations}",
    )
    rec = PublishedRecord(
        doi="10.1000/j.journal.6",
        title="Deep Learning for Schrödinger Equations",
        authors=[{"given": "Jane", "family": "Doe"}],
        journal="Physics Journal",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    assert updated["title"] == rec.title
    # Normalized versions should be semantically equivalent
    orig_norm = normalize_title_for_match(entry["title"])
    updated_norm = normalize_title_for_match(updated["title"])
    # Use fuzzy matching to verify semantic equivalence (handles diacritics differences)
    title_score = token_sort_ratio(orig_norm, updated_norm)
    assert title_score == 100, f"Title match score is {title_score}, expected 100 for semantically equivalent titles"


def test_update_author_consistency_jaccard():
    """Test that updated authors have high Jaccard similarity with original."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="jaccard_test",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        author="Smith, John and Doe, Jane",
    )
    rec = PublishedRecord(
        doi="10.1000/j.journal.7",
        title="Test Paper",
        authors=[{"given": "John", "family": "Smith"}, {"given": "Jane", "family": "Doe"}],
        journal="Test Journal",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    # Jaccard similarity between original and updated author sets should be 1.0
    orig_names = authors_last_names(entry["author"])
    updated_names = authors_last_names(updated["author"])
    similarity = jaccard_similarity(orig_names, updated_names)
    assert similarity == 1.0, f"Author Jaccard similarity is {similarity}, expected 1.0"


def test_update_title_consistency_fuzzy_match():
    """Test that updated title has high fuzzy match score with expected title."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="fuzzy_title",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        title="Neural Networks for Image Classification",
    )
    expected_title = "Neural Networks for Image Classification"
    rec = PublishedRecord(
        doi="10.1000/j.journal.8",
        title=expected_title,
        authors=[{"given": "Jane", "family": "Doe"}],
        journal="CV Journal",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    # Title score using token_sort_ratio should be 100 for identical titles
    title_score = token_sort_ratio(
        normalize_title_for_match(updated["title"]), normalize_title_for_match(expected_title)
    )
    assert title_score == 100, f"Title match score is {title_score}, expected 100"


def test_update_preserves_author_order():
    """Test that author order is preserved after update."""
    upd = Updater(keep_preprint_note=False, rekey=False)
    det = PreprintDetection(is_preprint=True, reason="url arXiv", arxiv_id="2001.01234")
    entry = _make_entry(
        ID="order_test",
        url="https://arxiv.org/abs/2001.01234",
        journal="arXiv preprint",
        author="Alpha, Ann and Beta, Bob and Gamma, Grace",
    )
    rec = PublishedRecord(
        doi="10.1000/j.journal.9",
        title="Order Matters Paper",
        authors=[
            {"given": "Ann", "family": "Alpha"},
            {"given": "Bob", "family": "Beta"},
            {"given": "Grace", "family": "Gamma"},
        ],
        journal="Order Journal",
        year=2023,
        type="journal-article",
        method="test",
        confidence=1.0,
    )
    updated = upd.update_entry(entry, rec, det)
    # Author order should be preserved: Alpha, Beta, Gamma
    updated_authors = split_authors_bibtex(updated["author"])
    assert len(updated_authors) == 3
    assert "Alpha" in updated_authors[0]
    assert "Beta" in updated_authors[1]
    assert "Gamma" in updated_authors[2]


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
