#!/usr/bin/env python3
"""Reference fact-checker: validate bibliographic entries against external APIs.

This tool validates that bibliographic entries in BibTeX files:
1. Actually exist in external databases (Crossref, DBLP, Semantic Scholar)
2. Have matching metadata (title, authors, year, venue)

It outputs detailed reports categorizing mismatches:
- VERIFIED: Entry matches an external record
- NOT_FOUND: No matching record found in any database
- HALLUCINATED: Very low match score, likely fabricated
- TITLE_MISMATCH: Title differs significantly
- AUTHOR_MISMATCH: Author list differs
- YEAR_MISMATCH: Publication year differs beyond tolerance
- VENUE_MISMATCH: Journal/venue differs
- PARTIAL_MATCH: Multiple fields differ
- API_ERROR: Errors during API queries

Usage:
    python reference_fact_checker.py input.bib --report report.json
    python reference_fact_checker.py *.bib --jsonl failures.jsonl --strict
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import bibtexparser

from bib_utils import (
    # Text normalization
    normalize_title_for_match,
    strip_diacritics,
    # Author handling
    authors_last_names,
    first_author_surname,
    # Matching
    jaccard_similarity,
    # HTTP infrastructure
    DiskCache,
    RateLimiter,
    HttpClient,
    # Data classes
    PublishedRecord,
    # API endpoints
    CROSSREF_API,
    DBLP_API_SEARCH,
    S2_API,
    # API converters
    crossref_message_to_record,
    dblp_hit_to_record,
    s2_data_to_record,
)

from rapidfuzz.fuzz import token_sort_ratio


# ------------- Enums & Data Classes -------------


class FactCheckStatus(Enum):
    """Status codes for fact check results."""

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    TITLE_MISMATCH = "title_mismatch"
    AUTHOR_MISMATCH = "author_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    VENUE_MISMATCH = "venue_mismatch"
    PARTIAL_MATCH = "partial_match"
    HALLUCINATED = "hallucinated"
    API_ERROR = "api_error"


@dataclass
class FieldComparison:
    """Result of comparing a single field between entry and API record."""

    field_name: str
    entry_value: Optional[str]
    api_value: Optional[str]
    similarity_score: float
    matches: bool
    note: Optional[str] = None


@dataclass
class FactCheckResult:
    """Complete result of fact-checking a single BibTeX entry."""

    entry_key: str
    entry_type: str
    status: FactCheckStatus
    overall_confidence: float
    field_comparisons: Dict[str, FieldComparison]
    best_match: Optional[PublishedRecord]
    api_sources_queried: List[str]
    api_sources_with_hits: List[str]
    errors: List[str]


@dataclass
class FactCheckerConfig:
    """Configuration for fact-checking thresholds."""

    title_threshold: float = 0.90
    author_threshold: float = 0.80
    year_tolerance: int = 1
    venue_threshold: float = 0.70
    hallucination_max_score: float = 0.50
    max_candidates_per_source: int = 10


# ------------- API Clients -------------


class CrossrefClient:
    """Crossref API client for bibliographic searches."""

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, query: str, rows: int = 10) -> List[Dict[str, Any]]:
        """Search Crossref for bibliographic records."""
        params = {"query.bibliographic": query, "rows": rows}
        try:
            resp = self.http._request("GET", CROSSREF_API, params=params, accept="application/json")
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
            return items
        except Exception:
            return []


class DBLPClient:
    """DBLP API client for computer science publications."""

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, query: str, max_hits: int = 10) -> List[Dict[str, Any]]:
        """Search DBLP for bibliographic records."""
        params = {"q": query, "h": max_hits, "format": "json"}
        try:
            resp = self.http._request("GET", DBLP_API_SEARCH, params=params, accept="application/json")
            if resp.status_code != 200:
                return []
            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(hits, dict):
                hits = [hits]
            return hits
        except Exception:
            return []


class SemanticScholarClient:
    """Semantic Scholar API client."""

    FIELDS = "title,authors,venue,year,publicationTypes,externalIds,doi,url"

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        url = f"{S2_API}/paper/search"
        try:
            resp = self.http._request("GET", url, params=params, accept="application/json")
            if resp.status_code != 200:
                return []
            return resp.json().get("data", []) or []
        except Exception:
            return []


# ------------- Fact Checker Core -------------


class FactChecker:
    """Validates bibliographic entries against external APIs."""

    API_SOURCES = ["crossref", "dblp", "semanticscholar"]

    def __init__(
        self,
        crossref: CrossrefClient,
        dblp: DBLPClient,
        s2: SemanticScholarClient,
        config: FactCheckerConfig,
        logger: logging.Logger,
    ):
        self.crossref = crossref
        self.dblp = dblp
        self.s2 = s2
        self.config = config
        self.logger = logger

    def check_entry(self, entry: Dict[str, Any]) -> FactCheckResult:
        """Fact-check a single bibliographic entry."""
        entry_key = entry.get("ID", "unknown")
        entry_type = entry.get("ENTRYTYPE", "misc").lower()
        errors: List[str] = []
        sources_queried: List[str] = []
        sources_with_hits: List[str] = []

        title = entry.get("title", "")
        title_norm = normalize_title_for_match(title)
        first_author = first_author_surname(entry)

        if not title_norm:
            return FactCheckResult(
                entry_key=entry_key,
                entry_type=entry_type,
                status=FactCheckStatus.API_ERROR,
                overall_confidence=0.0,
                field_comparisons={},
                best_match=None,
                api_sources_queried=[],
                api_sources_with_hits=[],
                errors=["No title available for search"],
            )

        query = f"{title_norm} {first_author}".strip()
        candidates = self._query_all_sources(
            entry, query, sources_queried, sources_with_hits, errors
        )

        if not candidates:
            status = FactCheckStatus.API_ERROR if errors else FactCheckStatus.NOT_FOUND
            return FactCheckResult(
                entry_key=entry_key,
                entry_type=entry_type,
                status=status,
                overall_confidence=0.0,
                field_comparisons={},
                best_match=None,
                api_sources_queried=sources_queried,
                api_sources_with_hits=sources_with_hits,
                errors=errors,
            )

        # Sort candidates by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match, source = candidates[0]

        field_comparisons = self._compare_all_fields(entry, best_match)
        status = self._determine_status(best_score, field_comparisons, sources_with_hits)

        return FactCheckResult(
            entry_key=entry_key,
            entry_type=entry_type,
            status=status,
            overall_confidence=best_score,
            field_comparisons=field_comparisons,
            best_match=best_match,
            api_sources_queried=sources_queried,
            api_sources_with_hits=sources_with_hits,
            errors=errors,
        )

    def _query_all_sources(
        self,
        entry: Dict[str, Any],
        query: str,
        sources_queried: List[str],
        sources_with_hits: List[str],
        errors: List[str],
    ) -> List[Tuple[float, PublishedRecord, str]]:
        """Query all API sources and collect scored candidates."""
        candidates: List[Tuple[float, PublishedRecord, str]] = []
        title_norm = normalize_title_for_match(entry.get("title", ""))
        authors_ref = authors_last_names(entry.get("author", ""), limit=3)

        # Crossref
        sources_queried.append("crossref")
        try:
            items = self.crossref.search(query, rows=self.config.max_candidates_per_source)
            if items:
                sources_with_hits.append("crossref")
                for item in items:
                    rec = crossref_message_to_record(item)
                    if rec:
                        score = self._score_candidate(title_norm, authors_ref, rec)
                        candidates.append((score, rec, "crossref"))
        except Exception as e:
            errors.append(f"Crossref: {e}")

        # DBLP
        sources_queried.append("dblp")
        try:
            hits = self.dblp.search(query, max_hits=self.config.max_candidates_per_source)
            if hits:
                sources_with_hits.append("dblp")
                for hit in hits:
                    rec = dblp_hit_to_record(hit)
                    if rec:
                        score = self._score_candidate(title_norm, authors_ref, rec)
                        candidates.append((score, rec, "dblp"))
        except Exception as e:
            errors.append(f"DBLP: {e}")

        # Semantic Scholar
        sources_queried.append("semanticscholar")
        try:
            data = self.s2.search(query, limit=self.config.max_candidates_per_source)
            if data:
                sources_with_hits.append("semanticscholar")
                for item in data:
                    rec = s2_data_to_record(item)
                    if rec:
                        score = self._score_candidate(title_norm, authors_ref, rec)
                        candidates.append((score, rec, "semanticscholar"))
        except Exception as e:
            errors.append(f"Semantic Scholar: {e}")

        return candidates

    def _score_candidate(
        self, title_norm: str, authors_ref: List[str], rec: PublishedRecord
    ) -> float:
        """Score a candidate record against the entry."""
        title_b = normalize_title_for_match(rec.title or "")
        title_score = token_sort_ratio(title_norm, title_b) / 100.0

        authors_b = [strip_diacritics(a.get("family", "")).lower() for a in rec.authors][:3]
        author_score = jaccard_similarity(authors_ref, authors_b)

        return 0.7 * title_score + 0.3 * author_score

    def _compare_all_fields(
        self, entry: Dict[str, Any], record: PublishedRecord
    ) -> Dict[str, FieldComparison]:
        """Compare all relevant fields between entry and record."""
        comparisons: Dict[str, FieldComparison] = {}
        cfg = self.config

        # Title
        entry_title = entry.get("title", "")
        api_title = record.title or ""
        title_score = (
            token_sort_ratio(
                normalize_title_for_match(entry_title), normalize_title_for_match(api_title)
            )
            / 100.0
        )
        comparisons["title"] = FieldComparison(
            "title",
            entry_title,
            api_title,
            title_score,
            title_score >= cfg.title_threshold,
        )

        # Author
        entry_authors = entry.get("author", "")
        entry_names = authors_last_names(entry_authors, limit=10)
        api_names = [strip_diacritics(a.get("family", "")).lower() for a in record.authors]
        author_score = jaccard_similarity(entry_names, api_names)
        api_authors_str = " and ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip() for a in record.authors
        )
        comparisons["author"] = FieldComparison(
            "author",
            entry_authors,
            api_authors_str,
            author_score,
            author_score >= cfg.author_threshold,
        )

        # Year
        entry_year = entry.get("year", "")
        api_year = str(record.year) if record.year else ""
        try:
            if entry_year and api_year:
                diff = abs(int(entry_year) - int(api_year))
                year_matches = diff <= cfg.year_tolerance
            else:
                year_matches = False
        except ValueError:
            year_matches = False
        comparisons["year"] = FieldComparison(
            "year",
            entry_year,
            api_year,
            1.0 if year_matches else 0.0,
            year_matches,
            f"Tolerance: ±{cfg.year_tolerance}",
        )

        # Venue
        entry_venue = entry.get("journal") or entry.get("booktitle") or ""
        api_venue = record.journal or ""
        if entry_venue and api_venue:
            venue_score = (
                token_sort_ratio(
                    normalize_title_for_match(entry_venue), normalize_title_for_match(api_venue)
                )
                / 100.0
            )
        else:
            venue_score = 1.0 if not entry_venue else 0.0
        comparisons["venue"] = FieldComparison(
            "venue",
            entry_venue,
            api_venue,
            venue_score,
            venue_score >= cfg.venue_threshold or not entry_venue,
        )

        return comparisons

    def _determine_status(
        self,
        best_score: float,
        comparisons: Dict[str, FieldComparison],
        sources_with_hits: List[str],
    ) -> FactCheckStatus:
        """Determine final status from score and comparisons."""
        if best_score < self.config.hallucination_max_score:
            return FactCheckStatus.HALLUCINATED

        mismatches = [name for name, c in comparisons.items() if not c.matches]

        if not mismatches:
            return FactCheckStatus.VERIFIED

        if len(mismatches) == 1:
            mismatch_map = {
                "title": FactCheckStatus.TITLE_MISMATCH,
                "author": FactCheckStatus.AUTHOR_MISMATCH,
                "year": FactCheckStatus.YEAR_MISMATCH,
                "venue": FactCheckStatus.VENUE_MISMATCH,
            }
            return mismatch_map.get(mismatches[0], FactCheckStatus.PARTIAL_MATCH)

        return FactCheckStatus.PARTIAL_MATCH


# ------------- Processor & Reporting -------------


class FactCheckProcessor:
    """Batch processing and reporting for fact-checking."""

    def __init__(self, checker: FactChecker, logger: logging.Logger):
        self.checker = checker
        self.logger = logger

    def process_entries(self, entries: List[Dict[str, Any]]) -> List[FactCheckResult]:
        """Process multiple entries and return results."""
        results = []
        for i, entry in enumerate(entries, 1):
            self.logger.info(
                "Checking %d/%d: %s", i, len(entries), entry.get("ID", "?")
            )
            results.append(self.checker.check_entry(entry))
        return results

    def generate_summary(self, results: List[FactCheckResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        counts = {s.value: 0 for s in FactCheckStatus}
        for r in results:
            counts[r.status.value] += 1

        field_mismatches: Dict[str, int] = {}
        for r in results:
            for name, c in r.field_comparisons.items():
                if not c.matches:
                    field_mismatches[name] = field_mismatches.get(name, 0) + 1

        problematic_statuses = [
            "not_found",
            "hallucinated",
            "title_mismatch",
            "author_mismatch",
            "year_mismatch",
            "venue_mismatch",
        ]

        return {
            "total": len(results),
            "status_counts": counts,
            "field_mismatch_counts": field_mismatches,
            "verified_rate": counts["verified"] / len(results) if results else 0,
            "problematic_count": sum(counts.get(s, 0) for s in problematic_statuses),
        }

    def generate_json_report(self, results: List[FactCheckResult]) -> Dict[str, Any]:
        """Generate full JSON report."""
        entries = []
        for r in results:
            entry_data = {
                "key": r.entry_key,
                "type": r.entry_type,
                "status": r.status.value,
                "confidence": r.overall_confidence,
                "field_comparisons": {
                    name: {
                        "entry_value": c.entry_value,
                        "api_value": c.api_value,
                        "similarity_score": c.similarity_score,
                        "matches": c.matches,
                        "note": c.note,
                    }
                    for name, c in r.field_comparisons.items()
                },
                "best_match": None,
                "api_sources_queried": r.api_sources_queried,
                "api_sources_with_hits": r.api_sources_with_hits,
                "errors": r.errors,
            }
            if r.best_match:
                entry_data["best_match"] = {
                    "doi": r.best_match.doi,
                    "title": r.best_match.title,
                    "journal": r.best_match.journal,
                    "year": r.best_match.year,
                }
            entries.append(entry_data)

        summary = self.generate_summary(results)
        summary["timestamp"] = datetime.datetime.now().isoformat()

        return {"summary": summary, "entries": entries}

    def generate_jsonl(self, results: List[FactCheckResult]) -> List[str]:
        """Generate JSONL format (one JSON object per line)."""
        lines = []
        for r in results:
            lines.append(
                json.dumps(
                    {
                        "key": r.entry_key,
                        "status": r.status.value,
                        "confidence": r.overall_confidence,
                        "mismatched_fields": [
                            n for n, c in r.field_comparisons.items() if not c.matches
                        ],
                        "api_sources": r.api_sources_with_hits,
                        "errors": r.errors,
                    },
                    ensure_ascii=False,
                )
            )
        return lines


# ------------- CLI -------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        description="Validate bibliographic entries against external APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reference_fact_checker.py input.bib
  python reference_fact_checker.py *.bib --report report.json
  python reference_fact_checker.py input.bib --jsonl failures.jsonl --strict
        """,
    )

    p.add_argument("bibfiles", nargs="+", help="BibTeX files to check")
    p.add_argument("--report", "-r", metavar="FILE", help="Write JSON report to FILE")
    p.add_argument("--jsonl", metavar="FILE", help="Write JSONL report to FILE")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 4 if NOT_FOUND or HALLUCINATED entries found",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    thresholds = p.add_argument_group("thresholds")
    thresholds.add_argument(
        "--title-threshold",
        type=float,
        default=0.90,
        help="Title similarity threshold (default: 0.90)",
    )
    thresholds.add_argument(
        "--author-threshold",
        type=float,
        default=0.80,
        help="Author similarity threshold (default: 0.80)",
    )
    thresholds.add_argument(
        "--year-tolerance",
        type=int,
        default=1,
        help="Year tolerance in years (default: 1)",
    )
    thresholds.add_argument(
        "--venue-threshold",
        type=float,
        default=0.70,
        help="Venue similarity threshold (default: 0.70)",
    )

    api_opts = p.add_argument_group("API options")
    api_opts.add_argument(
        "--cache-file",
        default=".cache.fact_checker.json",
        help="Cache file path (default: .cache.fact_checker.json)",
    )
    api_opts.add_argument(
        "--rate-limit",
        type=int,
        default=45,
        help="Requests per minute limit (default: 45)",
    )

    return p


def main() -> int:
    """Main entry point."""
    args = build_parser().parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("fact_checker")

    # Load entries from all BibTeX files
    entries = []
    for path in args.bibfiles:
        try:
            with open(path, encoding="utf-8") as f:
                db = bibtexparser.load(f)
                entries.extend(db.entries)
                logger.info("Loaded %d entries from %s", len(db.entries), path)
        except FileNotFoundError:
            logger.error("File not found: %s", path)
            return 1
        except Exception as e:
            logger.error("Failed to parse %s: %s", path, e)
            return 1

    if not entries:
        logger.error("No entries found in input files")
        return 1

    logger.info("Total entries to check: %d", len(entries))

    # Setup HTTP infrastructure
    cache = DiskCache(args.cache_file)
    limiter = RateLimiter(args.rate_limit)
    http = HttpClient(
        timeout=20.0,
        user_agent="BibtexFactChecker/1.0 (mailto:factchecker@example.com)",
        rate_limiter=limiter,
        cache=cache,
    )

    # Setup fact checker
    config = FactCheckerConfig(
        title_threshold=args.title_threshold,
        author_threshold=args.author_threshold,
        year_tolerance=args.year_tolerance,
        venue_threshold=args.venue_threshold,
    )

    checker = FactChecker(
        crossref=CrossrefClient(http),
        dblp=DBLPClient(http),
        s2=SemanticScholarClient(http),
        config=config,
        logger=logger,
    )

    processor = FactCheckProcessor(checker, logger)

    # Process entries
    results = processor.process_entries(entries)
    summary = processor.generate_summary(results)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY: %d entries checked", summary["total"])
    for status, count in summary["status_counts"].items():
        if count > 0:
            logger.info("  %s: %d", status.upper(), count)
    logger.info("Verified rate: %.1f%%", summary["verified_rate"] * 100)
    if summary["problematic_count"] > 0:
        logger.warning("Problematic entries: %d", summary["problematic_count"])

    # Write reports
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(processor.generate_json_report(results), f, indent=2, ensure_ascii=False)
        logger.info("JSON report written to %s", args.report)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for line in processor.generate_jsonl(results):
                f.write(line + "\n")
        logger.info("JSONL report written to %s", args.jsonl)

    # Exit code
    if args.strict:
        problem_count = (
            summary["status_counts"].get("not_found", 0)
            + summary["status_counts"].get("hallucinated", 0)
        )
        if problem_count > 0:
            logger.warning(
                "Strict mode: %d NOT_FOUND or HALLUCINATED entries found", problem_count
            )
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
