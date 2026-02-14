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
import concurrent.futures
import datetime
import json
import logging
import re
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import bibtexparser
import httpx
from rapidfuzz.fuzz import token_sort_ratio

from bibtex_updater.calibration import calibrate_result
from bibtex_updater.matching import (
    EXPANDED_VENUE_ALIASES,
    combined_author_score,
    get_canonical_venue,
    is_near_miss_title,
    title_edit_distance,
)
from bibtex_updater.utils import (
    # API endpoints
    CROSSREF_API,
    DBLP_API_SEARCH,
    S2_API,
    HttpClient,
    # Data classes
    PublishedRecord,
    RateLimiterRegistry,
    SqliteCache,
    # HTTP infrastructure
    authors_last_names,
    # API converters
    crossref_message_to_record,
    dblp_hit_to_record,
    first_author_surname,
    # Matching
    jaccard_similarity,
    # Text normalization
    normalize_title_for_match,
    s2_data_to_record,
    strip_diacritics,
)

# ------------- Enums & Data Classes -------------


class FactCheckStatus(Enum):
    """Status codes for fact check results."""

    # Academic verification statuses
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    TITLE_MISMATCH = "title_mismatch"
    AUTHOR_MISMATCH = "author_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    VENUE_MISMATCH = "venue_mismatch"
    PARTIAL_MATCH = "partial_match"
    HALLUCINATED = "hallucinated"
    API_ERROR = "api_error"

    # Pre-API validation statuses
    FUTURE_DATE = "future_date"  # Year is in the future
    INVALID_YEAR = "invalid_year"  # Year is missing, non-numeric, or implausible (<1800)
    DOI_NOT_FOUND = "doi_not_found"  # DOI doesn't resolve (HTTP 404 from doi.org)

    # Preprint-vs-published statuses
    PREPRINT_ONLY = "preprint_only"  # Paper found only as preprint, not at claimed venue
    PUBLISHED_VERSION_EXISTS = "published_version_exists"  # Informational: published version found

    # Web reference statuses
    URL_VERIFIED = "url_verified"  # URL accessible and content matches
    URL_ACCESSIBLE = "url_accessible"  # URL returns 200, no content check
    URL_NOT_FOUND = "url_not_found"  # 404 or domain unreachable
    URL_CONTENT_MISMATCH = "url_content_mismatch"  # Page content differs from entry

    # Book statuses
    BOOK_VERIFIED = "book_verified"  # Found in book API with matching metadata
    BOOK_NOT_FOUND = "book_not_found"  # Not in any book database

    # Working paper statuses
    WORKING_PAPER_VERIFIED = "working_paper_verified"
    WORKING_PAPER_NOT_FOUND = "working_paper_not_found"

    # General
    SKIPPED = "skipped"  # Entry type not verifiable


@dataclass
class FieldComparison:
    """Result of comparing a single field between entry and API record."""

    field_name: str
    entry_value: str | None
    api_value: str | None
    similarity_score: float
    matches: bool
    note: str | None = None


@dataclass
class FactCheckResult:
    """Complete result of fact-checking a single BibTeX entry."""

    entry_key: str
    entry_type: str
    status: FactCheckStatus
    overall_confidence: float
    field_comparisons: dict[str, FieldComparison]
    best_match: PublishedRecord | None
    api_sources_queried: list[str]
    api_sources_with_hits: list[str]
    errors: list[str]
    # New fields for extended verification
    category: EntryCategory | None = None
    url_check: URLCheckResult | None = None
    book_match: BookRecord | None = None


@dataclass
class FactCheckerConfig:
    """Configuration for fact-checking thresholds."""

    title_threshold: float = 0.90
    author_threshold: float = 0.80
    year_tolerance: int = 1
    venue_threshold: float = 0.70
    hallucination_max_score: float = 0.50
    max_candidates_per_source: int = 10
    check_years: bool = True
    check_dois: bool = True


# ------------- Entry Classification -------------


class EntryCategory(Enum):
    """Categories for BibTeX entry types."""

    ACADEMIC = "academic"  # journal articles, conference papers, preprints
    WEB_REFERENCE = "web_reference"  # blogs, podcasts, websites with URL
    BOOK = "book"  # @book entries
    WORKING_PAPER = "working_paper"  # NBER, HBS, non-peer-reviewed academic
    UNKNOWN = "unknown"  # fallback


@dataclass
class ClassificationResult:
    """Result of classifying a BibTeX entry."""

    category: EntryCategory
    reason: str
    extracted_url: str | None = None
    extracted_isbn: str | None = None


@dataclass
class URLCheckResult:
    """Result of checking URL accessibility."""

    url: str
    accessible: bool
    status_code: int | None = None
    is_redirect: bool = False
    final_url: str | None = None
    error: str | None = None


@dataclass
class BookRecord:
    """A book record from a book API."""

    title: str
    authors: list[str]
    publisher: str | None = None
    year: int | None = None
    isbn: str | None = None
    isbn_13: str | None = None
    source: str = ""  # "openlibrary" | "google_books"
    url: str | None = None


@dataclass
class WebVerifierConfig:
    """Configuration for web reference verification."""

    verify_content: bool = False
    content_threshold: float = 0.60
    timeout: float = 10.0
    follow_redirects: bool = True
    max_redirects: int = 5


@dataclass
class BookVerifierConfig:
    """Configuration for book verification."""

    use_google_books: bool = True
    google_books_api_key: str | None = None
    match_threshold: float = 0.80
    title_threshold: float = 0.85
    author_threshold: float = 0.70


@dataclass
class WorkingPaperConfig:
    """Configuration for working paper verification."""

    search_crossref: bool = True
    relaxed_thresholds: bool = True


class EntryClassifier:
    """Classifies BibTeX entries by type for appropriate verification."""

    # Domains that indicate academic content (should use academic verification)
    ACADEMIC_DOMAINS = {
        "arxiv.org",
        "doi.org",
        "dblp.org",
        "semanticscholar.org",
        "acm.org",
        "ieee.org",
        "springer.com",
        "nature.com",
        "sciencedirect.com",
        "wiley.com",
        "aps.org",
        "iopscience.iop.org",
    }

    # Indicators that an entry is a working paper
    WORKING_PAPER_INDICATORS = [
        "working paper",
        "discussion paper",
        "nber",
        "hbs working",
        "technical report",
        "ssrn",
    ]

    def classify(self, entry: dict[str, Any]) -> ClassificationResult:
        """Classify an entry into a category for verification."""
        entry_type = entry.get("ENTRYTYPE", "").lower()

        # Check for book first
        if entry_type in ("book", "inbook"):
            isbn = self._extract_isbn(entry)
            return ClassificationResult(
                category=EntryCategory.BOOK,
                reason=f"Entry type is {entry_type}",
                extracted_isbn=isbn,
            )

        # Check for working paper
        if self._is_working_paper(entry, entry_type):
            return ClassificationResult(
                category=EntryCategory.WORKING_PAPER,
                reason="Contains working paper indicators",
            )

        # Check for web reference (misc with URL in non-academic domain)
        url = self._extract_url(entry)
        if url and entry_type == "misc":
            if not self._is_academic_url(url):
                # Check if it looks like a preprint (has eprint/archiveprefix)
                if not entry.get("eprint") and not entry.get("archiveprefix"):
                    return ClassificationResult(
                        category=EntryCategory.WEB_REFERENCE,
                        reason="misc entry with non-academic URL",
                        extracted_url=url,
                    )

        # Check for academic indicators
        if self._is_academic(entry, entry_type):
            return ClassificationResult(
                category=EntryCategory.ACADEMIC,
                reason="Has academic indicators (DOI, known entry type, or academic URL)",
            )

        # Default: if has a title, try academic verification
        if entry.get("title"):
            return ClassificationResult(
                category=EntryCategory.ACADEMIC,
                reason="Default classification for entries with title",
            )

        return ClassificationResult(
            category=EntryCategory.UNKNOWN,
            reason="Could not classify entry",
        )

    def _extract_url(self, entry: dict[str, Any]) -> str | None:
        """Extract URL from entry fields."""
        # Direct url field
        if entry.get("url"):
            return entry["url"]

        # howpublished with \url{...}
        howpub = entry.get("howpublished", "")
        match = re.search(r"\\url\{([^}]+)\}", howpub)
        if match:
            return match.group(1)

        # note field sometimes contains URLs
        note = entry.get("note", "")
        match = re.search(r"\\url\{([^}]+)\}", note)
        if match:
            return match.group(1)

        # Plain URL in howpublished
        match = re.search(r"https?://[^\s}]+", howpub)
        if match:
            return match.group(0)

        return None

    def _extract_isbn(self, entry: dict[str, Any]) -> str | None:
        """Extract ISBN from entry fields."""
        # Direct isbn field
        isbn = entry.get("isbn", "")
        if isbn:
            # Clean up ISBN (remove dashes, spaces)
            return re.sub(r"[-\s]", "", isbn)

        # Check note field
        note = entry.get("note", "")
        match = re.search(r"ISBN[:\s]*([\d\-X]+)", note, re.IGNORECASE)
        if match:
            return re.sub(r"[-\s]", "", match.group(1))

        return None

    def _is_academic_url(self, url: str) -> bool:
        """Check if URL is from an academic domain."""
        try:
            domain = urlparse(url).netloc.lower()
            return any(academic in domain for academic in self.ACADEMIC_DOMAINS)
        except Exception:
            return False

    def _is_working_paper(self, entry: dict[str, Any], entry_type: str) -> bool:
        """Check if entry is a working paper."""
        if entry_type in ("techreport", "unpublished"):
            return True

        # Check for institution + number (typical working paper pattern)
        if entry.get("institution") and entry.get("number"):
            return True

        # Check text fields for working paper indicators
        text_to_check = " ".join(
            [
                entry.get("note", ""),
                entry.get("journal", ""),
                entry.get("series", ""),
                entry.get("type", ""),
            ]
        ).lower()

        return any(indicator in text_to_check for indicator in self.WORKING_PAPER_INDICATORS)

    def _is_academic(self, entry: dict[str, Any], entry_type: str) -> bool:
        """Check if entry has academic indicators."""
        # Has DOI
        if entry.get("doi"):
            return True

        # Has eprint/archiveprefix (arXiv)
        if entry.get("eprint") or entry.get("archiveprefix"):
            return True

        # Known academic entry types
        if entry_type in ("article", "inproceedings", "incollection", "phdthesis", "mastersthesis", "proceedings"):
            return True

        # Check if URL is academic
        url = self._extract_url(entry)
        if url and self._is_academic_url(url):
            return True

        return False


# ------------- Verifiers -------------


class BaseVerifier(ABC):
    """Abstract base class for entry verifiers."""

    @abstractmethod
    def verify(self, entry: dict[str, Any], classification: ClassificationResult) -> FactCheckResult:
        """Verify an entry and return a FactCheckResult."""
        pass

    @abstractmethod
    def supports(self, category: EntryCategory) -> bool:
        """Return True if this verifier handles the given category."""
        pass

    def _make_result(
        self,
        entry: dict[str, Any],
        status: FactCheckStatus,
        category: EntryCategory,
        confidence: float = 0.0,
        field_comparisons: dict[str, FieldComparison] | None = None,
        best_match: PublishedRecord | None = None,
        api_sources_queried: list[str] | None = None,
        api_sources_with_hits: list[str] | None = None,
        errors: list[str] | None = None,
        url_check: URLCheckResult | None = None,
        book_match: BookRecord | None = None,
    ) -> FactCheckResult:
        """Create a FactCheckResult with common fields."""
        return FactCheckResult(
            entry_key=entry.get("ID", "unknown"),
            entry_type=entry.get("ENTRYTYPE", "misc").lower(),
            status=status,
            overall_confidence=confidence,
            field_comparisons=field_comparisons or {},
            best_match=best_match,
            api_sources_queried=api_sources_queried or [],
            api_sources_with_hits=api_sources_with_hits or [],
            errors=errors or [],
            category=category,
            url_check=url_check,
            book_match=book_match,
        )


class WebVerifier(BaseVerifier):
    """Verifies web references (blogs, podcasts, websites)."""

    def __init__(self, http: HttpClient, config: WebVerifierConfig, logger: logging.Logger):
        self.http = http
        self.config = config
        self.logger = logger
        self.classifier = EntryClassifier()

    def supports(self, category: EntryCategory) -> bool:
        return category == EntryCategory.WEB_REFERENCE

    def verify(self, entry: dict[str, Any], classification: ClassificationResult) -> FactCheckResult:
        """Verify a web reference by checking URL accessibility."""
        url = classification.extracted_url or self.classifier._extract_url(entry)

        if not url:
            return self._make_result(
                entry,
                FactCheckStatus.API_ERROR,
                EntryCategory.WEB_REFERENCE,
                errors=["No URL found in entry"],
            )

        # Check URL accessibility
        url_result = self._check_url(url)

        if not url_result.accessible:
            return self._make_result(
                entry,
                FactCheckStatus.URL_NOT_FOUND,
                EntryCategory.WEB_REFERENCE,
                url_check=url_result,
                api_sources_queried=["url_check"],
                errors=[url_result.error] if url_result.error else [],
            )

        # Optionally verify content matches entry metadata
        if self.config.verify_content:
            content_result = self._verify_content(url, entry)
            if content_result:
                return self._make_result(
                    entry,
                    FactCheckStatus.URL_VERIFIED,
                    EntryCategory.WEB_REFERENCE,
                    confidence=content_result,
                    url_check=url_result,
                    api_sources_queried=["url_check", "content_verify"],
                    api_sources_with_hits=["url_check", "content_verify"],
                )

        # URL accessible but no content verification
        return self._make_result(
            entry,
            FactCheckStatus.URL_ACCESSIBLE,
            EntryCategory.WEB_REFERENCE,
            confidence=1.0,
            url_check=url_result,
            api_sources_queried=["url_check"],
            api_sources_with_hits=["url_check"],
        )

    def _check_url(self, url: str) -> URLCheckResult:
        """Check if URL is accessible via HEAD request."""
        import requests

        try:
            # Use HEAD request to minimize data transfer
            resp = requests.head(
                url,
                timeout=self.config.timeout,
                allow_redirects=self.config.follow_redirects,
                headers={"User-Agent": "BibtexFactChecker/1.0"},
            )

            is_redirect = len(resp.history) > 0
            final_url = resp.url if is_redirect else None

            return URLCheckResult(
                url=url,
                accessible=resp.status_code < 400,
                status_code=resp.status_code,
                is_redirect=is_redirect,
                final_url=final_url,
            )
        except requests.exceptions.SSLError as e:
            return URLCheckResult(url=url, accessible=False, error=f"SSL error: {e}")
        except requests.exceptions.ConnectionError as e:
            return URLCheckResult(url=url, accessible=False, error=f"Connection error: {e}")
        except requests.exceptions.Timeout:
            return URLCheckResult(url=url, accessible=False, error="Request timed out")
        except Exception as e:
            return URLCheckResult(url=url, accessible=False, error=str(e))

    def _verify_content(self, url: str, entry: dict[str, Any]) -> float | None:
        """Verify that page content matches entry metadata."""
        import requests

        try:
            resp = requests.get(
                url,
                timeout=self.config.timeout,
                headers={"User-Agent": "BibtexFactChecker/1.0"},
            )
            if resp.status_code != 200:
                return None

            content = resp.text.lower()
            title = entry.get("title", "").lower()

            # Simple check: does the title appear in the page?
            if title and title in content:
                return 1.0

            # Fuzzy match on title
            title_norm = normalize_title_for_match(entry.get("title", ""))
            if title_norm and title_norm in content:
                return 0.8

            return 0.5  # URL accessible but title not found

        except Exception:
            return None


class BookVerifier(BaseVerifier):
    """Verifies books using Open Library and Google Books APIs."""

    OPEN_LIBRARY_API = "https://openlibrary.org/search.json"
    GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"

    def __init__(self, http: HttpClient, config: BookVerifierConfig, logger: logging.Logger):
        self.http = http
        self.config = config
        self.logger = logger
        self.classifier = EntryClassifier()

    def supports(self, category: EntryCategory) -> bool:
        return category == EntryCategory.BOOK

    def verify(self, entry: dict[str, Any], classification: ClassificationResult) -> FactCheckResult:
        """Verify a book entry using book APIs."""
        title = entry.get("title", "")
        author = entry.get("author", "")
        isbn = classification.extracted_isbn or self.classifier._extract_isbn(entry)

        if not title:
            return self._make_result(
                entry,
                FactCheckStatus.API_ERROR,
                EntryCategory.BOOK,
                errors=["No title found for book search"],
            )

        candidates: list[tuple[float, BookRecord, str]] = []
        sources_queried: list[str] = []
        sources_with_hits: list[str] = []
        errors: list[str] = []

        # Try Open Library
        sources_queried.append("openlibrary")
        try:
            ol_results = self._search_open_library(title, author, isbn)
            if ol_results:
                sources_with_hits.append("openlibrary")
                for book in ol_results:
                    score = self._score_book_match(entry, book)
                    candidates.append((score, book, "openlibrary"))
        except Exception as e:
            errors.append(f"Open Library: {e}")

        # Try Google Books if enabled
        if self.config.use_google_books:
            sources_queried.append("google_books")
            try:
                gb_results = self._search_google_books(title, author, isbn)
                if gb_results:
                    sources_with_hits.append("google_books")
                    for book in gb_results:
                        score = self._score_book_match(entry, book)
                        candidates.append((score, book, "google_books"))
            except Exception as e:
                errors.append(f"Google Books: {e}")

        if not candidates:
            status = FactCheckStatus.API_ERROR if errors and not sources_with_hits else FactCheckStatus.BOOK_NOT_FOUND
            return self._make_result(
                entry,
                status,
                EntryCategory.BOOK,
                api_sources_queried=sources_queried,
                api_sources_with_hits=sources_with_hits,
                errors=errors,
            )

        # Find best match
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match, source = candidates[0]

        if best_score >= self.config.match_threshold:
            return self._make_result(
                entry,
                FactCheckStatus.BOOK_VERIFIED,
                EntryCategory.BOOK,
                confidence=best_score,
                book_match=best_match,
                api_sources_queried=sources_queried,
                api_sources_with_hits=sources_with_hits,
            )

        return self._make_result(
            entry,
            FactCheckStatus.BOOK_NOT_FOUND,
            EntryCategory.BOOK,
            confidence=best_score,
            book_match=best_match,
            api_sources_queried=sources_queried,
            api_sources_with_hits=sources_with_hits,
        )

    def _search_open_library(self, title: str, author: str, isbn: str | None) -> list[BookRecord]:
        """Search Open Library for books."""
        results = []

        # Build query
        query_parts = []
        if title:
            query_parts.append(f"title:{title}")
        if author:
            # Extract first author's last name
            first_author = first_author_surname({"author": author})
            if first_author:
                query_parts.append(f"author:{first_author}")

        if not query_parts:
            return []

        query = " ".join(query_parts)
        params = {"q": query, "limit": 5}

        try:
            resp = self.http._request(
                "GET", self.OPEN_LIBRARY_API, params=params, accept="application/json", service="openlibrary"
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            for doc in data.get("docs", [])[:5]:
                book = BookRecord(
                    title=doc.get("title", ""),
                    authors=doc.get("author_name", []),
                    publisher=doc.get("publisher", [""])[0] if doc.get("publisher") else None,
                    year=doc.get("first_publish_year"),
                    isbn=doc.get("isbn", [""])[0] if doc.get("isbn") else None,
                    source="openlibrary",
                    url=f"https://openlibrary.org{doc.get('key', '')}" if doc.get("key") else None,
                )
                results.append(book)

        except Exception as e:
            self.logger.debug("Open Library search failed: %s", e)

        return results

    def _search_google_books(self, title: str, author: str, isbn: str | None) -> list[BookRecord]:
        """Search Google Books for books."""
        results = []

        # Build query
        query_parts = []
        if isbn:
            query_parts.append(f"isbn:{isbn}")
        else:
            if title:
                query_parts.append(f"intitle:{title}")
            if author:
                first_author = first_author_surname({"author": author})
                if first_author:
                    query_parts.append(f"inauthor:{first_author}")

        if not query_parts:
            return []

        query = "+".join(query_parts)
        params = {"q": query, "maxResults": 5}

        if self.config.google_books_api_key:
            params["key"] = self.config.google_books_api_key

        try:
            resp = self.http._request(
                "GET", self.GOOGLE_BOOKS_API, params=params, accept="application/json", service="google_books"
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            for item in data.get("items", [])[:5]:
                vol = item.get("volumeInfo", {})
                identifiers = {i.get("type"): i.get("identifier") for i in vol.get("industryIdentifiers", [])}

                book = BookRecord(
                    title=vol.get("title", ""),
                    authors=vol.get("authors", []),
                    publisher=vol.get("publisher"),
                    year=(
                        int(vol.get("publishedDate", "")[:4])
                        if vol.get("publishedDate", "").isdigit()
                        or (len(vol.get("publishedDate", "")) >= 4 and vol.get("publishedDate", "")[:4].isdigit())
                        else None
                    ),
                    isbn=identifiers.get("ISBN_10"),
                    isbn_13=identifiers.get("ISBN_13"),
                    source="google_books",
                    url=vol.get("infoLink"),
                )
                results.append(book)

        except Exception as e:
            self.logger.debug("Google Books search failed: %s", e)

        return results

    def _score_book_match(self, entry: dict[str, Any], book: BookRecord) -> float:
        """Score how well a book matches the entry."""
        title_entry = normalize_title_for_match(entry.get("title", ""))
        title_book = normalize_title_for_match(book.title)
        title_score = token_sort_ratio(title_entry, title_book) / 100.0

        # Author matching
        entry_authors = authors_last_names(entry.get("author", ""), limit=3)
        book_authors = [strip_diacritics(a.split()[-1]).lower() for a in book.authors[:3] if a]
        author_score = jaccard_similarity(entry_authors, book_authors)

        # Year matching (bonus if matches)
        year_bonus = 0.0
        try:
            entry_year = int(entry.get("year", 0))
            if book.year and abs(entry_year - book.year) <= 1:
                year_bonus = 0.1
        except (ValueError, TypeError):
            pass

        return 0.6 * title_score + 0.3 * author_score + year_bonus


class WorkingPaperVerifier(BaseVerifier):
    """Verifies working papers using academic APIs with relaxed thresholds."""

    def __init__(
        self,
        crossref: CrossrefClient,
        config: WorkingPaperConfig,
        academic_config: FactCheckerConfig,
        logger: logging.Logger,
    ):
        self.crossref = crossref
        self.config = config
        self.academic_config = academic_config
        self.logger = logger

    def supports(self, category: EntryCategory) -> bool:
        return category == EntryCategory.WORKING_PAPER

    def verify(self, entry: dict[str, Any], classification: ClassificationResult) -> FactCheckResult:
        """Verify a working paper entry."""
        title = entry.get("title", "")
        if not title:
            return self._make_result(
                entry,
                FactCheckStatus.API_ERROR,
                EntryCategory.WORKING_PAPER,
                errors=["No title found for working paper search"],
            )

        sources_queried: list[str] = []
        sources_with_hits: list[str] = []
        errors: list[str] = []
        candidates: list[tuple[float, PublishedRecord]] = []

        # Search Crossref (often indexes working papers)
        if self.config.search_crossref:
            sources_queried.append("crossref")
            try:
                first_author = first_author_surname(entry)
                query = f"{normalize_title_for_match(title)} {first_author}".strip()
                items = self.crossref.search(query, rows=10)
                if items:
                    sources_with_hits.append("crossref")
                    for item in items:
                        rec = crossref_message_to_record(item)
                        if rec:
                            score = self._score_candidate(entry, rec)
                            candidates.append((score, rec))
            except Exception as e:
                errors.append(f"Crossref: {e}")

        if not candidates:
            status = FactCheckStatus.API_ERROR if errors else FactCheckStatus.WORKING_PAPER_NOT_FOUND
            return self._make_result(
                entry,
                status,
                EntryCategory.WORKING_PAPER,
                api_sources_queried=sources_queried,
                api_sources_with_hits=sources_with_hits,
                errors=errors,
            )

        # Find best match with relaxed thresholds
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match = candidates[0]

        # Use relaxed threshold for working papers (0.7 instead of 0.9)
        threshold = 0.70 if self.config.relaxed_thresholds else 0.90

        if best_score >= threshold:
            return self._make_result(
                entry,
                FactCheckStatus.WORKING_PAPER_VERIFIED,
                EntryCategory.WORKING_PAPER,
                confidence=best_score,
                best_match=best_match,
                api_sources_queried=sources_queried,
                api_sources_with_hits=sources_with_hits,
            )

        return self._make_result(
            entry,
            FactCheckStatus.WORKING_PAPER_NOT_FOUND,
            EntryCategory.WORKING_PAPER,
            confidence=best_score,
            best_match=best_match,
            api_sources_queried=sources_queried,
            api_sources_with_hits=sources_with_hits,
        )

    def _score_candidate(self, entry: dict[str, Any], rec: PublishedRecord) -> float:
        """Score a candidate record against the entry."""
        title_entry = normalize_title_for_match(entry.get("title", ""))
        title_rec = normalize_title_for_match(rec.title or "")
        title_score = token_sort_ratio(title_entry, title_rec) / 100.0

        authors_entry = authors_last_names(entry.get("author", ""), limit=3)
        authors_rec = [strip_diacritics(a.get("family", "")).lower() for a in rec.authors[:3]]
        author_score = jaccard_similarity(authors_entry, authors_rec)

        return 0.7 * title_score + 0.3 * author_score


# ------------- API Clients -------------


class CrossrefClient:
    """Crossref API client for bibliographic searches."""

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, query: str, rows: int = 10) -> list[dict[str, Any]]:
        """Search Crossref for bibliographic records."""
        params = {"query.bibliographic": query, "rows": rows}
        try:
            resp = self.http._request("GET", CROSSREF_API, params=params, accept="application/json", service="crossref")
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

    def search(self, query: str, max_hits: int = 10) -> list[dict[str, Any]]:
        """Search DBLP for bibliographic records."""
        params = {"q": query, "h": max_hits, "format": "json"}
        try:
            resp = self.http._request("GET", DBLP_API_SEARCH, params=params, accept="application/json", service="dblp")
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

    FIELDS = "title,authors,venue,year,publicationTypes,externalIds,url"

    def __init__(self, http: HttpClient):
        self.http = http

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        url = f"{S2_API}/paper/search"
        try:
            resp = self.http._request("GET", url, params=params, accept="application/json", service="semanticscholar")
            if resp.status_code != 200:
                return []
            return resp.json().get("data", []) or []
        except Exception:
            return []

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Get paper details by S2 paper ID, DOI, or arXiv ID.

        paper_id can be: S2 ID, "DOI:10.1234/...", "ARXIV:2301.00001", "CorpusId:12345".
        """
        fields = "title,authors,venue,year,publicationTypes,externalIds,publicationVenue,url"
        url = f"{S2_API}/paper/{paper_id}"
        params = {"fields": fields}
        try:
            resp = self.http._request("GET", url, params=params, accept="application/json", service="semanticscholar")
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None


# ------------- Venue Matching -------------

# Common venue name aliases for fuzzy matching
VENUE_ALIASES: dict[str, set[str]] = {
    "neurips": {
        "nips",
        "advances in neural information processing systems",
        "neural information processing systems",
    },
    "icml": {
        "international conference on machine learning",
        "proceedings of the international conference on machine learning",
    },
    "iclr": {"international conference on learning representations"},
    "aaai": {
        "association for the advancement of artificial intelligence",
        "proceedings of the aaai conference on artificial intelligence",
    },
    "cvpr": {
        "computer vision and pattern recognition",
        "ieee conference on computer vision and pattern recognition",
        "ieee/cvf conference on computer vision and pattern recognition",
    },
    "iccv": {
        "international conference on computer vision",
        "ieee international conference on computer vision",
        "ieee/cvf international conference on computer vision",
    },
    "eccv": {"european conference on computer vision"},
    "acl": {
        "association for computational linguistics",
        "annual meeting of the association for computational linguistics",
    },
    "emnlp": {
        "empirical methods in natural language processing",
        "conference on empirical methods in natural language processing",
    },
    "naacl": {"north american chapter of the association for computational linguistics"},
    "kdd": {"knowledge discovery and data mining"},
    "ijcai": {"international joint conference on artificial intelligence"},
    "uai": {"uncertainty in artificial intelligence"},
    "aistats": {"artificial intelligence and statistics"},
    "jmlr": {"journal of machine learning research"},
    "tmlr": {"transactions on machine learning research"},
}


def normalize_venue(venue: str) -> str:
    """Normalize a venue name for comparison."""
    venue = venue.lower().strip()
    for prefix in ["proceedings of the ", "proceedings of ", "proc. ", "in "]:
        if venue.startswith(prefix):
            venue = venue[len(prefix) :]
    venue = re.sub(r"\b\d{4}\b", "", venue)
    return " ".join(venue.split()).strip()


def _find_canonical_venue(norm_venue: str) -> str | None:
    """Find the canonical venue name for a normalized venue string.

    Note: This function is kept for backward compatibility.
    New code should use get_canonical_venue() from matching.py which uses EXPANDED_VENUE_ALIASES.
    """
    for canonical, aliases in VENUE_ALIASES.items():
        all_names = {canonical} | aliases
        if any(name in norm_venue or norm_venue in name for name in all_names if len(name) > 3):
            return canonical
    return None


def venues_match(venue_a: str, venue_b: str, threshold: float = 0.70) -> tuple[bool, float]:
    """Check if two venue names match, considering aliases. Returns (matches, score).

    P2.5: Now uses EXPANDED_VENUE_ALIASES from matching.py for better venue coverage.
    """
    if not venue_a or not venue_b:
        return (True, 1.0) if not venue_a else (False, 0.0)

    norm_a = normalize_venue(venue_a)
    norm_b = normalize_venue(venue_b)

    # P2.5: Use expanded venue aliases from matching.py
    canonical_a = get_canonical_venue(norm_a, EXPANDED_VENUE_ALIASES)
    canonical_b = get_canonical_venue(norm_b, EXPANDED_VENUE_ALIASES)
    if canonical_a and canonical_b:
        if canonical_a == canonical_b:
            return (True, 0.95)
        return (False, 0.0)  # Known different venues

    # Fall back to fuzzy matching
    score = token_sort_ratio(normalize_title_for_match(norm_a), normalize_title_for_match(norm_b)) / 100.0
    return (score >= threshold, score)


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

    def _validate_year(self, entry: dict[str, Any]) -> FactCheckStatus | None:
        """Pre-API year validation. Returns a status if year is invalid, None if OK."""
        year_str = entry.get("year", "")
        if not year_str:
            return None
        year_str = year_str.strip().strip("{}")
        try:
            year = int(year_str)
        except ValueError:
            return FactCheckStatus.INVALID_YEAR
        if year > datetime.datetime.now().year:
            return FactCheckStatus.FUTURE_DATE
        if year < 1800:
            return FactCheckStatus.INVALID_YEAR
        return None

    def _validate_doi(
        self, entry: dict[str, Any], pre_validated: dict[str, bool] | None = None
    ) -> FactCheckStatus | None:
        """Check if DOI resolves via HEAD request to doi.org.

        Args:
            entry: BibTeX entry with potential DOI field
            pre_validated: Optional dict mapping entry IDs to validation results from batch pre-check
        """
        entry_id = entry.get("ID", "")

        # P1.3: Use pre-validated result if available
        if pre_validated is not None and entry_id in pre_validated:
            return None if pre_validated[entry_id] else FactCheckStatus.DOI_NOT_FOUND

        doi = entry.get("doi", "")
        if not doi:
            return None
        doi = doi.strip()
        if doi.startswith("http"):
            doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        # P1.5: Use httpx instead of requests for better async compatibility
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.head(
                    f"https://doi.org/{doi}",
                    headers={"User-Agent": "BibtexFactChecker/1.0"},
                )
                # Only 404/410 indicate a DOI that truly doesn't exist.
                # Other 4xx (418 bot-detection, 403 access control, 429 rate limit)
                # are publisher-side blocks, not evidence of an invalid DOI.
                if resp.status_code in (404, 410):
                    return FactCheckStatus.DOI_NOT_FOUND
        except Exception:
            pass  # Network errors are not DOI validation failures
        return None

    def _check_preprint_status(self, entry: dict[str, Any], best_match: PublishedRecord) -> FactCheckStatus | None:
        """Check if entry claims venue but paper is only a preprint."""
        claimed_venue = entry.get("booktitle") or entry.get("journal") or ""
        if not claimed_venue:
            return None
        claimed_lower = claimed_venue.lower()
        if any(kw in claimed_lower for kw in ["arxiv", "biorxiv", "medrxiv", "preprint"]):
            return None
        paper_id = None
        if entry.get("doi"):
            paper_id = f"DOI:{entry['doi']}"
        elif entry.get("eprint"):
            paper_id = f"ARXIV:{entry['eprint']}"
        if not paper_id:
            return None
        s2_data = self.s2.get_paper(paper_id)
        if not s2_data:
            return None
        external_ids = s2_data.get("externalIds") or {}
        venue = s2_data.get("venue") or ""
        pub_venue = s2_data.get("publicationVenue")
        has_doi = bool(external_ids.get("DOI"))
        has_venue = bool(venue.strip()) or bool(pub_venue)
        is_only_arxiv = external_ids.get("ArXiv") and not has_doi
        if is_only_arxiv and not has_venue:
            return FactCheckStatus.PREPRINT_ONLY
        return None

    def check_entry(self, entry: dict[str, Any], pre_validated_dois: dict[str, bool] | None = None) -> FactCheckResult:
        """Fact-check a single bibliographic entry.

        Args:
            entry: BibTeX entry to check
            pre_validated_dois: Optional dict mapping entry IDs to DOI validation results
        """
        entry_key = entry.get("ID", "unknown")
        entry_type = entry.get("ENTRYTYPE", "misc").lower()
        errors: list[str] = []
        sources_queried: list[str] = []
        sources_with_hits: list[str] = []

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

        # Pre-API validation: year
        if self.config.check_years:
            year_status = self._validate_year(entry)
            if year_status is not None:
                return FactCheckResult(
                    entry_key=entry_key,
                    entry_type=entry_type,
                    status=year_status,
                    overall_confidence=0.0,
                    field_comparisons={},
                    best_match=None,
                    api_sources_queried=[],
                    api_sources_with_hits=[],
                    errors=[f"Year validation failed: {entry.get('year', 'missing')}"],
                )

        # Pre-API validation: DOI (P1.3: uses batch pre-validation if available)
        if self.config.check_dois:
            doi_status = self._validate_doi(entry, pre_validated=pre_validated_dois)
            if doi_status is not None:
                return FactCheckResult(
                    entry_key=entry_key,
                    entry_type=entry_type,
                    status=doi_status,
                    overall_confidence=0.0,
                    field_comparisons={},
                    best_match=None,
                    api_sources_queried=["doi.org"],
                    api_sources_with_hits=[],
                    errors=[f"DOI does not resolve: {entry.get('doi', '')}"],
                )

        query = f"{title_norm} {first_author}".strip()
        candidates = self._query_all_sources(entry, query, sources_queried, sources_with_hits, errors)

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

        # P2.4: Detect chimeric titles before sorting
        if self._detect_chimeric_title(entry, candidates):
            return FactCheckResult(
                entry_key=entry_key,
                entry_type=entry_type,
                status=FactCheckStatus.HALLUCINATED,
                overall_confidence=0.95,
                field_comparisons={},
                best_match=None,
                api_sources_queried=sources_queried,
                api_sources_with_hits=sources_with_hits,
                errors=["Chimeric title detected: tokens borrowed from multiple different papers"],
            )

        # Sort candidates by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match, source = candidates[0]

        field_comparisons = self._compare_all_fields(entry, best_match)
        status = self._determine_status(best_score, field_comparisons, sources_with_hits)

        # Post-match: check preprint status
        if status in (FactCheckStatus.VERIFIED, FactCheckStatus.VENUE_MISMATCH):
            preprint_status = self._check_preprint_status(entry, best_match)
            if preprint_status is not None:
                status = preprint_status

        # P3.1+P3.2+P3.3: Use calibrated confidence instead of raw best_score
        field_comp_dict = {
            name: {"score": c.similarity_score, "matches": c.matches} for name, c in field_comparisons.items()
        }
        confidence = calibrate_result(
            status=status.value,
            best_match_score=best_score,
            field_comparisons=field_comp_dict,
            sources_queried=sources_queried,
            sources_with_hits=sources_with_hits,
            errors=errors,
        )

        return FactCheckResult(
            entry_key=entry_key,
            entry_type=entry_type,
            status=status,
            overall_confidence=confidence,
            field_comparisons=field_comparisons,
            best_match=best_match,
            api_sources_queried=sources_queried,
            api_sources_with_hits=sources_with_hits,
            errors=errors,
        )

    def _query_all_sources(
        self,
        entry: dict[str, Any],
        query: str,
        sources_queried: list[str],
        sources_with_hits: list[str],
        errors: list[str],
    ) -> list[tuple[float, PublishedRecord, str]]:
        """Query all API sources and collect scored candidates (parallel version with early exit).

        P2.1: Early-exit optimization - if any source returns a very high-confidence match (>= 0.95),
        cancel remaining searches.
        """
        candidates: list[tuple[float, PublishedRecord, str]] = []
        title_norm = normalize_title_for_match(entry.get("title", ""))
        authors_ref = authors_last_names(entry.get("author", ""), limit=3)

        HIGH_CONFIDENCE_THRESHOLD = 0.95

        def _search_crossref() -> tuple[str, list[tuple[float, PublishedRecord, str]], bool, str | None]:
            """Search Crossref API."""
            local_candidates = []
            had_hits = False
            error = None
            try:
                items = self.crossref.search(query, rows=self.config.max_candidates_per_source)
                if items:
                    had_hits = True
                    for item in items:
                        rec = crossref_message_to_record(item)
                        if rec:
                            score = self._score_candidate(title_norm, authors_ref, rec)
                            local_candidates.append((score, rec, "crossref"))
            except Exception as e:
                error = f"Crossref: {e}"
            return ("crossref", local_candidates, had_hits, error)

        def _search_dblp() -> tuple[str, list[tuple[float, PublishedRecord, str]], bool, str | None]:
            """Search DBLP API."""
            local_candidates = []
            had_hits = False
            error = None
            try:
                hits = self.dblp.search(query, max_hits=self.config.max_candidates_per_source)
                if hits:
                    had_hits = True
                    for hit in hits:
                        rec = dblp_hit_to_record(hit)
                        if rec:
                            score = self._score_candidate(title_norm, authors_ref, rec)
                            local_candidates.append((score, rec, "dblp"))
            except Exception as e:
                error = f"DBLP: {e}"
            return ("dblp", local_candidates, had_hits, error)

        def _search_s2() -> tuple[str, list[tuple[float, PublishedRecord, str]], bool, str | None]:
            """Search Semantic Scholar API."""
            local_candidates = []
            had_hits = False
            error = None
            try:
                data = self.s2.search(query, limit=self.config.max_candidates_per_source)
                if data:
                    had_hits = True
                    for item in data:
                        rec = s2_data_to_record(item)
                        if rec:
                            score = self._score_candidate(title_norm, authors_ref, rec)
                            local_candidates.append((score, rec, "semanticscholar"))
            except Exception as e:
                error = f"Semantic Scholar: {e}"
            return ("semanticscholar", local_candidates, had_hits, error)

        # P2.1: Execute searches in parallel with early exit on high-confidence match
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_search_crossref): "crossref",
                executor.submit(_search_dblp): "dblp",
                executor.submit(_search_s2): "s2",
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    source, cands, had_hits, error = future.result()
                    sources_queried.append(source)
                    candidates.extend(cands)
                    if had_hits:
                        sources_with_hits.append(source)
                    if error:
                        errors.append(error)

                    # Note: With max_workers=3 and 3 tasks, all start immediately.
                    # The break skips processing remaining future results but doesn't stop running tasks.
                    if any(score >= HIGH_CONFIDENCE_THRESHOLD for score, _, _ in cands):
                        break
                except Exception:
                    pass  # Errors already captured in individual search functions

        return candidates

    def _score_candidate(self, title_norm: str, authors_ref: list[str], rec: PublishedRecord) -> float:
        """Score a candidate record against the entry."""
        title_b = normalize_title_for_match(rec.title or "")
        title_score = token_sort_ratio(title_norm, title_b) / 100.0

        authors_b = [strip_diacritics(a.get("family", "")).lower() for a in rec.authors][:3]
        author_score = jaccard_similarity(authors_ref, authors_b)

        return 0.7 * title_score + 0.3 * author_score

    def _detect_chimeric_title(
        self, entry: dict[str, Any], candidates: list[tuple[float, PublishedRecord, str]]
    ) -> bool:
        """Detect chimeric titles via multi-source cross-validation.

        P2.4: A chimeric title mixes tokens from multiple real papers. Detection:
        - Group candidates by API source
        - If different sources return different best-match titles, check if the
          entry title borrows tokens from multiple real papers.

        Returns:
            True if chimeric title detected, False otherwise
        """
        if len(candidates) < 2:
            return False

        entry_title = normalize_title_for_match(entry.get("title", ""))
        entry_tokens = set(entry_title.split())

        # Filter out common ML/academic stopwords to reduce false positives
        _TITLE_STOPWORDS = frozenset(
            {
                "a",
                "an",
                "the",
                "of",
                "for",
                "in",
                "on",
                "with",
                "and",
                "via",
                "using",
                "from",
                "to",
                "by",
                "its",
                "is",
                "are",
                "at",
                "as",
                "learning",
                "deep",
                "neural",
                "network",
                "networks",
                "model",
                "models",
                "training",
                "data",
                "based",
                "approach",
                "method",
                "methods",
            }
        )
        entry_tokens = entry_tokens - _TITLE_STOPWORDS

        # Get best match title per source
        by_source: dict[str, tuple[float, str]] = {}
        for score, rec, source in candidates:
            if source not in by_source or score > by_source[source][0]:
                by_source[source] = (score, normalize_title_for_match(rec.title or ""))

        if len(by_source) < 2:
            return False

        # Check if entry tokens are drawn from multiple different source titles
        source_overlaps: dict[str, set[str]] = {}
        for source, (_score, title) in by_source.items():
            api_tokens = set(title.split()) - _TITLE_STOPWORDS
            overlap = entry_tokens & api_tokens
            if len(overlap) >= 4:  # Require >= 4 overlapping tokens (increased from 3)
                source_overlaps[source] = overlap

        if len(source_overlaps) >= 2:
            # Check if different sources contribute different tokens
            all_overlaps = list(source_overlaps.values())
            for i in range(len(all_overlaps)):
                for j in range(i + 1, len(all_overlaps)):
                    unique_i = all_overlaps[i] - all_overlaps[j]
                    unique_j = all_overlaps[j] - all_overlaps[i]
                    if len(unique_i) >= 3 and len(unique_j) >= 3:  # Require >= 3 unique tokens (increased from 2)
                        # Different sources contribute distinct token sets - likely chimeric
                        return True

        return False

    def _compare_all_fields(self, entry: dict[str, Any], record: PublishedRecord) -> dict[str, FieldComparison]:
        """Compare all relevant fields between entry and record."""
        comparisons: dict[str, FieldComparison] = {}
        cfg = self.config

        # Title (P2.2: Near-miss detection)
        entry_title = entry.get("title", "")
        api_title = record.title or ""
        title_score = (
            token_sort_ratio(normalize_title_for_match(entry_title), normalize_title_for_match(api_title)) / 100.0
        )
        # Detect near-miss: high fuzzy score but character-level differences
        edit_dist = title_edit_distance(entry_title, api_title)
        near_miss = is_near_miss_title(entry_title, api_title, title_score, cfg.title_threshold)
        title_matches = title_score >= cfg.title_threshold and not near_miss
        comparisons["title"] = FieldComparison(
            "title",
            entry_title,
            api_title,
            title_score,
            title_matches,
            f"Edit distance: {edit_dist}" if near_miss else None,
        )

        # Author (P2.3: Ordered author comparison)
        entry_authors = entry.get("author", "")
        entry_names = authors_last_names(entry_authors, limit=10)
        api_names = [strip_diacritics(a.get("family", "")).lower() for a in record.authors]
        # Use combined score (Jaccard + sequence similarity)
        author_score = combined_author_score(entry_names, api_names, jaccard_weight=0.5, sequence_weight=0.5)
        api_authors_str = " and ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in record.authors)
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

        # Venue (alias-aware matching)
        entry_venue = entry.get("journal") or entry.get("booktitle") or ""
        api_venue = record.journal or ""
        venue_matches, venue_score = venues_match(entry_venue, api_venue, cfg.venue_threshold)
        comparisons["venue"] = FieldComparison(
            "venue",
            entry_venue,
            api_venue,
            venue_score,
            venue_matches,
        )

        return comparisons

    def _determine_status(
        self,
        best_score: float,
        comparisons: dict[str, FieldComparison],
        sources_with_hits: list[str],
    ) -> FactCheckStatus:
        """Determine final status from score and comparisons.

        P2.6: Venue mismatch is prioritized when title+author match.
        """
        if best_score < self.config.hallucination_max_score:
            return FactCheckStatus.HALLUCINATED

        mismatches = [name for name, c in comparisons.items() if not c.matches]

        if not mismatches:
            return FactCheckStatus.VERIFIED

        # P2.6: Prioritize venue mismatch when title+author match
        if "venue" in mismatches:
            title_ok = comparisons.get("title") and comparisons["title"].matches
            author_ok = comparisons.get("author") and comparisons["author"].matches
            if title_ok and author_ok:
                return FactCheckStatus.VENUE_MISMATCH

        if len(mismatches) == 1:
            mismatch_map = {
                "title": FactCheckStatus.TITLE_MISMATCH,
                "author": FactCheckStatus.AUTHOR_MISMATCH,
                "year": FactCheckStatus.YEAR_MISMATCH,
                "venue": FactCheckStatus.VENUE_MISMATCH,
            }
            return mismatch_map.get(mismatches[0], FactCheckStatus.PARTIAL_MATCH)

        return FactCheckStatus.PARTIAL_MATCH


class AcademicVerifier(BaseVerifier):
    """Verifies academic publications (journals, conferences, preprints).

    This wraps the existing FactChecker logic to implement the BaseVerifier interface.
    """

    def __init__(self, fact_checker: FactChecker):
        self.fact_checker = fact_checker

    def supports(self, category: EntryCategory) -> bool:
        return category == EntryCategory.ACADEMIC

    def verify(self, entry: dict[str, Any], classification: ClassificationResult) -> FactCheckResult:
        """Verify an academic entry using the existing FactChecker logic."""
        result = self.fact_checker.check_entry(entry)
        # Add category to result
        return FactCheckResult(
            entry_key=result.entry_key,
            entry_type=result.entry_type,
            status=result.status,
            overall_confidence=result.overall_confidence,
            field_comparisons=result.field_comparisons,
            best_match=result.best_match,
            api_sources_queried=result.api_sources_queried,
            api_sources_with_hits=result.api_sources_with_hits,
            errors=result.errors,
            category=EntryCategory.ACADEMIC,
            url_check=None,
            book_match=None,
        )


class UnifiedFactChecker:
    """Unified fact-checker that delegates to specialized verifiers based on entry category."""

    def __init__(
        self,
        http: HttpClient,
        crossref: CrossrefClient,
        dblp: DBLPClient,
        s2: SemanticScholarClient,
        config: FactCheckerConfig,
        web_config: WebVerifierConfig | None = None,
        book_config: BookVerifierConfig | None = None,
        working_paper_config: WorkingPaperConfig | None = None,
        logger: logging.Logger | None = None,
        skip_categories: list[EntryCategory] | None = None,
    ):
        self.logger = logger or logging.getLogger("unified_fact_checker")
        self.classifier = EntryClassifier()
        self.skip_categories = set(skip_categories or [])

        # Initialize verifiers
        academic_checker = FactChecker(crossref, dblp, s2, config, self.logger)
        self.verifiers: dict[EntryCategory, BaseVerifier] = {
            EntryCategory.ACADEMIC: AcademicVerifier(academic_checker),
            EntryCategory.WEB_REFERENCE: WebVerifier(http, web_config or WebVerifierConfig(), self.logger),
            EntryCategory.BOOK: BookVerifier(http, book_config or BookVerifierConfig(), self.logger),
            EntryCategory.WORKING_PAPER: WorkingPaperVerifier(
                crossref, working_paper_config or WorkingPaperConfig(), config, self.logger
            ),
        }

    def check_entry(self, entry: dict[str, Any]) -> FactCheckResult:
        """Fact-check a single entry using the appropriate verifier."""
        # Classify entry
        classification = self.classifier.classify(entry)
        self.logger.debug(
            "Entry %s classified as %s: %s",
            entry.get("ID"),
            classification.category.value,
            classification.reason,
        )

        # Check if category should be skipped
        if classification.category in self.skip_categories:
            return FactCheckResult(
                entry_key=entry.get("ID", "unknown"),
                entry_type=entry.get("ENTRYTYPE", "misc").lower(),
                status=FactCheckStatus.SKIPPED,
                overall_confidence=0.0,
                field_comparisons={},
                best_match=None,
                api_sources_queried=[],
                api_sources_with_hits=[],
                errors=[f"Category {classification.category.value} is skipped"],
                category=classification.category,
            )

        # Get appropriate verifier
        verifier = self.verifiers.get(classification.category)
        if not verifier:
            return FactCheckResult(
                entry_key=entry.get("ID", "unknown"),
                entry_type=entry.get("ENTRYTYPE", "misc").lower(),
                status=FactCheckStatus.SKIPPED,
                overall_confidence=0.0,
                field_comparisons={},
                best_match=None,
                api_sources_queried=[],
                api_sources_with_hits=[],
                errors=[f"No verifier for category: {classification.category.value}"],
                category=classification.category,
            )

        # Delegate to verifier
        return verifier.verify(entry, classification)


# ------------- Processor & Reporting -------------


class FactCheckProcessor:
    """Batch processing and reporting for fact-checking."""

    def __init__(self, checker: FactChecker | UnifiedFactChecker, logger: logging.Logger):
        self.checker = checker
        self.logger = logger

    def _batch_validate_dois(self, entries: list[dict[str, Any]]) -> dict[str, bool]:
        """Pre-validate all DOIs via concurrent HEAD requests.

        P1.3: Batch DOI pre-resolution - validates all DOIs in a single pass using
        concurrent HEAD requests, avoiding per-entry DOI checks later.

        Returns:
            dict mapping entry_id -> is_valid (True/False)
        """
        dois: dict[str, str] = {}
        for entry in entries:
            doi = entry.get("doi", "").strip()
            if doi:
                dois[entry.get("ID", "")] = doi

        if not dois:
            return {}

        def _check_doi(entry_id: str, doi: str, client: httpx.Client) -> tuple[str, bool]:
            """Check a single DOI via HEAD request."""
            try:
                # Normalize DOI URL
                if doi.startswith("http"):
                    url = doi
                else:
                    url = f"https://doi.org/{doi}"

                resp = client.head(url, headers={"User-Agent": "BibtexFactChecker/1.0"})
                # Only 404/410 mean the DOI doesn't exist.
                # 418/403/429 are publisher bot-detection, not invalid DOIs.
                return (entry_id, resp.status_code not in (404, 410))
            except Exception:
                # Assume valid on error (don't penalize network issues)
                return (entry_id, True)

        results: dict[str, bool] = {}
        # Execute DOI checks in parallel with shared httpx client
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(_check_doi, eid, doi, client) for eid, doi in dois.items()]
                for future in concurrent.futures.as_completed(futures):
                    entry_id, is_valid = future.result()
                    results[entry_id] = is_valid

        return results

    def process_entries(
        self, entries: list[dict[str, Any]], jsonl_path: str | None = None, max_workers: int = 8
    ) -> list[FactCheckResult]:
        """Process multiple entries and return results (concurrent version).

        If jsonl_path is provided, each result is flushed to the file immediately,
        so partial results survive timeouts and crashes.

        Args:
            entries: List of BibTeX entries to process
            jsonl_path: Optional path to write JSONL results as they complete
            max_workers: Number of concurrent workers (default: 8)
        """

        # P1.3: Batch DOI pre-resolution before main processing loop
        self.logger.info("Pre-validating DOIs for %d entries...", len(entries))
        pre_validated_dois = self._batch_validate_dois(entries)
        if pre_validated_dois:
            failed_count = sum(1 for valid in pre_validated_dois.values() if not valid)
            self.logger.info(
                "DOI pre-validation complete: %d checked, %d failed", len(pre_validated_dois), failed_count
            )

        results: list[FactCheckResult | None] = [None] * len(entries)  # Pre-allocate to preserve order

        jsonl_file = None
        jsonl_lock = threading.Lock()

        try:
            if jsonl_path:
                jsonl_file = open(jsonl_path, "a")  # noqa: SIM115

            def _process_one(index: int, entry: dict[str, Any]) -> FactCheckResult:
                """Process a single entry and write to JSONL if configured."""
                self.logger.info("Checking %d/%d: %s", index + 1, len(entries), entry.get("ID", "?"))
                try:
                    # Pass pre-validated DOI results if checker is FactChecker
                    if isinstance(self.checker, FactChecker):
                        result = self.checker.check_entry(entry, pre_validated_dois=pre_validated_dois)
                    else:
                        # UnifiedFactChecker doesn't support pre_validated_dois yet
                        result = self.checker.check_entry(entry)
                except Exception as exc:
                    self.logger.error("Exception checking entry %s: %s", entry.get("ID", "?"), exc)
                    result = FactCheckResult(
                        entry_key=entry.get("ID", "unknown"),
                        entry_type=entry.get("ENTRYTYPE", "misc").lower(),
                        status=FactCheckStatus.API_ERROR,
                        overall_confidence=0.0,
                        field_comparisons={},
                        best_match=None,
                        api_sources_queried=[],
                        api_sources_with_hits=[],
                        errors=[f"Exception: {exc}"],
                    )
                results[index] = result

                if jsonl_file:
                    line = (
                        json.dumps(
                            {
                                "key": result.entry_key,
                                "category": result.category.value if result.category else None,
                                "status": result.status.value,
                                "confidence": result.overall_confidence,
                                "mismatched_fields": [n for n, c in result.field_comparisons.items() if not c.matches],
                                "api_sources": result.api_sources_with_hits,
                                "errors": result.errors,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    with jsonl_lock:
                        jsonl_file.write(line)
                        jsonl_file.flush()

                return result

            # Process entries concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_one, i, entry): i for i, entry in enumerate(entries)}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        idx = futures[future]
                        self.logger.error("Error processing entry %d: %s", idx + 1, e)
                        # Don't re-raise — let other entries complete

        finally:
            if jsonl_file:
                jsonl_file.close()

        # Filter out None values (from failed entries)
        return [r for r in results if r is not None]

    def generate_summary(self, results: list[FactCheckResult]) -> dict[str, Any]:
        """Generate summary statistics from results."""
        counts = {s.value: 0 for s in FactCheckStatus}
        for r in results:
            counts[r.status.value] += 1

        # Count by category
        category_counts: dict[str, int] = {}
        for r in results:
            if r.category:
                cat_name = r.category.value
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        field_mismatches: dict[str, int] = {}
        for r in results:
            for name, c in r.field_comparisons.items():
                if not c.matches:
                    field_mismatches[name] = field_mismatches.get(name, 0) + 1

        # Include new problematic statuses
        problematic_statuses = [
            "not_found",
            "hallucinated",
            "title_mismatch",
            "author_mismatch",
            "year_mismatch",
            "venue_mismatch",
            "url_not_found",
            "url_content_mismatch",
            "book_not_found",
            "working_paper_not_found",
            "future_date",
            "invalid_year",
            "doi_not_found",
            "preprint_only",
        ]

        # Calculate verified rate including new verified statuses
        verified_statuses = ["verified", "url_verified", "url_accessible", "book_verified", "working_paper_verified"]
        verified_count = sum(counts.get(s, 0) for s in verified_statuses)

        return {
            "total": len(results),
            "status_counts": counts,
            "by_category": category_counts,
            "field_mismatch_counts": field_mismatches,
            "verified_rate": verified_count / len(results) if results else 0,
            "problematic_count": sum(counts.get(s, 0) for s in problematic_statuses),
        }

    def generate_json_report(self, results: list[FactCheckResult]) -> dict[str, Any]:
        """Generate full JSON report."""
        entries = []
        for r in results:
            entry_data = {
                "key": r.entry_key,
                "type": r.entry_type,
                "category": r.category.value if r.category else None,
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
            # Add URL check details for web references
            if r.url_check:
                entry_data["url_check"] = {
                    "url": r.url_check.url,
                    "accessible": r.url_check.accessible,
                    "status_code": r.url_check.status_code,
                    "is_redirect": r.url_check.is_redirect,
                    "final_url": r.url_check.final_url,
                    "error": r.url_check.error,
                }
            # Add book match details
            if r.book_match:
                entry_data["book_match"] = {
                    "title": r.book_match.title,
                    "authors": r.book_match.authors,
                    "publisher": r.book_match.publisher,
                    "year": r.book_match.year,
                    "isbn": r.book_match.isbn,
                    "source": r.book_match.source,
                    "url": r.book_match.url,
                }
            entries.append(entry_data)

        summary = self.generate_summary(results)
        summary["timestamp"] = datetime.datetime.now().isoformat()

        return {"summary": summary, "entries": entries}

    def generate_jsonl(self, results: list[FactCheckResult]) -> list[str]:
        """Generate JSONL format (one JSON object per line)."""
        lines = []
        for r in results:
            lines.append(
                json.dumps(
                    {
                        "key": r.entry_key,
                        "category": r.category.value if r.category else None,
                        "status": r.status.value,
                        "confidence": r.overall_confidence,
                        "mismatched_fields": [n for n, c in r.field_comparisons.items() if not c.matches],
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
    api_opts.add_argument(
        "--s2-api-key",
        metavar="KEY",
        help="Semantic Scholar API key for higher rate limits (or set S2_API_KEY env var)",
    )
    api_opts.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching",
    )
    api_opts.add_argument(
        "--no-check-dois",
        action="store_true",
        help="Disable DOI resolution verification",
    )
    api_opts.add_argument(
        "--no-check-years",
        action="store_true",
        help="Disable year validation (future dates, implausible years)",
    )
    api_opts.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of concurrent workers for parallel processing (default: 8)",
    )

    # Entry type filtering
    entry_types = p.add_argument_group("entry type filtering")
    entry_types.add_argument(
        "--skip-web",
        action="store_true",
        help="Skip web reference verification (blogs, websites)",
    )
    entry_types.add_argument(
        "--skip-books",
        action="store_true",
        help="Skip book verification",
    )
    entry_types.add_argument(
        "--skip-working-papers",
        action="store_true",
        help="Skip working paper verification",
    )
    entry_types.add_argument(
        "--academic-only",
        action="store_true",
        help="Only verify academic entries (skip web, books, working papers)",
    )

    # Web verification options
    web_opts = p.add_argument_group("web verification options")
    web_opts.add_argument(
        "--verify-url-content",
        action="store_true",
        help="Fetch web pages and verify content matches metadata",
    )
    web_opts.add_argument(
        "--url-timeout",
        type=float,
        default=10.0,
        help="Timeout for URL requests in seconds (default: 10)",
    )

    # Book verification options
    book_opts = p.add_argument_group("book verification options")
    book_opts.add_argument(
        "--google-books-api-key",
        metavar="KEY",
        help="Google Books API key for higher rate limits",
    )
    book_opts.add_argument(
        "--no-google-books",
        action="store_true",
        help="Disable Google Books API (use Open Library only)",
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
    import os

    s2_api_key = args.s2_api_key or os.environ.get("S2_API_KEY")
    if s2_api_key:
        logger.info("Using Semantic Scholar API key (authenticated rate limits)")

    cache = SqliteCache(args.cache_file) if not args.no_cache else None
    # Scale per-service limits proportionally to --rate-limit
    rate_scale = args.rate_limit / 45.0  # 45 is the default
    limiter = RateLimiterRegistry(
        {
            "crossref": max(10, int(50 * rate_scale)),
            "semanticscholar": 60 if s2_api_key else max(5, int(10 * rate_scale)),
            "dblp": max(10, int(30 * rate_scale)),
            "openlibrary": max(10, int(30 * rate_scale)),
            "google_books": max(10, int(30 * rate_scale)),
        }
    )
    http = HttpClient(
        timeout=20.0,
        user_agent="BibtexFactChecker/1.0 (mailto:factchecker@example.com)",
        rate_limiter=limiter,
        cache=cache,
        s2_api_key=s2_api_key,
    )

    # Setup fact checker
    config = FactCheckerConfig(
        title_threshold=args.title_threshold,
        author_threshold=args.author_threshold,
        year_tolerance=args.year_tolerance,
        venue_threshold=args.venue_threshold,
        check_dois=not args.no_check_dois,
        check_years=not args.no_check_years,
    )

    # Setup API clients
    crossref = CrossrefClient(http)
    dblp = DBLPClient(http)
    s2 = SemanticScholarClient(http)

    # Determine categories to skip
    skip_categories: list[EntryCategory] = []
    if args.academic_only:
        skip_categories = [EntryCategory.WEB_REFERENCE, EntryCategory.BOOK, EntryCategory.WORKING_PAPER]
    else:
        if args.skip_web:
            skip_categories.append(EntryCategory.WEB_REFERENCE)
        if args.skip_books:
            skip_categories.append(EntryCategory.BOOK)
        if args.skip_working_papers:
            skip_categories.append(EntryCategory.WORKING_PAPER)

    # Setup verifier configs
    web_config = WebVerifierConfig(
        verify_content=args.verify_url_content,
        timeout=args.url_timeout,
    )

    book_config = BookVerifierConfig(
        use_google_books=not args.no_google_books,
        google_books_api_key=args.google_books_api_key,
    )

    working_paper_config = WorkingPaperConfig(
        search_crossref=True,
        relaxed_thresholds=True,
    )

    # Create unified fact checker
    checker = UnifiedFactChecker(
        http=http,
        crossref=crossref,
        dblp=dblp,
        s2=s2,
        config=config,
        web_config=web_config,
        book_config=book_config,
        working_paper_config=working_paper_config,
        logger=logger,
        skip_categories=skip_categories,
    )

    processor = FactCheckProcessor(checker, logger)

    # Process entries (stream JSONL if path provided)
    results = processor.process_entries(entries, jsonl_path=args.jsonl, max_workers=args.workers)
    summary = processor.generate_summary(results)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY: %d entries checked", summary["total"])

    # Print category breakdown
    if summary.get("by_category"):
        logger.info("By category:")
        for cat, count in summary["by_category"].items():
            logger.info("  %s: %d", cat, count)

    # Print status counts
    logger.info("By status:")
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
        logger.info("JSONL report streamed to %s (%d entries)", args.jsonl, len(results))

    # Exit code
    if args.strict:
        problem_count = summary["status_counts"].get("not_found", 0) + summary["status_counts"].get("hallucinated", 0)
        if problem_count > 0:
            logger.warning("Strict mode: %d NOT_FOUND or HALLUCINATED entries found", problem_count)
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
