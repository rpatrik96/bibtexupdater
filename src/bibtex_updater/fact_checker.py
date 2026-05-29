#!/usr/bin/env python3
"""Reference fact-checker: validate bibliographic entries against external APIs.

This tool validates that bibliographic entries in BibTeX files:
1. Actually exist in external databases (Crossref, DBLP, Semantic Scholar)
2. Have matching metadata (title, authors, year, venue)

It outputs detailed reports categorizing mismatches:
- VERIFIED: Every claimed field positively confirmed against an external record
- NOT_FOUND: No matching record found in any database
- UNCONFIRMED: Record found and nothing contradicted, but a claimed field could
  not be positively confirmed (e.g. preprint-only venue, incomplete authors)
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import bibtexparser
import httpx
from rapidfuzz.fuzz import token_sort_ratio

from bibtex_updater.calibration import calibrate_result
from bibtex_updater.matching import (
    EXPANDED_VENUE_ALIASES,
    MatchOutcome,
    get_canonical_venue,
    is_near_miss_title,
    is_preprint_or_series_venue,
    symmetric_author_match,
    title_edit_distance,
)
from bibtex_updater.sources import (
    CASCADE_HIGH_CONFIDENCE,
    CASCADE_LOW_CONFIDENCE,
    DEFAULT_OPENALEX_MAILTO,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    AuthorIntersectionResult,
    OpenAlexClient,
    OpenReviewClient,
    cross_source_author_intersection,
    openalex_work_to_candidate_record,
    openreview_note_to_candidate_record,
    select_top_k_by_title_similarity,
)
from bibtex_updater.utils import (
    # API endpoints
    ARXIV_API,
    CROSSREF_API,
    DBLP_API_SEARCH,
    S2_API,
    HttpClient,
    # Data classes
    PublishedRecord,
    RateLimiterRegistry,
    SqliteCache,
    # API converters
    arxiv_atom_to_record,
    authors_last_names,
    crossref_message_to_record,
    dblp_hit_to_candidate_record,
    entry_surnames_against_structured,
    first_author_surname,
    is_valid_arxiv_id,
    # Matching
    jaccard_similarity,
    # DOI normalization (arXiv-version-aware)
    normalize_doi_for_resolution,
    # Text normalization
    normalize_title_for_match,
    s2_data_to_record,
    same_surname_given_order_violation,
    strip_diacritics,
)

# ------------- Numeric confidence tunables (CheckIfExist, Abbonato 2026) -------------
#
# Penalties / bonuses are intentionally module-level constants so callers and
# tests can override them without poking into class internals. Do NOT auto-fit
# these -- the values are taken from the published reference.

#: Title mismatch penalty (subtracted from numeric confidence, 0-100 scale).
PENALTY_TITLE_MISMATCH: float = 20.0

#: Author mismatch penalty.
PENALTY_AUTHOR_MISMATCH: float = 20.0

#: Journal/venue mismatch penalty (paper allows -10..-20 range; pick midpoint).
PENALTY_JOURNAL_MISMATCH: float = 15.0

#: Per-fabricated-author penalty.
PENALTY_PER_FABRICATED_AUTHOR: float = 10.0

#: Cap on cumulative fabricated-author penalty.
PENALTY_FABRICATED_AUTHOR_CAP: float = 20.0

#: Threshold above which the asymmetric high-title/low-author Case A applies.
CASE_A_TITLE_THRESHOLD: float = 80.0

#: Author similarity threshold below which Case A applies.
CASE_A_AUTHOR_THRESHOLD: float = 90.0

#: Multi-source confirmation bonus (β_ms in the paper, 0..10).
MULTI_SOURCE_BONUS: float = 10.0


# ------------- Non-generative-AI mode -------------
#
# When ``NON_GENERATIVE_MODE`` is enabled (CLI flag ``--non-generative`` or env
# var ``BIBTEX_CHECK_NON_GENERATIVE=1``), bibtex-check refuses to import any
# LLM-based backend. Today the package has no LLM backends, so the gate is a
# forward-compat guard plus a banner for venue-policy compliance
# (ACL ARR Apr-2026 LLM-in-review policy, ICML 2026 restrictions).

NON_GENERATIVE_MODE: bool = False

#: Substrings that, if present in a module name, mark it as an LLM backend.
_LLM_BACKEND_MARKERS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "llm",
    "huggingface",
    "transformers",
    "ollama",
)


def set_non_generative_mode(enabled: bool) -> None:
    """Toggle the global ``NON_GENERATIVE_MODE`` flag."""
    global NON_GENERATIVE_MODE
    NON_GENERATIVE_MODE = bool(enabled)


def is_non_generative_mode() -> bool:
    """Return the current non-generative-mode flag (env var falls through)."""
    import os

    if NON_GENERATIVE_MODE:
        return True
    return os.environ.get("BIBTEX_CHECK_NON_GENERATIVE", "").strip() in {"1", "true", "yes", "on"}


def assert_no_llm_backend(module_name: str) -> None:
    """Refuse to import an LLM-style backend when non-generative mode is on.

    Args:
        module_name: Name of the backend module being imported.

    Raises:
        RuntimeError: If non-generative mode is active and ``module_name``
            looks like an LLM backend.
    """
    if not is_non_generative_mode():
        return
    lower = (module_name or "").lower()
    if any(marker in lower for marker in _LLM_BACKEND_MARKERS):
        raise RuntimeError(
            "bibtex-check is in non-generative-AI mode (--non-generative or "
            "BIBTEX_CHECK_NON_GENERATIVE=1); refusing to load LLM backend "
            f"{module_name!r}. Disable the flag if you really need it."
        )


# ------------- Enums & Data Classes -------------


class FactCheckStatus(Enum):
    """Status codes for fact check results."""

    # Academic verification statuses
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    # A matching record was found and nothing contradicts the entry, but at
    # least one claimed field could not be POSITIVELY CONFIRMED (e.g. only a
    # preprint record was found so the claimed published venue is unconfirmable,
    # or the cited author list is a consistent-but-incomplete subset). This is a
    # "could not fully confirm / needs review" verdict -- abstention, NOT a
    # problem and NOT a verification.
    UNCONFIRMED = "unconfirmed"
    TITLE_MISMATCH = "title_mismatch"
    AUTHOR_MISMATCH = "author_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    VENUE_MISMATCH = "venue_mismatch"
    PARTIAL_MATCH = "partial_match"
    HALLUCINATED = "hallucinated"
    API_ERROR = "api_error"
    ARXIV_ID_MISMATCH = "arxiv_id_mismatch"  # Entry's cited arXiv ID resolves to a different paper
    DOI_MISMATCH = "doi_mismatch"  # Entry's cited DOI resolves to a different paper

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


# Fix B: statuses that mean "could not verify" (abstention) rather than
# "positive evidence of a problem". These are NOT hallucinations -- the tool
# simply failed to locate a matching record. Kept module-level so the JSONL
# writer and the summary buckets stay in sync.
ABSTAINED_STATUS_VALUES = frozenset(
    {
        FactCheckStatus.NOT_FOUND.value,
        # A record was found but a claimed field could not be positively
        # confirmed (preprint-only venue, incomplete author list). "Could not
        # fully confirm" is abstention, not a problem.
        FactCheckStatus.UNCONFIRMED.value,
        FactCheckStatus.BOOK_NOT_FOUND.value,
        FactCheckStatus.WORKING_PAPER_NOT_FOUND.value,
        FactCheckStatus.URL_NOT_FOUND.value,
    }
)


def _is_abstained_status(status: FactCheckStatus) -> bool:
    """True when ``status`` is an abstention (could-not-verify), not a problem."""
    return status.value in ABSTAINED_STATUS_VALUES


@dataclass
class FieldComparison:
    """Result of comparing a single field between entry and API record.

    ``matches`` is the legacy two-valued flag (kept for reporting/JSONL and
    backward compatibility): True only for a positive confirmation (MATCH).
    ``outcome`` carries the three-valued verdict so the status gate can tell a
    real MISMATCH apart from a NON_COMPARABLE / PARTIAL ("could not confirm").
    When ``outcome`` is None it defaults to MATCH if ``matches`` else MISMATCH.
    """

    field_name: str
    entry_value: str | None
    api_value: str | None
    similarity_score: float
    matches: bool
    note: str | None = None
    outcome: MatchOutcome | None = None

    @property
    def resolved_outcome(self) -> MatchOutcome:
        """Three-valued verdict, defaulting from ``matches`` when unset."""
        if self.outcome is not None:
            return self.outcome
        return MatchOutcome.MATCH if self.matches else MatchOutcome.MISMATCH

    @property
    def is_confirmed(self) -> bool:
        """True only for a positive confirmation (MATCH)."""
        return self.resolved_outcome is MatchOutcome.MATCH

    @property
    def is_mismatch(self) -> bool:
        """True only for a real contradiction (MISMATCH)."""
        return self.resolved_outcome is MatchOutcome.MISMATCH

    @property
    def is_non_confirming(self) -> bool:
        """True when the field is neither confirmed nor a mismatch.

        i.e. NON_COMPARABLE or PARTIAL: a record was found and nothing is
        contradicted, but the claimed field could not be positively confirmed.
        """
        return self.resolved_outcome in (MatchOutcome.NON_COMPARABLE, MatchOutcome.PARTIAL)


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
    # Per-entry verification state carried on the result instead of stashed on
    # the shared ``FactChecker`` instance. ``check_entry`` runs concurrently
    # across a ThreadPoolExecutor, so any per-entry value written to ``self``
    # would be clobbered by sibling entries. These let a caller build a rich
    # :class:`VerificationResult` from THIS entry's run without re-querying and
    # without racing on shared mutable state.
    author_intersection: AuthorIntersectionResult | None = None
    source_records: dict[str, PublishedRecord | None] = field(default_factory=dict)


@dataclass
class FactCheckerConfig:
    """Configuration for fact-checking thresholds."""

    title_threshold: float = 0.90
    author_threshold: float = 0.80
    year_tolerance: int = 1
    venue_threshold: float = 0.70
    hallucination_max_score: float = 0.50
    # Fix B (abstention): when the best title-search candidate scores below this,
    # the tool could not find the real paper -- it has *no* positive evidence of
    # fabrication. Such cases ABSTAIN (NOT_FOUND) instead of asserting
    # HALLUCINATED. Reserve HALLUCINATED for positive-evidence signals
    # (fabricated DOI, future/invalid year, arXiv-ID misattribution, chimeric
    # title) that fire *before* the score gate.
    abstention_below: float = 0.50
    max_candidates_per_source: int = 10
    check_years: bool = True
    check_dois: bool = True
    # Verify the entry's own arXiv ID points to the entry's paper (catches
    # misattributed identifiers that title/author search silently VERIFIES).
    check_arxiv_consistency: bool = True
    # Below this normalized title score (0-1), the entry's arXiv ID is treated
    # as pointing to a *different* paper. Deliberately low so only clear
    # different-paper cases trip it, not minor preprint/published title edits.
    arxiv_consistency_min_title: float = 0.50
    # Verify the entry's own DOI points to the entry's paper. Today _validate_doi
    # only checks the DOI *resolves* (doi.org HEAD); it never checks the DOI
    # points to the CITED paper. A copy-paste DOI that resolves to a different
    # work otherwise survives because title/author search VERIFIES the entry
    # against its real record.
    check_doi_consistency: bool = True
    # Below this normalized title score (0-1), the DOI's Crossref record is
    # treated as a *different* paper. Deliberately low (mirrors arXiv) so only
    # clear different-paper cases trip it.
    doi_consistency_min_title: float = 0.50
    # CheckIfExist additions (Item 1 + 2): cascading + top-K retrieval.
    # Verification uses the CrossRef -> OpenAlex -> DBLP -> Semantic Scholar
    # cascade, which short-circuits on a high-confidence match so the slow
    # keyless-S2 / specialist sources stay off the hot path for easy entries.
    top_k: int = DEFAULT_TOP_K
    cascade_low_confidence: float = CASCADE_LOW_CONFIDENCE
    cascade_high_confidence: float = CASCADE_HIGH_CONFIDENCE
    openalex_mailto: str = DEFAULT_OPENALEX_MAILTO


# ------------- Item 5: Rich VerificationResult -------------


@dataclass
class VerificationResult:
    """Rich per-entry verification result with similarity breakdown.

    This is the structured output for callers that want everything the
    fact-checker computed -- per-field similarity scores, confirmed/suspect
    authors, and the source provenance. The classic ``FactCheckResult`` is
    still produced and serialized to JSONL for backward compatibility; this
    type is purely additive.
    """

    bibtex_key: str
    status: str
    confidence_score: float
    similarity_breakdown: dict[str, float]
    confirmed_authors: list[str]
    suspect_authors: list[str]
    sources_consulted: list[str]
    sources_confirmed: list[str]
    issues: list[str]
    matched_metadata: dict[str, str] | None = None


# ------------- Item 4: Numeric confidence (0-100) -------------


def compute_numeric_confidence(
    title_score: float,
    author_score: float,
    journal_score: float,
    year_score: float,
    issues: list[str],
    multi_source_bonus: float = 0.0,
    fabricated_author_count: int = 0,
) -> float:
    """Compute the CheckIfExist numeric confidence in [0, 100].

    All similarity inputs are 0-100 scale.

    - Case A (asymmetric, real-paper-fake-authors detector):
      ``S_title > CASE_A_TITLE_THRESHOLD AND S_author < CASE_A_AUTHOR_THRESHOLD``
      => ``confidence = S_title - 0.5 * (100 - S_author)``
    - Case B (default):
      ``confidence = mean(S_title, S_author, S_journal, S_year) + β_ms``

    Then explicit penalties for any reported issues are applied.

    Args:
        title_score: 0-100.
        author_score: 0-100.
        journal_score: 0-100.
        year_score: 0-100.
        issues: List of issue tags (``"title_mismatch"``, ``"author_mismatch"``,
            ``"journal_mismatch"`` / ``"venue_mismatch"``).
        multi_source_bonus: ``β_ms`` from cross-source author intersection.
        fabricated_author_count: Number of "suspect" authors flagged.

    Returns:
        Float in ``[0.0, 100.0]``.
    """
    bonus = max(0.0, min(float(multi_source_bonus), 10.0))

    # Case A: high-title-low-author asymmetric
    if title_score > CASE_A_TITLE_THRESHOLD and author_score < CASE_A_AUTHOR_THRESHOLD:
        confidence = title_score - 0.5 * (100.0 - author_score)
    else:
        confidence = (title_score + author_score + journal_score + year_score) / 4.0 + bonus

    # Issue-based penalties (constants, not auto-fit)
    issue_set = {i.lower() for i in (issues or [])}
    if "title_mismatch" in issue_set:
        confidence -= PENALTY_TITLE_MISMATCH
    if "author_mismatch" in issue_set:
        confidence -= PENALTY_AUTHOR_MISMATCH
    if "journal_mismatch" in issue_set or "venue_mismatch" in issue_set:
        confidence -= PENALTY_JOURNAL_MISMATCH

    # Per-fabricated-author penalty, capped
    if fabricated_author_count > 0:
        fab_penalty = min(
            PENALTY_PER_FABRICATED_AUTHOR * float(fabricated_author_count),
            PENALTY_FABRICATED_AUTHOR_CAP,
        )
        confidence -= fab_penalty

    return max(0.0, min(100.0, confidence))


def build_verification_result(
    fc_result: FactCheckResult,
    intersection: AuthorIntersectionResult | None = None,
    source_records: dict[str, PublishedRecord | None] | None = None,
) -> VerificationResult:
    """Assemble a rich :class:`VerificationResult` from a :class:`FactCheckResult`.

    Item 5: callers that want per-field similarity scores plus the cross-source
    author intersection get them here without the legacy JSONL output changing.

    The per-entry intersection and source records are carried on ``fc_result``
    itself (``fc_result.author_intersection`` / ``fc_result.source_records``),
    so the default is to read them off the result -- this is thread-safe because
    ``check_entry`` runs concurrently and no longer stashes per-entry state on
    the shared ``FactChecker`` instance. The explicit ``intersection`` /
    ``source_records`` arguments still override for callers that compute them
    separately.

    Args:
        fc_result: The classic fact-check result from ``FactChecker.check_entry``.
        intersection: Optional cross-source author intersection. Defaults to
            ``fc_result.author_intersection`` (the value from the SAME run that
            produced ``fc_result``).
        source_records: Optional ``source_name -> PublishedRecord`` mapping.
            Defaults to ``fc_result.source_records`` from the same run.

    Returns:
        :class:`VerificationResult`. The numeric ``confidence_score`` falls
        back to ``getattr(fc_result, "confidence_score", overall_confidence*100)``
        so older paths still get a sensible score.
    """
    # Default to the per-entry state carried on the result itself (thread-safe:
    # it belongs to THIS entry's run, not to shared instance state).
    if intersection is None:
        intersection = fc_result.author_intersection
    if source_records is None:
        source_records = fc_result.source_records or None

    # Per-field similarity breakdown (0-100 scale, more useful for display).
    breakdown: dict[str, float] = {}
    issues: list[str] = []
    for name, comp in fc_result.field_comparisons.items():
        breakdown[name] = float(comp.similarity_score) * 100.0
        if not comp.matches:
            issues.append(f"{name}_mismatch")

    confidence_score = float(getattr(fc_result, "confidence_score", fc_result.overall_confidence * 100.0))

    confirmed = list(intersection.confirmed) if intersection else []
    suspect = list(intersection.suspect) if intersection else []
    if suspect:
        issues.append("potential_fabricated_authors")

    sources_consulted = list(fc_result.api_sources_queried)
    sources_confirmed = list(fc_result.api_sources_with_hits)

    matched: dict[str, str] | None = None
    if fc_result.best_match is not None:
        matched = {
            "title": fc_result.best_match.title or "",
            "doi": fc_result.best_match.doi or "",
            "journal": fc_result.best_match.journal or "",
            "year": str(fc_result.best_match.year) if fc_result.best_match.year else "",
        }

    # Annotate with source records if provided -- useful for debugging.
    if source_records:
        sources_consulted = sorted({*sources_consulted, *source_records.keys()})

    return VerificationResult(
        bibtex_key=fc_result.entry_key,
        status=fc_result.status.value,
        confidence_score=confidence_score,
        similarity_breakdown=breakdown,
        confirmed_authors=confirmed,
        suspect_authors=suspect,
        sources_consulted=sources_consulted,
        sources_confirmed=sources_confirmed,
        issues=issues,
        matched_metadata=matched,
    )


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
        authors_rec = rec.surname_keys(limit=3)
        author_score = jaccard_similarity(authors_entry, authors_rec)

        return 0.7 * title_score + 0.3 * author_score


# ------------- API Clients -------------


class CrossrefClient:
    """Crossref API client for bibliographic searches."""

    def __init__(self, http: HttpClient):
        self.http = http

    def search(
        self,
        query: str,
        rows: int = 10,
        title: str | None = None,
        author: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search Crossref for bibliographic records.

        Args:
            query: Free-text bibliographic query (``"<title> <author>"`` blob).
                Used as the ``query.bibliographic`` fallback.
            rows: Max records to retrieve.
            title: Raw (un-normalized, author-free) title. When provided, the
                client uses Crossref's *fielded* ``query.title`` instead of the
                generic ``query.bibliographic`` blob, which keeps DOI-less
                ML-conference titles ranked correctly rather than letting the
                appended surname pull in unrelated records.
            author: First-author surname, sent as ``query.author`` to tighten
                the fielded result set. Only used when ``title`` is supplied.

        Returns:
            List of Crossref message items, never None. Empty on any error.
        """
        if title and title.strip():
            params: dict[str, Any] = {"query.title": title.strip(), "rows": rows}
            if author and author.strip():
                params["query.author"] = author.strip()
        else:
            params = {"query.bibliographic": query, "rows": rows}
        try:
            resp = self.http._request("GET", CROSSREF_API, params=params, accept="application/json", service="crossref")
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
            return items
        except Exception:
            return []

    def get_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Fetch the Crossref ``message`` record a DOI resolves to.

        Uses the Crossref REST ``/works/{doi}`` endpoint for reliable metadata
        (unlike a doi.org HEAD, which only tells you the DOI resolves). Returns
        ``None`` on any non-200 / parse failure / network error so callers can
        treat "cannot determine" as no evidence (FPR-safe), never a flag.
        """
        from urllib.parse import quote

        doi = normalize_doi_for_resolution(doi) or (doi or "").strip()
        if not doi:
            return None
        url = f"{CROSSREF_API}/{quote(doi, safe='')}"
        try:
            resp = self.http._request("GET", url, accept="application/json", service="crossref")
            if resp.status_code != 200:
                return None
            return resp.json().get("message", {}) or None
        except Exception:
            return None


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


class ArxivClient:
    """arXiv export API client for authoritative lookup by arXiv ID.

    Unlike Crossref/DBLP/Semantic Scholar, arXiv has the record for any valid
    arXiv ID immediately, so this is the reliable source for brand-new preprints
    that the aggregators have not indexed yet.
    """

    def __init__(self, http: HttpClient):
        self.http = http

    def fetch_atom(self, arxiv_id: str) -> str | None:
        """Fetch the raw Atom feed for a single arXiv ID, or None on failure."""
        params = {"id_list": arxiv_id}
        try:
            resp = self.http._request("GET", ARXIV_API, params=params, accept="application/atom+xml", service="arxiv")
            if resp.status_code != 200:
                return None
            return resp.text
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


@dataclass(frozen=True)
class VenueMatchResult:
    """Trichotomy result of :func:`venues_match`.

    ``outcome`` is one of MATCH / MISMATCH / NON_COMPARABLE. NON_COMPARABLE means
    the claimed published venue cannot be *confirmed* (blank on a side, or the
    matched record is only a preprint/series) -- it is NOT a match.
    """

    outcome: MatchOutcome
    score: float

    @property
    def is_confirmed(self) -> bool:
        return self.outcome is MatchOutcome.MATCH

    @property
    def is_mismatch(self) -> bool:
        return self.outcome is MatchOutcome.MISMATCH


def venues_match(venue_a: str, venue_b: str, threshold: float = 0.70) -> VenueMatchResult:
    """Compare two venue names (alias-aware) into a three-valued outcome.

    P2.5: Uses EXPANDED_VENUE_ALIASES from matching.py for venue coverage.

    The comparison is three-valued so a blank/preprint record can no longer
    masquerade as positive confirmation:

    - NON_COMPARABLE when either side is empty, OR when the matched record's
      venue is a preprint/publisher-series (arXiv/CoRR, bioRxiv, PMLR, JMLR
      W&CP). A preprint record cannot *confirm* the published venue the entry
      claims -- it says nothing about it -- so it must not read as a match.
    - MATCH when both sides are populated real venues that canonicalize equal
      (or fuzzy >= threshold).
    - MISMATCH when both sides are populated real venues that differ.

    Returns a :class:`VenueMatchResult`.
    """
    # Empty on either side: nothing to confirm -> non-comparable.
    if not venue_a or not venue_b:
        return VenueMatchResult(MatchOutcome.NON_COMPARABLE, 1.0)

    # Preprint / non-specific series on either side: the published venue the
    # entry cites cannot be confirmed from a preprint record -> non-comparable.
    if is_preprint_or_series_venue(venue_a) or is_preprint_or_series_venue(venue_b):
        return VenueMatchResult(MatchOutcome.NON_COMPARABLE, 1.0)

    norm_a = normalize_venue(venue_a)
    norm_b = normalize_venue(venue_b)

    # P2.5: Use expanded venue aliases from matching.py
    canonical_a = get_canonical_venue(norm_a, EXPANDED_VENUE_ALIASES)
    canonical_b = get_canonical_venue(norm_b, EXPANDED_VENUE_ALIASES)
    if canonical_a and canonical_b:
        if canonical_a == canonical_b:
            return VenueMatchResult(MatchOutcome.MATCH, 0.95)
        return VenueMatchResult(MatchOutcome.MISMATCH, 0.0)  # Known different venues

    # Fall back to fuzzy matching
    score = token_sort_ratio(normalize_title_for_match(norm_a), normalize_title_for_match(norm_b)) / 100.0
    if score >= threshold:
        return VenueMatchResult(MatchOutcome.MATCH, score)
    return VenueMatchResult(MatchOutcome.MISMATCH, score)


#: A record whose title matches the entry but whose publication year is at least
#: this many years away, AND whose claimed venue cannot be positively confirmed,
#: is treated as a DIFFERENT edition/reprint of the same work (or a same-title
#: decoy from free-text retrieval) rather than positive evidence the entry is
#: wrong. Its venue/year fields abstain (NON_COMPARABLE) instead of flagging a
#: mismatch. Small year slips (typos) stay below this gap and still surface as a
#: YEAR_MISMATCH; a genuinely matching venue or a wrong author is never masked.
_EDITION_YEAR_GAP = 4


def _doi_is_preprint(doi: str | None) -> bool:
    """True when ``doi`` is a preprint DOI: arXiv (``10.48550/arXiv...``) or
    bioRxiv/medRxiv (``10.1101...``).

    The matched record's identifier is the authoritative preprint signal -- more
    reliable than the venue STRING, which APIs sometimes fill with an institutional
    repository name (e.g. 'UvA-DARE (University of Amsterdam)' for an arXiv DataCite
    DOI) that the venue-string heuristic does not recognize. A preprint record
    cannot confirm a claimed *published* venue.
    """
    if not doi:
        return False
    return doi.lower().startswith(("10.48550/arxiv", "10.1101"))


def _doi_resolves(client: httpx.Client, doi: str) -> bool | None:
    """Probe whether a DOI resolves at doi.org.

    Returns:
        False  -- the DOI definitively does not exist (404/410, confirmed by a
                  GET retry to rule out HEAD-hostile hosts).
        True   -- it resolves (2xx/3xx) or is blocked by a publisher (418/403/
                  429): a block is not evidence of an invalid DOI.
        None   -- network error; the caller should treat this as non-evidence
                  (do not penalize).
    """
    url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    headers = {"User-Agent": "BibtexFactChecker/1.0"}
    try:
        resp = client.head(url, headers=headers)
    except Exception:
        return None  # Network errors are not DOI validation failures.
    # Only 404/410 indicate a DOI that truly doesn't exist. Other 4xx (418
    # bot-detection, 403 access control, 429 rate limit) are publisher-side
    # blocks, not evidence of an invalid DOI.
    if resp.status_code not in (404, 410):
        return True
    # Some hosts are HEAD-hostile (return 404/410 to HEAD but resolve to GET).
    # Retry once with a tiny ranged GET before concluding the DOI is missing.
    try:
        retry = client.get(url, headers={**headers, "Range": "bytes=0-0"})
    except Exception:
        return None
    # Definitively missing only if the GET also returns 404/410.
    return retry.status_code not in (404, 410)


# ------------- Fact Checker Core -------------


class FactChecker:
    """Validates bibliographic entries against external APIs."""

    API_SOURCES = ["crossref", "dblp", "semanticscholar", "openalex", "openreview"]

    def __init__(
        self,
        crossref: CrossrefClient,
        dblp: DBLPClient,
        s2: SemanticScholarClient,
        config: FactCheckerConfig,
        logger: logging.Logger,
        openalex: OpenAlexClient | None = None,
        arxiv: ArxivClient | None = None,
        openreview: OpenReviewClient | None = None,
    ):
        self.crossref = crossref
        self.dblp = dblp
        self.s2 = s2
        self.config = config
        self.logger = logger
        # Authoritative lookup-by-arXiv-ID source. Optional so existing callers
        # and tests that don't need preprint verification keep working.
        self.arxiv: ArxivClient | None = arxiv
        # OpenAlex client is optional; tests can pass a fake. The cascade falls
        # back to creating a default client if not provided.
        self.openalex: OpenAlexClient | None = openalex
        # OpenReview client is optional and lazily built from the shared HTTP
        # client inside the cascade (mirroring OpenAlex). It is the authoritative
        # source for ICLR/NeurIPS/TMLR submissions the other sources miss.
        self.openreview: OpenReviewClient | None = openreview
        # Per-entry verification state (cross-source author intersection,
        # per-source records) is NOT stored on the shared instance: check_entry
        # runs concurrently across a ThreadPoolExecutor, so a sibling entry would
        # clobber it. It is returned on the FactCheckResult instead. See
        # FactCheckResult.author_intersection / .source_records.
        #
        # Memoize fetched+parsed arXiv records by ID: the consistency pre-check
        # and the by-ID candidate query both look up the entry's arXiv ID in one
        # verification pass, so this avoids a duplicate network fetch + parse.
        # This IS a cross-entry cache (not per-entry verdict state), shared by
        # all concurrent check_entry calls, so its dict mutation is guarded by a
        # lock. The actual fetch happens OUTSIDE the lock (double-checked insert)
        # so a slow network call never serializes the other workers.
        self._arxiv_record_cache: dict[str, PublishedRecord | None] = {}
        self._arxiv_cache_lock = threading.Lock()

    def _arxiv_record(self, arxiv_id: str) -> PublishedRecord | None:
        """Fetch + parse the arXiv record for an ID, memoized per checker.

        Thread-safe: the cache dict is shared across concurrent ``check_entry``
        workers. We read/insert under ``_arxiv_cache_lock`` but perform the
        network fetch+parse outside it (double-checked locking) so a slow arXiv
        request does not stall sibling workers.
        """
        with self._arxiv_cache_lock:
            if arxiv_id in self._arxiv_record_cache:
                return self._arxiv_record_cache[arxiv_id]
        rec: PublishedRecord | None = None
        if self.arxiv is not None:
            xml = self.arxiv.fetch_atom(arxiv_id)
            if xml:
                rec = arxiv_atom_to_record(xml)
        with self._arxiv_cache_lock:
            # Another worker may have populated the same ID while we fetched;
            # keep the first cached value so the memo stays a stable cache.
            if arxiv_id in self._arxiv_record_cache:
                return self._arxiv_record_cache[arxiv_id]
            self._arxiv_record_cache[arxiv_id] = rec
        return rec

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

        raw_doi = entry.get("doi", "")
        if not raw_doi:
            return None
        # Normalize: strip URL prefix/lowercase, and for arXiv DataCite DOIs drop
        # a trailing version suffix (the versioned DOI 404s at doi.org but the
        # unversioned one resolves). Fall back to the raw string if normalization
        # yields nothing.
        doi = normalize_doi_for_resolution(raw_doi) or raw_doi.strip()

        # Reuse the shared httpx.Client to avoid per-entry TCP/TLS overhead.
        if _doi_resolves(self.crossref.http.client, doi) is False:
            return FactCheckStatus.DOI_NOT_FOUND
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

        # Pre-search consistency: the entry's own arXiv ID must point to *this*
        # paper. A wrong ID otherwise survives because title/author search
        # VERIFIES the entry against the real paper from Crossref/DBLP/S2,
        # silently leaving the misattributed identifier in place.
        if self.config.check_arxiv_consistency:
            arxiv_status = self._check_arxiv_id_consistency(entry)
            if arxiv_status is not None:
                return arxiv_status

        # Pre-search consistency: the entry's own DOI must point to *this* paper.
        # A copy-paste DOI that resolves to a different work otherwise survives
        # because title/author search VERIFIES the entry against its real record.
        if self.config.check_doi_consistency:
            doi_consistency_status = self._check_doi_consistency(entry)
            if doi_consistency_status is not None:
                return doi_consistency_status

        query = f"{title_norm} {first_author}".strip()
        # Item 1: cascading source order (CrossRef -> OpenAlex -> DBLP -> S2).
        candidates = self._query_cascade(entry, query, sources_queried, sources_with_hits, errors)

        # Authoritative arXiv-by-ID lookup. Added as an extra candidate so valid
        # but not-yet-indexed preprints verify instead of being flagged
        # HALLUCINATED/NOT_FOUND from a failed title search.
        candidates.extend(self._query_arxiv_by_id(entry, sources_queried, sources_with_hits, errors))

        if not candidates:
            status = FactCheckStatus.API_ERROR if errors else FactCheckStatus.NOT_FOUND
            # No candidates -> no per-entry intersection/source records. These
            # ride on the result (not self) so concurrent entries don't clobber
            # each other.
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
                author_intersection=None,
                source_records={},
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
                author_intersection=None,
                source_records={},
            )

        # Sort candidates by score descending (used below for per-source author
        # intersection, which takes the top record per source).
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Pick the best match preferring fuller positive confirmation: among the
        # candidates that tie at the top of the title+author score, choose the one
        # that confirms the most claimed fields, so a proceedings record that
        # confirms the venue wins over a tied preprint that cannot.
        best_score, best_match, _source = self._select_best_candidate(entry, candidates)

        # Item 3: cross-source author intersection -- pick the best record from
        # each source and intersect their author lists. Carried on the returned
        # FactCheckResult (NOT on self) so callers can build a rich
        # VerificationResult without re-querying and without racing concurrent
        # check_entry calls.
        best_per_source: dict[str, PublishedRecord | None] = {}
        for _cand_score, cand_rec, cand_source in candidates:
            current = best_per_source.get(cand_source)
            if current is None:
                best_per_source[cand_source] = cand_rec
            # The list is already sorted desc; first entry per source wins.
        intersection = cross_source_author_intersection(best_per_source, multi_source_bonus=MULTI_SOURCE_BONUS)

        field_comparisons = self._compare_all_fields(entry, best_match)
        status = self._determine_status(best_score, field_comparisons, sources_with_hits)

        # FPR guard (Task 2b): an AUTHOR_MISMATCH driven by a candidate from a
        # source WITHOUT authoritative given/family names (S2 flat names, a DBLP
        # hit, an OpenAlex display_name) may be a NAME-PARSING artifact rather
        # than a real author discrepancy. Before trusting it, re-check the SAME
        # paper against a STRUCTURED source (Crossref by DOI, else Crossref title
        # search). If the structured comparison MATCHES, the mismatch was a parse
        # artifact -> recompute the comparison/status against the structured
        # record. If it still mismatches (or no structured source is reachable),
        # the AUTHOR_MISMATCH stands. This only changes WHICH surname tokens are
        # compared; it never relaxes ordering or the match threshold.
        if status is FactCheckStatus.AUTHOR_MISMATCH and not best_match.structured_names:
            structured_rec = self._structured_author_recheck(entry, best_match)
            if structured_rec is not None:
                best_match = structured_rec
                field_comparisons = self._compare_all_fields(entry, best_match)
                status = self._determine_status(best_score, field_comparisons, sources_with_hits)

        # Post-match: check preprint status. UNCONFIRMED is included because a
        # venue we could not confirm is exactly the case where an independent
        # preprint-only signal (arXiv-only, no DOI/venue) upgrades the verdict to
        # the positive-evidence PREPRINT_ONLY.
        if status in (
            FactCheckStatus.VERIFIED,
            FactCheckStatus.VENUE_MISMATCH,
            FactCheckStatus.UNCONFIRMED,
        ):
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

        # Item 4: numeric (0-100) confidence with explicit penalties/bonuses.
        # Stored alongside the legacy ``overall_confidence`` (0-1 calibrated)
        # via the ``confidence_score`` attribute so existing JSONL output keys
        # remain untouched.
        title_pct = (field_comparisons["title"].similarity_score if "title" in field_comparisons else 0.0) * 100.0
        author_pct = (field_comparisons["author"].similarity_score if "author" in field_comparisons else 0.0) * 100.0
        venue_pct = (field_comparisons["venue"].similarity_score if "venue" in field_comparisons else 0.0) * 100.0
        year_pct = (field_comparisons["year"].similarity_score if "year" in field_comparisons else 0.0) * 100.0

        issues: list[str] = []
        if "title" in field_comparisons and not field_comparisons["title"].matches:
            issues.append("title_mismatch")
        if "author" in field_comparisons and not field_comparisons["author"].matches:
            issues.append("author_mismatch")
        if "venue" in field_comparisons and not field_comparisons["venue"].matches:
            issues.append("venue_mismatch")
        if "year" in field_comparisons and not field_comparisons["year"].matches:
            issues.append("year_mismatch")

        numeric_conf = compute_numeric_confidence(
            title_score=title_pct,
            author_score=author_pct,
            journal_score=venue_pct,
            year_score=year_pct,
            issues=issues,
            multi_source_bonus=intersection.bonus,
            fabricated_author_count=len(intersection.suspect),
        )

        result = FactCheckResult(
            entry_key=entry_key,
            entry_type=entry_type,
            status=status,
            overall_confidence=confidence,
            field_comparisons=field_comparisons,
            best_match=best_match,
            api_sources_queried=sources_queried,
            api_sources_with_hits=sources_with_hits,
            errors=errors,
            # Per-entry state carried on the result (not stashed on self) so
            # concurrent check_entry calls don't clobber each other.
            author_intersection=intersection,
            source_records=best_per_source,
        )
        # Stash the numeric (0-100) confidence as an attribute -- additive only,
        # not part of the JSONL schema so existing consumers keep working.
        result.confidence_score = numeric_conf  # type: ignore[attr-defined]
        return result

    @staticmethod
    def _arxiv_id_from_entry(entry: dict[str, Any]) -> str | None:
        """Extract a bare arXiv ID from an entry's eprint/url/howpublished fields.

        Recognizes both modern (``2602.01031``) and legacy (``cs/0001001``) IDs,
        from an explicit ``eprint`` (with an arXiv ``archivePrefix``/``journal``)
        or from an ``arxiv.org/abs/<id>`` URL / ``arXiv:<id>`` string.
        """
        eprint = (entry.get("eprint") or "").strip()
        archive = (entry.get("archiveprefix") or entry.get("archivePrefix") or "").strip().lower()
        if eprint and (archive == "arxiv" or re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", eprint)):
            bare = re.sub(r"v\d+$", "", eprint)
            # A legacy-scheme eprint (e.g. "math.GT/0309136") is structurally
            # valid; a modern "YYMM.NNNNN" must have a real month.
            if is_valid_arxiv_id(bare):
                return bare
            return None

        for field_name in ("url", "howpublished", "journal", "note"):
            value = entry.get(field_name) or ""
            m = re.search(r"arxiv\.org/abs/([^\s,}{]+)", value, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"arxiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", value, flags=re.IGNORECASE)
            if m:
                bare = re.sub(r"v\d+$", "", m.group(1).strip())
                if is_valid_arxiv_id(bare):
                    return bare
        return None

    def _id_anchored_author_mismatch(
        self,
        entry: dict[str, Any],
        rec: PublishedRecord,
        source: str,
        identifier: str,
        id_kind: str,
    ) -> FactCheckResult | None:
        """Flag an ID-resolved record whose title matches but authors are wrong.

        The caller has already confirmed that ``rec`` IS the cited paper (its
        title matches the entry). This catches the residual case where a
        hallucinated citation carries a *correct* DOI/arXiv-ID resolving to the
        exact real paper, but lists SWAPPED or PLACEHOLDER authors.

        Both author sides are reduced to canonical surname keys with the same
        machinery used everywhere else -- the entry via ``authors_last_names``
        and the resolved record via ``PublishedRecord.surname_keys`` (both route
        each name through ``last_name_from_person``) -- then compared with
        ``symmetric_author_match``.

        FPR-safe gating: a flag fires ONLY on a genuine ``MISMATCH`` (different
        lead author / transposition / placeholder authors). ``MATCH`` (correct
        authors), ``PARTIAL`` (consistent-but-incomplete, e.g. "and others"),
        and ``NON_COMPARABLE`` (missing authors on either side) all return
        ``None`` so valid citations are never touched.
        """
        entry_names = self._entry_surname_keys(entry, rec, limit=10_000)
        api_names = rec.surname_keys(limit=10_000)
        # Nothing comparable on either side -> cannot refute (FPR-safe).
        if not entry_names or not api_names:
            return None
        author_result = symmetric_author_match(
            entry_names, api_names, threshold=self.config.author_threshold, order_reliable=rec.order_reliable
        )
        if not author_result.is_mismatch:
            return None

        self.logger.warning(
            "%s %s for entry %r resolves to the cited paper but with mismatched "
            "authors: entry authors %r vs %s authors %r",
            id_kind,
            identifier,
            entry.get("ID", "?"),
            entry.get("author", ""),
            source,
            api_names,
        )
        return FactCheckResult(
            entry_key=entry.get("ID", "unknown"),
            entry_type=entry.get("ENTRYTYPE", "misc").lower(),
            status=FactCheckStatus.AUTHOR_MISMATCH,
            overall_confidence=0.0,
            field_comparisons=self._compare_all_fields(entry, rec),
            best_match=rec,
            api_sources_queried=[source],
            api_sources_with_hits=[source],
            errors=[
                f"{id_kind} {identifier} resolves to the cited paper "
                f"{rec.title!r}, but the entry's authors do not match the "
                f"record's authors ({api_names})"
            ],
        )

    def _check_arxiv_id_consistency(self, entry: dict[str, Any]) -> FactCheckResult | None:
        """Flag entries whose cited arXiv ID resolves to a *different* paper.

        Title/author search happily VERIFIES a misattributed entry against the
        real paper returned by Crossref/DBLP/S2, so a wrong arXiv ID (a
        copy-paste or lookup error) survives unnoticed. Here we fetch the
        entry's own arXiv ID and require its title to match the entry; a clear
        mismatch is a misattributed identifier, not a valid preprint, and is
        reported as :class:`FactCheckStatus.ARXIV_ID_MISMATCH`.

        Returns ``None`` when there is nothing to check (no arXiv client, no
        arXiv ID, lookup failed, or the titles are consistent) so the normal
        verification flow proceeds.
        """
        if self.arxiv is None:
            return None
        arxiv_id = self._arxiv_id_from_entry(entry)
        if not arxiv_id:
            return None

        try:
            rec = self._arxiv_record(arxiv_id)
        except Exception:
            return None
        if rec is None or not rec.title:
            return None

        entry_title = normalize_title_for_match(entry.get("title", ""))
        if not entry_title:
            return None
        arxiv_title = normalize_title_for_match(rec.title)
        title_score = token_sort_ratio(entry_title, arxiv_title) / 100.0
        if title_score >= self.config.arxiv_consistency_min_title:
            # Title confirms this IS the cited paper. The arXiv ID is the entry's
            # OWN identifier, so a genuine author mismatch here is positive
            # evidence of ID-anchored author fabrication (swapped/placeholder
            # authors on an otherwise-correct preprint).
            return self._id_anchored_author_mismatch(
                entry, rec, source="arxiv", identifier=arxiv_id, id_kind="arXiv ID"
            )

        self.logger.warning(
            "arXiv ID %s for entry %r points to a different paper: entry title "
            "%r vs arXiv title %r (title score %.2f)",
            arxiv_id,
            entry.get("ID", "?"),
            entry.get("title", ""),
            rec.title,
            title_score,
        )
        return FactCheckResult(
            entry_key=entry.get("ID", "unknown"),
            entry_type=entry.get("ENTRYTYPE", "misc").lower(),
            status=FactCheckStatus.ARXIV_ID_MISMATCH,
            overall_confidence=0.0,
            field_comparisons={},
            best_match=rec,
            api_sources_queried=["arxiv"],
            api_sources_with_hits=["arxiv"],
            errors=[
                f"arXiv ID {arxiv_id} resolves to {rec.title!r}, which does not "
                f"match entry title {entry.get('title', '')!r}"
            ],
        )

    def _check_doi_consistency(self, entry: dict[str, Any]) -> FactCheckResult | None:
        """Flag entries whose cited DOI resolves to a *different* paper.

        ``_validate_doi`` only checks that the DOI *resolves* (doi.org HEAD); it
        never checks that the DOI points to the CITED paper. Title/author search
        then happily VERIFIES a misattributed entry against its real record, so a
        copy-paste DOI (e.g. "IBRNet" carrying the DOI of a 3D-detection paper)
        survives unnoticed. Here we fetch the DOI's Crossref record and require
        its title to match the entry; a clear mismatch is reported as
        :class:`FactCheckStatus.DOI_MISMATCH`.

        FPR-safe: returns ``None`` (no flag) when there is nothing to check (no
        DOI, feature off) OR the determination is uncertain (Crossref fetch
        failed / non-200 / no record / no title -- e.g. IEEE bot-block or a DOI
        Crossref doesn't index). Only a SUCCESSFULLY fetched record whose title
        CLEARLY differs (score below ``doi_consistency_min_title``) trips it.
        """
        if not self.config.check_doi_consistency:
            return None
        raw_doi = entry.get("doi", "")
        if not raw_doi:
            return None

        entry_title = normalize_title_for_match(entry.get("title", ""))
        if not entry_title:
            return None

        rec = self._structured_record_by_doi(raw_doi)
        # Cannot determine -> do NOT flag (keeps FPR low).
        if rec is None or not rec.title:
            return None

        doi_title = normalize_title_for_match(rec.title)
        title_score = token_sort_ratio(entry_title, doi_title) / 100.0
        if title_score >= self.config.doi_consistency_min_title:
            # Title confirms this IS the cited paper. The DOI is the entry's OWN
            # identifier, so a genuine author mismatch here is positive evidence
            # of ID-anchored author fabrication (swapped/placeholder authors on
            # an entry that otherwise carries the correct DOI).
            return self._id_anchored_author_mismatch(entry, rec, source="crossref", identifier=raw_doi, id_kind="DOI")

        self.logger.warning(
            "DOI %s for entry %r points to a different paper: entry title " "%r vs DOI title %r (title score %.2f)",
            raw_doi,
            entry.get("ID", "?"),
            entry.get("title", ""),
            rec.title,
            title_score,
        )
        return FactCheckResult(
            entry_key=entry.get("ID", "unknown"),
            entry_type=entry.get("ENTRYTYPE", "misc").lower(),
            status=FactCheckStatus.DOI_MISMATCH,
            overall_confidence=0.0,
            field_comparisons={},
            best_match=rec,
            api_sources_queried=["crossref"],
            api_sources_with_hits=["crossref"],
            errors=[
                f"DOI {raw_doi} resolves to {rec.title!r}, which does not "
                f"match entry title {entry.get('title', '')!r}"
            ],
        )

    def _structured_record_by_doi(self, raw_doi: str) -> PublishedRecord | None:
        """Fetch the Crossref ``message`` for ``raw_doi`` as a structured record.

        Shared by ``_check_doi_consistency`` (DOI-target consistency) and
        ``_structured_author_recheck`` (Task 2b name-parse FPR guard) so the
        ``get_by_doi`` round-trip is never duplicated. Crossref returns
        authoritative ``given``/``family`` fields, so the resulting record has
        ``structured_names=True``. Returns ``None`` on any non-200 / parse / no
        DOI so callers treat "cannot determine" as no evidence (FPR-safe).
        """
        if not raw_doi:
            return None
        try:
            msg = self.crossref.get_by_doi(raw_doi)
        except Exception:
            return None
        if msg is None:
            return None
        return crossref_message_to_record(msg)

    def _structured_author_recheck(self, entry: dict[str, Any], best_match: PublishedRecord) -> PublishedRecord | None:
        """Re-fetch the cited paper from a STRUCTURED source to vet an AUTHOR_MISMATCH.

        Called only when the candidate that produced an AUTHOR_MISMATCH came from
        a source WITHOUT authoritative given/family names (``best_match`` has
        ``structured_names`` False). A family-first CJK name flattened by such a
        source can be mis-tokenized (entry "Chen Xing" vs a flat "Xing Chen"
        record), so the mismatch may be a parsing artifact rather than a real
        author discrepancy.

        Strategy (reuses existing fetch paths, no duplicate work):
          1. If the entry carries a DOI, fetch the structured Crossref record via
             the shared ``_structured_record_by_doi`` (same call
             ``_check_doi_consistency`` uses).
          2. Otherwise, if the entry has a confident title, run a fielded
             Crossref title+author search and take the single best
             title-matching hit.

        The returned structured record is handed back ONLY when (a) its title
        confirms it is the cited paper and (b) the author comparison against its
        authoritative given/family names is a positive MATCH. In every other
        case (no structured source reachable, title doesn't confirm, or authors
        STILL mismatch) we return ``None`` so the original AUTHOR_MISMATCH stands.
        This narrows surname tokenization only; it never relaxes author ordering
        or the match threshold.
        """
        entry_title = normalize_title_for_match(entry.get("title", ""))
        if not authors_last_names(entry.get("author", ""), limit=10_000):
            return None

        # ----- Path 1: DOI present -> authoritative Crossref record. -----
        raw_doi = entry.get("doi", "") or ""
        structured: PublishedRecord | None = None
        if raw_doi:
            rec = self._structured_record_by_doi(raw_doi)
            if rec is not None and rec.structured_names and rec.title and entry_title:
                doi_title = normalize_title_for_match(rec.title)
                if token_sort_ratio(entry_title, doi_title) / 100.0 >= self.config.doi_consistency_min_title:
                    structured = rec

        # ----- Path 2: no usable DOI hit -> confident-title Crossref search. -----
        if structured is None and entry_title:
            raw_title = entry.get("title", "") or ""
            first_author = first_author_surname(entry)
            try:
                items = self.crossref.search(raw_title, rows=5, title=raw_title, author=first_author)
            except Exception:
                items = []
            best_struct: PublishedRecord | None = None
            best_title_score = 0.0
            for item in items or []:
                rec = crossref_message_to_record(item)
                if rec is None or not rec.structured_names or not rec.title:
                    continue
                ts = token_sort_ratio(entry_title, normalize_title_for_match(rec.title)) / 100.0
                if ts > best_title_score:
                    best_title_score, best_struct = ts, rec
            # Require a CONFIDENT title to be sure we re-checked the same paper.
            if best_struct is not None and best_title_score >= self.config.title_threshold:
                structured = best_struct

        if structured is None:
            return None

        # Re-run the author comparison against authoritative given/family keys.
        # Same matcher, same threshold, same ordering rules -- only the surname
        # tokens are now trustworthy. The entry side is disambiguated against the
        # structured family set (resolving family-first CJK names).
        api_names = structured.surname_keys(limit=10_000)
        if not api_names:
            return None
        entry_names = self._entry_surname_keys(entry, structured, limit=10_000)
        author_result = symmetric_author_match(
            entry_names, api_names, threshold=self.config.author_threshold, order_reliable=structured.order_reliable
        )
        if not author_result.is_confirmed:
            # Still not a positive MATCH (genuine different/swapped/placeholder
            # authors, or only a partial confirmation) -> keep AUTHOR_MISMATCH.
            return None

        self.logger.debug(
            "Author mismatch for entry %r suppressed: structured Crossref record "
            "confirms authors %r match entry %r (was mis-tokenized by an "
            "unstructured source)",
            entry.get("ID", "?"),
            api_names,
            entry_names,
        )
        return structured

    def _query_arxiv_by_id(
        self,
        entry: dict[str, Any],
        sources_queried: list[str],
        sources_with_hits: list[str],
        errors: list[str],
    ) -> list[tuple[float, PublishedRecord, str]]:
        """Look the entry up on arXiv by its ID and return it as a scored candidate.

        This rescues valid arXiv-only preprints that title/author search misses
        because Crossref/DBLP/Semantic Scholar have not indexed them yet, which
        otherwise produced false HALLUCINATED/NOT_FOUND verdicts.
        """
        if self.arxiv is None:
            return []
        arxiv_id = self._arxiv_id_from_entry(entry)
        if not arxiv_id:
            return []

        sources_queried.append("arxiv")
        try:
            rec = self._arxiv_record(arxiv_id)
        except Exception as e:
            errors.append(f"arXiv: {e}")
            return []
        if rec is None:
            return []

        sources_with_hits.append("arxiv")
        title_norm = normalize_title_for_match(entry.get("title", ""))
        authors_ref = authors_last_names(entry.get("author", ""), limit=3)
        score = self._score_candidate(title_norm, authors_ref, rec)
        return [(score, rec, "arxiv")]

    def _query_cascade(
        self,
        entry: dict[str, Any],
        query: str,
        sources_queried: list[str],
        sources_with_hits: list[str],
        errors: list[str],
    ) -> list[tuple[float, PublishedRecord, str]]:
        """Source order: CrossRef -> OpenAlex -> DBLP -> OpenReview -> Semantic Scholar.

        Item 1 (CheckIfExist Algorithm 1, Abbonato 2026). Each step retrieves
        ``config.top_k`` candidates and re-ranks them by Levenshtein title
        similarity (Item 2). The cascade short-circuits as soon as a source
        returns a candidate at or above ``cascade_high_confidence``.

        Retrieval (this fix): Crossref and OpenAlex are queried with *fielded
        title* searches (``query.title`` / ``filter=title.search:``) using the
        raw, author-free title rather than the normalized ``"title + surname"``
        blob. The blob fed to the free-text BM25 endpoints returned unrelated
        papers for DOI-less ML-conference titles (ICML/ICLR/NeurIPS), causing
        ~61% of valid references to be falsely flagged. DBLP -- which
        authoritatively indexes those venues -- is added as a cascade step
        after OpenAlex.

        Order rationale (throughput): the fast, broad sources come first so the
        slow specialist is only reached on hard entries. OpenAlex runs on the
        polite pool (~100 req/min) and aggregates Crossref + others; DBLP
        (~30 req/min) is the CS-conference authority; OpenReview (~30 req/min) is
        the ICLR/NeurIPS/TMLR submission authority queried before the slow
        keyless Semantic Scholar (~10 req/min), which is queried last to keep it
        off the hot path for the easy majority of entries.

        Returns:
            List of ``(score, record, source_name)`` tuples, possibly from
            multiple sources if intermediate matches were below threshold.
        """
        raw_title = entry.get("title", "") or ""
        title_norm = normalize_title_for_match(raw_title)
        first_author = first_author_surname(entry)
        authors_ref = authors_last_names(entry.get("author", ""), limit=3)
        top_k = max(1, min(int(self.config.top_k), MAX_TOP_K))

        all_candidates: list[tuple[float, PublishedRecord, str]] = []

        def _ingest(source_name: str, records: list[PublishedRecord]) -> float:
            """Score + add records under ``source_name``; return best score."""
            best_local = 0.0
            ranked = select_top_k_by_title_similarity(entry.get("title", ""), records, k=top_k)
            for _title_score, rec in ranked:
                score = self._score_candidate(title_norm, authors_ref, rec)
                all_candidates.append((score, rec, source_name))
                if score > best_local:
                    best_local = score
            return best_local

        # ----- Step 1: CrossRef (fielded query.title + query.author) -----
        sources_queried.append("crossref")
        try:
            cr_items = self.crossref.search(query, rows=top_k, title=raw_title, author=first_author)
        except Exception as exc:
            cr_items = []
            errors.append(f"Crossref: {exc}")
        cr_records: list[PublishedRecord] = []
        for item in cr_items or []:
            rec = crossref_message_to_record(item)
            if rec:
                cr_records.append(rec)
        if cr_records:
            sources_with_hits.append("crossref")
        _ingest("crossref", cr_records)
        if self._has_full_confirmation(entry, all_candidates):
            return all_candidates

        # ----- Step 2: OpenAlex (high-rate aggregator, broad coverage) -----
        if self.openalex is None:
            # Lazily build a default OpenAlex client, reusing the shared HTTP
            # client reachable through the Crossref client. Without a shared
            # client we skip OpenAlex rather than fabricate a bare, unthrottled
            # connection -- this keeps tests hermetic and avoids impolite
            # off-pool traffic.
            shared_http = getattr(self.crossref, "http", None)
            if shared_http is not None:
                self.openalex = OpenAlexClient(
                    http=shared_http,
                    mailto=self.config.openalex_mailto,
                )
        if self.openalex is not None:
            sources_queried.append("openalex")
            try:
                # Fielded filter=title.search:<raw title>, free-text fallback.
                oa_items = self.openalex.search(query, limit=top_k, title=raw_title)
            except Exception as exc:
                oa_items = []
                errors.append(f"OpenAlex: {exc}")
            oa_records: list[PublishedRecord] = []
            for item in oa_items or []:
                rec = openalex_work_to_candidate_record(item)
                if rec:
                    oa_records.append(rec)
            if oa_records:
                sources_with_hits.append("openalex")
            _ingest("openalex", oa_records)
            if self._has_full_confirmation(entry, all_candidates):
                return all_candidates

        # ----- Step 3: DBLP (authoritative ICML/ICLR/NeurIPS index) -----
        # DBLP's q= is a token-AND matcher (not BM25 relevance), so the raw
        # title + surname locates the exact paper. Uses the permissive
        # dblp_hit_to_candidate_record so DOI-less / CoRR conference hits are
        # kept as scorable candidates (the strict resolver converter drops them).
        if self.dblp is not None:
            sources_queried.append("dblp")
            dblp_query = f"{raw_title} {first_author}".strip()
            try:
                dblp_hits = self.dblp.search(dblp_query, max_hits=top_k)
            except Exception as exc:
                dblp_hits = []
                errors.append(f"DBLP: {exc}")
            dblp_records: list[PublishedRecord] = []
            for hit in dblp_hits or []:
                rec = dblp_hit_to_candidate_record(hit)
                if rec:
                    dblp_records.append(rec)
            if dblp_records:
                sources_with_hits.append("dblp")
            _ingest("dblp", dblp_records)
            if self._has_full_confirmation(entry, all_candidates):
                return all_candidates

        # ----- Step 4: OpenReview (authoritative ICLR/NeurIPS/TMLR registry) -----
        # OpenReview owns the submission record for most ML conferences, which the
        # DOI/CS-index sources above frequently fail to *positively* confirm
        # (these land in the "could-not-verify" bucket). It exposes ~Given_Family
        # profile handles, yielding authoritative family names. Queried before the
        # slow keyless Semantic Scholar so the ML-venue authority is consulted
        # first. Lazily built from the shared HTTP client (reachable via Crossref),
        # mirroring the OpenAlex step; skipped if no shared client is available so
        # tests stay hermetic and no impolite off-pool traffic is created.
        if self.openreview is None:
            shared_http = getattr(self.crossref, "http", None)
            if shared_http is not None:
                self.openreview = OpenReviewClient(http=shared_http)
        if self.openreview is not None:
            sources_queried.append("openreview")
            try:
                or_notes = self.openreview.search(query, limit=top_k, title=raw_title, first_author=first_author)
            except Exception as exc:
                or_notes = []
                errors.append(f"OpenReview: {exc}")
            or_records: list[PublishedRecord] = []
            for note in or_notes or []:
                rec = openreview_note_to_candidate_record(note)
                if rec:
                    or_records.append(rec)
            if or_records:
                sources_with_hits.append("openreview")
            _ingest("openreview", or_records)
            if self._has_full_confirmation(entry, all_candidates):
                return all_candidates

        # ----- Step 5: Semantic Scholar (preprint coverage; slowest w/o key) -----
        sources_queried.append("semanticscholar")
        try:
            # Title-led query keeps S2 relevance focused on the paper title.
            s2_query = f"{raw_title} {first_author}".strip() or query
            s2_data = self.s2.search(s2_query, limit=top_k)
        except Exception as exc:
            s2_data = []
            errors.append(f"Semantic Scholar: {exc}")
        s2_records: list[PublishedRecord] = []
        for item in s2_data or []:
            rec = s2_data_to_record(item)
            if rec:
                s2_records.append(rec)
        if s2_records:
            sources_with_hits.append("semanticscholar")
        _ingest("semanticscholar", s2_records)
        return all_candidates

    def _score_candidate(self, title_norm: str, authors_ref: list[str], rec: PublishedRecord) -> float:
        """Score a candidate record against the entry."""
        title_b = normalize_title_for_match(rec.title or "")
        title_score = token_sort_ratio(title_norm, title_b) / 100.0

        authors_b = rec.surname_keys(limit=3)
        author_score = jaccard_similarity(authors_ref, authors_b)

        return 0.7 * title_score + 0.3 * author_score

    def _has_full_confirmation(
        self, entry: dict[str, Any], all_candidates: list[tuple[float, PublishedRecord, str]]
    ) -> bool:
        """True when some high-confidence candidate positively confirms EVERY
        claimed field (title, author, year, venue) -- i.e. it would verdict
        VERIFIED.

        This is the cascade stop condition. The general principle: do not stop
        while a claimed field is still unconfirmed and a remaining source could
        resolve it. A DOI-less conference paper matches its arXiv preprint
        perfectly on title+author, but the preprint cannot confirm the claimed
        published venue -> not a full confirmation -> the cascade keeps going to
        DBLP/OpenReview (which carry the proceedings venue) instead of returning a
        could-not-verify the preprint forced. If no source ever fully confirms,
        the cascade exhausts its sources and returns normally.
        """
        threshold = self.config.cascade_high_confidence
        for score, rec, _src in all_candidates:
            if score < threshold:
                continue
            comparisons = self._compare_all_fields(entry, rec)
            if all(c.is_confirmed for c in comparisons.values()):
                return True
        return False

    #: Title+author score window: candidates within this margin of the top score
    #: are treated as equally-good title/author matches, and the tie is broken in
    #: favour of the one that positively confirms the most claimed fields. Wide
    #: enough to span a preprint vs its published-proceedings twin (identical
    #: title+author), narrow enough not to promote an unrelated lower-ranked paper.
    _SELECTION_SCORE_BAND = 0.05

    def _select_best_candidate(
        self, entry: dict[str, Any], candidates: list[tuple[float, PublishedRecord, str]]
    ) -> tuple[float, PublishedRecord, str]:
        """Pick the best candidate, preferring fuller positive confirmation.

        Primary signal stays the title+author score; among the candidates that
        tie at the top of that score (within ``_SELECTION_SCORE_BAND``), choose
        the one that confirms the MOST claimed fields (and fewest mismatches), so
        a published record that also confirms the venue/year is chosen over a
        preprint that cannot. This is the selection half of "resolve what can be
        resolved": reaching the proceedings record is useless unless it is then
        actually selected over the tied preprint.
        """
        ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
        top = ordered[0][0]
        band = [c for c in ordered if c[0] >= top - self._SELECTION_SCORE_BAND]
        if len(band) == 1:
            return band[0]

        def confirmation_key(cand: tuple[float, PublishedRecord, str]) -> tuple[int, int, float]:
            comparisons = self._compare_all_fields(entry, cand[1])
            confirmed = sum(1 for c in comparisons.values() if c.is_confirmed)
            mismatches = sum(1 for c in comparisons.values() if c.is_mismatch)
            return (confirmed, -mismatches, cand[0])

        return max(band, key=confirmation_key)

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

    @staticmethod
    def _entry_surname_keys(entry: dict[str, Any], record: PublishedRecord, limit: int = 10_000) -> list[str]:
        """Entry surname keys to compare against ``record``.

        When ``record`` has AUTHORITATIVE given/family names (Crossref), use the
        record's family set to disambiguate order-ambiguous comma-less entry
        names (family-first CJK names): ``entry_surnames_against_structured``
        picks the entry token that matches a known family. Otherwise the record
        cannot disambiguate anything, so use the plain ``authors_last_names``
        heuristic. Either way each entry author maps to one surname at its own
        position -- ordering/first-author checks downstream are untouched.
        """
        if record.structured_names:
            family_keys = set(record.surname_keys(limit=limit))
            if family_keys:
                return entry_surnames_against_structured(entry.get("author", ""), family_keys, limit=limit)
        return authors_last_names(entry.get("author", ""), limit=limit)

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

        # Author (symmetric comparison; see FIX C / symmetric_author_match).
        # Both sides go through the same canonical surname reduction
        # (last_name_from_person) with a generous limit, then the matcher slices
        # both sides symmetrically and applies ordered-containment. The legacy
        # code sliced the entry side to 10 but left the API side at 10_000, so a
        # correctly cited paper that lists fewer authors (or "and others") scored
        # below threshold purely from the length asymmetry.
        entry_authors = entry.get("author", "")
        entry_names = self._entry_surname_keys(entry, record, limit=10_000)
        api_names = record.surname_keys(limit=10_000)
        author_result = symmetric_author_match(
            entry_names, api_names, threshold=cfg.author_threshold, order_reliable=record.order_reliable
        )
        api_authors_str = " and ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in record.authors)
        # Mirror the venue "no claim" rule: if the entry lists no authors there is
        # nothing to confirm (vacuously MATCH). A PARTIAL (consistent-but-
        # incomplete) confirmation is NOT a full confirmation and routes to
        # UNCONFIRMED; a NON_COMPARABLE result with authors on the entry side but
        # none in the record also could-not-confirm.
        if not entry_names:
            author_outcome = MatchOutcome.MATCH
            author_confirmed = True
            author_note = "No authors claimed"
        else:
            author_outcome = author_result.outcome
            author_confirmed = author_result.is_confirmed
            if author_result.outcome is MatchOutcome.PARTIAL:
                author_note = "Authors consistent but incomplete"
            elif author_result.outcome is MatchOutcome.NON_COMPARABLE:
                author_note = "Authors could not be confirmed (no author data in record)"
            else:
                author_note = None
        comparisons["author"] = FieldComparison(
            "author",
            entry_authors,
            api_authors_str,
            author_result.score,
            # ``matches`` is positive-confirmation only: a PARTIAL/NON_COMPARABLE
            # author check is not a full confirmation, so it is not "matched".
            author_confirmed,
            note=author_note,
            outcome=author_outcome,
        )

        # Same-surname given-name swap. Surname-only matching is blind to a swap of
        # two co-authors who share a surname (e.g. 'Yang Song' <-> 'Jiaming Song' --
        # both reduce to 'song'). When the matched record preserves author order,
        # compare the given-name initials of each shared-surname run; a difference
        # is a real corruption that an otherwise-confirming author check missed.
        if (
            comparisons["author"].resolved_outcome in (MatchOutcome.MATCH, MatchOutcome.PARTIAL)
            and same_surname_given_order_violation(entry_authors, record)
        ):
            comparisons["author"].outcome = MatchOutcome.MISMATCH
            comparisons["author"].matches = False
            comparisons["author"].note = "Same-surname co-authors in a different given-name order (swapped authors)"

        # Year (three-valued, mirroring venue/author). An empty or unparseable
        # year on either side cannot confirm OR refute the claim -> NON_COMPARABLE,
        # not a mismatch (the old two-valued flag read a blank record year as a
        # YEAR_MISMATCH). An entry that claims no year has nothing to confirm
        # (vacuous MATCH). Only two populated, parseable years beyond tolerance are
        # a real MISMATCH. ``year_diff`` is kept for the different-edition guard.
        entry_year = entry.get("year", "")
        api_year = str(record.year) if record.year else ""
        year_diff: int | None = None
        if not entry_year:
            year_outcome = MatchOutcome.MATCH
            year_score = 1.0
            year_note = "No year claimed"
        elif not api_year:
            year_outcome = MatchOutcome.NON_COMPARABLE
            year_score = 1.0
            year_note = "Year could not be confirmed (no year in record)"
        else:
            try:
                year_diff = abs(int(entry_year) - int(api_year))
                year_outcome = MatchOutcome.MATCH if year_diff <= cfg.year_tolerance else MatchOutcome.MISMATCH
                year_score = 1.0 if year_outcome is MatchOutcome.MATCH else 0.0
                year_note = f"Tolerance: ±{cfg.year_tolerance}"
            except ValueError:
                year_outcome = MatchOutcome.NON_COMPARABLE
                year_score = 1.0
                year_note = "Year could not be compared (unparseable)"
        comparisons["year"] = FieldComparison(
            "year",
            entry_year,
            api_year,
            year_score,
            year_outcome is MatchOutcome.MATCH,
            year_note,
            outcome=year_outcome,
        )

        # Venue (alias-aware matching, three-valued).
        entry_venue = entry.get("journal") or entry.get("booktitle") or ""
        api_venue = record.journal or ""
        # A matched record that is itself a PREPRINT cannot confirm the published
        # venue the entry claims. Detect the preprint from the authoritative
        # identifier (arXiv/bioRxiv DOI) as well as the venue string, because APIs
        # sometimes return a junk repository name for a preprint's venue (e.g.
        # 'UvA-DARE (University of Amsterdam)' for an arXiv DataCite DOI) that the
        # string heuristic alone does not catch -> a false venue MISMATCH.
        record_is_preprint = _doi_is_preprint(record.doi) or is_preprint_or_series_venue(api_venue)
        # Positive-confirmation gate distinguishes "no claim" from "claim we
        # could not confirm":
        #  - The entry makes NO venue claim (preprint @misc/@article with no
        #    journal/booktitle) -> there is nothing to confirm, so the field is
        #    vacuously confirmed (MATCH) and must not block VERIFIED.
        #  - The entry CLAIMS a published venue but the matched record is a
        #    preprint or blank -> the claim cannot be confirmed -> NON_COMPARABLE,
        #    which routes the verdict to UNCONFIRMED (could not verify).
        if not entry_venue:
            venue_outcome = MatchOutcome.MATCH
            venue_score = 1.0
            venue_confirmed = True
            venue_note = "No venue claimed"
        elif record_is_preprint:
            venue_outcome = MatchOutcome.NON_COMPARABLE
            venue_score = 1.0
            venue_confirmed = False
            venue_note = "Claimed venue could not be confirmed (preprint record)"
        else:
            venue_result = venues_match(entry_venue, api_venue, cfg.venue_threshold)
            venue_outcome = venue_result.outcome
            venue_score = venue_result.score
            venue_confirmed = venue_result.is_confirmed
            venue_note = (
                "Claimed venue could not be confirmed (preprint/blank record)"
                if venue_result.outcome is MatchOutcome.NON_COMPARABLE
                else None
            )
        comparisons["venue"] = FieldComparison(
            "venue",
            entry_venue,
            api_venue,
            venue_score,
            venue_confirmed,
            note=venue_note,
            outcome=venue_outcome,
        )

        # Different-edition / reprint guard. A record with essentially the SAME
        # title (>= title_threshold) but published >= _EDITION_YEAR_GAP years away,
        # whose claimed venue is NOT positively confirmed, is almost certainly a
        # different edition/reprint of the same work -- or a same-title decoy from
        # free-text retrieval. It can neither confirm nor refute the entry's
        # published venue/year, so those fields abstain (NON_COMPARABLE) rather
        # than reading as positive evidence of a problem. A genuinely matching
        # venue keeps a year mismatch (a real contradiction in the same venue),
        # and the author check is untouched -- so a wrong author still flags.
        title_cmp = comparisons["title"]
        venue_cmp = comparisons["venue"]
        if (
            title_cmp.similarity_score >= cfg.title_threshold
            and year_diff is not None
            and year_diff >= _EDITION_YEAR_GAP
            and not venue_cmp.is_confirmed
        ):
            comparisons["year"].outcome = MatchOutcome.NON_COMPARABLE
            comparisons["year"].matches = False
            comparisons["year"].note = "Different edition/reprint (same title, different year/venue)"
            comparisons["venue"].outcome = MatchOutcome.NON_COMPARABLE
            comparisons["venue"].matches = False
            # A high-fuzzy near-miss title on a different edition ('Nets' vs
            # 'networks') is a stylistic variant of the same work, not a chimera
            # -> it must not read as a TITLE_MISMATCH. Chimeric/welded titles are
            # caught earlier by _detect_chimeric_title, before this point.
            if not title_cmp.is_confirmed:
                comparisons["title"].outcome = MatchOutcome.NON_COMPARABLE
                comparisons["title"].matches = False

        return comparisons

    def _determine_status(
        self,
        best_score: float,
        comparisons: dict[str, FieldComparison],
        sources_with_hits: list[str],
    ) -> FactCheckStatus:
        """Determine final status from score and field comparisons.

        VERIFIED means POSITIVE CONFIRMATION of every claimed field, not merely
        "nothing was contradicted". Each field comparison is three-valued
        (MATCH / MISMATCH / NON_COMPARABLE|PARTIAL), and the gate routes them:

        - Any MISMATCH (two different real venues, a swapped/wrong author, a
          different title/year) is positive evidence of a problem -> the usual
          PROBLEMATIC statuses (VENUE_MISMATCH / AUTHOR_MISMATCH / ... /
          PARTIAL_MATCH).
        - No MISMATCH but some field is NON_COMPARABLE or PARTIAL (preprint-only
          venue, incomplete author list) -> UNCONFIRMED: a record was found and
          nothing conflicts, but a claimed field could not be positively
          confirmed. This is abstention ("could not fully confirm / needs
          review"), distinct from both VERIFIED and PROBLEMATIC.
        - Every field CONFIRMED -> VERIFIED.

        Fix B (abstention): a weak best candidate means the title search returned
        an *unrelated* paper -- the tool simply could not find the real one. That
        is "I couldn't verify this", NOT "this is fabricated", so we ABSTAIN with
        NOT_FOUND rather than asserting HALLUCINATED. This sits AFTER the
        positive-evidence checks in ``check_entry`` (``_validate_year``,
        ``_validate_doi``, ``_check_arxiv_id_consistency``, ``_detect_chimeric_title``),
        each of which ``return``s before ``_determine_status`` is ever reached --
        so abstention can never suppress a true HALLUCINATED verdict backed by
        positive evidence.
        """
        # Wrong-paper signature: the best match is too weak to trust. Either the
        # blended score is below the abstention threshold, or the title itself is
        # essentially unrelated (very low title score) AND neither title nor
        # author corroborate the entry. In both cases there is no positive
        # evidence of fabrication, only a failed lookup -> abstain.
        title_cmp = comparisons.get("title")
        author_cmp = comparisons.get("author")
        title_score = title_cmp.similarity_score if title_cmp else 0.0
        title_confirmed = bool(title_cmp and title_cmp.is_confirmed)
        author_confirmed = bool(author_cmp and author_cmp.is_confirmed)
        wrong_paper_signature = title_score < 0.30 and not title_confirmed and not author_confirmed

        if best_score < self.config.abstention_below or wrong_paper_signature:
            return FactCheckStatus.NOT_FOUND

        # Positive evidence of a problem: a field that is a real MISMATCH (both
        # sides populated and conflicting). These take priority over abstention.
        mismatches = [name for name, c in comparisons.items() if c.is_mismatch]

        if mismatches:
            # P2.6: Prioritize venue mismatch when title+author are confirmed.
            if "venue" in mismatches and title_confirmed and author_confirmed:
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

        # No mismatch, but require POSITIVE confirmation of every claimed field.
        # A NON_COMPARABLE venue (preprint/blank record) or a PARTIAL author
        # confirmation (consistent-but-incomplete) is not a full confirmation:
        # the claim could not be verified, so abstain with UNCONFIRMED rather
        # than reporting VERIFIED.
        non_confirming = [name for name, c in comparisons.items() if c.is_non_confirming]
        if non_confirming:
            return FactCheckStatus.UNCONFIRMED

        return FactCheckStatus.VERIFIED


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
        academic_checker = FactChecker(crossref, dblp, s2, config, self.logger, arxiv=ArxivClient(http))
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
            """Check a single DOI via HEAD (with a GET fallback)."""
            # Normalize: strip URL prefix/lowercase, and for arXiv DataCite DOIs
            # drop a trailing version suffix (the versioned DOI 404s at doi.org
            # but the unversioned one resolves).
            normalized = normalize_doi_for_resolution(doi) or doi.strip()
            # None (network error) -> assume valid (don't penalize network issues).
            return (entry_id, _doi_resolves(client, normalized) is not False)

        results: dict[str, bool] = {}
        # Reuse shared httpx.Client if available (avoids TCP/TLS overhead)
        if isinstance(self.checker, FactChecker):
            client = self.checker.crossref.http.client
        else:
            client = httpx.Client(timeout=10.0, follow_redirects=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(_check_doi, eid, doi, client) for eid, doi in dois.items()]
                for future in concurrent.futures.as_completed(futures):
                    entry_id, is_valid = future.result()
                    results[entry_id] = is_valid
        finally:
            # Only close if we created the client ourselves
            if not isinstance(self.checker, FactChecker):
                client.close()

        return results

    def _batch_warm_crossref_records(self, entries: list[dict[str, Any]]) -> int:
        """Pre-fetch every entry DOI's Crossref ``/works`` record in parallel.

        Latency optimization (mirrors :meth:`_batch_validate_dois`). The per-entry
        DOI checks (``_check_doi_consistency`` and the structured-name author
        recheck) each call ``CrossrefClient.get_by_doi`` for the entry's DOI. Run
        serially across the worker pool, those round-trips are gated by the
        crossref rate limiter (50/min) and dominate wall-clock on large
        bibliographies.

        Here we issue the SAME ``get_by_doi`` calls once, up front, in a bounded
        thread pool. ``get_by_doi`` routes through the shared ``HttpClient``, whose
        responses are stored in the thread-safe SqliteCache keyed by request URL.
        Warming that cache turns the later per-entry ``get_by_doi`` calls into
        cache hits, so the same records are used for the same comparisons -- this
        is purely a change to WHEN the fetch happens, never WHICH record is used
        (verdict-neutral).

        Thread-safety: this runs BEFORE the main worker pool starts and writes
        only to the SqliteCache (already thread-safe). It adds no mutable state on
        ``self`` or the checker, so concurrent ``check_entry`` calls only ever READ
        the warmed cache.

        Fallback: a DOI whose pre-fetch fails (network error / non-200) is simply
        not cached. The per-entry ``get_by_doi`` then performs its own fetch
        exactly as before -- correct result, no speedup for that one DOI.

        Returns:
            Number of distinct DOIs warmed (best-effort; for logging/tests).
        """
        # Only FactChecker has a crossref client whose cache we can warm.
        if not isinstance(self.checker, FactChecker):
            return 0
        crossref = self.checker.crossref
        # No shared response cache -> warming would not be observed by the
        # per-entry calls, so skip (each call would re-fetch anyway).
        if getattr(crossref.http, "cache", None) is None:
            return 0

        # Dedupe DOIs across entries: identical DOIs share one cache key.
        dois: set[str] = set()
        for entry in entries:
            doi = (entry.get("doi", "") or "").strip()
            if doi:
                dois.add(doi)
        if not dois:
            return 0

        def _warm_one(doi: str) -> None:
            # Same call the per-entry path uses; result lands in the shared
            # cache. Swallow everything so one bad DOI never aborts warming.
            try:
                crossref.get_by_doi(doi)
            except Exception:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_warm_one, doi) for doi in dois]
            for future in concurrent.futures.as_completed(futures):
                # Results are written straight to the cache; nothing to collect.
                try:
                    future.result()
                except Exception:
                    pass

        return len(dois)

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

        # Latency: warm the Crossref /works response cache for all entry DOIs in
        # parallel up front, so the per-entry DOI checks (consistency + structured
        # author recheck) hit the cache instead of each making a serial,
        # rate-limited round-trip. Verdict-neutral; failures fall back to the
        # existing per-entry fetch. (Mirrors _batch_validate_dois.)
        warmed = self._batch_warm_crossref_records(entries)
        if warmed:
            self.logger.info("Pre-fetched Crossref records for %d DOIs", warmed)

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
                                # Fix B: distinct flag so abstentions ("could not
                                # verify") are unambiguously identifiable and never
                                # read as confirmed hallucinations downstream.
                                "abstained": _is_abstained_status(result.status),
                                "confidence": result.overall_confidence,
                                # Additive: 0-100 numeric confidence (Item 4).
                                "confidence_score": float(getattr(result, "confidence_score", 0.0)),
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

        # Three buckets (Fix B). ABSTAINED ("could not verify") is reported
        # separately from PROBLEMATIC ("positive evidence of a problem"): a
        # not-found / uncertain entry is the tool failing to locate a record, NOT
        # evidence of fabrication, so it must never read as a confirmed
        # hallucination.
        abstained_statuses = sorted(ABSTAINED_STATUS_VALUES)
        # Positive-evidence problems only. (not_found / *_not_found moved to the
        # abstained bucket above.)
        problematic_statuses = [
            "hallucinated",
            "title_mismatch",
            "author_mismatch",
            "year_mismatch",
            "venue_mismatch",
            # Multiple confirmed mismatches -> positive evidence of a problem.
            # (Previously omitted, so PARTIAL_MATCH entries fell through all three
            # buckets and the bucket counts under-summed.)
            "partial_match",
            "url_content_mismatch",
            "future_date",
            "invalid_year",
            "doi_not_found",
            "arxiv_id_mismatch",
            "doi_mismatch",
            "preprint_only",
        ]

        # Calculate verified rate including new verified statuses
        verified_statuses = ["verified", "url_verified", "url_accessible", "book_verified", "working_paper_verified"]
        verified_count = sum(counts.get(s, 0) for s in verified_statuses)
        abstained_count = sum(counts.get(s, 0) for s in abstained_statuses)

        return {
            "total": len(results),
            "status_counts": counts,
            "by_category": category_counts,
            "field_mismatch_counts": field_mismatches,
            "verified_rate": verified_count / len(results) if results else 0,
            "verified_count": verified_count,
            # Distinct "could not verify" bucket -- abstentions, not hallucinations.
            "could_not_verify_rate": abstained_count / len(results) if results else 0,
            "abstained_count": abstained_count,
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
                # Additive: 0-100 numeric confidence (Item 4).
                "confidence_score": float(getattr(r, "confidence_score", 0.0)),
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
                        # Fix B: distinct abstention flag (see process_entries).
                        "abstained": _is_abstained_status(r.status),
                        "confidence": r.overall_confidence,
                        # Additive: 0-100 numeric confidence (Item 4).
                        "confidence_score": float(getattr(r, "confidence_score", 0.0)),
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

    # CheckIfExist additions
    cascade_opts = p.add_argument_group("cascading source order (CheckIfExist)")
    cascade_opts.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"Top-K candidates per source (default: {DEFAULT_TOP_K}, max: {MAX_TOP_K}).",
    )
    cascade_opts.add_argument(
        "--openalex-mailto",
        default=DEFAULT_OPENALEX_MAILTO,
        metavar="EMAIL",
        help="OpenAlex polite-pool email (default: %(default)s).",
    )

    # Non-generative-AI mode (venue policy compliance)
    policy = p.add_argument_group("policy / compliance")
    policy.add_argument(
        "--non-generative",
        action="store_true",
        help=(
            "Run in non-generative-AI mode (no LLM calls). Compliant with "
            "ICML 2026 / ACL ARR LLM-in-review policies. Also settable via "
            "BIBTEX_CHECK_NON_GENERATIVE=1."
        ),
    )

    return p


def main() -> int:
    """Main entry point."""
    args = build_parser().parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    # Quiet the noisy per-request HTTP client logs (one INFO line per API call)
    # unless --verbose: they bury the tool's own progress and the final summary.
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = logging.getLogger("fact_checker")

    # Item 6: non-generative-AI mode (CLI flag wins; env var also honored).
    import os as _os

    env_flag = _os.environ.get("BIBTEX_CHECK_NON_GENERATIVE", "").strip() in {"1", "true", "yes", "on"}
    if args.non_generative or env_flag:
        set_non_generative_mode(True)
        sys.stderr.write(
            "bibtex-check running in non-generative mode (no LLM calls). "
            "Compliant with ICML 2026 / ACL ARR LLM-in-review policies.\n"
        )
        logger.info("non-generative-AI mode active")

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
            "openreview": max(10, int(30 * rate_scale)),
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
    top_k = max(1, min(int(args.top_k), MAX_TOP_K))
    config = FactCheckerConfig(
        title_threshold=args.title_threshold,
        author_threshold=args.author_threshold,
        year_tolerance=args.year_tolerance,
        venue_threshold=args.venue_threshold,
        check_dois=not args.no_check_dois,
        check_years=not args.no_check_years,
        top_k=top_k,
        openalex_mailto=args.openalex_mailto,
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

    # Three clearly distinct buckets (Fix B): VERIFIED, COULD NOT VERIFY
    # (abstained -- no matching record found, NOT evidence of fabrication), and
    # PROBLEMATIC (positive evidence of a problem). "Could not verify" must never
    # be lumped in with either "verified" or "hallucinated".
    total = summary["total"]
    verified_n = summary.get("verified_count", 0)
    abstained_n = summary.get("abstained_count", 0)
    problematic_n = summary.get("problematic_count", 0)
    logger.info("Results by bucket:")
    logger.info(
        "  (1) VERIFIED:            %d  (%.1f%%)",
        verified_n,
        summary["verified_rate"] * 100,
    )
    logger.info(
        "  (2) COULD NOT VERIFY:    %d  (%.1f%%)  -- no matching record found; not evidence of fabrication",
        abstained_n,
        summary["could_not_verify_rate"] * 100,
    )
    logger.info(
        "  (3) PROBLEMATIC:         %d  (%.1f%%)  -- positive evidence of a problem",
        problematic_n,
        (problematic_n / total * 100) if total else 0.0,
    )
    logger.info("Verified rate: %.1f%%", summary["verified_rate"] * 100)
    logger.info("Could-not-verify rate: %.1f%%", summary["could_not_verify_rate"] * 100)
    if problematic_n > 0:
        logger.warning("Problematic entries (positive evidence): %d", problematic_n)
    if abstained_n > 0:
        logger.info(
            "Could-not-verify entries (abstained, no matching record found): %d",
            abstained_n,
        )

    # Item 4: numeric confidence summary in the text report. ``confidence_score``
    # is the optional calibrated 0-100 score (populated in generative mode); when
    # it is absent/zero, fall back to the always-populated ``overall_confidence``
    # (0-1) scaled to 0-100 so the line reports the real confidence instead of 0.
    numeric_scores = [
        float(getattr(r, "confidence_score", 0.0)) or float(r.overall_confidence) * 100.0 for r in results
    ]
    if numeric_scores:
        mean_score = sum(numeric_scores) / len(numeric_scores)
        min_score = min(numeric_scores)
        max_score = max(numeric_scores)
        logger.info(
            "Numeric confidence (0-100): mean=%.1f, min=%.1f, max=%.1f",
            mean_score,
            min_score,
            max_score,
        )

    # Write reports
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(processor.generate_json_report(results), f, indent=2, ensure_ascii=False)
        logger.info("JSON report written to %s", args.report)

    if args.jsonl:
        logger.info("JSONL report streamed to %s (%d entries)", args.jsonl, len(results))

    # Exit code
    if args.strict:
        # Fix B: only POSITIVE-evidence problems gate the exit code. Abstentions
        # (could-not-verify) are reported on their own line and do NOT count as
        # confirmed hallucinations, so they no longer fail strict mode.
        problem_count = summary["problematic_count"]
        if abstained_n > 0:
            logger.info(
                "Strict mode: %d could-not-verify entries (abstained; not counted as failures)",
                abstained_n,
            )
        if problem_count > 0:
            logger.warning("Strict mode: %d PROBLEMATIC entries (positive evidence of a problem)", problem_count)
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
