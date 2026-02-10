"""Shared bibliographic utilities for BibTeX tools.

This module provides common functionality used by:
- bibtex_updater.py (preprint-to-published resolver)
- reference_fact_checker.py (bibliographic validation)

Includes text normalization, author parsing, DOI/arXiv handling,
HTTP infrastructure with caching and rate limiting, and API client utilities.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

# ------------- Constants & Regex -------------

ARXIV_ID_RE = re.compile(
    r"""
    (?:
        arxiv[:\s/]?   # prefix
    )?
    (?P<id>
        (?:\d{4}\.\d{4,5})(?:v\d+)?   # new style
        |
        (?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?  # old style e.g., cs/0301001
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

ARXIV_HOST_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>[^?/]+)", re.IGNORECASE)

PREPRINT_HOSTS = ("arxiv", "biorxiv", "medrxiv")

# API endpoints
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
DBLP_API_SEARCH = "https://dblp.org/search/publ/api"
S2_API = "https://api.semanticscholar.org/graph/v1"
ACL_ANTHOLOGY_URL = "https://aclanthology.org"
ACL_DOI_PREFIX = "10.18653/v1/"
ACL_ANTHOLOGY_ID_RE = re.compile(r"https?://aclanthology\.org/([A-Z0-9][\w.-]+?)(?:\.pdf|\.bib)?/?$", re.IGNORECASE)
OPENALEX_API = "https://api.openalex.org"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


# ------------- Text Normalization -------------


def safe_lower(x: str | None) -> str:
    """Null-safe lowercase and strip."""
    return (x or "").lower().strip()


def strip_diacritics(text: str) -> str:
    """Remove diacritics from text (e.g., 'café' -> 'cafe')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(\s*\[[^\]]*\])?(\s*\{[^}]*\})?")
_LATEX_MATH_RE = re.compile(r"\$[^$]*\$")
_BRACES_RE = re.compile(r"[{}]")


def latex_to_plain(text: str) -> str:
    """Remove LaTeX commands, math, and braces from text."""
    if not text:
        return ""
    t = _LATEX_MATH_RE.sub(" ", text)
    t = _LATEX_CMD_RE.sub(" ", t)
    t = _BRACES_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_title_for_match(title: str) -> str:
    """Normalize a title for fuzzy matching.

    Removes LaTeX, diacritics, punctuation, and extra whitespace.
    Converts to lowercase.
    """
    t = latex_to_plain(title)
    t = strip_diacritics(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------- Author Handling -------------


def split_authors_bibtex(author_field: str) -> list[str]:
    """Split BibTeX 'A and B and C' author string into individual names."""
    if not author_field:
        return []
    parts = [p.strip() for p in re.split(r"\s+\band\b\s+", author_field, flags=re.IGNORECASE) if p.strip()]
    return parts


def last_name_from_person(name: str) -> str:
    """Extract last name from a person name.

    Handles both 'Family, Given' and 'Given Family' formats.
    """
    name = latex_to_plain(name)
    if "," in name:
        last = name.split(",", 1)[0].strip()
    else:
        toks = name.split()
        last = toks[-1].strip() if toks else ""
    last = strip_diacritics(last).lower()
    last = re.sub(r"[^a-z0-9\s-]", "", last)
    return last


def authors_last_names(author_field: str, limit: int = 3) -> list[str]:
    """Extract last names from BibTeX author field.

    Args:
        author_field: BibTeX author string (e.g., 'Smith, John and Doe, Jane')
        limit: Maximum number of authors to extract

    Returns:
        List of normalized last names
    """
    authors = split_authors_bibtex(author_field)
    last_names = [last_name_from_person(a) for a in authors][:limit]
    return [ln for ln in last_names if ln]


def first_author_surname(entry: dict[str, Any]) -> str:
    """Get the first author's surname from a BibTeX entry."""
    return authors_last_names(entry.get("author", ""), limit=1)[0] if entry.get("author") else ""


# ------------- Matching Utilities -------------


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    """Compute Jaccard similarity between two iterables of strings."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ------------- DOI & arXiv Utilities -------------


def doi_normalize(doi: str | None) -> str | None:
    """Normalize a DOI by removing URL prefix and lowercasing."""
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    return d.lower()


def doi_url(doi: str) -> str:
    """Convert a DOI to a URL."""
    return f"https://doi.org/{doi}"


def extract_arxiv_id_from_text(text: str) -> str | None:
    """Extract arXiv ID from a text string (URL, eprint field, note, etc.)."""
    if not text:
        return None
    m = ARXIV_HOST_RE.search(text)
    if m:
        return m.group("id")
    m = ARXIV_ID_RE.search(text)
    if m:
        return m.group("id")
    return None


def extract_acl_anthology_id(doi_or_url: str) -> str | None:
    """Extract ACL Anthology ID from a DOI or URL.

    Handles:
    - DOIs like '10.18653/v1/2022.acl-long.220'
    - URLs like 'https://aclanthology.org/2022.acl-long.220'

    Returns:
        Anthology ID (e.g., '2022.acl-long.220') or None
    """
    if not doi_or_url:
        return None
    s = doi_or_url.strip()
    # Check DOI prefix
    doi_stripped = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    if doi_stripped.lower().startswith(ACL_DOI_PREFIX):
        anthology_id = doi_stripped[len(ACL_DOI_PREFIX) :]
        if anthology_id:
            return anthology_id
    # Check URL
    m = ACL_ANTHOLOGY_ID_RE.search(s)
    if m:
        return m.group(1)
    return None


def acl_anthology_bib_to_record(bib_text: str) -> PublishedRecord | None:
    """Parse BibTeX text from ACL Anthology into a PublishedRecord.

    Args:
        bib_text: Raw BibTeX string from aclanthology.org/{id}.bib

    Returns:
        PublishedRecord or None if parsing fails
    """
    if not bib_text or not bib_text.strip():
        return None

    # BibTeX field extractor that handles nested braces
    def _extract_field(field_name: str, text: str) -> str | None:
        # Find "field_name = " then extract the brace- or quote-delimited value
        pattern = rf"{field_name}\s*=\s*"
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        rest = text[m.end() :]
        if not rest:
            return None
        delim = rest[0]
        if delim == "{":
            # Count brace depth to find matching close
            depth = 0
            for i, ch in enumerate(rest):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        val = rest[1:i]  # strip outer braces
                        val = re.sub(r"\s+", " ", val).strip()
                        return val
            return None
        elif delim == '"':
            # Find matching closing quote (ignore escaped quotes)
            end = rest.find('"', 1)
            if end == -1:
                return None
            val = rest[1:end]
            val = re.sub(r"\s+", " ", val).strip()
            return val
        return None

    title = _extract_field("title", bib_text)
    booktitle = _extract_field("booktitle", bib_text)
    journal = _extract_field("journal", bib_text)
    doi = doi_normalize(_extract_field("doi", bib_text))
    url = _extract_field("url", bib_text)
    year_str = _extract_field("year", bib_text)
    pages = _extract_field("pages", bib_text)
    publisher = _extract_field("publisher", bib_text)
    volume = _extract_field("volume", bib_text)
    number = _extract_field("number", bib_text)
    author_str = _extract_field("author", bib_text)

    if not title:
        return None

    year: int | None = None
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            pass

    # Parse authors: "Last, First and Last2, First2"
    authors: list[dict[str, str]] = []
    if author_str:
        for part in re.split(r"\s+and\s+", author_str):
            part = part.strip()
            if not part:
                continue
            if "," in part:
                pieces = part.split(",", 1)
                authors.append({"given": pieces[1].strip(), "family": pieces[0].strip()})
            else:
                pieces = part.split()
                if len(pieces) >= 2:
                    authors.append({"given": " ".join(pieces[:-1]), "family": pieces[-1]})
                elif pieces:
                    authors.append({"given": "", "family": pieces[0]})

    # Determine venue: prefer booktitle (conference), fall back to journal
    venue = booktitle or journal

    # Determine type from entry type in BibTeX
    entry_type_match = re.match(r"@(\w+)\s*\{", bib_text.strip(), re.IGNORECASE)
    entry_type = entry_type_match.group(1).lower() if entry_type_match else ""
    if entry_type == "inproceedings" or booktitle:
        record_type = "proceedings-article"
    elif entry_type == "article":
        record_type = "journal-article"
    else:
        record_type = "proceedings-article"  # ACL Anthology is predominantly proceedings

    return PublishedRecord(
        doi=doi or "",
        url=url,
        title=title,
        authors=authors,
        journal=venue,
        publisher=publisher,
        year=year,
        volume=volume,
        number=number,
        pages=pages,
        type=record_type,
    )


# ------------- Rate Limiting & Caching -------------


class RateLimiter:
    """Thread-safe rate limiter for API requests."""

    def __init__(self, req_per_min: int) -> None:
        self.req_per_min = max(req_per_min, 1)
        self.lock = threading.Lock()
        self.timestamps: list[float] = []

    def wait(self) -> None:
        """Block until a request can be made within the rate limit."""
        with self.lock:
            now = time.time()
            window = 60.0
            self.timestamps = [t for t in self.timestamps if now - t < window]
            if len(self.timestamps) >= self.req_per_min:
                earliest = min(self.timestamps)
                sleep_for = window - (now - earliest) + 0.01
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    now = time.time()
                    self.timestamps = [t for t in self.timestamps if now - t < window]
            self.timestamps.append(time.time())


class RateLimiterRegistry:
    """Manages per-service rate limiters.

    This allows different API services to have their own rate limits,
    optimizing throughput while respecting each service's constraints.
    """

    DEFAULT_LIMITS = {
        "crossref": 50,  # Crossref: 50/min polite pool
        "semanticscholar": 100,  # S2: 100/min (1000 with API key)
        "dblp": 30,  # DBLP: 30/min (conservative)
        "arxiv": 30,  # arXiv: 30/min
        "scholarly": 10,  # Scholar: very conservative
        "aclanthology": 30,  # ACL Anthology: 30/min (conservative)
        "openalex": 100,  # OpenAlex: polite pool (~10 req/sec max)
        "europepmc": 20,  # Europe PMC: conservative rate limit
    }

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        """Initialize the registry with optional custom limits.

        Args:
            limits: Optional dict of service name to requests per minute.
                   Overrides DEFAULT_LIMITS for specified services.
        """
        self._limits = {**self.DEFAULT_LIMITS, **(limits or {})}
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def get(self, service: str) -> RateLimiter:
        """Get or create rate limiter for service.

        Args:
            service: Name of the API service (e.g., 'crossref', 'dblp')

        Returns:
            RateLimiter instance for the service
        """
        with self._lock:
            if service not in self._limiters:
                limit = self._limits.get(service, 30)  # Default 30/min
                self._limiters[service] = RateLimiter(limit)
            return self._limiters[service]

    def wait(self, service: str) -> None:
        """Wait for rate limit on specified service.

        Args:
            service: Name of the API service
        """
        self.get(service).wait()


class AdaptiveRateLimiterRegistry(RateLimiterRegistry):
    """Rate limiter registry that adapts based on API response headers.

    This extends RateLimiterRegistry to dynamically adjust rate limits based on
    feedback from API responses (e.g., X-RateLimit-Remaining headers or 429 responses).
    """

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        """Initialize the adaptive registry.

        Args:
            limits: Optional dict of service name to requests per minute.
                   Overrides DEFAULT_LIMITS for specified services.
        """
        super().__init__(limits)
        self._min_limits: dict[str, int] = {
            k: max(5, v // 4) for k, v in self._limits.items()  # Minimum is 25% of default
        }
        self._backoff_until: dict[str, float] = {}  # Service -> timestamp when backoff ends

    def adapt(self, service: str, response: Any) -> None:
        """Adjust rate limit based on API feedback.

        Call this after each API response to adapt the rate limit based on
        headers like X-RateLimit-Remaining, X-RateLimit-Limit, or 429 responses.

        Args:
            service: Name of the API service
            response: httpx.Response object from the API call
        """
        # Check for rate limit headers
        remaining = response.headers.get("X-RateLimit-Remaining")
        # Note: X-RateLimit-Limit and X-RateLimit-Reset are available but not currently used
        # They could be used for more sophisticated adaptive logic in the future

        if remaining is not None:
            try:
                remaining_int = int(remaining)
                if remaining_int < 10:
                    # Getting close to limit, slow down
                    current_limit = self._limits.get(service, 30)
                    new_limit = max(current_limit // 2, self._min_limits.get(service, 5))
                    if new_limit != current_limit:
                        self._limits[service] = new_limit
                        # Recreate limiter with new limit
                        with self._lock:
                            self._limiters[service] = RateLimiter(new_limit)
            except (ValueError, TypeError):
                pass

        # Handle 429 Too Many Requests
        if hasattr(response, "status_code") and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_seconds = int(retry_after)
                    self._backoff_until[service] = time.time() + retry_seconds
                except (ValueError, TypeError):
                    # Default 60 second backoff
                    self._backoff_until[service] = time.time() + 60.0
            else:
                # Default 60 second backoff
                self._backoff_until[service] = time.time() + 60.0

            # Also reduce the rate limit
            current_limit = self._limits.get(service, 30)
            new_limit = max(current_limit // 2, self._min_limits.get(service, 5))
            self._limits[service] = new_limit
            with self._lock:
                self._limiters[service] = RateLimiter(new_limit)

    def wait(self, service: str) -> None:
        """Wait for rate limit, including any backoff period.

        Args:
            service: Name of the API service
        """
        # Check for backoff period
        backoff_end = self._backoff_until.get(service, 0)
        now = time.time()
        if now < backoff_end:
            time.sleep(backoff_end - now)

        # Regular rate limiting
        super().wait(service)

    def reset_limit(self, service: str) -> None:
        """Reset a service's rate limit to its default value.

        Args:
            service: Name of the API service
        """
        default = self.DEFAULT_LIMITS.get(service, 30)
        self._limits[service] = default
        with self._lock:
            if service in self._limiters:
                self._limiters[service] = RateLimiter(default)


class DiskCache:
    """Thread-safe on-disk JSON cache for API responses."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key: str) -> Any | None:
        """Get a cached value by key."""
        if not self.path:
            return None
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a cached value."""
        if not self.path:
            return
        with self.lock:
            self.data[key] = value
            tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".json", prefix=".tmp_cache_")
            try:
                json.dump(self.data, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            finally:
                tmp.close()
            os.replace(tmp.name, self.path)


@dataclass
class ResolutionCacheEntry:
    """Cached resolution result for a preprint."""

    arxiv_id: str | None
    preprint_doi: str | None
    resolved_doi: str | None  # None = no published version found
    method: str | None  # Resolution method used (e.g., "arXiv->Semantic Scholar")
    confidence: float
    timestamp: float
    status: str  # "resolved" or "no_match"
    ttl_days: int = 30  # Re-check after 30 days


class ResolutionCache:
    """Semantic-level cache for preprint resolution results.

    This cache stores resolution results at a semantic level, caching whether
    a preprint has been resolved to a published version (and which one) or
    whether no match was found. This is separate from the HTTP-level DiskCache
    which caches raw API responses.

    Features:
    - Thread-safe implementation using locks
    - TTL-based expiration (default 30 days)
    - Atomic file writes using temp file + os.replace
    - Caches both positive results (resolved) and negative results (no_match)
    """

    def __init__(self, path: str | None, ttl_days: int = 30) -> None:
        """Initialize the resolution cache.

        Args:
            path: Path to the cache file. If None, caching is disabled.
            ttl_days: Time-to-live in days for cache entries. Default 30 days.
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
            "w", delete=False, encoding="utf-8", suffix=".json", prefix=".tmp_resolution_cache_"
        )
        try:
            json.dump(self.data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, self.path)

    def _make_key(self, arxiv_id: str | None, doi: str | None) -> str:
        """Create cache key from identifiers."""
        return f"arxiv:{arxiv_id or ''}_doi:{doi or ''}"

    def get(self, arxiv_id: str | None, doi: str | None) -> ResolutionCacheEntry | None:
        """Get cached resolution result if valid.

        Args:
            arxiv_id: The arXiv ID of the preprint (if available)
            doi: The DOI of the preprint (if available)

        Returns:
            ResolutionCacheEntry if a valid cached result exists, None otherwise.
        """
        if not self.path:
            return None
        key = self._make_key(arxiv_id, doi)
        with self.lock:
            if key not in self.data:
                return None
            entry_data = self.data[key]
            # Check TTL
            age_days = (time.time() - entry_data["timestamp"]) / 86400
            if age_days > entry_data.get("ttl_days", self.ttl_days):
                return None  # Expired
            return ResolutionCacheEntry(**entry_data)

    def set(self, entry: ResolutionCacheEntry) -> None:
        """Store resolution result.

        Args:
            entry: The ResolutionCacheEntry to store.
        """
        if not self.path:
            return
        key = self._make_key(entry.arxiv_id, entry.preprint_doi)
        with self.lock:
            # Convert dataclass to dict for JSON serialization
            self.data[key] = {
                "arxiv_id": entry.arxiv_id,
                "preprint_doi": entry.preprint_doi,
                "resolved_doi": entry.resolved_doi,
                "method": entry.method,
                "confidence": entry.confidence,
                "timestamp": entry.timestamp,
                "status": entry.status,
                "ttl_days": entry.ttl_days,
            }
            self._save()

    def set_no_match(self, arxiv_id: str | None, doi: str | None) -> None:
        """Cache a negative result (no published version found).

        Args:
            arxiv_id: The arXiv ID of the preprint (if available)
            doi: The DOI of the preprint (if available)
        """
        entry = ResolutionCacheEntry(
            arxiv_id=arxiv_id,
            preprint_doi=doi,
            resolved_doi=None,
            method=None,
            confidence=0.0,
            timestamp=time.time(),
            status="no_match",
            ttl_days=self.ttl_days,
        )
        self.set(entry)


# ------------- HTTP Client -------------


class HttpClient:
    """HTTP client with caching, rate limiting, and retry logic."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        timeout: float,
        user_agent: str,
        rate_limiter: RateLimiter | RateLimiterRegistry,
        cache: DiskCache,
        verbose: bool = False,
        s2_api_key: str | None = None,
    ):
        """Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds
            user_agent: User-Agent header value
            rate_limiter: Either a single RateLimiter (for backward compatibility)
                         or a RateLimiterRegistry for per-service rate limiting
            cache: DiskCache instance for caching responses
            verbose: Enable verbose logging
            s2_api_key: Optional Semantic Scholar API key for authenticated requests
        """
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self._rate_limiter = rate_limiter
        self._uses_registry = isinstance(rate_limiter, RateLimiterRegistry)
        self.cache = cache
        self.verbose = verbose
        self.s2_api_key = s2_api_key

    @property
    def rate_limiter(self) -> RateLimiter | RateLimiterRegistry:
        """Return the rate limiter for backward compatibility."""
        return self._rate_limiter

    def _get_limiter_for_service(self, service: str | None) -> RateLimiter:
        """Get the appropriate rate limiter for a service.

        Args:
            service: Service name (e.g., 'crossref', 'dblp'). If None or
                    if using a single RateLimiter, returns the default limiter.

        Returns:
            RateLimiter instance to use for rate limiting
        """
        if self._uses_registry and service:
            return self._rate_limiter.get(service)  # type: ignore[union-attr]
        elif self._uses_registry:
            # Default to crossref if no service specified with registry
            return self._rate_limiter.get("crossref")  # type: ignore[union-attr]
        else:
            return self._rate_limiter  # type: ignore[return-value]

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
        service: str | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with caching and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: Query parameters
            accept: Accept header value
            json_body: JSON body for POST requests
            service: Optional service name for per-service rate limiting
                    (e.g., 'crossref', 'dblp', 'semanticscholar', 'arxiv')
        """
        cache_key = None
        if self.cache:
            cache_key = json.dumps({"m": method, "u": url, "p": params, "a": accept, "j": json_body}, sort_keys=True)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return httpx.Response(
                    200,
                    content=json.dumps(cached).encode("utf-8"),
                    headers={"X-From-Cache": "1"},
                )
        backoff = 1.0
        limiter = self._get_limiter_for_service(service)
        for _ in range(6):
            limiter.wait()
            try:
                headers = {"Accept": accept} if accept else {}
                if json_body is not None:
                    headers["Content-Type"] = "application/json"
                # Add Semantic Scholar API key if available
                if service == "semanticscholar" and self.s2_api_key:
                    headers["x-api-key"] = self.s2_api_key
                resp = self.client.request(method, url, params=params, headers=headers, json=json_body)
                if resp.status_code in self.RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError("Retryable status", request=resp.request, response=resp)
                if self.cache and cache_key and resp.headers.get("Content-Type", "").startswith("application/json"):
                    try:
                        self.cache.set(cache_key, resp.json())
                    except Exception:
                        pass
                return resp
            except httpx.HTTPError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 16.0)
        raise RuntimeError(f"Network failure after retries for {url}")


# ------------- Data Classes -------------


@dataclass
class PublishedRecord:
    """A published bibliographic record from an API source."""

    doi: str
    url: str | None = None
    title: str | None = None
    authors: list[dict[str, str]] = field(default_factory=list)  # [{"given":..., "family":...}]
    journal: str | None = None
    publisher: str | None = None
    year: int | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    type: str | None = None
    method: str | None = None  # how found
    confidence: float = 0.0


# ------------- API Response Converters -------------


def crossref_message_to_record(msg: dict[str, Any]) -> PublishedRecord | None:
    """Convert a Crossref works message to a PublishedRecord."""
    typ = msg.get("type")
    doi = doi_normalize(msg.get("DOI"))
    if not doi:
        return None
    url = msg.get("URL")
    titles = msg.get("title") or []
    title = titles[0] if titles else None
    if title:
        title = re.sub(r"<[^>]*>", "", title)  # strip HTML tags

    # Authors - handle given/family and literal formats
    authors = []
    for a in msg.get("author", []) or []:
        given = a.get("given") or ""
        family = a.get("family") or ""
        # Handle literal name format when given/family are missing
        if not (given or family) and a.get("literal"):
            lit = a["literal"]
            parts = lit.split()
            if parts:
                family = parts[-1]
                given = " ".join(parts[:-1])
        authors.append({"given": given, "family": family})

    # Container/journal
    container = msg.get("container-title", [])
    journal = container[0] if container else None

    # Publication date - check multiple date fields
    pubyear = None
    for dt_key in ("published-print", "published-online", "created", "issued", "published"):
        if msg.get(dt_key, {}).get("date-parts"):
            y = msg[dt_key]["date-parts"][0][0]
            if y:
                pubyear = int(y)
                break

    # Issue can be in different locations
    issue = msg.get("issue") or msg.get("journal-issue", {}).get("issue")

    return PublishedRecord(
        doi=doi,
        url=url,
        title=title,
        authors=authors,
        journal=journal,
        publisher=msg.get("publisher"),
        year=pubyear,
        volume=msg.get("volume"),
        number=issue,
        pages=msg.get("page"),
        type=typ,
    )


def dblp_hit_to_record(hit: dict[str, Any]) -> PublishedRecord | None:
    """Convert a DBLP hit to a PublishedRecord."""
    info = hit.get("info", {})
    title = info.get("title") or ""
    title = re.sub(r"<[^>]*>", "", title)  # strip HTML tags

    # Parse authors
    authors_field = info.get("authors", {}).get("author")
    authors_list: list[str] = []
    if isinstance(authors_field, list):
        for a in authors_field:
            if isinstance(a, dict):
                authors_list.append(a.get("text") or a.get("name") or "")
            elif isinstance(a, str):
                authors_list.append(a)
    elif isinstance(authors_field, dict):
        authors_list.append(authors_field.get("text") or authors_field.get("name") or "")

    authors = []
    for full in authors_list:
        full = full.strip()
        if not full:
            continue
        parts = full.split()
        if len(parts) >= 2:
            authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
        else:
            authors.append({"given": "", "family": parts[0]})

    venue = info.get("journal") or info.get("venue")
    year = None
    try:
        year = int(info.get("year")) if info.get("year") else None
    except Exception:
        pass

    doi = doi_normalize(info.get("doi"))
    ee = info.get("ee")
    volume = info.get("volume")
    number = info.get("number")
    pages = info.get("pages")
    typ = safe_lower(info.get("type"))

    # Reject preprint venues (CoRR is arXiv's journal name in DBLP)
    venue_lower = safe_lower(venue) if venue else ""
    if re.search(r"arxiv|biorxiv|medrxiv|^corr$", venue_lower):
        return None
    if not venue or not year:
        return None
    # Accept if DOI present or at least URL present
    if not (doi or ee):
        return None

    # Determine record type: journal-article or proceedings-article
    is_conference = (
        "conference" in typ
        or "proceedings" in typ
        or re.search(r"proceedings|conference|workshop|symposium", venue_lower)
    )
    record_type = "proceedings-article" if is_conference else "journal-article"

    return PublishedRecord(
        doi=doi or "",
        url=ee,
        title=title or None,
        authors=authors,
        journal=venue,
        year=year,
        volume=volume,
        number=number,
        pages=pages,
        type=record_type,
    )


def s2_data_to_record(data: dict[str, Any]) -> PublishedRecord | None:
    """Convert Semantic Scholar data to a PublishedRecord."""
    doi = doi_normalize(data.get("doi") or (data.get("externalIds") or {}).get("DOI"))

    # Parse authors
    authors = []
    for a in data.get("authors") or []:
        name = a.get("name") or ""
        parts = name.split()
        if len(parts) >= 2:
            authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
        elif parts:
            authors.append({"given": "", "family": parts[0]})

    # Get venue
    venue = (data.get("publicationVenue") or {}).get("name") or data.get("venue")

    # Reject arXiv/preprint venues
    venue_lower = safe_lower(venue) if venue else ""
    if re.search(r"arxiv|biorxiv|medrxiv", venue_lower):
        return None

    pub_types = data.get("publicationTypes") or []
    s2_type = pub_types[0].lower() if pub_types else ""

    # Map Semantic Scholar types to standard types
    # S2 uses: JournalArticle, Conference, Review, Book, BookSection, etc.
    s2_type_map = {
        "journalarticle": "journal-article",
        "conference": "proceedings-article",
        "book": "book",
        "booksection": "book-chapter",
        "review": "journal-article",
    }
    record_type = s2_type_map.get(s2_type, s2_type)

    return PublishedRecord(
        doi=doi or "",
        url=data.get("url"),
        title=data.get("title"),
        authors=authors,
        journal=venue,
        year=data.get("year"),
        type=record_type,
    )


def openalex_work_to_record(work: dict[str, Any]) -> PublishedRecord | None:
    """Convert an OpenAlex work to a PublishedRecord.

    OpenAlex tracks preprint-to-published version relationships. This function
    extracts metadata from the published version of a work.

    Args:
        work: OpenAlex work object from API response

    Returns:
        PublishedRecord if the work is a published version, None otherwise
    """
    if not work:
        return None

    # Extract DOI (OpenAlex DOIs come as full URLs)
    doi = doi_normalize(work.get("doi"))
    if not doi:
        return None

    title = work.get("title")
    year = work.get("publication_year")
    work_type = work.get("type")

    # Map OpenAlex type to our types
    type_map = {
        "article": "journal-article",
        "proceedings-article": "proceedings-article",
        "book-chapter": "book-chapter",
    }
    record_type = type_map.get(work_type, work_type)

    # Reject posted-content (preprints)
    if work_type == "posted-content":
        return None

    # Parse authors from authorships
    authors: list[dict[str, str]] = []
    for authorship in work.get("authorships", []):
        author_obj = authorship.get("author", {})
        display_name = author_obj.get("display_name", "")
        if display_name:
            parts = display_name.split()
            if len(parts) >= 2:
                authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
            elif parts:
                authors.append({"given": "", "family": parts[0]})

    # Get journal/venue from primary_location
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    journal = source.get("display_name")
    version = primary_location.get("version")

    # Check all locations for a published version
    has_published_version = version == "publishedVersion"
    for loc in work.get("locations", []):
        if loc.get("version") == "publishedVersion":
            has_published_version = True
            if not journal:
                loc_source = loc.get("source") or {}
                journal = loc_source.get("display_name")
            break

    # Reject if journal name contains preprint indicators
    if journal:
        journal_lower = safe_lower(journal)
        if re.search(r"arxiv|biorxiv|medrxiv|preprint", journal_lower):
            return None
    elif record_type in ("journal-article", "proceedings-article"):
        return None  # No journal name for what should be a published work

    # Only accept if we have evidence of a published version
    if record_type not in ("journal-article", "proceedings-article", "book-chapter"):
        return None
    if not has_published_version and record_type in ("journal-article", "proceedings-article"):
        # For articles without explicit publishedVersion, require a real journal
        if not journal:
            return None

    # Extract bibliographic info
    biblio = work.get("biblio") or {}
    volume = biblio.get("volume")
    issue = biblio.get("issue")
    first_page = biblio.get("first_page")
    last_page = biblio.get("last_page")

    pages = None
    if first_page:
        if last_page and last_page != first_page:
            pages = f"{first_page}-{last_page}"
        else:
            pages = str(first_page)

    publisher = source.get("host_organization_name")

    return PublishedRecord(
        doi=doi,
        url=doi_url(doi) if doi else None,
        title=title,
        authors=authors,
        journal=journal,
        publisher=publisher,
        year=year,
        volume=volume,
        number=issue,
        pages=pages,
        type=record_type,
    )


def europepmc_result_to_record(result: dict[str, Any]) -> PublishedRecord | None:
    """Convert a Europe PMC search result to a PublishedRecord.

    Europe PMC is particularly useful for bioRxiv/medRxiv preprint resolution
    as it maintains active preprint-to-published linking for life sciences.

    Args:
        result: Europe PMC result object from search API

    Returns:
        PublishedRecord if the result is a published article, None otherwise
    """
    if not result:
        return None

    doi = doi_normalize(result.get("doi"))
    title = result.get("title")
    if not title:
        return None

    # Check source - we want published articles (MED=PubMed, PMC=PubMed Central)
    source = result.get("source", "")
    if source not in ("MED", "PMC", "AGR", "CBA", "CTX", "ETH", "HIR", "PAT"):
        # Skip preprint sources
        if source == "PPR":
            return None

    # Parse authors from authorString: "Smith J, Doe J, ..."
    authors: list[dict[str, str]] = []
    author_str = result.get("authorString", "")
    if author_str:
        # Remove trailing period if present
        author_str = author_str.rstrip(".")
        for author_part in author_str.split(","):
            author_part = author_part.strip()
            if not author_part:
                continue
            parts = author_part.split()
            if len(parts) >= 2:
                # Last token(s) are initials, first token(s) are family name
                # Europe PMC format: "FamilyName Initials" e.g., "Smith JA"
                # Check if last parts are initials (all uppercase, short)
                family_parts = []
                given_parts = []
                for p in parts:
                    if len(p) <= 3 and p.isupper():
                        given_parts.append(p)
                    else:
                        family_parts.append(p)
                family = " ".join(family_parts) if family_parts else parts[0]
                given = " ".join(given_parts) if given_parts else parts[-1]
                authors.append({"given": given, "family": family})
            elif parts:
                authors.append({"given": "", "family": parts[0]})

    journal = result.get("journalTitle")
    year_str = result.get("pubYear")
    year = int(year_str) if year_str and year_str.isdigit() else None
    volume = result.get("journalVolume")
    issue = result.get("issue")
    page_info = result.get("pageInfo")

    # Reject if journal looks like a preprint server
    if journal:
        journal_lower = safe_lower(journal)
        if re.search(r"arxiv|biorxiv|medrxiv|preprint", journal_lower):
            return None

    return PublishedRecord(
        doi=doi or "",
        url=doi_url(doi) if doi else None,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        volume=volume,
        number=issue,
        pages=page_info,
        type="journal-article",
    )


# ------------- Async Rate Limiting -------------


class AsyncRateLimiter:
    """Async-compatible rate limiter using sliding window.

    This rate limiter uses an asyncio lock and sleep for non-blocking
    rate limiting in async contexts. It maintains a sliding window of
    timestamps to enforce the rate limit.
    """

    def __init__(self, req_per_min: int) -> None:
        """Initialize the async rate limiter.

        Args:
            req_per_min: Maximum number of requests allowed per minute.
                        Minimum value is 1.
        """
        import asyncio

        self.req_per_min = max(req_per_min, 1)
        self.lock = asyncio.Lock()
        self.timestamps: list[float] = []

    async def wait(self) -> None:
        """Async wait until a request can be made within the rate limit.

        This method will sleep asynchronously if the rate limit has been
        exceeded, allowing other coroutines to run while waiting.
        """
        import asyncio

        async with self.lock:
            now = time.time()
            window = 60.0
            self.timestamps = [t for t in self.timestamps if now - t < window]

            if len(self.timestamps) >= self.req_per_min:
                earliest = min(self.timestamps)
                sleep_for = window - (now - earliest) + 0.01
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                    now = time.time()
                    self.timestamps = [t for t in self.timestamps if now - t < window]

            self.timestamps.append(now)


class AsyncRateLimiterRegistry:
    """Manages per-service async rate limiters.

    This registry creates and manages AsyncRateLimiter instances for
    different API services, allowing each service to have its own
    rate limit configuration.
    """

    DEFAULT_LIMITS = {
        "crossref": 50,  # Crossref: 50/min polite pool
        "semanticscholar": 100,  # S2: 100/min (1000 with API key)
        "dblp": 30,  # DBLP: 30/min (conservative)
        "arxiv": 30,  # arXiv: 30/min
        "scholarly": 10,  # Scholar: very conservative
        "aclanthology": 30,  # ACL Anthology: 30/min (conservative)
        "openalex": 100,  # OpenAlex: polite pool (~10 req/sec max)
        "europepmc": 20,  # Europe PMC: conservative rate limit
    }

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        """Initialize the registry with optional custom limits.

        Args:
            limits: Optional dict of service name to requests per minute.
                   Overrides DEFAULT_LIMITS for specified services.
        """
        self._limits = {**self.DEFAULT_LIMITS, **(limits or {})}
        self._limiters: dict[str, AsyncRateLimiter] = {}

    def get(self, service: str) -> AsyncRateLimiter:
        """Get or create async rate limiter for service.

        Args:
            service: Name of the API service (e.g., 'crossref', 'dblp')

        Returns:
            AsyncRateLimiter instance for the service
        """
        if service not in self._limiters:
            limit = self._limits.get(service, 30)  # Default 30/min
            self._limiters[service] = AsyncRateLimiter(limit)
        return self._limiters[service]

    async def wait(self, service: str) -> None:
        """Async wait for rate limit on specified service.

        Args:
            service: Name of the API service
        """
        await self.get(service).wait()


# ------------- Async HTTP Client -------------


class AsyncHttpClient:
    """Async HTTP client with rate limiting, caching, and retry logic.

    This client provides async HTTP requests with:
    - Per-service rate limiting via AsyncRateLimiterRegistry
    - Response caching via DiskCache
    - Automatic retry with exponential backoff for transient failures
    """

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        rate_limiters: AsyncRateLimiterRegistry,
        cache: DiskCache | None = None,
        timeout: float = 20.0,
        user_agent: str = "bibtex-updater/1.0 (async)",
    ) -> None:
        """Initialize the async HTTP client.

        Args:
            rate_limiters: AsyncRateLimiterRegistry for per-service rate limiting
            cache: Optional DiskCache for caching responses
            timeout: Request timeout in seconds
            user_agent: User-Agent header value
        """
        self.rate_limiters = rate_limiters
        self.cache = cache
        self.timeout = timeout
        self.user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client instance."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._client

    def _mock_response(self, data: Any) -> httpx.Response:
        """Create a mock response from cached data.

        Args:
            data: Cached JSON data to wrap in a response

        Returns:
            httpx.Response with the cached data
        """
        return httpx.Response(
            200,
            content=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-From-Cache": "1"},
        )

    async def request(
        self,
        method: str,
        url: str,
        service: str = "default",
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
        headers: dict[str, str] | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        """Make async HTTP request with rate limiting and retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            service: Service name for rate limiting (e.g., 'crossref', 'dblp')
            params: Query parameters
            json_body: JSON body for POST requests
            headers: Additional headers
            accept: Accept header value

        Returns:
            httpx.Response object

        Raises:
            RuntimeError: If request fails after all retry attempts
        """
        import asyncio

        # Check cache first for GET requests
        cache_key = None
        if method == "GET" and self.cache:
            cache_key = json.dumps(
                {"m": method, "u": url, "p": params, "a": accept},
                sort_keys=True,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._mock_response(cached)

        request_headers = {**(headers or {}), "Accept": accept}
        if json_body is not None:
            request_headers["Content-Type"] = "application/json"

        backoff = 1.0

        for attempt in range(6):
            await self.rate_limiters.wait(service)
            try:
                resp = await self.client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=request_headers,
                )
                if resp.status_code in self.RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"Status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                # Cache successful JSON responses for GET requests
                if (
                    method == "GET"
                    and self.cache
                    and cache_key
                    and "application/json" in resp.headers.get("content-type", "")
                ):
                    try:
                        self.cache.set(cache_key, resp.json())
                    except Exception:
                        pass  # Ignore cache errors

                return resp
            except httpx.HTTPError:
                if attempt < 5:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 16.0)

        raise RuntimeError(f"Network failure after retries for {url}")

    async def get(
        self,
        url: str,
        service: str = "default",
        params: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        """Convenience method for GET requests.

        Args:
            url: Request URL
            service: Service name for rate limiting
            params: Query parameters
            accept: Accept header value

        Returns:
            httpx.Response object
        """
        return await self.request("GET", url, service=service, params=params, accept=accept)

    async def post(
        self,
        url: str,
        service: str = "default",
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
        accept: str = "application/json",
    ) -> httpx.Response:
        """Convenience method for POST requests.

        Args:
            url: Request URL
            service: Service name for rate limiting
            params: Query parameters
            json_body: JSON body
            accept: Accept header value

        Returns:
            httpx.Response object
        """
        return await self.request("POST", url, service=service, params=params, json_body=json_body, accept=accept)

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> AsyncHttpClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        await self.close()
