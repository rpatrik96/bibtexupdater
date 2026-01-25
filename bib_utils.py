"""Shared bibliographic utilities for BibTeX tools.

This module provides common functionality used by:
- bibtex_updater.py (preprint-to-published resolver)
- reference_fact_checker.py (bibliographic validation)

Includes text normalization, author parsing, DOI/arXiv handling,
HTTP infrastructure with caching and rate limiting, and API client utilities.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

ARXIV_HOST_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>[^?/]+)", re.IGNORECASE
)

PREPRINT_HOSTS = ("arxiv", "biorxiv", "medrxiv")

# API endpoints
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
DBLP_API_SEARCH = "https://dblp.org/search/publ/api"
S2_API = "https://api.semanticscholar.org/graph/v1"


# ------------- Text Normalization -------------


def safe_lower(x: Optional[str]) -> str:
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


def split_authors_bibtex(author_field: str) -> List[str]:
    """Split BibTeX 'A and B and C' author string into individual names."""
    if not author_field:
        return []
    parts = [
        p.strip()
        for p in re.split(r"\s+\band\b\s+", author_field, flags=re.IGNORECASE)
        if p.strip()
    ]
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


def authors_last_names(author_field: str, limit: int = 3) -> List[str]:
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


def first_author_surname(entry: Dict[str, Any]) -> str:
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


def doi_normalize(doi: Optional[str]) -> Optional[str]:
    """Normalize a DOI by removing URL prefix and lowercasing."""
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    return d.lower()


def doi_url(doi: str) -> str:
    """Convert a DOI to a URL."""
    return f"https://doi.org/{doi}"


def extract_arxiv_id_from_text(text: str) -> Optional[str]:
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


# ------------- Rate Limiting & Caching -------------


class RateLimiter:
    """Thread-safe rate limiter for API requests."""

    def __init__(self, req_per_min: int) -> None:
        self.req_per_min = max(req_per_min, 1)
        self.lock = threading.Lock()
        self.timestamps: List[float] = []

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


class DiskCache:
    """Thread-safe on-disk JSON cache for API responses."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key: str) -> Optional[Any]:
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
            tmp = tempfile.NamedTemporaryFile(
                "w", delete=False, encoding="utf-8", suffix=".json", prefix=".tmp_cache_"
            )
            try:
                json.dump(self.data, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
            finally:
                tmp.close()
            os.replace(tmp.name, self.path)


# ------------- HTTP Client -------------


class HttpClient:
    """HTTP client with caching, rate limiting, and retry logic."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        timeout: float,
        user_agent: str,
        rate_limiter: RateLimiter,
        cache: DiskCache,
        verbose: bool = False,
    ):
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.verbose = verbose

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        accept: Optional[str] = None,
    ) -> httpx.Response:
        """Make an HTTP request with caching and retries."""
        cache_key = None
        if self.cache:
            cache_key = json.dumps(
                {"m": method, "u": url, "p": params, "a": accept}, sort_keys=True
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return httpx.Response(
                    200,
                    content=json.dumps(cached).encode("utf-8"),
                    headers={"X-From-Cache": "1"},
                )
        backoff = 1.0
        for _ in range(6):
            self.rate_limiter.wait()
            try:
                headers = {"Accept": accept} if accept else {}
                resp = self.client.request(method, url, params=params, headers=headers)
                if resp.status_code in self.RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        "Retryable status", request=resp.request, response=resp
                    )
                if (
                    self.cache
                    and cache_key
                    and resp.headers.get("Content-Type", "").startswith("application/json")
                ):
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
    url: Optional[str] = None
    title: Optional[str] = None
    authors: List[Dict[str, str]] = field(default_factory=list)  # [{"given":..., "family":...}]
    journal: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    pages: Optional[str] = None
    type: Optional[str] = None
    method: Optional[str] = None  # how found
    confidence: float = 0.0


# ------------- API Response Converters -------------


def crossref_message_to_record(msg: Dict[str, Any]) -> Optional[PublishedRecord]:
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


def dblp_hit_to_record(hit: Dict[str, Any]) -> Optional[PublishedRecord]:
    """Convert a DBLP hit to a PublishedRecord."""
    info = hit.get("info", {})
    title = info.get("title") or ""
    title = re.sub(r"<[^>]*>", "", title)  # strip HTML tags

    # Parse authors
    authors_field = info.get("authors", {}).get("author")
    authors_list: List[str] = []
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

    # Consider only journal-like entries
    is_journal = ("journal" in typ) or (
        venue and not re.search(r"proceedings|conference|arxiv|biorxiv|medrxiv", safe_lower(venue))
    )
    if not is_journal:
        return None
    if not venue or not year:
        return None
    # Accept if DOI present or at least URL present
    if not (doi or ee):
        return None

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
        type="journal-article",
    )


def s2_data_to_record(data: Dict[str, Any]) -> Optional[PublishedRecord]:
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

    pub_types = data.get("publicationTypes") or []

    return PublishedRecord(
        doi=doi or "",
        url=data.get("url"),
        title=data.get("title"),
        authors=authors,
        journal=venue,
        year=data.get("year"),
        type=pub_types[0].lower() if pub_types else None,
    )
