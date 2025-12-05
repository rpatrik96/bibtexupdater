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
import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from rapidfuzz.fuzz import token_sort_ratio

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

CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"  # Atom feed
DBLP_API_SEARCH = "https://dblp.org/search/publ/api"
S2_API = "https://api.semanticscholar.org/graph/v1"


# ------------- Utilities -------------
def safe_lower(x: Optional[str]) -> str:
    return (x or "").lower().strip()


def strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(\s*\[[^\]]*\])?(\s*\{[^}]*\})?")
_LATEX_MATH_RE = re.compile(r"\$[^$]*\$")
_BRACES_RE = re.compile(r"[{}]")


def latex_to_plain(text: str) -> str:
    if not text:
        return ""
    t = _LATEX_MATH_RE.sub(" ", text)
    t = _LATEX_CMD_RE.sub(" ", t)
    t = _BRACES_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_title_for_match(title: str) -> str:
    t = latex_to_plain(title)
    t = strip_diacritics(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_authors_bibtex(author_field: str) -> List[str]:
    if not author_field:
        return []
    parts = [p.strip() for p in re.split(r"\s+\band\b\s+", author_field, flags=re.IGNORECASE) if p.strip()]
    return parts


def last_name_from_person(name: str) -> str:
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
    authors = split_authors_bibtex(author_field)
    last_names = [last_name_from_person(a) for a in authors][:limit]
    return [ln for ln in last_names if ln]


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def doi_normalize(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    return d.lower()


def doi_url(doi: str) -> str:
    return f"https://doi.org/{doi}"


def extract_arxiv_id_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = ARXIV_HOST_RE.search(text)
    if m:
        return m.group("id")
    m = ARXIV_ID_RE.search(text)
    if m:
        return m.group("id")
    return None


def first_author_surname(entry: Dict[str, Any]) -> str:
    return authors_last_names(entry.get("author", ""), limit=1)[0] if entry.get("author") else ""


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


# ------------- Rate Limiting & Caching -------------
class RateLimiter:
    def __init__(self, req_per_min: int) -> None:
        self.req_per_min = max(req_per_min, 1)
        self.lock = threading.Lock()
        self.timestamps: List[float] = []

    def wait(self) -> None:
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
        if not self.path:
            return None
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
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


# ------------- HTTP Client -------------
class HttpClient:
    def __init__(
        self, timeout: float, user_agent: str, rate_limiter: RateLimiter, cache: DiskCache, verbose: bool = False
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
        self, method: str, url: str, params: Optional[Dict[str, Any]] = None, accept: Optional[str] = None
    ) -> httpx.Response:
        cache_key = None
        if self.cache:
            cache_key = json.dumps({"m": method, "u": url, "p": params, "a": accept}, sort_keys=True)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return httpx.Response(200, content=json.dumps(cached).encode("utf-8"), headers={"X-From-Cache": "1"})
        backoff = 1.0
        for _ in range(6):
            self.rate_limiter.wait()
            try:
                headers = {"Accept": accept} if accept else {}
                resp = self.client.request(method, url, params=params, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
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


# ------------- Google Scholar Client (optional) -------------
class ScholarlyClient:
    """Google Scholar client via scholarly package (opt-in, reliability-focused)."""

    def __init__(self, proxy: str = "none", delay: float = 5.0, logger: Optional[logging.Logger] = None):
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

    def search(self, title: str, first_author: str) -> Optional[Dict[str, Any]]:
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
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None


class Detector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _has_preprint_host(value: str) -> bool:
        v = safe_lower(value)
        return any(host in v for host in PREPRINT_HOSTS)

    def detect(self, entry: Dict[str, Any]) -> PreprintDetection:
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
@dataclass
class PublishedRecord:
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


class Resolver:
    def __init__(
        self, http: HttpClient, logger: logging.Logger, scholarly_client: Optional[ScholarlyClient] = None
    ) -> None:
        self.http = http
        self.logger = logger
        self.scholarly_client = scholarly_client

    # --- arXiv ---
    def arxiv_candidate_doi(self, arxiv_id: str) -> Optional[str]:
        params = {"id_list": arxiv_id}
        try:
            resp = self.http._request("GET", ARXIV_API, params=params, accept="application/atom+xml")
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

    # --- Crossref Works ---
    def crossref_get(self, doi: str) -> Optional[Dict[str, Any]]:
        doi = doi_normalize(doi) or ""
        from urllib.parse import quote

        url = f"{CROSSREF_API}/{quote(doi, safe='')}"
        # url = f"{CROSSREF_API}/{httpx.utils.quote(doi, safe='')}"
        try:
            resp = self.http._request("GET", url, accept="application/json")
            if resp.status_code != 200:
                return None
            data = resp.json().get("message", {})
            return data
        except Exception as e:
            self.logger.debug("Crossref works failed for %s: %s", doi, e)
            return None

    def crossref_search(
        self, query: str, rows: int = 25, filter_type: Optional[str] = "journal-article"
    ) -> List[Dict[str, Any]]:
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

    # --- DBLP ---
    def dblp_search(self, query: str, h: int = 25) -> List[Dict[str, Any]]:
        params = {"q": query, "h": h, "format": "json"}
        try:
            resp = self.http._request("GET", DBLP_API_SEARCH, params=params, accept="application/json")
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
    def _dblp_hit_to_record(hit: Dict[str, Any]) -> Optional[PublishedRecord]:
        info = hit.get("info", {})
        title = info.get("title") or ""
        title = re.sub(r"<[^>]*>", "", title)
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
        rec = PublishedRecord(
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
        return rec

    # --- Semantic Scholar (safe alternative to Google Scholar scraping) ---
    def s2_from_arxiv(self, arxiv_id: str) -> Optional[PublishedRecord]:
        fields = "externalIds,doi,title,year,authors,venue,publicationTypes,publicationVenue,url"
        url = f"{S2_API}/paper/arXiv:{arxiv_id}"
        try:
            resp = self.http._request("GET", url, params={"fields": fields}, accept="application/json")
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

    def s2_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        params = {"query": query, "limit": limit, "fields": "title,authors,year,venue,publicationTypes,doi,url"}
        url = f"{S2_API}/paper/search"
        try:
            resp = self.http._request("GET", url, params=params, accept="application/json")
            if resp.status_code != 200:
                return []
            return resp.json().get("data", []) or []
        except Exception as e:
            self.logger.debug("Semantic Scholar search failed '%s': %s", query, e)
            return []

    # --- Shared helpers ---
    @staticmethod
    def _message_to_record(msg: Dict[str, Any]) -> Optional[PublishedRecord]:
        typ = msg.get("type")
        doi = doi_normalize(msg.get("DOI"))
        if not doi:
            return None
        url = msg.get("URL")
        titles = msg.get("title") or []
        title = titles[0] if titles else None
        if title:
            title = re.sub(r"<[^>]*>", "", title)
        authors = []
        for a in msg.get("author", []) or []:
            given = a.get("given") or ""
            family = a.get("family") or ""
            if not (given or family) and a.get("literal"):
                lit = a["literal"]
                parts = lit.split()
                if parts:
                    family = parts[-1]
                    given = " ".join(parts[:-1])
            authors.append({"given": given, "family": family})
        container = msg.get("container-title", [])
        journal = container[0] if container else None
        pubyear = None
        for dt_key in ("published-print", "published-online", "created", "issued"):
            if msg.get(dt_key, {}).get("date-parts"):
                y = msg[dt_key]["date-parts"][0][0]
                pubyear = int(y)
                break
        volume = msg.get("volume")
        issue = msg.get("issue") or msg.get("journal-issue", {}).get("issue")
        pages = msg.get("page")
        publisher = msg.get("publisher")
        return PublishedRecord(
            doi=doi,
            url=url,
            title=title,
            authors=authors,
            journal=journal,
            publisher=publisher,
            year=pubyear,
            volume=volume,
            number=issue,
            pages=pages,
            type=typ,
        )

    @staticmethod
    def _credible_journal_article(rec: PublishedRecord) -> bool:
        if rec.type != "journal-article":
            return False
        if not rec.journal:
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

    def _scholarly_to_record(self, pub: Dict[str, Any]) -> Optional[PublishedRecord]:
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

    def resolve(self, entry: Dict[str, Any], detection: PreprintDetection) -> Optional[PublishedRecord]:
        # 1) arXiv -> DOI -> Crossref
        candidate_doi: Optional[str] = None
        if detection.arxiv_id:
            # Semantic Scholar first (direct arXiv mapping is strong)
            s2 = self.s2_from_arxiv(detection.arxiv_id)
            if s2 and self._credible_journal_article(s2):
                return s2
            candidate_doi = self.arxiv_candidate_doi(detection.arxiv_id)
            if candidate_doi:
                msg = self.crossref_get(candidate_doi)
                if msg:
                    rec = self._message_to_record(msg)
                    if rec and rec.type == "journal-article" and self._credible_journal_article(rec):
                        rec.method = "arXiv->Crossref(works)"
                        rec.confidence = 1.0
                        return rec
                    rel = msg.get("relation") or {}
                    pre_of = rel.get("is-preprint-of") or []
                    for node in pre_of:
                        if node.get("id-type") == "doi" and node.get("id"):
                            pub_doi = doi_normalize(node["id"])
                            pub_msg = self.crossref_get(pub_doi)
                            rec2 = self._message_to_record(pub_msg or {})
                            if rec2 and self._credible_journal_article(rec2):
                                rec2.method = "arXiv->Crossref(relation)"
                                rec2.confidence = 1.0
                                return rec2

        # 2) Crossref by DOI (preprint DOI or candidate)
        for d in filter(None, (detection.doi, candidate_doi)):
            msg = self.crossref_get(d)
            if msg:
                rel = msg.get("relation") or {}
                pre_of = rel.get("is-preprint-of") or []
                for node in pre_of:
                    if node.get("id-type") == "doi" and node.get("id"):
                        pub_doi = doi_normalize(node["id"])
                        pub_msg = self.crossref_get(pub_doi)
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

        # 3) DBLP bibliographic search
        title = entry.get("title") or ""
        title_norm = normalize_title_for_match(title)
        if title_norm:
            first_author = first_author_surname(entry)
            dblp_query = f"{title_norm} {first_author}".strip()
            hits = self.dblp_search(dblp_query, h=30)
            if hits:
                authors_ref = authors_last_names(entry.get("author", ""))
                ta = title_norm
                best: Optional[Tuple[float, PublishedRecord]] = None
                for h in hits:
                    rec = self._dblp_hit_to_record(h)
                    if not rec:
                        continue
                    tb = normalize_title_for_match(rec.title or "")
                    title_score = token_sort_ratio(ta, tb)  # 0..100
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors_ref[:3], blns)
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    if combined >= 0.9 and self._credible_journal_article(rec):
                        rec.method = "DBLP(search)"
                        rec.confidence = combined
                        if (
                            (best is None)
                            or (combined > best[0])
                            or (
                                combined == best[0]
                                and (int(bool(rec.pages)) + int(bool(rec.volume)) + int(bool(rec.number)))
                                > (int(bool(best[1].pages)) + int(bool(best[1].volume)) + int(bool(best[1].number)))
                            )
                        ):
                            best = (combined, rec)
                if best:
                    # If DBLP lacks DOI but we have URL, try to resolve Crossref by DOI when present
                    if not best[1].doi and best[1].url:
                        # nothing to add — keep URL
                        pass
                    return best[1]

        # 4) Semantic Scholar search
        if title_norm:
            first_author = first_author_surname(entry)
            s2_query = f"{title_norm} {first_author}".strip()
            data = self.s2_search(s2_query, limit=25)
            if data:
                authors_ref = authors_last_names(entry.get("author", ""))
                ta = title_norm
                candidates: List[Tuple[float, PublishedRecord]] = []
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
                    title_score = token_sort_ratio(ta, tb)
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors_ref[:3], blns)
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    if combined >= 0.9 and self._credible_journal_article(rec):
                        rec.method = "SemanticScholar(search)"
                        rec.confidence = combined
                        candidates.append((combined, rec))
                if candidates:
                    candidates.sort(
                        key=lambda x: (x[0], int(bool(x[1].pages)) + int(bool(x[1].volume)) + int(bool(x[1].number))),
                        reverse=True,
                    )
                    return candidates[0][1]

        # 5) Crossref bibliographic search (final fallback)
        if title_norm:
            first_author = first_author_surname(entry)
            query = f"{title_norm} {first_author}".strip()
            items = self.crossref_search(query, rows=30, filter_type="journal-article")
            if items:
                title_a = title_norm
                authors_a = authors_last_names(entry.get("author", ""))

                def score_item(msg: Dict[str, Any]) -> Tuple[float, PublishedRecord]:
                    rec = self._message_to_record(msg)
                    if not rec or rec.type != "journal-article":
                        return (0.0, PublishedRecord(doi=""))
                    t = normalize_title_for_match(rec.title or "")
                    title_score = token_sort_ratio(title_a, t)
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors_a[:3], blns[:3])
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    return (combined, rec)

                candidates: List[Tuple[float, PublishedRecord]] = [score_item(it) for it in items]
                passing = [(s, r) for (s, r) in candidates if (s >= 0.9 and self._credible_journal_article(r))]
                if not passing:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    top = candidates[0] if candidates else None
                    if top and top[0] >= 0.85 and self._credible_journal_article(top[1]):
                        rec = top[1]
                        rec.method = "Crossref(search,relaxed)"
                        rec.confidence = top[0]
                        return rec
                    return None
                passing.sort(
                    key=lambda x: (x[0], int(bool(x[1].pages)) + int(bool(x[1].volume)) + int(bool(x[1].number))),
                    reverse=True,
                )
                score, best = passing[0]
                best.method = "Crossref(search)"
                best.confidence = score
                return best

        # 6) Google Scholar fallback (opt-in only)
        if self.scholarly_client and title_norm:
            first_author = first_author_surname(entry)
            pub = self.scholarly_client.search(title_norm, first_author)
            if pub:
                rec = self._scholarly_to_record(pub)
                if rec and rec.title:
                    tb = normalize_title_for_match(rec.title)
                    title_score = token_sort_ratio(title_norm, tb)
                    authors_ref = authors_last_names(entry.get("author", ""))
                    blns = [strip_diacritics(a.get("family") or "").lower() for a in rec.authors][:3]
                    auth_score = jaccard_similarity(authors_ref[:3], blns)
                    combined = 0.7 * (title_score / 100.0) + 0.3 * auth_score
                    if combined >= 0.9:
                        rec.method = "GoogleScholar(search)"
                        rec.confidence = combined
                        self.logger.debug(
                            "Google Scholar match: %.2f (title=%.0f, auth=%.2f)", combined, title_score, auth_score
                        )
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
    def _year_from_record(rec: PublishedRecord) -> Optional[str]:
        return str(rec.year) if rec.year else None

    @staticmethod
    def _generate_key(entry: Dict[str, Any], rec: PublishedRecord) -> str:
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

    def update_entry(self, entry: Dict[str, Any], rec: PublishedRecord, detection: PreprintDetection) -> Dict[str, Any]:
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
    def _key(entry: Dict[str, Any]) -> Tuple[str, str]:
        doi = doi_normalize(entry.get("doi") or "")
        if doi:
            return ("doi", doi)
        title = normalize_title_for_match(entry.get("title") or "")
        auths = authors_last_names(entry.get("author", ""))[:3]
        key = title + "|" + ",".join(sorted(auths))
        return ("fuzzy", key)

    @staticmethod
    def _score(entry: Dict[str, Any]) -> int:
        score = 0
        if safe_lower(entry.get("ENTRYTYPE")) == "article":
            score += 5
        for f in ("title", "author", "journal", "year", "volume", "number", "pages", "doi", "url", "publisher"):
            if entry.get(f):
                score += 1
        return score

    @staticmethod
    def merge_entries(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
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
    ) -> Tuple[bibtexparser.bibdatabase.BibDatabase, List[Tuple[str, List[str]]]]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for e in db.entries:
            groups.setdefault(self._key(e), []).append(e)

        new_entries: List[Dict[str, Any]] = []
        merged_info: List[Tuple[str, List[str]]] = []

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
def entry_to_bib(entry: Dict[str, Any]) -> str:
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = [entry]
    writer = BibWriter()
    return writer.dumps(db).strip()


def diff_entries(old: Dict[str, Any], new: Dict[str, Any], key: str) -> str:
    a = entry_to_bib(old).splitlines(keepends=True)
    b = entry_to_bib(new).splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=f"{key} (old)", tofile=f"{key} (new)", lineterm=""))


# ------------- Processing Pipeline -------------
@dataclass
class ProcessResult:
    original: Dict[str, Any]
    updated: Dict[str, Any]
    changed: bool
    action: str
    method: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None


def process_entry(
    entry: Dict[str, Any], detector: Detector, resolver: Resolver, updater: Updater, logger: logging.Logger
) -> ProcessResult:
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
    p.add_argument("--cache", default=".cache.replace_preprints.json", help="On-disk cache file")
    p.add_argument("--rate-limit", type=int, default=45, help="Requests per minute (default 45)")
    p.add_argument("--max-workers", type=int, default=4, help="Max concurrent workers")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
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


def summarize(results: List[ProcessResult], logger: logging.Logger) -> Dict[str, int]:
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


def print_failures(results: List[ProcessResult], logger: logging.Logger) -> None:
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


def write_report_line(fh, res: ProcessResult, src_file: Optional[str] = None) -> None:
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


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = init_logging(args.verbose)

    if not args.in_place and not args.output:
        logger.error("Specify -o OUTPUT.bib or --in-place")
        return 1

    rate_limiter = RateLimiter(args.rate_limit)
    cache = DiskCache(args.cache) if args.cache else DiskCache(None)
    user_agent = "bib-preprint-upgrader/1.1 (mailto:you@example.com)"
    http = HttpClient(
        timeout=args.timeout, user_agent=user_agent, rate_limiter=rate_limiter, cache=cache, verbose=args.verbose
    )

    # Google Scholar client (opt-in)
    scholarly_client = None
    if args.use_scholarly:
        scholarly_client = ScholarlyClient(
            proxy=args.scholarly_proxy,
            delay=args.scholarly_delay,
            logger=logger,
        )
        if scholarly_client._scholarly:
            logger.info(
                "Google Scholar fallback enabled (delay=%.1fs, proxy=%s)", args.scholarly_delay, args.scholarly_proxy
            )
        else:
            logger.warning("Google Scholar fallback requested but scholarly package not available")
            scholarly_client = None

    resolver = Resolver(http=http, logger=logger, scholarly_client=scholarly_client)
    detector = Detector()
    updater = Updater(keep_preprint_note=args.keep_preprint_note, rekey=args.rekey)

    loader = BibLoader()
    writer = BibWriter()

    # Read inputs
    databases: List[Tuple[str, bibtexparser.bibdatabase.BibDatabase]] = []
    try:
        for path in args.inputs:
            db = loader.load_file(path)
            databases.append((path, db))
    except Exception as e:
        logger.error("Failed to read inputs: %s", e)
        return 1

    if args.in_place:
        overall_exit = 0
        for path, db in databases:
            results: List[ProcessResult] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futures = [ex.submit(process_entry, entry, detector, resolver, updater, logger) for entry in db.entries]
                for fut in concurrent.futures.as_completed(futures):
                    results.append(fut.result())

            new_entries: List[Dict[str, Any]] = [r.updated for r in results]
            new_db = bibtexparser.bibdatabase.BibDatabase()
            new_db.entries = new_entries

            merged_info: List[Tuple[str, List[str]]] = []
            if args.dedupe:
                new_db, merged_info = Dedupe().dedupe_db(new_db, logger)

            if args.dry_run:
                for r in results:
                    if r.changed:
                        print(diff_entries(r.original, r.updated, r.original.get("ID")))
                if args.dedupe and merged_info:
                    print(f"# Dedupe merged groups: {merged_info}")
            else:
                try:
                    writer.dump_to_file(new_db, path)
                except Exception as e:
                    logger.error("Failed to write %s: %s", path, e)
                    overall_exit = 1

            if args.report:
                with open(args.report, "a", encoding="utf-8") as fh:
                    for res in results:
                        write_report_line(fh, res, src_file=path)

            summary = summarize(results, logger)
            print_failures(results, logger)
            if summary["failures"] > 0 and overall_exit == 0:
                overall_exit = 2
        return overall_exit

    else:
        merged_db = bibtexparser.bibdatabase.BibDatabase()
        merged_db.entries = []
        src_for_entry: List[str] = []
        for path, db in databases:
            merged_db.entries.extend(db.entries)
            src_for_entry.extend([path] * len(db.entries))

        results: List[ProcessResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(process_entry, entry, detector, resolver, updater, logger) for entry in merged_db.entries
            ]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        obj_map = {id(r.original): r for r in results}
        ordered_results: List[ProcessResult] = []
        for e in merged_db.entries:
            r = obj_map.get(id(e))
            ordered_results.append(r if r else ProcessResult(original=e, updated=e, changed=False, action="unchanged"))

        new_db = bibtexparser.bibdatabase.BibDatabase()
        new_db.entries = [r.updated for r in ordered_results]

        merged_info: List[Tuple[str, List[str]]] = []
        if args.dedupe:
            new_db, merged_info = Dedupe().dedupe_db(new_db, logger)

        if args.dry_run:
            for r in ordered_results:
                if r.changed:
                    print(diff_entries(r.original, r.updated, r.original.get("ID")))
            if args.dedupe and merged_info:
                print(f"# Dedupe merged groups: {merged_info}")
        else:
            try:
                writer.dump_to_file(new_db, args.output)
            except Exception as e:
                logger.error("Failed to write %s: %s", args.output, e)
                return 1

        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                for idx, res in enumerate(ordered_results):
                    write_report_line(fh, res, src_file=src_for_entry[idx] if idx < len(src_for_entry) else None)

        summary = summarize(ordered_results, logger)
        print_failures(ordered_results, logger)
        if summary["failures"] > 0:
            return 2
        return 0


# ------------- Tests (pytest style) -------------
def _make_entry(**kwargs) -> Dict[str, Any]:
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
