"""Cascading source clients and cross-source author cross-validation.

This module consolidates calls to external bibliographic databases
(CrossRef, Semantic Scholar, OpenAlex) for the fact-checker.

The cascading order is intentional:

1. **CrossRef** -- broadest DOI-registered coverage, no API key, generous
   polite-pool rate limits.
2. **Semantic Scholar** -- best preprint coverage (arXiv, bioRxiv, ...).
3. **OpenAlex** -- aggregator that catches everything CrossRef and S2 miss.

The cascade short-circuits as soon as one source returns a candidate above the
``CASCADE_HIGH_CONFIDENCE`` threshold, which avoids wasted API calls.

Reference: Abbonato, "CheckIfExist: lightweight verification of academic
references" (2026), Algorithm 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
from rapidfuzz.fuzz import token_sort_ratio

from bibtex_updater.utils import (
    OPENALEX_API,
    OPENREVIEW_API,
    PublishedRecord,
    last_name_from_person,
    normalize_title_for_match,
    strip_diacritics,
)

__all__ = [
    "OpenAlexClient",
    "openalex_work_to_candidate_record",
    "OpenReviewClient",
    "openreview_note_to_candidate_record",
    "build_openreview_paperhash",
    "select_top_k_by_title_similarity",
    "cross_source_author_intersection",
    "AuthorIntersectionResult",
    "CASCADE_LOW_CONFIDENCE",
    "CASCADE_HIGH_CONFIDENCE",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "DEFAULT_OPENALEX_MAILTO",
]


# ------------- Cascade tuning constants -------------

#: Below this score, fall through to the next cascade source.
CASCADE_LOW_CONFIDENCE: float = 0.50

#: At/above this score, short-circuit the cascade -- we have a good match.
CASCADE_HIGH_CONFIDENCE: float = 0.95

#: Default number of candidates to retrieve per source before fuzzy ranking.
DEFAULT_TOP_K: int = 3

#: Hard cap on ``--top-k`` to keep API usage sane.
MAX_TOP_K: int = 10

#: Used for OpenAlex polite-pool routing when no email is configured.
DEFAULT_OPENALEX_MAILTO: str = "bibtex-check@example.org"


# ------------- OpenAlex client -------------


class OpenAlexClient:
    """Minimal OpenAlex search client.

    Endpoint: ``https://api.openalex.org/works?search=...``.

    OpenAlex returns 25 works per page by default; we cap to ``per_page`` to
    keep responses small. Including ``mailto`` opts into the polite pool, which
    bumps rate limits without requiring an API key.
    """

    def __init__(
        self,
        http: Any | None = None,
        mailto: str = DEFAULT_OPENALEX_MAILTO,
        timeout: float = 20.0,
    ) -> None:
        self.http = http
        self.mailto = mailto
        self.timeout = timeout

    def search(
        self,
        query: str,
        limit: int = DEFAULT_TOP_K,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search OpenAlex works.

        Args:
            query: Free-text search string (typically ``"<title> <author>"``).
                Used for the ``?search=`` fallback path.
            limit: Max records to retrieve (capped at ``MAX_TOP_K``).
            title: Raw (un-normalized, author-free) title. When provided, the
                client first issues a *fielded* ``filter=title.search:<title>``
                query, which matches the exact paper at rank #1 far more often
                than the BM25 ``?search=`` relevance endpoint -- the latter
                returns unrelated papers for DOI-less ML-conference titles.
                The free-text ``?search=`` path is used only as a fallback when
                the fielded query yields zero results.

        Returns:
            List of OpenAlex work dicts, never None. Empty on any error.
        """
        per_page = max(1, min(int(limit), MAX_TOP_K))

        # ----- Fielded title.search path (preferred) -----
        if title and title.strip():
            fielded_params = {
                "filter": f"title.search:{title.strip()}",
                "per-page": per_page,
                "mailto": self.mailto,
            }
            fielded = self._fetch(fielded_params)
            if fielded:
                return fielded
            # Zero fielded results -> fall through to free-text below.

        # ----- Free-text ?search= path (fallback / legacy) -----
        if not query:
            return []
        params = {
            "search": query,
            "per-page": per_page,
            "mailto": self.mailto,
        }
        return self._fetch(params)

    def _fetch(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Issue a single OpenAlex /works request and return the result list.

        Preserves the shared-HttpClient-vs-bare-httpx routing (rate limiting,
        caching, polite User-Agent on the shared path; hermetic fallback
        otherwise). Returns ``[]`` on any non-200 status or exception.
        """
        url = f"{OPENALEX_API}/works"
        try:
            if self.http is not None and hasattr(self.http, "_request"):
                resp = self.http._request(
                    "GET",
                    url,
                    params=params,
                    accept="application/json",
                    service="openalex",
                )
                if resp.status_code != 200:
                    return []
                data = resp.json() or {}
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(url, params=params)
                    if resp.status_code != 200:
                        return []
                    data = resp.json() or {}
        except Exception:
            return []
        results = data.get("results") or []
        if not isinstance(results, list):
            return []
        return results


def openalex_work_to_candidate_record(work: dict[str, Any]) -> PublishedRecord | None:
    """Permissive OpenAlex -> ``PublishedRecord`` conversion for cascade search.

    Unlike ``utils.openalex_work_to_record`` (which is preprint-strict for
    publication-resolution), this version retains preprints and works without
    a venue, since the cascade just needs *candidates* to score against.

    Args:
        work: OpenAlex work dict from the ``/works`` endpoint.

    Returns:
        ``PublishedRecord`` or ``None`` if the work lacks a usable title.
    """
    if not work:
        return None
    title = work.get("title") or work.get("display_name")
    if not title:
        return None

    raw_doi = work.get("doi") or ""
    doi: str | None = None
    if raw_doi:
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(raw_doi), flags=re.IGNORECASE)

    authors: list[dict[str, str]] = []
    for authorship in work.get("authorships") or []:
        author_obj = authorship.get("author") or {}
        display_name = author_obj.get("display_name", "")
        if not display_name:
            continue
        parts = display_name.split()
        if len(parts) >= 2:
            authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
        elif parts:
            authors.append({"given": "", "family": parts[0]})

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    journal = source.get("display_name")
    year = work.get("publication_year")
    work_type = work.get("type")

    return PublishedRecord(
        doi=doi,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        type=work_type,
    )


# ------------- OpenReview client -------------


#: OpenReview "tilde" profile-id pattern, e.g. ``~Diederik_P_Kingma1`` or
#: ``~Aidan_N._Gomez1`` (a middle initial may carry a trailing period). The
#: family name is the underscore token immediately before the trailing digits.
_OPENREVIEW_TILDE_ID = re.compile(r"^~([A-Za-z][\w.]*?)(\d+)$")


def _content_value(content: dict[str, Any], key: str) -> Any:
    """Read an OpenReview note ``content`` field across API v1/v2 shapes.

    API v2 wraps every field as ``{"value": <v>}``; API v1 stores the bare
    value. This returns the inner value either way (or ``None`` if absent).
    """
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw.get("value")
    return raw


def build_openreview_paperhash(title: str, first_author_last_name: str) -> str | None:
    """Construct OpenReview's ``paperhash`` exact-match key for a paper.

    OpenReview indexes every note under ``<firstauthor_lastname>|<title>`` where
    the title is lowercased, stripped of all non-alphanumeric characters (colons,
    hyphens, apostrophes are *removed*, not spaced -- "Few-Shot" -> "fewshot",
    "BERT:" -> "bert"), whitespace-collapsed, and spaces become underscores. The
    author prefix is the first author's last name, lowercased + diacritics-free.
    Verified live against the Kingma/He/Vaswani/Devlin/Brown/Ho papers.

    Returns ``None`` when either component is empty (an un-hashable entry).
    """
    last = strip_diacritics(first_author_last_name or "").lower().strip()
    last = re.sub(r"[^a-z0-9]", "", last)
    norm_title = strip_diacritics(title or "").lower()
    norm_title = re.sub(r"[^a-z0-9\s]", "", norm_title)
    norm_title = "_".join(norm_title.split())
    if not last or not norm_title:
        return None
    return f"{last}|{norm_title}"


class OpenReviewClient:
    """Minimal OpenReview lookup client (legacy ``api.openreview.net``).

    OpenReview is the authoritative submission registry for ICLR, NeurIPS, TMLR
    and many other ML venues, which the rest of the cascade (CrossRef/OpenAlex/
    DBLP/S2) frequently fails to *positively* confirm. The legacy ``/notes``
    endpoint exposes a ``paperhash`` filter that does an exact title + first-author
    match -- far more precise than the full-text ``term=``/``query=`` search,
    which returns reviews and unrelated public articles. ``content.authors`` is a
    flat name list, but ``content.authorids`` carries ``~Given_Family<N>`` profile
    handles from which an authoritative family name is recoverable.

    Mirrors :class:`OpenAlexClient`: optional shared ``http`` (rate limiting +
    caching via ``service="openreview"``), bare ``httpx`` fallback for hermetic
    use, and ``[]`` on any error / non-200.
    """

    def __init__(self, http: Any | None = None, timeout: float = 20.0) -> None:
        self.http = http
        self.timeout = timeout

    def search(
        self,
        query: str,
        limit: int = DEFAULT_TOP_K,
        title: str | None = None,
        first_author: str | None = None,
    ) -> list[dict[str, Any]]:
        """Look up OpenReview notes by exact ``paperhash`` (title + first author).

        Args:
            query: Free-text blob (unused by the paperhash path; accepted for a
                signature symmetric with the other cascade clients).
            limit: Max notes to retrieve (capped at ``MAX_TOP_K``).
            title: Raw paper title. Required to build the ``paperhash``.
            first_author: First author's *last* name (already reduced). Required
                to build the ``paperhash``; without it no lookup is possible.

        Returns:
            List of OpenReview note dicts (never ``None``). Empty on any error,
            on a missing title/author, or when nothing matches.
        """
        per_page = max(1, min(int(limit), MAX_TOP_K))
        paperhash = build_openreview_paperhash(title or "", first_author or "")
        if not paperhash:
            return []
        params = {"paperhash": paperhash, "limit": per_page}
        return self._fetch(params)

    def _fetch(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Issue a single OpenReview ``/notes`` request, return the note list.

        Preserves shared-HttpClient-vs-bare-httpx routing (rate limiting + caching
        on the shared path). Returns ``[]`` on any non-200 status or exception.
        """
        url = f"{OPENREVIEW_API}/notes"
        try:
            if self.http is not None and hasattr(self.http, "_request"):
                resp = self.http._request(
                    "GET",
                    url,
                    params=params,
                    accept="application/json",
                    service="openreview",
                )
                if resp.status_code != 200:
                    return []
                data = resp.json() or {}
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(url, params=params)
                    if resp.status_code != 200:
                        return []
                    data = resp.json() or {}
        except Exception:
            return []
        notes = data.get("notes") or []
        if not isinstance(notes, list):
            return []
        return notes


def _family_from_tilde_id(authorid: str) -> str | None:
    """Recover an authoritative family name from a ``~Given_Family<N>`` handle.

    OpenReview profile ids encode the name as underscore-separated tokens with a
    trailing disambiguation digit, e.g. ``~Diederik_P_Kingma1`` -> ``Kingma``,
    ``~Aidan_N._Gomez1`` -> ``Gomez``. Non-tilde ids (raw emails, DBLP-search
    URLs) yield ``None`` so the caller can fall back to the flat display name.
    """
    if not authorid or not authorid.startswith("~"):
        return None
    m = _OPENREVIEW_TILDE_ID.match(authorid)
    if not m:
        return None
    tokens = [t for t in m.group(1).split("_") if t]
    if not tokens:
        return None
    return tokens[-1]


def openreview_note_to_candidate_record(note: dict[str, Any]) -> PublishedRecord | None:
    """Convert an OpenReview note to a ``PublishedRecord`` for cascade scoring.

    Authors come from the flat ``content.authors`` list; whenever the parallel
    ``content.authorids`` entry is a ``~Given_Family<N>`` profile handle we lift
    the AUTHORITATIVE family name out of it and set ``structured_names=True`` so
    ``surname_keys`` trusts the family verbatim. If *any* author lacks a usable
    tilde id we leave ``structured_names=False`` and the synthesized last-token
    family is used (and the verdict logic stays conservative). The venue is taken
    from ``content.venue`` (falling back to ``content.venueid``).

    Returns ``None`` if the note has no usable title.
    """
    if not note:
        return None
    content = note.get("content") or {}
    if not isinstance(content, dict):
        return None

    title = _content_value(content, "title")
    if not title or not str(title).strip():
        return None
    title = re.sub(r"<[^>]*>", "", str(title)).strip()

    raw_authors = _content_value(content, "authors") or []
    raw_authorids = _content_value(content, "authorids") or []
    if not isinstance(raw_authors, list):
        raw_authors = []
    if not isinstance(raw_authorids, list):
        raw_authorids = []

    authors: list[dict[str, str]] = []
    all_structured = bool(raw_authors)
    for idx, full_name in enumerate(raw_authors):
        if not isinstance(full_name, str) or not full_name.strip():
            continue
        authorid = raw_authorids[idx] if idx < len(raw_authorids) else ""
        family = _family_from_tilde_id(authorid if isinstance(authorid, str) else "")
        if family:
            parts = full_name.strip().split()
            given = " ".join(parts[:-1]) if len(parts) > 1 else ""
            authors.append({"given": given, "family": family})
        else:
            # No authoritative handle: synthesize a family via the same last-token
            # heuristic the entry side uses, and mark the record unstructured.
            all_structured = False
            family = last_name_from_person(full_name)
            parts = full_name.strip().split()
            given = " ".join(parts[:-1]) if len(parts) > 1 else ""
            authors.append({"given": given, "family": family})

    if not authors:
        all_structured = False

    venue = _content_value(content, "venue") or _content_value(content, "venueid")

    year = None
    raw_year = _content_value(content, "year")
    try:
        if raw_year is not None:
            year = int(str(raw_year)[:4])
    except (ValueError, TypeError):
        year = None
    # OpenReview rarely sets an explicit ``year`` field, but its venue strings
    # almost always embed one ("ICLR 2024", "NeurIPS 2023 Conference"). Recover it
    # so the record can positively confirm the entry's claimed year instead of
    # leaving it unconfirmable -- resolve what can be resolved.
    if year is None and venue:
        m = re.search(r"\b(19|20)\d{2}\b", str(venue))
        if m:
            year = int(m.group(0))

    return PublishedRecord(
        doi=None,
        title=title,
        authors=authors,
        journal=venue,
        year=year,
        type="conference",
        structured_names=all_structured,
    )


# ------------- Top-K candidate selection -------------


def select_top_k_by_title_similarity(
    query_title: str,
    candidates: list[PublishedRecord],
    k: int = DEFAULT_TOP_K,
) -> list[tuple[float, PublishedRecord]]:
    """Re-rank candidates by Levenshtein title similarity, keep top-K.

    Implements step 3 of CheckIfExist Algorithm 1: from each source, retrieve a
    handful of candidates and pick the best ones by fuzzy title score before
    doing the more expensive author/venue/year cross-checks.

    Args:
        query_title: The entry's title (any normalization is fine).
        candidates: Records to rank.
        k: How many top candidates to return.

    Returns:
        ``(score_0_to_1, record)`` pairs sorted descending by score.
    """
    if not candidates:
        return []
    query_norm = normalize_title_for_match(query_title or "")
    scored: list[tuple[float, PublishedRecord]] = []
    for rec in candidates:
        rec_title = normalize_title_for_match(rec.title or "")
        if not query_norm or not rec_title:
            score = 0.0
        else:
            score = token_sort_ratio(query_norm, rec_title) / 100.0
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    k_eff = max(1, min(int(k), MAX_TOP_K))
    return scored[:k_eff]


# ------------- Cross-source author intersection -------------


@dataclass
class AuthorIntersectionResult:
    """Outcome of intersecting author lists across multiple matched sources.

    Attributes:
        confirmed: Family names present in *every* contributing source.
        suspect: Family names appearing in some but not all sources -- these
            are the author-fabrication candidates.
        sources_consulted: Names of the sources that contributed an author list.
        bonus: ``+10`` confidence bonus when ``len(confirmed) >= 2``, else 0.
    """

    confirmed: list[str] = field(default_factory=list)
    suspect: list[str] = field(default_factory=list)
    sources_consulted: list[str] = field(default_factory=list)
    bonus: float = 0.0


def _normalize_family_name(name: str) -> str:
    """Lowercase + strip diacritics for stable cross-source comparison.

    Single-letter tokens (initials) are dropped -- "J. Smith" should match
    "James Smith" once we normalize the family-name only.
    """
    if not name:
        return ""
    cleaned = strip_diacritics(name).lower().strip()
    cleaned = re.sub(r"[^\w\s'-]", "", cleaned)
    cleaned = " ".join(part for part in cleaned.split() if len(part) > 1)
    return cleaned


def _extract_family_names(record: PublishedRecord | None) -> set[str]:
    """Pull a normalized family-name set from a candidate record."""
    if record is None or not record.authors:
        return set()
    out: set[str] = set()
    for author in record.authors:
        family = author.get("family") if isinstance(author, dict) else ""
        norm = _normalize_family_name(family or "")
        if norm:
            out.add(norm)
    return out


def cross_source_author_intersection(
    source_records: dict[str, PublishedRecord | None],
    multi_source_bonus: float = 10.0,
) -> AuthorIntersectionResult:
    """Cross-validate authors across 2+ sources (CheckIfExist core mechanism).

    - ``confirmed`` = intersection of all non-empty author sets.
    - ``suspect`` = union minus confirmed.
    - When at least two confirmed authors agree across sources, contribute a
      ``multi_source_bonus`` (default ``+10``) to the final confidence score.

    Args:
        source_records: Mapping ``source_name -> PublishedRecord``. ``None``
            values are dropped.
        multi_source_bonus: ``β_ms`` from the CheckIfExist paper, in [0, 10].

    Returns:
        ``AuthorIntersectionResult``. With fewer than two contributing sources
        the result has empty confirmed/suspect lists and zero bonus.
    """
    contributing: list[tuple[str, set[str]]] = []
    for source, record in source_records.items():
        names = _extract_family_names(record)
        if names:
            contributing.append((source, names))

    if len(contributing) < 2:
        return AuthorIntersectionResult(
            confirmed=[],
            suspect=[],
            sources_consulted=[s for s, _ in contributing],
            bonus=0.0,
        )

    confirmed_set: set[str] = set.intersection(*(names for _, names in contributing))
    union_set: set[str] = set.union(*(names for _, names in contributing))
    suspect_set = union_set - confirmed_set

    bonus = float(multi_source_bonus) if len(confirmed_set) >= 2 else 0.0
    bonus = max(0.0, min(bonus, 10.0))

    return AuthorIntersectionResult(
        confirmed=sorted(confirmed_set),
        suspect=sorted(suspect_set),
        sources_consulted=[s for s, _ in contributing],
        bonus=bonus,
    )


# ------------- Convenience: build polite OpenAlex query -------------


def build_polite_openalex_url(query: str, mailto: str, per_page: int = DEFAULT_TOP_K) -> str:
    """Construct the OpenAlex polite-pool URL (used for logging/debugging)."""
    safe_query = quote(query, safe="")
    safe_mailto = quote(mailto, safe="")
    per_page = max(1, min(int(per_page), MAX_TOP_K))
    return f"{OPENALEX_API}/works?search={safe_query}&per-page={per_page}&mailto={safe_mailto}"


# Re-export for tests / external callers that prefer to import here.
__all__.extend(
    [
        "crossref_message_to_record",
        "s2_data_to_record",
        "build_polite_openalex_url",
    ]
)
