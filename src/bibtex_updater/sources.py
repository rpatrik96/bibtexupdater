"""Cascading source clients and cross-source author cross-validation.

This module consolidates calls to external bibliographic databases
(CrossRef, OpenAlex, DBLP, OpenReview, Semantic Scholar) for the fact-checker.

The cascading order is intentional and throughput-aware:

1. **CrossRef** -- broadest DOI-registered coverage, no API key, generous
   polite-pool rate limits.
2. **OpenAlex** -- high-rate aggregator (polite pool) with broad coverage that
   catches works CrossRef misses.
3. **DBLP** -- authoritative CS/ML-conference index (token-AND title search).
4. **OpenReview** -- authoritative ICLR/NeurIPS/TMLR submission registry; it
   positively confirms ML-conference papers the DOI- and CS-index sources can
   only leave unconfirmed.
5. **Semantic Scholar** -- strong preprint coverage (arXiv, bioRxiv, ...) but the
   slowest source without an API key, so it is queried last.

The cascade short-circuits as soon as one source returns a candidate above the
``CASCADE_HIGH_CONFIDENCE`` threshold, which avoids wasted API calls.

Reference: Abbonato, "CheckIfExist: lightweight verification of academic
references" (2026), Algorithm 1.
"""

from __future__ import annotations

import html
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
from rapidfuzz.fuzz import token_sort_ratio

from bibtex_updater.utils import (
    OPENALEX_API,
    OPENREVIEW_API,
    OPENREVIEW_API_V2,
    OPENREVIEW_AUTH_REFRESH_STATUS,
    OPENREVIEW_SEARCH_SERVICE,
    OPENREVIEW_SERVICE,
    PublishedRecord,
    SourceUnavailableError,
    _reduce_trailing_to_surname,
    arxiv_id_from_datacite_doi,
    as_source_failure,
    decode_latex_accents,
    extract_arxiv_id_from_text,
    is_preprint_venue,
    last_name_from_person,
    latex_to_plain,
    normalize_issn,
    normalize_title_for_match,
    raise_for_failed_lookup,
    strip_diacritics,
)

__all__ = [
    "OpenAlexClient",
    "openalex_work_to_candidate_record",
    "OpenReviewClient",
    "openreview_note_to_candidate_record",
    "openreview_acceptance",
    "OR_ACCEPTED",
    "OR_NOT_ACCEPTED",
    "OR_PREPRINT",
    "OR_UNKNOWN",
    "build_openreview_paperhash",
    "build_openreview_paperhashes",
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
    bumps rate limits without requiring an API key. Since Feb 2026 keyless
    traffic is additionally metered against a shared daily credit budget that
    429s for the rest of the day once exhausted; a premium ``api_key`` (or the
    ``OPENALEX_API_KEY`` env var) lifts requests out of that pool.
    """

    def __init__(
        self,
        http: Any | None = None,
        mailto: str = DEFAULT_OPENALEX_MAILTO,
        timeout: float = 20.0,
        api_key: str | None = None,
    ) -> None:
        self.http = http
        self.mailto = mailto
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY") or None

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
            List of OpenAlex work dicts, never None. Empty when OpenAlex
            answered and reported no works.

        Raises:
            SourceUnavailableError: the lookup ended without an answer
                (unreachable host, error status, unparseable body).
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
        otherwise).

        Returns ``[]`` only when OpenAlex answered and reported no works. A
        lookup that ends without an answer -- unreachable host, error status,
        unparseable body -- raises :class:`SourceUnavailableError` instead, so
        the cascade can record that this source was never consulted rather than
        counting it as a source that knows nothing about the entry.
        """
        url = f"{OPENALEX_API}/works"
        if self.api_key:
            params.setdefault("api_key", self.api_key)
        try:
            if self.http is not None and hasattr(self.http, "_request"):
                resp = self.http._request(
                    "GET",
                    url,
                    params=params,
                    accept="application/json",
                    service="openalex",
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(url, params=params)
            raise_for_failed_lookup("openalex", url, resp.status_code)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
        except Exception as exc:
            raise as_source_failure("openalex", url, exc) from exc
        results = data.get("results") or []
        if not isinstance(results, list):
            return []
        return results

    def search_sources(self, query: str, limit: int = 10) -> list[dict[str, Any]] | None:
        """Search the OpenAlex *sources* registry (journals / conference series).

        Endpoint: ``GET /sources?search=<venue>&per-page=<limit>`` (polite-pool
        ``mailto`` included). Used by the venue-existence check, so errors must
        be distinguishable from zero hits: returns ``None`` on any non-200 /
        parse / network failure (callers MUST treat that as "could not check",
        never as "venue missing") and a possibly-empty source list only when
        OpenAlex answered successfully.
        """
        if not query or not query.strip():
            return []
        per_page = max(1, min(int(limit), 25))
        url = f"{OPENALEX_API}/sources"
        params: dict[str, Any] = {"search": query.strip(), "per-page": per_page, "mailto": self.mailto}
        if self.api_key:
            params.setdefault("api_key", self.api_key)
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
                    return None
                data = resp.json() or {}
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(url, params=params)
                    if resp.status_code != 200:
                        return None
                    data = resp.json() or {}
        except Exception:
            return None
        results = data.get("results")
        if results is None:
            results = []
        return results if isinstance(results, list) else None


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
    if not isinstance(primary_location, dict):
        primary_location = {}
    source = primary_location.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    journal = source.get("display_name")
    year = work.get("publication_year")
    work_type = work.get("type")

    # arXiv identity: OpenAlex has no first-class arXiv field on works; the ID
    # is recoverable from a DataCite arXiv DOI (``10.48550/arXiv.<id>``) or
    # from an arxiv.org landing/PDF URL on the primary location / locations
    # list. Defensive throughout: any unexpected shape just leaves it None.
    arxiv_id = arxiv_id_from_datacite_doi(doi)
    if arxiv_id is None:
        locations: list[Any] = [primary_location]
        raw_locations = work.get("locations")
        if isinstance(raw_locations, list):
            locations.extend(raw_locations)
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            for url_field in ("landing_page_url", "pdf_url"):
                url_val = loc.get(url_field)
                if isinstance(url_val, str) and "arxiv.org" in url_val.lower():
                    arxiv_id = extract_arxiv_id_from_text(url_val)
                    if arxiv_id:
                        break
            if arxiv_id:
                break

    # Venue identity: the OpenAlex source id is a stable venue identifier, and
    # ``issn``/``issn_l`` carry the journal's ISSNs. Defensive: any unexpected
    # shape yields empty identity fields rather than a parse error.
    raw_source_id = source.get("id")
    venue_source_id = raw_source_id if isinstance(raw_source_id, str) and raw_source_id.strip() else None
    raw_issns = source.get("issn") or []
    if isinstance(raw_issns, str):
        raw_issns = [raw_issns]
    issns: list[str] = []
    issn_candidates = list(raw_issns) if isinstance(raw_issns, list) else []
    issn_candidates.append(source.get("issn_l"))
    for raw_issn in issn_candidates:
        norm_issn = normalize_issn(raw_issn)
        if norm_issn and norm_issn not in issns:
            issns.append(norm_issn)

    return PublishedRecord(
        doi=doi,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        type=work_type,
        order_reliable=True,
        issn=tuple(issns),
        venue_source_id=venue_source_id,
        arxiv_id=arxiv_id,
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


def _coerce_venue_string(raw: Any) -> str | None:
    """Reduce an OpenReview venue field of any shape to a usable string.

    ``venue``/``venueid`` come back as a list on some notes. Every venue helper
    downstream (``is_preprint_venue``, ``is_preprint_or_series_venue``,
    ``_normalize_venue_for_matching``) calls ``.lower()``, so a list raised
    ``AttributeError`` mid-cascade; the per-entry handler swallowed it and the
    entry vanished from the report. Returns the first non-empty string in a
    list, the string itself, a stringified scalar, or ``None``.
    """
    if raw is None or isinstance(raw, str):
        return raw or None
    if isinstance(raw, list | tuple):
        for item in raw:
            if isinstance(item, str) and item.strip():
                return item
        return None
    if isinstance(raw, int | float):
        return str(raw)
    return None


#: Latin-1 Supplement letters (U+00DF-U+00FF) that OpenReview's paperhash index
#: KEEPS. Everything above that block -- Latin Extended-A and beyond -- is
#: DROPPED, not folded: ``Karlaš`` indexes as ``karla`` and ``Jovanović`` as
#: ``jovanovi``, while ``Akyürek`` keeps its ``ü``. Measured across 6,698 live
#: notes on both hosts (ICLR, ICML, NeurIPS, TMLR, COLM).
_OR_LATIN1_LETTERS = "\u00df-\u00ff"

#: Characters the title slug keeps beyond letters and digits. OpenReview does
#: NOT strip the maths that survives into a title: ``$\ell_p$`` indexes as
#: ``\ell_p`` and ``RoboMP$^2$`` as ``robomp^2``, so the backslash, caret,
#: brackets and underscore are part of the key.
_OR_TITLE_KEEP = rf"a-z0-9{_OR_LATIN1_LETTERS}_\\\^\[\]"
_OR_NAME_KEEP = rf"a-z0-9{_OR_LATIN1_LETTERS}"

#: Font/emphasis wrappers whose *content* belongs in the title. Dropping the
#: command but keeping the braced text is what OpenReview's own plain-text
#: title carries.
_OR_MARKUP_CMD_RE = re.compile(r"\\(?:emph|textbf|textit|texttt|textsc|textrm|textsf|mbox|text|rm|it|bf|sc|tt)\s*\{")

#: LaTeX-escaped literals that survive into the indexed title as the bare
#: character.
_OR_ESCAPED_LITERALS = {r"\&": "&", r"\%": "%", r"\$": "$", r"\#": "#", r"\_": "_"}


def _openreview_delatex(text: str) -> str:
    r"""De-LaTeX a field the way OpenReview's stored plain text reads.

    Unlike :func:`~bibtex_updater.utils.latex_to_plain` this does NOT delete
    maths: ``latex_to_plain`` drops everything between ``$``…``$``, which erases
    exactly the ``\ell_p`` and ``^2`` fragments OpenReview keeps in the index.
    Accent macros still decode to real Unicode (``{\"u}`` -> ``ü``), because the
    folded ``u`` form returns nothing.
    """
    if not text:
        return ""
    out = html.unescape(text)
    out = decode_latex_accents(out)
    out = _OR_MARKUP_CMD_RE.sub("{", out)
    for escaped, literal in _OR_ESCAPED_LITERALS.items():
        out = out.replace(escaped, literal)
    out = out.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", out).strip()


def _openreview_slug(text: str, keep: str) -> str:
    """Lowercase, drop every character outside ``keep``, join words with ``_``.

    Punctuation is REMOVED rather than spaced, which is why "Few-Shot" indexes
    as ``fewshot`` and "BERT:" as ``bert``.
    """
    lowered = text.lower()
    lowered = re.sub(rf"[^{keep} ]+", "", lowered, flags=re.UNICODE)
    return "_".join(lowered.split())


def _openreview_surname_token(first_author: str) -> str:
    """Reduce a first-author name to the token OpenReview hashes on.

    OpenReview keys on the LAST whitespace token of the name and discards
    nobiliary particles with it: ``Marine Le Morvan`` -> ``morvan``,
    ``Julius von Kügelgen`` -> ``kügelgen``, ``Aaron van den Oord`` -> ``oord``.
    Accepts either BibTeX order ("Le Morvan, Marine") or display order, and is
    idempotent on an already-reduced surname. Diacritics survive here; the
    caller decides which of the two indexed forms to issue.
    """
    plain = _openreview_delatex(first_author or "")
    last = plain.split(",", 1)[0] if "," in plain else plain
    # Punctuation is deleted, not spaced, so ``O'Brien`` stays one token and
    # reduces to ``obrien`` rather than to ``brien``.
    last = re.sub(r"[^\w\s-]", "", last.lower(), flags=re.UNICODE)
    tokens = _reduce_trailing_to_surname([t for t in last.split() if t])
    return tokens[-1] if tokens else ""


def build_openreview_paperhashes(title: str, first_author: str) -> list[str]:
    """Every ``paperhash`` form under which OpenReview may have indexed a paper.

    OpenReview indexes each note as ``<firstauthor_surname>|<title>``, both
    slugified: lowercased, punctuation deleted rather than spaced ("Few-Shot"
    -> ``fewshot``, "BERT:" -> ``bert``), whitespace collapsed, spaces to
    underscores.

    Two forms come back, in query order:

    1. **Diacritic-preserving.** ``Akyürek`` indexes as ``akyürek`` and the
       ASCII-folded ``akyurek`` returns nothing.
    2. **ASCII-folded.** OpenReview keeps Latin-1 but DROPS Latin Extended, so
       ``Karlaš`` indexes as ``karla`` -- while the DBLP mirror of the same
       paper, whose name arrives already transliterated, indexes as ``karlas``.
       Both are live records with different note ids. A client cannot know which
       range a surname falls in, so it issues both.

    The two forms collapse to one entry for the overwhelmingly common
    unaccented case, which keeps the request budget at one hash per host.
    Returns ``[]`` when either component slugs to nothing.
    """
    surname = _openreview_surname_token(first_author)
    plain_title = _openreview_delatex(title or "")
    hashes: list[str] = []
    for name, text in ((surname, plain_title), (strip_diacritics(surname), strip_diacritics(plain_title))):
        name_slug = _openreview_slug(name, _OR_NAME_KEEP)
        title_slug = _openreview_slug(text, _OR_TITLE_KEEP)
        if not name_slug or not title_slug:
            continue
        candidate = f"{name_slug}|{title_slug}"
        if candidate not in hashes:
            hashes.append(candidate)
    return hashes


def build_openreview_paperhash(title: str, first_author_last_name: str) -> str | None:
    """The primary (diacritic-preserving) ``paperhash`` for a paper.

    Thin wrapper over :func:`build_openreview_paperhashes` for callers that want
    a single key. ``None`` when the entry is un-hashable. Prefer the plural form
    in a lookup: an ASCII-transliterated mirror of the same paper is indexed
    under the folded hash alone.
    """
    hashes = build_openreview_paperhashes(title, first_author_last_name)
    return hashes[0] if hashes else None


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

    OpenReview keeps ``/notes`` behind a browser challenge, so an anonymous
    caller gets ``403 ChallengeRequiredError`` there on both hosts and the
    paperhash lookup cannot run. Credentials
    (:class:`~bibtex_updater.utils.OpenReviewAuth`, opt-in through
    ``OPENREVIEW_USERNAME`` / ``OPENREVIEW_PASSWORD`` or
    ``--openreview-username``, and carried by the shared ``http`` client)
    restore it. Without them every lookup here fails, which is the honest
    outcome: an endpoint that declined to answer can never support the
    exhaustive ``not_found`` claim. The refusal is latched so the rest of the
    run costs nothing.
    """

    def __init__(self, http: Any | None = None, timeout: float = 20.0) -> None:
        self.http = http
        self.timeout = timeout
        # Endpoints that refused this run (HTTP 401/403). OpenReview gates
        # ``/notes`` behind a browser challenge for anonymous callers, which is a
        # configuration state rather than a blip: re-issuing the same refused
        # request once per entry buys a round trip and a circuit-breaker tick for
        # an answer that is not coming. The latch is per endpoint, not per
        # service, so an authenticated run whose token covers one host is not
        # punished for the other, and so the source's circuit stays free to
        # describe what the transport is actually doing.
        self._refused_urls: set[str] = set()
        self._refused_lock = threading.Lock()

    def _is_refused(self, url: str) -> bool:
        """Has this endpoint already refused us in this run?"""
        with self._refused_lock:
            return url in self._refused_urls

    def _latch_refusal(self, url: str, exc: SourceUnavailableError) -> None:
        """Remember a 401/403 so later entries skip straight past this endpoint.

        Only a refusal latches. A timeout, a 5xx or an exhausted retry budget is
        transient and belongs to the circuit breaker, which already paces it.
        """
        if exc.status_code in OPENREVIEW_AUTH_REFRESH_STATUS:
            with self._refused_lock:
                self._refused_urls.add(url)

    #: Host order for every ``/notes`` lookup. v2 first: it holds the 2023+
    #: venues (ICLR 2024 alone has 2,260 notes there and 0 on v1), which is the
    #: dominant shape in a modern ML bibliography.
    NOTES_HOSTS = (OPENREVIEW_API_V2, OPENREVIEW_API)

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
            first_author: First author's name -- the raw BibTeX name where the
                caller has it, so the diacritics and the nobiliary particles
                OpenReview keys on survive. An already-reduced surname still
                works. Required to build the ``paperhash``.

        Returns:
            List of OpenReview note dicts (never ``None``). Empty on a missing
            title/author, or when every host answered and nothing matched.

        Raises:
            SourceUnavailableError: the lookup ended without an answer from at
                least one host and nothing was found.

        **The two hosts are disjoint, and the paperhash lookup runs against
        both.** Counted live under an authenticated session: ICLR 2024 has 0
        notes on v1 and 2,260 on v2, NeurIPS 2024 0 and 4,035, TMLR 0 and 4,639,
        while ICLR 2021 has 860 on v1 and 0 on v2. v1 holds the pre-2023 venues
        and v2 everything from 2023 on, so querying v1 alone can never confirm a
        modern ML paper. A miss is exhaustive only once both have answered.
        (One token authenticates both hosts. The two differ on the ``count``
        parameter -- v2 requires ``count=true`` to return a total, v1 rejects it
        with ``400 AdditionalPropertiesError`` and returns ``count`` by default
        -- which is why no lookup here asks for one.)

        Up to two hashes are issued per host, covering the diacritic-preserving
        and ASCII-folded index forms (see
        :func:`build_openreview_paperhashes`); an unaccented name collapses them
        to one.

        FIX B1: when every paperhash returns 0 notes (author-name spelling
        drift, a title OpenReview stores differently), fall back to
        ``/notes/search?term=`` with the plain title on both hosts so the
        cascade still gets a chance to confirm the venue. A fabricated paper
        still returns 0 notes under every query: the index is closed-world over
        OpenReview-hosted submissions.

        Challenge gate: ``/notes`` answers ``403 ChallengeRequiredError`` to an
        anonymous caller on both hosts, so without credentials the paperhash
        lookup cannot run. The refusal is latched per host, and every later
        entry fails that host without issuing a request: 68 of 68 sampled
        lookups in one screening run were refused alike, and re-asking a gated
        endpoint per entry buys a round trip and a circuit-breaker tick for an
        answer that is not coming. The failure is still raised per entry, so a
        refused OpenReview lookup keeps blocking the exhaustive ``not_found``
        claim exactly as any other failed lookup does. Credentials through
        :class:`~bibtex_updater.utils.OpenReviewAuth` restore the exact lookup.
        """
        per_page = max(1, min(int(limit), MAX_TOP_K))
        paperhashes = build_openreview_paperhashes(title or "", first_author or "")
        # The first lookup that ended without an answer. Held rather than raised
        # so a host that IS answering still gets its chance to confirm the
        # paper; re-raised at the end when nothing was found, which is what
        # keeps a half-answered miss out of the exhaustive ``not_found`` claim.
        failure: SourceUnavailableError | None = None

        for host in self.NOTES_HOSTS:
            notes_url = f"{host}/notes"
            if self._is_refused(notes_url):
                # Already gated this run: record the failure without a request.
                failure = failure or SourceUnavailableError(
                    "openreview",
                    notes_url,
                    "challenge-gated for anonymous callers; set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD",
                    transport_failure=False,
                    status_code=403,
                )
                continue
            for paperhash in paperhashes:
                try:
                    notes = self._fetch({"paperhash": paperhash, "limit": per_page}, url=notes_url)
                except SourceUnavailableError as exc:
                    self._latch_refusal(notes_url, exc)
                    failure = failure or exc
                    # This host is out for this entry; the other one may answer.
                    break
                if notes:
                    return notes

        # FIX B1 fallback: term= search against the LaTeX-stripped title.
        # Gated to require first_author (and a built paperhash) so this only
        # fires when paperhash was attempted and missed; never on author-less
        # searches where the cascade already gave up. Also skipped when a host
        # failed to answer: an anonymous run is challenge-gated on every
        # ``/notes`` request, and ``/notes/search`` is paced at 5 requests per
        # minute, so walking it per entry would cost hours for a lookup the
        # failure has already disqualified from claiming ``not_found``.
        if failure is None and first_author and paperhashes:
            plain_title = latex_to_plain(title or "").strip()
            if plain_title:
                term_params = {"term": plain_title, "limit": per_page}
                for host in self.NOTES_HOSTS:
                    try:
                        notes = self._fetch(term_params, url=f"{host}/notes/search")
                    except SourceUnavailableError as exc:
                        failure = failure or exc
                        continue
                    if notes:
                        return notes

        if failure is not None:
            raise failure
        return []

    def _fetch(self, params: dict[str, Any], url: str | None = None) -> list[dict[str, Any]]:
        """Issue a single OpenReview request, return the note list.

        ``url`` defaults to the legacy v1 ``/notes`` endpoint; the term-search
        fallbacks pass the v1 and v2 ``/notes/search`` URLs instead (all three
        answer ``{"notes": [...]}``). Preserves
        shared-HttpClient-vs-bare-httpx routing (rate limiting, caching and
        OpenReview credentials on the shared path; the bare-``httpx`` fallback
        is anonymous and hermetic by design).

        Returns ``[]`` only when OpenReview answered and reported no notes; a
        lookup that ends without an answer raises
        :class:`SourceUnavailableError` (see :meth:`OpenAlexClient._fetch`).
        """
        if url is None:
            url = f"{OPENREVIEW_API}/notes"
        # OpenReview paces its two endpoint families very differently: 180
        # requests per minute on ``/notes``, against 5 on the v1 ``/notes/search``
        # and 20 on the v2 one. They stay one source for the circuit breaker and
        # for ``sources_failed``, and get separate rate limiters.
        rate_limit_service = OPENREVIEW_SEARCH_SERVICE if url.endswith("/notes/search") else OPENREVIEW_SERVICE
        try:
            if self.http is not None and hasattr(self.http, "_request"):
                resp = self.http._request(
                    "GET",
                    url,
                    params=params,
                    accept="application/json",
                    service=OPENREVIEW_SERVICE,
                    rate_limit_service=rate_limit_service,
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(url, params=params)
            raise_for_failed_lookup("openreview", url, resp.status_code)
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
        except Exception as exc:
            raise as_source_failure("openreview", url, exc) from exc
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


# OpenReview acceptance-status classification. OpenReview hosts a submission
# whatever its outcome, so presence alone is not publication: the venue/venueid
# separate an accepted paper from a rejected/withdrawn/under-review one and from a
# preprint mirror (CoRR). Gates resolution (ACCEPTED only) and powers the
# verification existence + unpublished-at-claimed-venue signals.
OR_ACCEPTED = "accepted"
OR_NOT_ACCEPTED = "not_accepted"
OR_PREPRINT = "preprint"
OR_UNKNOWN = "unknown"

_OR_NOT_ACCEPTED_RE = re.compile(r"withdrawn|rejected|desk[_-]?reject", re.IGNORECASE)
#: Venue ids under which OpenReview hosts SELF-CLAIMED metadata rather than a
#: submission it ran: ``Public_Article`` is what an author's ORCID/Crossref
#: profile import writes (251,508 notes live), ``Archive`` is a self-uploaded
#: record (26,809). Their ``venue`` strings look exactly like an accepted
#: paper's -- "WWW 2026", "Information Sciences" -- so the year rule below would
#: read them as acceptance. OpenReview did not review these and cannot vouch for
#: them, so they never confirm a venue. This matters more now that the paperhash
#: lookup reaches v2, where the whole self-claimed pool lives.
_OR_SELF_CLAIMED_VENUEID_RE = re.compile(r"^openreview\.net/(?:public_article|archive)\b", re.IGNORECASE)
_OR_DBLP_CONF_RE = re.compile(r"dblp\.org/conf/", re.IGNORECASE)
_OR_NATIVE_CONF_RE = re.compile(r"\.cc/\d{4}/conference\b", re.IGNORECASE)
_OR_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def openreview_acceptance(note: dict[str, Any]) -> str:
    """Classify an OpenReview note's acceptance status from ``venue``/``venueid``.

    Returns one of :data:`OR_ACCEPTED`, :data:`OR_NOT_ACCEPTED`,
    :data:`OR_PREPRINT`, or :data:`OR_UNKNOWN`. OpenReview hosts a submission
    whatever its outcome, so presence is not publication: the venue strings
    separate an accepted paper ("ICLR 2021", ``dblp.org/conf/ICLR/2021``) from a
    rejected/withdrawn/under-review one (``…/Withdrawn_Submission``, ``…/Rejected``,
    "Submitted to ICLR 2024") and from a preprint mirror (CoRR). ``UNKNOWN`` is
    returned when the status cannot be determined; callers treat it conservatively
    (never a resolution, never a problematic verdict).
    """
    content = note.get("content") or {}
    if not isinstance(content, dict):
        return OR_UNKNOWN
    # Coerced because OpenReview returns these as a LIST on some notes; the
    # raw value used to reach ``is_preprint_venue`` below and raise
    # ``'list' object has no attribute 'lower'``.
    venue = _coerce_venue_string(_content_value(content, "venue"))
    venueid = _coerce_venue_string(_content_value(content, "venueid"))
    venue_s = venue or ""
    venueid_l = (venueid or "").lower()
    venue_l = venue_s.lower()

    # 0. Self-claimed profile metadata -- OpenReview is a mirror here, not a
    #    registry, so it can say nothing about acceptance.
    if _OR_SELF_CLAIMED_VENUEID_RE.match(venueid_l):
        return OR_UNKNOWN
    # 1. Preprint mirror (CoRR / arXiv) -- not a publication.
    if is_preprint_venue(venue) or "journals/corr" in venueid_l:
        return OR_PREPRINT
    # 2. Not accepted -- withdrawn / rejected / still under review.
    if _OR_NOT_ACCEPTED_RE.search(venueid_l) or venue_l.startswith("submitted to"):
        return OR_NOT_ACCEPTED
    # 3. Accepted -- a DBLP conference import, a native ``…/Conference`` venueid,
    #    or a clean venue string carrying a year and none of the markers above.
    if _OR_DBLP_CONF_RE.search(venueid_l) or _OR_NATIVE_CONF_RE.search(venueid_l):
        return OR_ACCEPTED
    if venue_s and _OR_YEAR_RE.search(venue_s):
        return OR_ACCEPTED
    return OR_UNKNOWN


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

    # OpenReview returns ``venue``/``venueid`` as a LIST on some notes, and the
    # downstream venue helpers all assume a string -- an uncoerced list reached
    # ``is_preprint_venue`` and raised ``'list' object has no attribute 'lower'``,
    # which the per-entry handler swallowed, silently dropping the entry from the
    # report. Coerce to the first usable string, mirroring the defensive handling
    # the neighbouring title/author/year fields already get.
    raw_venueid = _coerce_venue_string(_content_value(content, "venueid"))
    venue = _coerce_venue_string(_content_value(content, "venue") or raw_venueid)
    # A preprint-labelled venue (e.g. a DBLP "CoRR" import surfaced on OpenReview)
    # is not a published-venue confirmation; drop it so the verifier treats the
    # venue as unconfirmable rather than matching against "CoRR".
    if is_preprint_venue(venue):
        venue = None
    # Same treatment, for the same reason, for a self-claimed ORCID/Crossref
    # profile import: its "WWW 2026" is the author's own assertion, so it must
    # not confirm an entry's venue claim (and the year recovered from it below
    # must not confirm the year either). The record can still corroborate title
    # and authors, which is all OpenReview actually knows here.
    if _OR_SELF_CLAIMED_VENUEID_RE.match((raw_venueid or "").lower()):
        venue = None

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
        order_reliable=True,
        acceptance=openreview_acceptance(note),
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
