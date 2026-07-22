"""Shared bibliographic utilities for BibTeX tools.

This module provides common functionality used by:
- bibtex_updater.py (preprint-to-published resolver)
- reference_fact_checker.py (bibliographic validation)

Includes text normalization, author parsing, DOI/arXiv handling,
HTTP infrastructure with caching and rate limiting, and API client utilities.
"""

from __future__ import annotations

import collections
import errno
import html
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when a service's circuit breaker is open, to skip the request fast.

    Source clients catch this via their broad ``except`` and return an empty
    result, so a throttled service stops being hammered while the cascade falls
    through to the next source.
    """

    def __init__(self, service: str) -> None:
        super().__init__(f"circuit open for service {service!r}")
        self.service = service


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

ARXIV_HOST_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>[^\s?#]+)", re.IGNORECASE)

PREPRINT_HOSTS = ("arxiv", "biorxiv", "medrxiv")

# DBLP labels arXiv mirrors "CoRR" (Computing Research Repository) -- a preprint
# venue, not a real publication venue. Match it on a word boundary so real venues
# that merely contain the letters (e.g. "Corrosion Science") are not rejected.
_CORR_VENUE_RE = re.compile(r"\bcorr\b", re.IGNORECASE)


def is_preprint_venue(venue: str | None) -> bool:
    """True if ``venue`` names a preprint host (arXiv/bioRxiv/medRxiv) or CoRR.

    Single source of truth for the "is this a real published venue?" check,
    shared by the resolver credibility gate and the source converters so a
    preprint-labelled record -- e.g. an OpenReview note whose venue is
    "CoRR 2017" -- can never be mistaken for a published venue.
    """
    if not venue:
        return False
    v = venue.lower()
    if any(host in v for host in PREPRINT_HOSTS):
        return True
    return bool(_CORR_VENUE_RE.search(v))


def retry_after_seconds(exc: httpx.HTTPError, fallback: float, cap: float = 60.0) -> float:
    """Retry sleep: honor a server ``Retry-After`` header when present (integer
    seconds, capped at ``cap``), else the exponential ``fallback``. Lets the
    client respect DBLP/Crossref 429/503 backoff requests instead of hammering;
    the caller adds jitter."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return min(float(int(header)), cap)
            except (ValueError, TypeError):
                pass
    return fallback


# API endpoints
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
DBLP_API_SEARCH = "https://dblp.org/search/publ/api"
DBLP_API_VENUE_SEARCH = "https://dblp.org/search/venue/api"
S2_API = "https://api.semanticscholar.org/graph/v1"
ACL_ANTHOLOGY_URL = "https://aclanthology.org"
ACL_DOI_PREFIX = "10.18653/v1/"
ACL_ANTHOLOGY_ID_RE = re.compile(r"https?://aclanthology\.org/([A-Z0-9][\w.-]+?)(?:\.pdf|\.bib)?/?$", re.IGNORECASE)
OPENALEX_API = "https://api.openalex.org"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
# Legacy OpenReview API. The v2 host (api2.openreview.net) does NOT serve the
# ``paperhash`` exact-match filter, but the legacy ``api.openreview.net/notes``
# endpoint does -- and that is the authoritative title+first-author lookup the
# cascade relies on. Public read is keyless.
OPENREVIEW_API = "https://api.openreview.net"
# OpenReview API v2. Venues that migrated to v2 (ICLR 2024+, NeurIPS 2023+,
# most 2024+ venues) are INVISIBLE on the legacy v1 host, so the client falls
# back to ``GET /notes/search?term=<title>`` here when both v1 lookups miss.
# v2 wraps every note content field as ``{"value": ...}`` (the converters
# accept both shapes via ``_content_value``). Public read is keyless.
OPENREVIEW_API_V2 = "https://api2.openreview.net"


# ------------- Atomic File Replace -------------


def atomic_replace(src: str, dst: str) -> None:
    """Atomically move src to dst, with a copy+unlink fallback on cross-device errors.

    os.replace is atomic on the same filesystem but raises OSError EXDEV when
    src and dst live on different filesystems — common when $TMPDIR is on
    tmpfs and dst is on disk. In that case we fall back to a non-atomic
    shutil.copyfile + os.unlink.
    """
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV:
            shutil.copyfile(src, dst)
            os.unlink(src)
        else:
            raise


# ------------- Text Normalization -------------


def safe_lower(x: str | None) -> str:
    """Null-safe lowercase and strip."""
    return (x or "").lower().strip()


# Letters with no NFKD decomposition (no combining mark) that should still fold
# to an ASCII base so the SAME name matches across sources, e.g. "Reiß" -> "reiss".
# This only equates variant spellings of one name; it never merges distinct names.
_NONDECOMPOSING_FOLD = str.maketrans(
    {
        "ß": "ss",
        "ẞ": "ss",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ł": "l",
        "Ł": "L",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "TH",
        "ı": "i",
        "ŧ": "t",
    }
)


def strip_diacritics(text: str) -> str:
    """Remove diacritics from text (e.g., 'café' -> 'cafe').

    Also folds letters that lack an NFKD decomposition (ß, ø, ł, æ, ...) to an
    ASCII base, so variant spellings of the same name produce the same key.
    """
    nfkd = unicodedata.normalize("NFKD", text.translate(_NONDECOMPOSING_FOLD))
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(\s*\[[^\]]*\])?(\s*\{[^}]*\})?")
_LATEX_MATH_RE = re.compile(r"\$[^$]*\$")
_BRACES_RE = re.compile(r"[{}]")

# LaTeX accent macros -> the Unicode COMBINING mark they apply to the next
# character. Decoded before ``_LATEX_CMD_RE`` runs: that regex replaces a
# command with a SPACE, and a letter-named accent macro is separated from its
# base character by whitespace (``{\H u}``), so stripping it blindly split the
# word -- "Heged{\H u}s" became "Heged us", whose surname key is "us". The same
# defect made Erd{\H o}s reduce to "os" and Ak{\c c}ay to "cay", so authors with
# Hungarian, Polish, Turkish/Romanian or Scandinavian names could never match
# their own records in Crossref/DBLP. Punctuation-named macros (``\'``, ``\"``)
# escaped the bug only because ``'``/``"`` are not in ``[a-zA-Z]``.
_LATEX_ACCENTS: dict[str, str] = {
    "`": "̀",  # grave
    "'": "́",  # acute
    "^": "̂",  # circumflex
    "~": "̃",  # tilde
    '"': "̈",  # diaeresis
    "=": "̄",  # macron
    ".": "̇",  # dot above
    "u": "̆",  # breve
    "r": "̊",  # ring above
    "H": "̋",  # double acute (Hungarian o"/u")
    "v": "̌",  # caron
    "d": "̣",  # dot below
    "c": "̧",  # cedilla
    "k": "̨",  # ogonek
    "b": "̱",  # macron below
}

#: LaTeX macros naming a whole glyph rather than an accent. ``strip_diacritics``
#: already folds the resulting characters to ASCII via ``_NONDECOMPOSING_FOLD``,
#: so decoding to real Unicode keeps one folding table instead of two.
_LATEX_GLYPHS: dict[str, str] = {
    "ss": "ß",
    "aa": "å",
    "AA": "Å",
    "ae": "æ",
    "AE": "Æ",
    "oe": "œ",
    "OE": "Œ",
    "dh": "ð",
    "DH": "Ð",
    "th": "þ",
    "TH": "Þ",
    "dj": "đ",
    "DJ": "Đ",
    "l": "ł",
    "L": "Ł",
    "o": "ø",
    "O": "Ø",
    "i": "ı",
    "j": "ȷ",
}

# Longest-first so "\ae" wins over "\a"-prefixes and "\dh" over "\d".
_LATEX_GLYPH_RE = re.compile(
    r"\\(" + "|".join(sorted((re.escape(k) for k in _LATEX_GLYPHS), key=len, reverse=True)) + r")(?![A-Za-z])"
)

# The accented base is any single LETTER, not just ASCII: glyph macros are
# decoded first, so "{\'\i}" (the standard BibTeX spelling of í) reaches this
# pass as "{\'ı}" -- an ASCII-only class silently left it undecoded.
_ACCENT_BASE = r"[^\W\d_]"

# Punctuation-named accents: the base character may follow immediately ("\'a"),
# so no lookahead is possible or needed.
_LATEX_PUNCT_ACCENT_RE = re.compile(
    r"\\([" + re.escape("`'^~\"=.") + r"])\s*(?:\{\s*(" + _ACCENT_BASE + r")\s*\}|(" + _ACCENT_BASE + r"))",
)

# Letter-named accents: require a non-letter after the macro name so "\Huge" is
# not read as "\H" + "uge". The base character may be braced ("\H{u}"), space
# separated ("\H u") or inside a group ("{\H u}").
_LATEX_LETTER_ACCENT_RE = re.compile(
    r"\\(["
    + "".join(c for c in _LATEX_ACCENTS if c.isalpha())
    + r"])(?![A-Za-z])\s*(?:\{\s*("
    + _ACCENT_BASE
    + r")\s*\}|("
    + _ACCENT_BASE
    + r"))",
)


def _apply_latex_accent(match: re.Match[str]) -> str:
    """Compose ``base + combining mark`` into a single precomposed character."""
    combining = _LATEX_ACCENTS[match.group(1)]
    base = match.group(2) or match.group(3) or ""
    # Dotless i/j exist only to carry an accent ("\'{\i}" -> í); the dot returns
    # with the accent, and only the dotted forms compose under NFC.
    base = {"ı": "i", "ȷ": "j"}.get(base, base)
    return unicodedata.normalize("NFC", base + combining)


def decode_latex_accents(text: str) -> str:
    """Decode LaTeX accent and glyph macros to precomposed Unicode.

    ``Heged{\\H u}s`` -> ``Hegedűs``, ``Ak{\\c c}ay`` -> ``Akçay``,
    ``Wa{\\l}{\\k e}sa`` -> ``Wałęsa``. Glyph macros are decoded first so a
    dotless-i base (``\\'{\\i}``) is already a character when the accent applies.
    """
    if "\\" not in text:
        return text
    t = _LATEX_GLYPH_RE.sub(lambda m: _LATEX_GLYPHS[m.group(1)], text)
    t = _LATEX_PUNCT_ACCENT_RE.sub(_apply_latex_accent, t)
    return _LATEX_LETTER_ACCENT_RE.sub(_apply_latex_accent, t)


def latex_to_plain(text: str) -> str:
    """Convert markup to plain text for matching/display.

    Decodes HTML/XML entities first (e.g. ``d&apos;Amore`` -> ``d'Amore``,
    ``A &amp; B`` -> ``A & B``) so DBLP/XML-scraped fields match clean records,
    then decodes LaTeX accent macros to real Unicode, and finally removes the
    remaining LaTeX commands, math, and braces.
    """
    if not text:
        return ""
    text = html.unescape(text)
    t = _LATEX_MATH_RE.sub(" ", text)
    t = decode_latex_accents(t)
    t = _LATEX_CMD_RE.sub(" ", t)
    t = _BRACES_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


#: A ``howpublished`` matching any of these is a web reference or a citation-style
#: marker, not a venue name. Promoting one to a venue claim would invent a claim
#: the entry never made -- and a claim the checker can then "mismatch" against.
_NON_VENUE_HOWPUBLISHED_RE = re.compile(
    r"^\s*(?:\\url\b|\\href\b|https?://|www\.|ftp://|\[online\]|available\b|retrieved\b|accessed\b)",
    re.IGNORECASE,
)


def entry_venue(entry: Mapping[str, Any]) -> str:
    """The venue an entry claims, in descending order of authority.

    ``journal``/``booktitle`` first, then ``howpublished`` -- where ``@misc``
    front matter (editorials, magazine columns) names its journal -- then
    ``series``. Without the ``howpublished`` arm, 24 entries of the Varga
    bibliography reported "No venue claimed", so the venue field matched
    vacuously and the journal name never constrained retrieval.

    A URL-valued ``howpublished`` is a web reference and is NOT a venue claim.
    """
    for field_name in ("journal", "booktitle"):
        value = (entry.get(field_name) or "").strip()
        if value:
            return value
    howpublished = (entry.get("howpublished") or "").strip()
    if howpublished and not _NON_VENUE_HOWPUBLISHED_RE.match(howpublished):
        return howpublished
    return (entry.get("series") or "").strip()


def entry_authors(entry: Mapping[str, Any]) -> str:
    """The people an entry credits, falling back to ``editor``.

    ``@proceedings`` and ``@book`` name editors rather than authors, so reading
    ``author`` alone produced an EMPTY author list for every volume-level entry
    -- which both starves the search query and makes the author comparison
    non-comparable.
    """
    author = (entry.get("author") or "").strip()
    if author:
        return author
    return (entry.get("editor") or "").strip()


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


#: Generational/lineage suffixes that trail a surname ("John Smith Jr.",
#: "Forsyth III"). Like single-letter initials and 4-digit DBLP homonym
#: suffixes, they are NOT the distinctive family token and must be dropped
#: before taking the last token -- otherwise "John Smith Jr." reduces to "jr"
#: and spuriously MISMATCHES the suffix-less "John Smith" of the same author.
#: Multi-character only: a bare "V"/"I" generational numeral is already removed
#: by the single-letter rule, and listing it here could eat a real initial.
_GENERATIONAL_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv"})


def _reduce_trailing_to_surname(tokens: list[str]) -> list[str]:
    """Drop trailing non-surname tokens so the last token is the family name.

    Strips, from the end: 4-digit DBLP homonym suffixes ("Sun 0020"), trailing
    single-letter initials ("Mallikarjun B. R." -> the naive last token would be
    "r"), and generational suffixes ("John Smith Jr." -> "jr"). Guarded by
    ``len > 1`` so a name that is *only* an initial/number/suffix is never
    emptied. Shared by :func:`last_name_from_person` and
    :func:`_normalize_surname_key` so the entry side and the authoritative
    record side reduce IDENTICALLY (the symmetry the whole comparison relies on).
    """
    while len(tokens) > 1 and (
        re.fullmatch(r"\d{4}", tokens[-1]) or len(tokens[-1]) == 1 or tokens[-1] in _GENERATIONAL_SUFFIXES
    ):
        tokens.pop()
    return tokens


def last_name_from_person(name: str) -> str:
    """Extract a comparable surname key from a person name.

    Handles both 'Family, Given' and 'Given Family' formats, reducing a
    multi-token family (including nobiliary particles like "van den") to its
    final, most distinctive token. This keeps the key symmetric regardless of
    citation style -- both "van den Oord, Aaron" and "Aaron van den Oord"
    reduce to "oord" -- as long as callers run the entry side and the API-record
    side through this same function.
    """
    name = latex_to_plain(name)
    if "," in name:
        last = name.split(",", 1)[0].strip()
    else:
        last = name.strip()
    last = strip_diacritics(last).lower()
    last = re.sub(r"[^a-z0-9\s-]", "", last).strip()

    tokens = _reduce_trailing_to_surname(last.split())
    if tokens:
        last = tokens[-1]
    return last


def _normalize_surname_key(family: str) -> str:
    """Normalize an AUTHORITATIVE structured ``family`` field to a comparison key.

    The key insight versus :func:`last_name_from_person` is the INPUT, not the
    reduction. ``last_name_from_person`` takes a *flattened* "Given Family"
    string and must guess which token is the surname by taking the LAST one --
    that guess is wrong for family-first CJK names ("Chen Xing" -> "xing"). Here
    the input is ALREADY just the family component the source split out, so we
    never have to strip a given name. We apply the SAME character normalization
    and the SAME trailing-token reduction the entry side uses (drop 4-digit DBLP
    homonym suffixes and single-letter initials, then keep the last token), so a
    multi-token Western family ("van den Oord") still reduces to "oord" -- which
    is exactly what the entry side ``last_name_from_person`` produces for
    "Aaron van den Oord". A single-token CJK family ("Chen") is a no-op and
    stays "chen". This makes the authoritative side symmetric with the entry
    side WITHOUT inheriting the family-first misparse.
    """
    family = latex_to_plain(family or "")
    # A structured family may itself arrive comma-first ("Chen, ...") on rare
    # malformed inputs; take the part before the comma defensively.
    if "," in family:
        family = family.split(",", 1)[0].strip()
    key = strip_diacritics(family).lower()
    key = re.sub(r"[^a-z0-9\s-]", "", key).strip()
    # Drop trailing junk (4-digit DBLP suffix / single-letter initial /
    # generational suffix) then take the distinctive last family token --
    # identical reduction to ``last_name_from_person`` (same shared helper),
    # but applied to a family-only string so the lead given name of a CJK name
    # can never masquerade as the surname.
    tokens = _reduce_trailing_to_surname(key.split())
    if tokens:
        return tokens[-1]
    return ""


def _given_initial_from_full(name: str) -> str:
    """First alphabetic initial of the GIVEN name in a flat person string.

    Handles 'Family, Given' (given = after the comma) and 'Given ... Family'
    (given = everything but the last token). Returns '' when no given name is
    present (mononym / surname-only).
    """
    name = latex_to_plain(name or "").strip()
    if not name:
        return ""
    if "," in name:
        given = name.split(",", 1)[1].strip()
    else:
        toks = name.split()
        given = " ".join(toks[:-1]) if len(toks) > 1 else ""
    given = strip_diacritics(given).lower().strip()
    for ch in given:
        if ch.isalpha():
            return ch
    return ""


def _given_initial(given: str) -> str:
    """First alphabetic initial of an already-isolated given-name string."""
    g = strip_diacritics(latex_to_plain(given or "")).lower().strip()
    for ch in g:
        if ch.isalpha():
            return ch
    return ""


def record_looks_alphabetized(record: PublishedRecord) -> bool:
    """True if the record's author list looks SORTED rather than publication order.

    Some order-reliable sources alphabetize their contributor lists instead of
    preserving title-page order (Crossref NeurIPS/ICML proceedings deposits,
    e.g. the 10.52202 prefix) -- sometimes by family name, sometimes by the
    full given-first display string ("Anh Tuan Tran" < "Khoi Nguyen" < "Quang
    Ho Nguyen" < "Truong Thanh Vu"). Such a record's author ORDER carries no
    publication-order signal, so order-sensitive author checks (shared-surname
    run comparison, positional given-name pairing across a repeated surname,
    same-multiset swap detection) must not anchor on it. Mirrors
    ``matching._looks_alphabetized`` (surname keys) and extends it with the
    display-string sort that surname keys cannot see. Requires >= 3 usable
    names: with fewer, sorted order coincides too often to carry any signal.
    """
    authors = record.authors or []
    if len(authors) < 3:
        return False
    surnames = [k for k in record.surname_keys(limit=10_000) if k]
    if len(surnames) >= 3 and surnames == sorted(surnames):
        return True
    display: list[str] = []
    for a in authors:
        name = f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
        key = strip_diacritics(latex_to_plain(name)).lower()
        key = re.sub(r"[^a-z0-9\s]", " ", key)
        key = " ".join(key.split())
        if key:
            display.append(key)
    return len(display) >= 3 and display == sorted(display)


def same_surname_given_order_violation(entry_author_field: str, record: PublishedRecord) -> bool:
    """Detect a swap/mismatch between two authors who share a surname.

    Surname-only matching is blind to a swap of two co-authors with the SAME
    surname (e.g. 'Yang Song' <-> 'Jiaming Song'): both reduce to 'song', so the
    ordered surname lists look identical. When the matched record is from an
    order-preserving source (``order_reliable``), compare the GIVEN-name initials
    of each shared-surname run, in document order; a difference is a real
    author-order/identity corruption the surname check cannot see.

    Conservative by construction -- fires ONLY when, for a surname appearing >= 2
    times, BOTH sides have the SAME count (aligned runs) AND every author in the
    run has a usable given initial on both sides. A dropped/added same-surname
    author (unequal counts) or any missing given name leaves this to the
    surname-level logic instead. A record whose author list looks ALPHABETIZED
    (``record_looks_alphabetized``) never anchors this check: alphabetization
    re-orders shared-surname runs (the two 'Nguyen's of an A-Z-sorted Crossref
    proceedings deposit need not follow publication order), so a run-order
    difference against it is a record-side sort artifact, not a swap.
    """
    if not getattr(record, "order_reliable", False) or not record.authors:
        return False
    if record_looks_alphabetized(record):
        return False

    entry_pairs = [
        (last_name_from_person(n), _given_initial_from_full(n)) for n in split_authors_bibtex(entry_author_field)
    ]
    rec_pairs: list[tuple[str, str]] = []
    for a in record.authors:
        family = (a.get("family") or "").strip()
        if record.structured_names and family:
            surname = _normalize_surname_key(family)
        else:
            surname = last_name_from_person(f"{a.get('given', '') or ''} {family}".strip())
        rec_pairs.append((surname, _given_initial(a.get("given", ""))))

    from collections import Counter

    entry_counts = Counter(sk for sk, _ in entry_pairs if sk)
    rec_counts = Counter(sk for sk, _ in rec_pairs if sk)
    for surname, n in entry_counts.items():
        if n < 2 or rec_counts.get(surname, 0) != n:
            continue  # not a shared-surname run aligned on both sides
        entry_initials = [gi for sk, gi in entry_pairs if sk == surname]
        rec_initials = [gi for sk, gi in rec_pairs if sk == surname]
        if all(entry_initials) and all(rec_initials) and entry_initials != rec_initials:
            return True
    return False


# Graded given-name comparison (runs only after surnames already align). Each
# tier is a benign-convention equivalence except the last; only a full-vs-full
# incompatible first given token escalates. Tier -> verdict class is in
# GIVEN_VARIETY_CLASS below.
class GivenNameVariety:
    EXACT = "given_exact"
    DIACRITIC_HYPHEN = "given_diacritic_hyphen_variant"
    INITIAL_COMPATIBLE = "given_initial_form"
    ABBREVIATION = "given_abbreviation_variant"
    MIDDLE_NAME = "given_middle_name_delta"
    TRANSLITERATION = "given_transliteration_variant"
    NICKNAME = "given_nickname_variant"
    INITIAL_CONFLICT = "given_initial_conflict"
    SUBSTITUTION = "given_name_substitution"
    NON_COMPARABLE = "given_non_comparable"


#: Verdict class per variety. "confirmed" = benign, verdict unchanged (note only);
#: "soften" = low-confidence, route to could-not-verify with a note (never a hard
#: flag); "escalate" = genuine substitution -> PROBLEMATIC; "skip" = nothing to say.
GIVEN_VARIETY_CLASS: dict[str, str] = {
    GivenNameVariety.EXACT: "confirmed",
    GivenNameVariety.DIACRITIC_HYPHEN: "confirmed",
    GivenNameVariety.INITIAL_COMPATIBLE: "confirmed",
    GivenNameVariety.ABBREVIATION: "confirmed",
    GivenNameVariety.MIDDLE_NAME: "confirmed",
    GivenNameVariety.TRANSLITERATION: "soften",
    GivenNameVariety.NICKNAME: "soften",
    GivenNameVariety.INITIAL_CONFLICT: "soften",
    GivenNameVariety.SUBSTITUTION: "escalate",
    GivenNameVariety.NON_COMPARABLE: "skip",
}

#: Close-spelling / romanization variants of a first given token within this edit
#: distance are treated as a low-confidence TRANSLITERATION (softened, never
#: escalated), so romanizations like "Sergey"/"Serguei" or "Aleksandr"/"Alexander"
#: are not hard-flagged. A genuine different name (Yue/Yujing, Ramon/Rafael) is
#: well beyond this. Tunable via the empirical FP check on the HALLMARK valid set.
_GIVEN_SOFTEN_MAX_EDIT = 3

#: Curated, finite hypocorism (nickname) groups; any two members are treated as a
#: benign NICKNAME variant. Deliberately conservative.
_NICKNAME_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"william", "will", "bill", "billy"}),
    frozenset({"robert", "rob", "bob", "bobby"}),
    frozenset({"richard", "rich", "rick", "dick"}),
    frozenset({"james", "jim", "jimmy"}),
    frozenset({"john", "jack", "johnny"}),
    frozenset({"joseph", "joe", "joey"}),
    frozenset({"thomas", "tom", "tommy"}),
    frozenset({"michael", "mike", "mikey"}),
    frozenset({"charles", "charlie", "chuck"}),
    frozenset({"christopher", "chris"}),
    frozenset({"matthew", "matt"}),
    frozenset({"anthony", "tony"}),
    frozenset({"daniel", "dan", "danny"}),
    frozenset({"david", "dave"}),
    frozenset({"edward", "ed", "eddie", "ted"}),
    frozenset({"alexander", "alex", "sasha", "aleksandr"}),
    frozenset({"benjamin", "ben"}),
    frozenset({"samuel", "sam"}),
    frozenset({"nicholas", "nick"}),
    frozenset({"andrew", "andy", "drew"}),
    frozenset({"katherine", "kate", "katie", "kathy"}),
    frozenset({"elizabeth", "liz", "beth", "betty"}),
    frozenset({"margaret", "maggie", "meg", "peggy"}),
)


def _given_tokens(given: str) -> list[str]:
    """Normalize a given-name string to comparison tokens: LaTeX-decoded,
    diacritic-folded, lowercased, hyphens/dots split to spaces."""
    g = strip_diacritics(latex_to_plain(given or "")).lower()
    g = re.sub(r"[.\-]", " ", g)
    g = re.sub(r"[^a-z0-9\s]", " ", g)
    return [t for t in g.split() if t]


def _is_initial_tokens(tokens: list[str]) -> bool:
    return bool(tokens) and all(len(t) == 1 for t in tokens)


def _nickname_equiv(a: str, b: str) -> bool:
    return any(a in grp and b in grp for grp in _NICKNAME_GROUPS)


def _deglue_initials(given: str) -> str:
    """Expand a glued, separator-less all-caps initial run into spaced initials.

    PubMed/biomedical sources write given names as concatenated initials with no
    dots or spaces -- "ME" for *Maria Elisabetta*, "RMF" for *Robin Maria
    Francisca*. Without this, ``_given_tokens("ME")`` yields the single 2-char
    token ``["me"]``, which is not all-length-1 so it escapes the initials branch
    of :func:`classify_given_pair` and is graded as a *full* given token -- "me"
    vs "maria" then escalates to a spurious SUBSTITUTION (a false author flag).

    Case is the disambiguator: only an all-uppercase 2-5 letter run with no
    separator is treated as glued initials, so a real short given ("Bo", "Wei",
    mixed-case) is left untouched. The conservative failure mode if a genuinely
    all-caps short name ever slips through is INITIAL_CONFLICT (-> soften/abstain),
    never a false flag. Returns ``given`` unchanged when it is not such a run.
    """
    s = strip_diacritics(latex_to_plain(given or "")).strip()
    if re.fullmatch(r"[A-Z]{2,5}", s):
        return " ".join(s)
    return given


def classify_given_pair(given_entry: str, given_record: str) -> str:
    """Classify a single (entry, record) given-name pair into a GivenNameVariety.

    First-hit-wins cascade. Only SUBSTITUTION (two FULL, non-initial first given
    tokens that are neither a nickname pair nor within a small edit distance)
    escalates; everything else is a benign convention or a low-confidence variant.

    Glued separator-less initial runs ("ME", "RMF") are expanded to spaced
    initials first (see :func:`_deglue_initials`) so the initials branch handles
    them instead of mis-grading them as full given tokens.
    """
    te, tr = _given_tokens(_deglue_initials(given_entry)), _given_tokens(_deglue_initials(given_record))
    if not te or not tr:
        return GivenNameVariety.NON_COMPARABLE
    if te == tr:
        return GivenNameVariety.EXACT
    # Separator/diacritic fold: "Jun-Yan" vs "Junyan", "Jun Yan" vs "Jun-Yan".
    if "".join(te) == "".join(tr):
        return GivenNameVariety.DIACRITIC_HYPHEN
    # Initials on either side: an initials-only string is a benign abbreviation
    # of the other ONLY if its letters are the leading initials in order.
    if _is_initial_tokens(te) or _is_initial_tokens(tr):
        short, long = (te, tr) if _is_initial_tokens(te) else (tr, te)
        if len(short) <= len(long) and all(short[i] == long[i][0] for i in range(len(short))):
            return GivenNameVariety.INITIAL_COMPATIBLE
        return GivenNameVariety.INITIAL_CONFLICT
    # MIXED initial+name forms: a given like "J. Westerborn" is NOT all-initials
    # (so the branch above does not catch it) but its FIRST token is still a
    # bare initial. Comparing that single letter against a full first given
    # token ("Johan") as if both were full names mis-graded the pair as a
    # SUBSTITUTION (HALLMARK FP: "Johan Alenlov" vs a record's "J. Westerborn
    # Alenlov"). A one-letter first token can only ever testify about the
    # initial: same letter -> benign initial form; different letter -> the same
    # low-confidence INITIAL_CONFLICT the all-initials branch yields. Never a
    # full-name substitution.
    if len(te[0]) == 1 or len(tr[0]) == 1:
        if te[0][0] == tr[0][0]:
            return GivenNameVariety.INITIAL_COMPATIBLE
        return GivenNameVariety.INITIAL_CONFLICT
    # Both sides carry a full (non-initial) given name from here.
    if te[0] == tr[0]:
        # Same first given token, differing middle/extra tokens -> middle-name
        # presence/absence (benign): "Diederik P." vs "Diederik".
        return GivenNameVariety.MIDDLE_NAME
    # Abbreviation/diminutive: one full first token is a leading character prefix
    # of the other ("Tim" of "Timothy", "Chris" of "Christopher", "Dan" of
    # "Daniel") -- a shortened form, not a different person. Genuine substitutions
    # (Yue/Yujing, Ramon/Rafael) are NOT prefixes of each other, so they still
    # escalate. Both sides require >= 2 chars (bare initials are Tier 2 already).
    if len(te[0]) >= 2 and len(tr[0]) >= 2 and (te[0].startswith(tr[0]) or tr[0].startswith(te[0])):
        return GivenNameVariety.ABBREVIATION
    if _nickname_equiv(te[0], tr[0]):
        return GivenNameVariety.NICKNAME
    if Levenshtein.distance(te[0], tr[0]) <= _GIVEN_SOFTEN_MAX_EDIT:
        return GivenNameVariety.TRANSLITERATION
    return GivenNameVariety.SUBSTITUTION


def given_name_position_audit(entry_author_field: str, record: PublishedRecord) -> tuple[str, list[dict]]:
    """Grade given names at every surname-confirmed position against an
    order-preserving, structured record. Returns ``(worst_class, findings)`` where
    worst_class is one of "escalate" / "soften" / "confirmed" / "skip" and findings
    is a per-position list of ``{position, variety, entry_given, record_given}``.

    Gated on ``order_reliable`` (Crossref/OpenAlex/DBLP/OpenReview). Note it does
    NOT require ``structured_names``: a leaked hallucinated author is far worse
    than a spurious flag, and a genuine substitution (e.g. d67418 'Yue'->'Yujing'
    Zhao) often surfaces only via a DBLP/OpenAlex record whose given/family split
    is synthesized. Several guards keep the synthesized split from causing false
    positives: (1) a position is audited only where its surname is already
    positionally confirmed AND (2) that surname is UNIQUE on both sides -- so a
    repeated surname (two co-authors named "Liu") or a non-publication record
    order (some sources alphabetize) can never cross-pair two different
    same-surname authors; and (3) the given-name cascade folds initials,
    diacritics, hyphenation and middle names before any escalation, so only a
    full-vs-full incompatible leading given token escalates.
    """
    if not getattr(record, "order_reliable", False) or not record.authors:
        return "skip", []

    entry_names = split_authors_bibtex(entry_author_field)
    entry_pairs = [(last_name_from_person(n), _entry_given_of(n)) for n in entry_names]
    # Surname-key derivation mirrors ``PublishedRecord.surname_keys``: a trusted
    # ``family`` is normalized verbatim; an empty family (Crossref ``literal``
    # author with no given/family split) falls back to ``last_name_from_person``
    # on the reconstructed full name so the audit still pairs position 0 against
    # a record whose lead author arrived as a literal. Without this fallback the
    # audit silently skipped a real lead-author given-name SUBSTITUTION (Leak B:
    # "Shunyu Zhou" cited where the real lead is "Denny Zhou" and Crossref only
    # had a ``literal`` for the lead).
    rec_pairs: list[tuple[str, str]] = []
    for a in record.authors:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if record.structured_names and family:
            surname_key = _normalize_surname_key(family)
        else:
            surname_key = last_name_from_person(f"{given} {family}".strip())
        rec_pairs.append((surname_key, given))

    from collections import Counter

    entry_sur_counts = Counter(s for s, _ in entry_pairs if s)
    rec_sur_counts = Counter(s for s, _ in rec_pairs if s)

    findings: list[dict] = []
    rank = {"skip": 0, "confirmed": 1, "soften": 2, "escalate": 3}
    worst = "skip"
    record_alphabetized = record_looks_alphabetized(record)
    for i in range(min(len(entry_pairs), len(rec_pairs))):
        e_sur, e_giv = entry_pairs[i]
        r_sur, r_giv = rec_pairs[i]
        if not e_sur or not r_sur or e_sur != r_sur:
            continue  # surname not positionally confirmed here -> not our job
        # Repeated-surname ambiguity guard. The OLD rule -- skip if EITHER side
        # had the surname repeated -- was over-conservative: a true cross-pairing
        # risk only exists when BOTH sides have the surname multiple times (e.g.
        # the canonical 'Yang Song' / 'Jiaming Song' co-authors that both appear
        # in entry AND record, where the positional pairing could be flipped).
        # When ONLY ONE side repeats the surname, the OTHER side's unique
        # occurrence pins the pairing -- and at position 0 (LEAD AUTHOR) the
        # natural pairing is unambiguous regardless. Specifically: the
        # Least-to-Most leak ("Shunyu Zhou" at entry pos 0 vs canonical lead
        # "Denny Zhou") was silently skipped because the entry had TWO 'zhou's
        # (Shunyu at 0, Denny re-listed at the tail) while the record had ONE
        # 'zhou' (Denny at 0). The OLD guard tripped on the entry-side repeat
        # even though the record-side 'zhou' uniquely pinned the comparison.
        rec_repeats = rec_sur_counts[r_sur] > 1
        both_repeat = entry_sur_counts[e_sur] > 1 and rec_repeats
        if both_repeat and i != 0:
            continue
        # Positional-ambiguity guard: when the RECORD carries the surname more
        # than once AND the positional pairing cannot be trusted -- at the lead
        # with both sides repeating (two real 'Song' lead-co-authors), or
        # ANYWHERE against an ALPHABETIZED record (sorting re-orders a shared-
        # surname run, so position i may hold the OTHER same-surname author,
        # e.g. an A-Z Crossref deposit putting 'Khoi Nguyen' where the entry
        # cites 'Quang Nguyen') -- grade the entry given against EVERY record
        # author with that surname instead of the positional one. If ANY of
        # them explains it benignly (or nothing is comparable), abstain; only
        # when every comparable candidate grades as a substitution is the
        # entry's author genuinely none of them -> escalate.
        if (both_repeat and i == 0) or (rec_repeats and record_alphabetized):
            same_surname_givens = [g for s, g in rec_pairs if s == r_sur]
            cand_classes = {GIVEN_VARIETY_CLASS.get(classify_given_pair(e_giv, g), "skip") for g in same_surname_givens}
            if "escalate" not in cand_classes or "confirmed" in cand_classes or "soften" in cand_classes:
                # Some record author with that surname is a benign/low-confidence
                # match (or none is comparable) -> abstain.
                continue
        variety = classify_given_pair(e_giv, r_giv)
        cls = GIVEN_VARIETY_CLASS.get(variety, "skip")
        if cls == "skip":
            continue
        findings.append({"position": i, "variety": variety, "entry_given": e_giv, "record_given": r_giv})
        if rank[cls] > rank[worst]:
            worst = cls
    return worst, findings


def _entry_given_of(name: str) -> str:
    """Best-effort given-name string for a flat entry author ('Given Family' or
    'Family, Given'). Mirrors last_name_from_person's surname extraction."""
    name = latex_to_plain(name or "").strip()
    if not name:
        return ""
    if "," in name:
        return name.split(",", 1)[1].strip()
    toks = name.split()
    return " ".join(toks[:-1]) if len(toks) > 1 else ""


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


def _name_tokens(name: str) -> list[str]:
    """Normalized comparable tokens of a single person name (no last-token pick).

    Splits a "Family, Given" or "Given Family" string into individual
    normalized tokens (diacritics stripped, lowercased, punctuation removed,
    4-digit DBLP suffixes and bare single letters dropped). Used to disambiguate
    which token is the surname when an authoritative family set is available.
    """
    name = latex_to_plain(name or "")
    name = name.replace(",", " ")
    name = strip_diacritics(name).lower()
    name = re.sub(r"[^a-z0-9\s-]", " ", name)
    return [t for t in name.split() if t and len(t) > 1 and not re.fullmatch(r"\d{4}", t)]


def entry_surnames_against_structured(author_field: str, family_keys: set[str], limit: int = 3) -> list[str]:
    """Entry surname keys, disambiguated by an AUTHORITATIVE family-name set.

    The comma-less entry name "Chen Xing" is order-ambiguous: ``last_name_from_person``
    must guess the surname is the LAST token ("xing"), which is wrong for a
    family-first CJK name whose family is "Chen". When we hold an authoritative
    family set (the ``family`` keys of a structured Crossref record for the SAME
    paper), we can resolve the ambiguity *per author*: if exactly one of an
    entry author's normalized tokens is a known family key, that token is the
    surname; otherwise fall back to ``last_name_from_person``.

    This is NOT order-insensitive matching across authors -- each entry author
    still maps to exactly one surname at its own position, and the downstream
    ``symmetric_author_match`` still enforces first-author identity and ordering.
    It only fixes WHICH token within a single ambiguous name is treated as the
    surname, using the authoritative record as the disambiguator. A genuinely
    different author cannot be rescued: none of "John Smith"'s tokens are in a
    {"chen"} family set, so it still reduces to "smith" and still mismatches.

    Args:
        author_field: BibTeX author string.
        family_keys: Set of authoritative normalized family keys (from a
            structured record's ``surname_keys``).
        limit: Max authors to extract.

    Returns:
        List of normalized surname keys, disambiguated where possible.
    """
    authors = split_authors_bibtex(author_field)[:limit]
    resolved: list[str] = []
    for a in authors:
        default_key = last_name_from_person(a)
        # A comma-form name ("Chen, Xing") is already unambiguous -- the family
        # precedes the comma -- so only disambiguate comma-less names.
        if "," not in a and family_keys:
            tokens = _name_tokens(a)
            matches = [t for t in tokens if t in family_keys]
            # Use the authoritative token only when it is UNAMBIGUOUS: exactly
            # one token of this name is a known family key. (Two matches would be
            # ambiguous; zero means this author isn't in the record -> keep the
            # heuristic so a genuine extra/different author still surfaces.)
            if len(matches) == 1:
                resolved.append(matches[0])
                continue
        if default_key:
            resolved.append(default_key)
    return [k for k in resolved if k]


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


#: arXiv DataCite DOI prefix. arXiv mints versioned DOIs
#: (``10.48550/arXiv.2010.11929v1``), but only the *unversioned* DOI
#: (``10.48550/arXiv.2010.11929``) resolves via doi.org -- the versioned form
#: 404s. Stripping the version suffix is therefore safe ONLY for this prefix;
#: other DOIs may legitimately end in a letter+digit token.
_ARXIV_DOI_PREFIX = "10.48550/arxiv."


def normalize_doi_for_resolution(doi: str | None) -> str | None:
    """Normalize a DOI for resolution against doi.org.

    Builds on ``doi_normalize`` (strips URL prefix, lowercases). Additionally,
    for arXiv DataCite DOIs (prefix ``10.48550/arXiv.``) strips a trailing
    version suffix (``v1``, ``v2``, ...): the versioned DOI 404s at doi.org but
    the unversioned one resolves (302). The version strip is scoped to the arXiv
    prefix so non-arXiv DOIs that legitimately end in ``vN`` are left untouched.
    """
    normalized = doi_normalize(doi)
    if not normalized:
        return normalized
    if normalized.startswith(_ARXIV_DOI_PREFIX):
        normalized = re.sub(r"v\d+$", "", normalized)
    return normalized


#: arXiv DataCite DOI shape: ``10.48550/arXiv.YYMM.NNNNN`` with an optional
#: version suffix. Case-insensitive because the ``arXiv`` casing varies across
#: sources. Single source of truth for "this DOI *is* an arXiv ID" -- reused by
#: ``arxiv_id_from_datacite_doi`` and the fact-checker's entry-side extractor.
_ARXIV_DATACITE_DOI_RE = re.compile(r"^10\.48550/arxiv\.(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)


def arxiv_id_from_datacite_doi(doi: str | None) -> str | None:
    """Extract the bare arXiv ID from an arXiv DataCite DOI.

    ``10.48550/arXiv.2301.00001`` (or ``...v2``) -> ``2301.00001``: the version
    suffix is stripped and the ID is month-validated via ``is_valid_arxiv_id``.
    Any other DOI shape (including legacy-scheme arXiv IDs, which DataCite does
    not mint under this prefix in the wild) returns ``None``.
    """
    if not doi:
        return None
    m = _ARXIV_DATACITE_DOI_RE.match(doi.strip())
    if not m:
        return None
    bare = m.group(1)
    return bare if is_valid_arxiv_id(bare) else None


def doi_url(doi: str) -> str:
    """Convert a DOI to a URL."""
    return f"https://doi.org/{doi}"


_ARXIV_NEW_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$")


def is_valid_arxiv_id(arxiv_id: str | None) -> bool:
    """Validate a bare arXiv identifier.

    Modern IDs are ``YYMM.NNNNN`` where ``MM`` is a real month (01-12); a
    number with an impossible month (e.g. a DOI fragment like ``5678.9012``)
    is not a valid arXiv ID. Legacy IDs (``hep-th/9901001``) are accepted
    structurally. Returns False for None/empty.
    """
    if not arxiv_id:
        return False
    m = _ARXIV_NEW_ID_RE.match(arxiv_id.strip())
    if m:
        month = int(m.group(2))
        return 1 <= month <= 12
    # Legacy scheme handled by ARXIV_ID_RE old-style alternative; accept as-is.
    return True


def extract_arxiv_id_from_text(text: str) -> str | None:
    """Extract arXiv ID from a text string (URL, eprint field, note, etc.).

    Modern IDs with an impossible month are rejected so that DOI fragments or
    arbitrary ``NNNN.NNNNN`` numbers are not mistaken for arXiv identifiers.
    """
    if not text:
        return None

    def _normalize(raw: str) -> str | None:
        # Drop a .pdf suffix and version, re-extract the canonical ID, and keep
        # it only if the month is real. Returns None for non-arXiv numbers.
        if raw.lower().endswith(".pdf"):
            raw = raw[:-4]
        idm = ARXIV_ID_RE.search(raw)
        candidate = re.sub(r"v\d+$", "", idm.group("id") if idm else raw)
        return candidate if is_valid_arxiv_id(candidate) else None

    # Prefer the ID from an explicit arxiv.org URL over a coincidental number
    # elsewhere in the text; legacy URLs (.../abs/hep-th/9901001) keep their slash.
    m = ARXIV_HOST_RE.search(text)
    if m:
        candidate = _normalize(m.group("id"))
        if candidate:
            return candidate
    for match in ARXIV_ID_RE.finditer(text):
        candidate = _normalize(match.group("id"))
        if candidate:
            return candidate
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
        self.timestamps: collections.deque[float] = collections.deque()

    def wait(self) -> None:
        """Block until a request can be made within the rate limit."""
        window = 60.0
        while True:
            sleep_for = 0.0
            with self.lock:
                now = time.time()
                cutoff = now - window
                # O(1) popleft from deque (timestamps are chronological)
                while self.timestamps and self.timestamps[0] < cutoff:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.req_per_min:
                    self.timestamps.append(time.time())
                    return  # Slot acquired
                # Need to wait — compute sleep duration but release lock first
                sleep_for = window - (now - self.timestamps[0]) + 0.01
            # Sleep WITHOUT holding the lock; add jitter to prevent
            # synchronized wake-ups across threads (burst-stall pattern)
            if sleep_for > 0:
                time.sleep(sleep_for + random.uniform(0, 0.5))


class RateLimiterRegistry:
    """Manages per-service rate limiters.

    This allows different API services to have their own rate limits,
    optimizing throughput while respecting each service's constraints.
    """

    # Conservative *library* defaults for embedders that never tune limits.
    # The fact-checker CLI overrides these with scaled-and-capped values much
    # closer to the documented service ceilings (see
    # ``fact_checker._cli_service_rate_limits``).
    DEFAULT_LIMITS = {
        "crossref": 50,  # Conservative default; Crossref's polite-pool ceiling is ~50 req/SECOND
        "semanticscholar": 100,  # Keyless S2 is a SHARED global pool; keyed search allowance is ~1 req/s
        "dblp": 30,  # DBLP: 30/min (conservative)
        "arxiv": 20,  # arXiv asks for ~1 request per 3 seconds (politeness)
        "scholarly": 10,  # Scholar: very conservative
        "aclanthology": 30,  # ACL Anthology: 30/min (conservative)
        "openalex": 100,  # Conservative default; OpenAlex polite pool allows ~10 req/sec
        "europepmc": 20,  # Europe PMC: conservative rate limit
        "openreview": 30,  # OpenReview: 30/min (conservative; keyless public read)
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
                    # Getting close to limit, slow down. The read-modify-write
                    # on ``_limits`` must share the lock with the paired
                    # ``_limiters`` swap, or a concurrent adapt can tear them.
                    with self._lock:
                        current_limit = self._limits.get(service, 30)
                        new_limit = max(current_limit // 2, self._min_limits.get(service, 5))
                        if new_limit != current_limit:
                            self._limits[service] = new_limit
                            self._limiters[service] = RateLimiter(new_limit)
            except (ValueError, TypeError):
                pass

        # Handle 429 Too Many Requests
        if hasattr(response, "status_code") and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            backoff_seconds = 60.0  # Default backoff when header absent/unparseable
            if retry_after:
                try:
                    backoff_seconds = float(int(retry_after))
                except (ValueError, TypeError):
                    backoff_seconds = 60.0
            # ``_backoff_until`` is read by ``wait`` on every worker thread, so
            # the write must happen under the registry lock (it was previously
            # unguarded; a torn read could miss/garble a fresh backoff).
            with self._lock:
                self._backoff_until[service] = time.time() + backoff_seconds

            # Also reduce the rate limit (same locking rationale as above).
            with self._lock:
                current_limit = self._limits.get(service, 30)
                new_limit = max(current_limit // 2, self._min_limits.get(service, 5))
                self._limits[service] = new_limit
                self._limiters[service] = RateLimiter(new_limit)

    def wait(self, service: str) -> None:
        """Wait for rate limit, including any backoff period.

        Args:
            service: Name of the API service
        """
        # Check for backoff period (read under the lock; see ``adapt``).
        with self._lock:
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
        with self._lock:
            self._limits[service] = default
            if service in self._limiters:
                self._limiters[service] = RateLimiter(default)


class DiskCache:
    """Thread-safe on-disk JSON cache for API responses."""

    def __init__(self, path: str | None, max_age_days: int = 30) -> None:
        self.path = path
        self.max_age_days = max_age_days
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
            atomic_replace(tmp.name, self.path)

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if not self.path:
            return False
        with self.lock:
            return key in self.data


class SqliteCache:
    """Thread-safe SQLite-backed cache for API responses.

    Drop-in replacement for DiskCache with:
    - Thread-safe concurrent reads/writes via SQLite WAL mode
    - No full-file rewrite on every insert
    - Automatic expiry of stale entries
    """

    def __init__(self, path: str | None, max_age_days: int = 30) -> None:
        """Initialize the SQLite cache.

        Args:
            path: Path to the cache file. If None, caching is disabled.
                 Will use .db extension instead of .json.
            max_age_days: Maximum age of cache entries in days. Default 30 days.
        """
        self.path = path
        self.max_age_days = max_age_days
        self._local = threading.local()

        if path:
            # Convert .json path to .db path if needed
            if path.endswith(".json"):
                self.path = path[:-5] + ".db"

            # Initialize database
            self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        if not self.path:
            return

        # Ensure directory exists
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Create connection with WAL mode for concurrent access
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        if not self.path:
            raise RuntimeError("Cannot create connection: path is None")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Any | None:
        """Get a cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if not self.path:
            return None

        conn = self._get_connection()
        cursor = conn.execute("SELECT value, timestamp FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row is None:
            return None

        value_json, timestamp = row

        # Check expiry
        age_days = (time.time() - timestamp) / 86400
        if age_days > self.max_age_days:
            # Delete expired entry
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None

        try:
            return json.loads(value_json)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any) -> None:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.path:
            return

        conn = self._get_connection()
        value_json = json.dumps(value)
        timestamp = time.time()

        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)", (key, value_json, timestamp)
        )
        conn.commit()

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache (and is not expired).

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired, False otherwise
        """
        if not self.path:
            return False

        conn = self._get_connection()
        cursor = conn.execute("SELECT timestamp FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()

        if row is None:
            return False

        timestamp = row[0]
        age_days = (time.time() - timestamp) / 86400
        if age_days > self.max_age_days:
            # Delete expired entry
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return False

        return True

    def clear_expired(self) -> int:
        """Delete all expired entries from the cache.

        Returns:
            Number of deleted entries
        """
        if not self.path:
            return 0

        conn = self._get_connection()
        cutoff_timestamp = time.time() - (self.max_age_days * 86400)

        cursor = conn.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff_timestamp,))
        deleted_count = cursor.rowcount
        conn.commit()

        return deleted_count

    def close(self) -> None:
        """Close the database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        """Cleanup: close connection when object is destroyed."""
        try:
            self.close()
        except Exception:
            pass


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
        atomic_replace(tmp.name, self.path)

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

    # Versioned envelope marker for cached NON-JSON text responses (e.g. arXiv
    # Atom XML), stored as {"__btu_cache_v2__": true, "ct": <content-type>,
    # "text": <body>}. JSON responses keep the legacy format (the raw decoded
    # body), so existing cache files remain fully readable in both directions.
    CACHE_ENVELOPE_KEY = "__btu_cache_v2__"

    # Circuit breaker: after this many consecutive fully-failed requests to a
    # service (sustained 429/5xx -- typically DBLP/Crossref IP-throttling), stop
    # hammering it for CIRCUIT_COOLDOWN seconds so the throttle can clear instead
    # of bursting into it. State is persisted to the cache for cross-run pacing.
    CIRCUIT_FAIL_THRESHOLD = 4
    CIRCUIT_COOLDOWN = 90.0
    _CIRCUIT_CACHE_KEY = "__bibtex_updater_circuit_open_until__"

    def __init__(
        self,
        timeout: float,
        user_agent: str,
        rate_limiter: RateLimiter | RateLimiterRegistry,
        cache: DiskCache | SqliteCache,
        verbose: bool = False,
        s2_api_key: str | None = None,
    ):
        """Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds
            user_agent: User-Agent header value
            rate_limiter: Either a single RateLimiter (for backward compatibility)
                         or a RateLimiterRegistry for per-service rate limiting
            cache: DiskCache or SqliteCache instance for caching responses
            verbose: Enable verbose logging
            s2_api_key: Optional Semantic Scholar API key for authenticated requests
        """
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        self._rate_limiter = rate_limiter
        self._uses_registry = isinstance(rate_limiter, RateLimiterRegistry)
        self.cache = cache
        self.verbose = verbose
        self.s2_api_key = s2_api_key
        # Per-service circuit-breaker state (see CIRCUIT_* and _request).
        self._circuit_fail_streak: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}
        self._tripped_services: set[str] = set()
        self._load_circuit_state()

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

    # --- Circuit breaker (per service) ---
    @property
    def tripped_services(self) -> set[str]:
        """Services whose circuit opened this run (sustained throttling / 5xx)."""
        return set(self._tripped_services)

    def _load_circuit_state(self) -> None:
        """Load any still-open per-service cooldowns persisted by an earlier run."""
        if not self.cache:
            return
        try:
            data = self.cache.get(self._CIRCUIT_CACHE_KEY)
        except Exception:
            return
        if isinstance(data, dict):
            now = time.time()
            kept: dict[str, float] = {}
            for svc, ts in data.items():
                try:
                    ts_f = float(ts)
                except (ValueError, TypeError):
                    continue
                if ts_f > now:
                    kept[svc] = ts_f
            self._circuit_open_until = kept

    def _persist_circuit_state(self) -> None:
        if not self.cache:
            return
        try:
            self.cache.set(self._CIRCUIT_CACHE_KEY, self._circuit_open_until)
        except Exception:
            pass

    def _circuit_is_open(self, service: str | None) -> bool:
        return bool(service) and time.time() < self._circuit_open_until.get(service, 0.0)

    def _record_service_success(self, service: str | None) -> None:
        if service:
            self._circuit_fail_streak[service] = 0

    def _record_service_failure(self, service: str | None) -> None:
        if not service:
            return
        streak = self._circuit_fail_streak.get(service, 0) + 1
        self._circuit_fail_streak[service] = streak
        if streak >= self.CIRCUIT_FAIL_THRESHOLD and not self._circuit_is_open(service):
            self._circuit_open_until[service] = time.time() + self.CIRCUIT_COOLDOWN
            self._tripped_services.add(service)
            self._persist_circuit_state()
            logger.warning(
                "Service %r is rate-limited/unavailable (sustained 429/5xx); pausing it for %ds. "
                "Entries needing it may be left UNRESOLVED due to throttling, not absence -- "
                "re-run after the cooldown to retry them.",
                service,
                int(self.CIRCUIT_COOLDOWN),
            )

    @staticmethod
    def _is_cacheable_text(content_type: str) -> bool:
        """True for non-JSON *text* bodies worth caching (arXiv Atom XML, ...).

        Conservative allow-list: ``text/*`` plus XML media types. Binary
        responses are never cached (the JSON-valued SqliteCache cannot hold
        them faithfully).
        """
        mime = content_type.split(";", 1)[0].strip().lower()
        return mime.startswith("text/") or mime.endswith("+xml") or mime == "application/xml"

    @staticmethod
    def _cache_key(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        accept: str | None,
        json_body: dict[str, Any] | list[Any] | None,
    ) -> str:
        """Single source of truth for response-cache keys.

        Shared by :meth:`_request` (read/write on real traffic) and
        :meth:`prime_cache` (bulk pre-population) so the two can never drift:
        a primed entry is guaranteed to be the exact key the equivalent
        ``_request`` call would look up.
        """
        return json.dumps({"m": method, "u": url, "p": params, "a": accept, "j": json_body}, sort_keys=True)

    def prime_cache(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
        *,
        value: Any,
    ) -> None:
        """Pre-populate the response cache for an equivalent ``_request`` call.

        ``value`` is stored exactly as a 200 JSON response body would be (the
        legacy raw-decoded-JSON format), so a later ``_request(method, url,
        params=..., accept=..., json_body=...)`` is served from cache without
        touching the network. Public sibling of ``_request`` used by bulk
        prefetchers (e.g. the Semantic Scholar ``/paper/batch`` warm-up). A
        cache-less client makes this a no-op; storage failures are swallowed
        (priming is best-effort by definition).
        """
        if not self.cache:
            return
        try:
            self.cache.set(self._cache_key(method, url, params, accept, json_body), value)
        except Exception:
            pass

    def _adapt_rate_limiter(self, service: str | None, resp: httpx.Response) -> None:
        """Feed a REAL (non-cache-hit) response to an adaptive rate limiter.

        Called once per transport response -- including a retryable 429/5xx
        before the retry exception is raised -- so a server ``Retry-After`` /
        ``X-RateLimit-Remaining`` lands in the registry's backoff state.
        Cache-hit synthetic responses never reach this (the cache
        short-circuits before the request loop). No-op when the configured
        limiter has no ``adapt`` attribute (plain RateLimiter / registry) or
        no service was given.
        """
        if not service:
            return
        adapt = getattr(self._rate_limiter, "adapt", None)
        if adapt is None:
            return
        try:
            adapt(service, resp)
        except Exception:
            # Adaptation is advisory; it must never lose a fetched response.
            logger.debug("rate-limiter adapt() failed for service %r", service, exc_info=True)

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
            cache_key = self._cache_key(method, url, params, accept, json_body)
            cached = self.cache.get(cache_key)
            if cached is not None:
                # v2 envelope: a cached non-JSON text response (arXiv Atom XML,
                # ...). Reconstruct it with its original Content-Type so callers
                # that inspect the header / use .text behave identically.
                if isinstance(cached, dict) and cached.get(self.CACHE_ENVELOPE_KEY) is True:
                    return httpx.Response(
                        200,
                        content=str(cached.get("text") or "").encode("utf-8"),
                        headers={
                            "Content-Type": str(cached.get("ct") or "text/plain"),
                            "X-From-Cache": "1",
                        },
                    )
                # Legacy format: the raw decoded JSON body (kept for backward
                # compatibility with existing cache files).
                return httpx.Response(
                    200,
                    content=json.dumps(cached).encode("utf-8"),
                    headers={"X-From-Cache": "1"},
                )
        if self._circuit_is_open(service):
            raise CircuitOpenError(service or "")
        backoff = 1.0
        attempts = 6
        limiter = self._get_limiter_for_service(service)
        for attempt in range(attempts):
            limiter.wait()
            try:
                headers = {"Accept": accept} if accept else {}
                if json_body is not None:
                    headers["Content-Type"] = "application/json"
                # Add Semantic Scholar API key if available
                if service == "semanticscholar" and self.s2_api_key:
                    headers["x-api-key"] = self.s2_api_key
                resp = self.client.request(method, url, params=params, headers=headers, json=json_body)
                # Adaptive rate limiting feedback for every REAL response,
                # including a retryable 429/5xx before it is raised below.
                self._adapt_rate_limiter(service, resp)
                if resp.status_code in self.RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError("Retryable status", request=resp.request, response=resp)
                if self.cache and cache_key and resp.status_code == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    if content_type.startswith("application/json"):
                        # Legacy format: raw decoded JSON body (kept so old and
                        # new cache files stay mutually compatible).
                        try:
                            self.cache.set(cache_key, resp.json())
                        except Exception:
                            pass
                    elif self._is_cacheable_text(content_type):
                        # Non-JSON text (arXiv Atom XML, ...): versioned envelope
                        # preserving the Content-Type for faithful replay.
                        try:
                            self.cache.set(
                                cache_key,
                                {self.CACHE_ENVELOPE_KEY: True, "ct": content_type, "text": resp.text},
                            )
                        except Exception:
                            pass
                self._record_service_success(service)
                return resp
            except httpx.HTTPError as exc:
                # Don't sleep after the final attempt -- we're about to give up, so
                # the backoff would just be a wasted stall on a dead or rate-limited
                # source. Honor a server Retry-After (DBLP/Crossref 429/503) over the
                # blind exponential backoff, with jitter to avoid synchronized retries.
                if attempt < attempts - 1:
                    time.sleep(retry_after_seconds(exc, backoff) + random.uniform(0, 0.5))
                    backoff = min(backoff * 2, 16.0)
        self._record_service_failure(service)
        raise RuntimeError(f"Network failure after retries for {url}")


# ------------- Venue-identity helpers -------------


#: ISSN body after stripping the hyphen: 7 digits + a digit-or-X check char.
_ISSN_BODY_RE = re.compile(r"^\d{7}[\dX]$")

#: DBLP record-key prefixes that name a venue *stream* (``conf/icml/Smith23``
#: -> stream ``conf/icml``). Other prefixes (``books``, ``phd``, ``homepages``)
#: do not identify a recurring venue and are ignored.
_DBLP_STREAM_TYPES = frozenset({"conf", "journals", "series"})


def normalize_issn(value: Any) -> str | None:
    """Normalize an ISSN to the standard hyphenated uppercase ``NNNN-NNNX`` form.

    Accepts hyphenated/unhyphenated, any case (``1532-4435``, ``15324435``,
    ``2167-647x``). Returns ``None`` for anything that is not 7 digits plus a
    digit-or-X check character, so junk values never pollute identifier-based
    venue grouping.
    """
    if not value:
        return None
    body = str(value).strip().upper().replace("-", "").replace(" ", "")
    if not _ISSN_BODY_RE.match(body):
        return None
    return f"{body[:4]}-{body[4:]}"


def dblp_stream_key(info: dict[str, Any]) -> str | None:
    """Derive the DBLP venue *stream* key from a search-hit ``info`` dict.

    DBLP record keys embed the venue stream as their first two segments:
    ``conf/icml/Smith23`` -> ``conf/icml``; ``journals/corr/abs-2301-00001`` ->
    ``journals/corr``. The same shape appears in the record URL after ``/rec/``
    (``https://dblp.org/rec/conf/icml/Smith23``), which is consulted as a
    fallback when ``key`` is absent. Parsing is defensive: anything that does
    not look like ``<stream-type>/<stream>/<record-id>`` with a known stream
    type yields ``None`` rather than a bogus venue identity.
    """
    if not isinstance(info, dict):
        return None
    candidates: list[str] = []
    key = info.get("key")
    if isinstance(key, str) and key.strip():
        candidates.append(key.strip())
    url = info.get("url")
    if isinstance(url, str) and "/rec/" in url:
        candidates.append(url.split("/rec/", 1)[1])
    for cand in candidates:
        parts = [p for p in cand.strip("/").split("/") if p]
        if len(parts) >= 3 and parts[0].lower() in _DBLP_STREAM_TYPES:
            return f"{parts[0].lower()}/{parts[1].lower()}"
    return None


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
    # True when ``authors[*]["family"]`` comes from an AUTHORITATIVE structured
    # source -- Crossref's separate ``given``/``family`` fields -- False when the
    # split was SYNTHESIZED by last-token tokenization of a flat name string
    # (Semantic Scholar, DBLP, OpenAlex ``display_name``, Crossref ``literal``).
    # OpenAlex exposes only a flat ``display_name`` in the works response (no
    # given/family split), so it is treated as unstructured here. Drives
    # ``surname_keys``:
    # an authoritative ``family`` is trusted verbatim (only normalized), while a
    # synthesized one is unreliable for CJK/family-first names and is re-derived
    # from the full name via ``last_name_from_person``.
    structured_names: bool = False
    # True when this source returns authors in PUBLICATION ORDER (Crossref,
    # OpenAlex, DBLP, OpenReview -- verified empirically). The author matcher then
    # treats a reordered author list as a real swapped-authors mismatch. Semantic
    # Scholar (synthesized names, less reliable ordering) leaves this False so
    # order is ignored.
    order_reliable: bool = False
    # OpenReview acceptance status (accepted/not_accepted/preprint/unknown) stamped
    # by the OpenReview converter; None for every other source. Lets the verifier
    # act on a not-accepted match without re-fetching the note.
    acceptance: str | None = None
    # ----- Venue-identity fields (cross-source venue consensus) -----
    # All default-empty/None so every existing constructor call keeps working.
    # Two records that share any of these identifiers refer to the SAME venue
    # even when their venue *strings* differ and neither canonicalizes through
    # the hand-curated alias map.
    #: Venue ISSNs in the standard hyphenated uppercase ``NNNN-NNNX`` form
    #: (``normalize_issn``). Crossref ``ISSN`` list / OpenAlex source ``issn``.
    issn: tuple[str, ...] = ()
    #: OpenAlex source id URL (e.g. ``https://openalex.org/S1983995261``).
    venue_source_id: str | None = None
    #: DBLP venue stream key (e.g. ``conf/icml``, ``journals/corr``); see
    #: ``dblp_stream_key``. ``journals/corr`` marks the arXiv/CoRR preprint
    #: stream and must never anchor a published-venue claim.
    venue_key: str | None = None
    #: Bare arXiv ID (version stripped) when the source ties this record to an
    #: arXiv preprint: the arXiv Atom feed entry id, S2 ``externalIds.ArXiv``,
    #: an OpenAlex arXiv landing page, or the record's own DataCite arXiv DOI.
    #: Lets the preprint check pivot onto the MATCHED record's identity when
    #: the entry itself carries no doi/eprint (HALLMARK preprint_as_published
    #: entries strip identifiers; real-world offenders often cite none).
    arxiv_id: str | None = None

    def surname_keys(self, limit: int = 3) -> list[str]:
        """Canonical surname comparison keys derived from ``self.authors``.

        Single source of truth for turning a record author into a surname key.
        How a key is derived depends on whether the source gave us an
        AUTHORITATIVE family name:

        * ``structured_names`` is True (Crossref/OpenAlex separate
          ``given``/``family`` fields): trust the explicit ``family`` verbatim,
          only NORMALIZING it (strip diacritics, lowercase, drop a trailing
          4-digit DBLP homonym suffix) -- but WITHOUT the last-token reduction.
          That reduction corrupts family-first CJK names: e.g. an entry
          "Chen Xing" reduces to ``xing`` while a Crossref record
          ``{"given": "Xing", "family": "Chen"}`` is authoritatively ``chen`` --
          same person, two keys, a false AUTHOR_MISMATCH. Trusting ``family``
          keeps the authoritative side correct.
        * Otherwise (flat ``name`` from S2/DBLP that we tokenized ourselves):
          there is no reliable family split, so reconstruct the full name and
          route it through ``last_name_from_person`` exactly as the BibTeX-entry
          side does, so the two sides stay symmetric.

        Truncate to ``limit`` first, then drop empties, mirroring
        ``authors_last_names`` so neither side can drift.
        """
        keys: list[str] = []
        for a in self.authors[:limit]:
            family = (a.get("family") or "").strip()
            if self.structured_names and family:
                keys.append(_normalize_surname_key(family))
            else:
                # No authoritative family split: rebuild the flattened name and
                # reduce it with the same last-token heuristic the entry uses.
                full = f"{a.get('given', '') or ''} {family}".strip()
                keys.append(last_name_from_person(full))
        return [k for k in keys if k]

    @property
    def canonical_venue(self) -> str | None:
        """Canonical venue id for ``self.journal`` (or None if unrecognized)."""
        # Lazy import: matching.py imports from utils, so a module-level import
        # here would be a cycle.
        from bibtex_updater.matching import get_canonical_venue

        return get_canonical_venue(self.journal or "")


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
    # ACM/IEEE deposit colon-titles split across title/subtitle (the CACM NeRF
    # record is title=["NeRF"], subtitle=["Representing scenes as ..."]).
    # Dropping the subtitle left the record title as the bare head, so the
    # DOI-consistency check compared the entry's FULL title against "NeRF"
    # alone and flagged a CORRECT DOI as DOI_MISMATCH. Re-join them unless the
    # subtitle is already contained in the title (some publishers repeat it).
    subtitles = msg.get("subtitle") or []
    subtitle = subtitles[0] if subtitles else None
    if subtitle:
        subtitle = re.sub(r"<[^>]*>", "", subtitle).strip()  # strip HTML tags
    if title and subtitle and subtitle.lower() not in title.lower():
        title = f"{title}: {subtitle}"

    # Authors - handle given/family and literal formats. Crossref returns
    # AUTHORITATIVE separate given/family fields, so a real ``family`` is
    # trusted verbatim by ``surname_keys`` (no last-token misparse). A ``literal``
    # name we had to split ourselves is NOT authoritative -- if every author came
    # from a literal we leave ``structured_names`` False and let ``surname_keys``
    # fall back to ``last_name_from_person``.
    authors = []
    has_structured_family = False
    for a in msg.get("author", []) or []:
        given = a.get("given") or ""
        family = a.get("family") or ""
        if family:
            has_structured_family = True
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

    # Publication date - check multiple date fields. ``created`` is the DOI
    # *deposit* date, which can be years off the publication date for
    # backfilled archives, so it is consulted strictly LAST.
    pubyear = None
    for dt_key in ("published-print", "published-online", "issued", "published", "created"):
        if msg.get(dt_key, {}).get("date-parts"):
            y = msg[dt_key]["date-parts"][0][0]
            if y:
                pubyear = int(y)
                break

    # Issue can be in different locations
    issue = msg.get("issue") or msg.get("journal-issue", {}).get("issue")

    # Venue identity: Crossref carries the container's ISSNs (print + online).
    # Normalized + deduped; junk entries are dropped by ``normalize_issn``.
    raw_issns = msg.get("ISSN") or []
    if isinstance(raw_issns, str):
        raw_issns = [raw_issns]
    issns: list[str] = []
    for raw_issn in raw_issns if isinstance(raw_issns, list) else []:
        norm_issn = normalize_issn(raw_issn)
        if norm_issn and norm_issn not in issns:
            issns.append(norm_issn)

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
        structured_names=has_structured_family,
        order_reliable=True,
        issn=tuple(issns),
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
        # DBLP appends a 4-digit disambiguation suffix to homonymous author
        # names ("Yu Sun 0020", "Chuan Guo 0001"). Drop it so the surname is the
        # real family name rather than the number, which would otherwise score a
        # false author mismatch against the bib entry.
        if len(parts) >= 2 and re.fullmatch(r"\d{4}", parts[-1]):
            parts = parts[:-1]
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
    if is_preprint_venue(venue):
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
        venue_key=dblp_stream_key(info),
    )


def dblp_hit_to_candidate_record(hit: dict[str, Any]) -> PublishedRecord | None:
    """Permissive DBLP hit -> ``PublishedRecord`` for cascade candidate search.

    Unlike :func:`dblp_hit_to_record` (which is preprint-strict for
    publication-*resolution*: it drops hits lacking a DOI/``ee`` URL, lacking a
    venue/year, or labelled with an arXiv/CoRR venue), this converter is built
    for the fact-checker *cascade*, which only needs candidates to *score*
    against. DBLP authoritatively indexes ICML/ICLR/NeurIPS papers that are
    frequently DOI-less, so we keep any hit with a clear title + authors and we
    retain the arXiv/CoRR copy rather than discarding it outright.

    Args:
        hit: A DBLP search ``hit`` dict (``{"info": {...}}``).

    Returns:
        ``PublishedRecord`` or ``None`` if the hit lacks a usable title or any
        author.
    """
    if not hit:
        return None
    info = hit.get("info", {}) or {}
    title = info.get("title") or ""
    title = re.sub(r"<[^>]*>", "", title).strip()  # strip HTML tags
    if not title:
        return None

    # Parse authors (same logic as dblp_hit_to_record, including the 4-digit
    # homonym-disambiguation-suffix stripping).
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

    authors: list[dict[str, str]] = []
    for full in authors_list:
        full = full.strip()
        if not full:
            continue
        parts = full.split()
        if len(parts) >= 2 and re.fullmatch(r"\d{4}", parts[-1]):
            parts = parts[:-1]
        if len(parts) >= 2:
            authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
        else:
            authors.append({"given": "", "family": parts[0]})

    if not authors:
        return None

    venue = info.get("journal") or info.get("venue")
    year = None
    try:
        year = int(info.get("year")) if info.get("year") else None
    except Exception:
        pass

    doi = doi_normalize(info.get("doi"))
    ee = info.get("ee")
    venue_lower = safe_lower(venue) if venue else ""
    typ = safe_lower(info.get("type"))
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
        volume=info.get("volume"),
        number=info.get("number"),
        pages=info.get("pages"),
        type=record_type,
        order_reliable=True,
        venue_key=dblp_stream_key(info),
    )


def s2_data_to_record(data: dict[str, Any]) -> PublishedRecord | None:
    """Convert Semantic Scholar data to a PublishedRecord."""
    external_ids = data.get("externalIds") or {}
    doi = doi_normalize(data.get("doi") or external_ids.get("DOI"))

    # arXiv identity: S2 exposes the bare ID in ``externalIds.ArXiv`` (kept on
    # published papers too, since S2 merges preprint + published versions).
    # Fall back to a DataCite arXiv DOI. Version-stripped + month-validated so
    # junk values never pollute the preprint-twin pivot downstream.
    arxiv_id: str | None = None
    raw_arxiv = external_ids.get("ArXiv")
    if isinstance(raw_arxiv, str) and raw_arxiv.strip():
        bare = re.sub(r"v\d+$", "", raw_arxiv.strip())
        if is_valid_arxiv_id(bare):
            arxiv_id = bare
    if arxiv_id is None:
        arxiv_id = arxiv_id_from_datacite_doi(doi)

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
        arxiv_id=arxiv_id,
    )


_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def arxiv_atom_to_record(xml: str) -> PublishedRecord | None:
    """Convert an arXiv Atom API response (``id_list`` query) to a PublishedRecord.

    The arXiv export API returns the authoritative record for a given arXiv ID,
    even for very recent papers that Crossref/DBLP/Semantic Scholar have not yet
    indexed. For a bad/unknown ID arXiv still answers ``200`` with a single
    sentinel ``Error`` entry; that case returns ``None``.

    Args:
        xml: Raw Atom feed text from ``http://export.arxiv.org/api/query``.

    Returns:
        PublishedRecord for the first real entry, or ``None`` if the feed has no
        usable entry or cannot be parsed.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None

    entry = root.find("atom:entry", _ATOM_NS)
    if entry is None:
        return None

    # arXiv signals bad IDs with an entry whose <id> points at the errors doc.
    id_el = entry.find("atom:id", _ATOM_NS)
    entry_id = (id_el.text or "").strip() if id_el is not None else ""
    if "/api/errors" in entry_id:
        return None

    title_el = entry.find("atom:title", _ATOM_NS)
    title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""
    if not title:
        return None

    authors: list[dict[str, str]] = []
    for author_el in entry.findall("atom:author", _ATOM_NS):
        name_el = author_el.find("atom:name", _ATOM_NS)
        name = (name_el.text or "").strip() if name_el is not None else ""
        parts = name.split()
        if len(parts) >= 2:
            authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
        elif parts:
            authors.append({"given": "", "family": parts[0]})

    year: int | None = None
    published_el = entry.find("atom:published", _ATOM_NS)
    if published_el is not None and published_el.text:
        m = re.match(r"\s*(\d{4})", published_el.text)
        if m:
            year = int(m.group(1))

    doi_el = entry.find("arxiv:doi", _ATOM_NS)
    doi = doi_normalize(doi_el.text) if doi_el is not None and doi_el.text else ""

    # The Atom <id> is the abs URL (``http://arxiv.org/abs/2301.00001v2``);
    # extract the bare, version-stripped arXiv ID so the record carries its own
    # preprint identity even when the entry that matched it cites no identifier.
    arxiv_id = extract_arxiv_id_from_text(entry_id) if entry_id else None

    return PublishedRecord(
        doi=doi or "",
        url=entry_id or None,
        title=title,
        authors=authors,
        journal=None,
        year=year,
        type="preprint",
        method="arXiv(id_list)",
        arxiv_id=arxiv_id,
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
    type_map: dict[str, str] = {
        "article": "journal-article",
        "proceedings-article": "proceedings-article",
        "book-chapter": "book-chapter",
    }
    record_type = type_map.get(work_type, work_type) if work_type else work_type

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
        self.timestamps: collections.deque[float] = collections.deque()

    async def wait(self) -> None:
        """Async wait until a request can be made within the rate limit.

        This method will sleep asynchronously if the rate limit has been
        exceeded, allowing other coroutines to run while waiting.
        """
        import asyncio

        window = 60.0
        while True:
            sleep_for = 0.0
            async with self.lock:
                now = time.time()
                cutoff = now - window
                # O(1) popleft from deque (timestamps are chronological)
                while self.timestamps and self.timestamps[0] < cutoff:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.req_per_min:
                    self.timestamps.append(time.time())
                    return  # Slot acquired
                # Need to wait — compute sleep duration but release lock first
                sleep_for = window - (now - self.timestamps[0]) + 0.01
            # Sleep WITHOUT holding the lock; add jitter to prevent
            # synchronized wake-ups across coroutines (burst-stall pattern)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for + random.uniform(0, 0.5))


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
        "openreview": 30,  # OpenReview: 30/min (conservative; keyless public read)
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
        cache: DiskCache | SqliteCache | None = None,
        timeout: float = 20.0,
        user_agent: str = "bibtex-updater/1.0 (async)",
    ) -> None:
        """Initialize the async HTTP client.

        Args:
            rate_limiters: AsyncRateLimiterRegistry for per-service rate limiting
            cache: Optional DiskCache or SqliteCache for caching responses
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
            except httpx.HTTPError as exc:
                if attempt < 5:
                    await asyncio.sleep(retry_after_seconds(exc, backoff) + random.uniform(0, 0.5))
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
