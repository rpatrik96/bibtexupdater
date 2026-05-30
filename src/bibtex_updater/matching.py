"""Standalone matching utilities for bibliographic comparison.

This module provides advanced matching functions for:
- P2.2: Near-miss title detection (high fuzzy score but character-level differences)
- P2.3: Ordered author comparison (LCS-based sequence similarity)
- P2.5: Expanded venue aliases (comprehensive venue normalization)

These functions are standalone and can be used by fact_checker.py or other modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from rapidfuzz.distance import Levenshtein

from bibtex_updater.utils import jaccard_similarity, normalize_title_for_match

__all__ = [
    "title_edit_distance",
    "is_near_miss_title",
    "word_level_diff",
    "author_sequence_similarity",
    "combined_author_score",
    "MatchOutcome",
    "AuthorMatchResult",
    "symmetric_author_match",
    "has_explicit_truncation_indicator",
    "EXPANDED_VENUE_ALIASES",
    "get_canonical_venue",
    "is_preprint_or_series_venue",
]


class MatchOutcome(Enum):
    """Three-valued result of a field comparison.

    The verifier distinguishes positive confirmation from mere absence of
    contradiction. A field is only allowed to contribute to a VERIFIED verdict
    when it is CONFIRMED (MATCH); a real contradiction is a MISMATCH; and
    "I had nothing comparable / could not positively confirm" is NON_COMPARABLE
    (no data either side) or PARTIAL (consistent-but-incomplete confirmation).
    """

    MATCH = "match"  # both sides populated and positively agree
    MISMATCH = "mismatch"  # both sides populated real values that conflict
    NON_COMPARABLE = "non_comparable"  # empty/blank, or a preprint/series record
    PARTIAL = "partial"  # consistent but incomplete (e.g. dropped authors)


@dataclass(frozen=True)
class AuthorMatchResult:
    """Trichotomy result of :func:`symmetric_author_match`.

    ``outcome`` carries the three-valued verdict; ``score`` is the legacy
    0-1 similarity for confidence/reporting. ``is_confirmed`` is true only for
    a full positive confirmation (exact match or a leading subset explicitly
    elided with an "and others"/"et al" sentinel).
    """

    outcome: MatchOutcome
    score: float

    @property
    def is_confirmed(self) -> bool:
        return self.outcome is MatchOutcome.MATCH

    @property
    def is_mismatch(self) -> bool:
        return self.outcome is MatchOutcome.MISMATCH


# ------------- P2.2: Near-Miss Title Detection -------------


def title_edit_distance(title_a: str, title_b: str) -> int:
    """Compute Levenshtein edit distance on normalized titles.

    Uses normalize_title_for_match() from utils before computing distance.
    A near-miss title has high token_sort_ratio (≥0.90) but non-zero edit distance.

    Args:
        title_a: First title string
        title_b: Second title string

    Returns:
        Edit distance (number of character insertions/deletions/substitutions)
    """
    norm_a = normalize_title_for_match(title_a)
    norm_b = normalize_title_for_match(title_b)
    return Levenshtein.distance(norm_a, norm_b)


def is_near_miss_title(
    title_a: str,
    title_b: str,
    fuzzy_score: float,
    fuzzy_threshold: float = 0.90,
    min_edit_distance: int = 3,
) -> bool:
    """Detect near-miss titles: high fuzzy score but significant character-level differences.

    Args:
        title_a: First title
        title_b: Second title
        fuzzy_score: Pre-computed fuzzy match score (0.0-1.0)
        fuzzy_threshold: Minimum fuzzy score to consider
        min_edit_distance: Minimum edit distance to flag as near-miss (default: 3)

    Returns:
        True if fuzzy_score >= threshold AND edit_distance >= min_edit_distance
    """
    if fuzzy_score < fuzzy_threshold:
        return False
    edit_dist = title_edit_distance(title_a, title_b)
    return edit_dist >= min_edit_distance


def word_level_diff(title_a: str, title_b: str) -> dict:
    """Compute word-level differences between two titles.

    Returns dict with:
    - added: words in b not in a
    - removed: words in a not in b
    - common: words in both
    - substitution_count: estimated word substitutions

    Args:
        title_a: First title string
        title_b: Second title string

    Returns:
        Dict with word-level difference statistics
    """
    norm_a = normalize_title_for_match(title_a)
    norm_b = normalize_title_for_match(title_b)

    words_a = set(norm_a.split())
    words_b = set(norm_b.split())

    common = words_a & words_b
    removed = words_a - words_b
    added = words_b - words_a

    # Estimate substitutions: pairs of removed/added words
    # (This is a heuristic - true substitution detection would need alignment)
    substitution_count = min(len(removed), len(added))

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "common": sorted(common),
        "substitution_count": substitution_count,
    }


# ------------- P2.3: Ordered Author Comparison -------------


def author_sequence_similarity(list_a: list[str], list_b: list[str]) -> float:
    """LCS-based ordered author comparison.

    Unlike Jaccard (set-based, order-blind), this uses longest common subsequence
    to detect author order changes and insertions/deletions.

    Returns similarity 0.0-1.0 based on LCS length / max(len(a), len(b)).

    Args:
        list_a: First list of author last names (normalized)
        list_b: Second list of author last names (normalized)

    Returns:
        Sequence similarity score 0.0-1.0
    """
    if not list_a and not list_b:
        return 1.0
    if not list_a or not list_b:
        return 0.0

    # Dynamic programming LCS
    m, n = len(list_a), len(list_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list_a[i - 1] == list_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    max_len = max(m, n)
    return lcs_length / max_len if max_len > 0 else 0.0


def combined_author_score(
    list_a: list[str], list_b: list[str], jaccard_weight: float = 0.5, sequence_weight: float = 0.5
) -> float:
    """Combine Jaccard (unordered) and LCS (ordered) author similarity.

    Uses both metrics to catch:
    - Missing/extra authors (Jaccard detects well)
    - Swapped author order (LCS detects well)

    Args:
        list_a: First list of author last names (normalized)
        list_b: Second list of author last names (normalized)
        jaccard_weight: Weight for Jaccard similarity (default 0.5)
        sequence_weight: Weight for sequence similarity (default 0.5)

    Returns:
        Combined similarity score 0.0-1.0
    """
    jaccard_score = jaccard_similarity(list_a, list_b)
    sequence_score = author_sequence_similarity(list_a, list_b)
    return jaccard_weight * jaccard_score + sequence_weight * sequence_score


#: Sentinel surnames that BibTeX/citation tooling inserts for elided author
#: lists ("and others" -> "others"; "et al."). They are not real authors and
#: must be stripped before any author-set comparison, otherwise a correctly
#: cited paper that uses "and others" is punished for a phantom mismatch.
_AUTHOR_SENTINELS: frozenset[str] = frozenset({"others", "et al", "etal", "al"})


def _has_author_sentinel(names: list[str]) -> bool:
    """True if a surname list contains an elision sentinel ("others"/"et al")."""
    return any(n and n.lower() in _AUTHOR_SENTINELS for n in names)


#: Explicit truncation indicators that a citation may use OUTSIDE the structured
#: author field to disclose that the author list is incomplete. ``--strict``'s
#: "silent truncation" rule (rule 5) does NOT flag entries that disclose their
#: truncation: a citation that says "..." or ``\ldots`` or a trailing
#: ``, et al.`` outside the author field has *already announced* the omission.
#: That is the cited author's responsibility, not a hallucination signal.
_EXPLICIT_TRUNCATION_MARKERS: tuple[str, ...] = (
    "...",
    "\\ldots",
    "\\dots",
    "et al",
    "et al.",
)


def has_explicit_truncation_indicator(*fields: str | None) -> bool:
    """True if any of ``fields`` carries a disclosed-truncation marker.

    Catches the cases the structured ``and others`` / ``et al`` sentinel inside
    the BibTeX ``author`` field does not: a trailing ``...`` / ``\\ldots`` in
    the rendered citation, or a trailing ``, et al.`` placed in a sibling
    field (``note``, ``howpublished``, even the title) rather than as a proper
    author token. Used by ``--strict`` to refuse to escalate a leading-prefix
    author list to AUTHOR_TRUNCATED when the citation already discloses that
    its author list is truncated.
    """
    for raw in fields:
        if not raw:
            continue
        text = raw.lower()
        # Cheap substring check; the markers are short and distinctive enough
        # that a false positive on real prose is exceedingly unlikely.
        if any(marker in text for marker in _EXPLICIT_TRUNCATION_MARKERS):
            return True
    return False


def _strip_author_sentinels(names: list[str]) -> list[str]:
    """Drop "others"/"et al" sentinels from a surname list."""
    return [n for n in names if n and n.lower() not in _AUTHOR_SENTINELS]


def _is_ordered_subsequence(short: list[str], long: list[str]) -> bool:
    """True if ``short`` appears in ``long`` in order (not necessarily contiguous)."""
    if not short:
        return True
    it = iter(long)
    return all(name in it for name in short)


def _is_leading_prefix(short: list[str], long: list[str]) -> bool:
    """True if ``short`` is a contiguous *leading* prefix of ``long``."""
    return len(short) <= len(long) and short == long[: len(short)]


def _looks_alphabetized(names: list[str]) -> bool:
    """True if surname keys are sorted A-Z over >=3 names.

    A record whose authors are in alphabetical order has very likely *sorted*
    its contributor list (Crossref NeurIPS/ICML proceedings deposits do this,
    e.g. the 10.52202 prefix) rather than preserving title-page order, so its
    author *order* cannot be trusted to detect a publication-order swap. Require
    >=3 names: with one or two authors, alphabetical order coincides too often to
    carry any signal.
    """
    return len(names) >= 3 and names == sorted(names)


def symmetric_author_match(
    entry_names: list[str],
    api_names: list[str],
    threshold: float = 0.80,
    prefix_n: int = 5,
    order_reliable: bool = False,
    strict: bool = False,
) -> AuthorMatchResult:
    """Compare entry vs API author surnames on a *symmetric* basis (trichotomy).

    Containment proves the cited authors are *consistent with* the real list, not
    that the citation is *complete*. We therefore distinguish a full positive
    confirmation from a consistent-but-incomplete one:

    1. Strip "others"/"et al" sentinels from both sides (but remember whether a
       sentinel was present -- it signals a deliberate elision).
    2. NON_COMPARABLE: either side has no usable surnames -- nothing to confirm
       or refute.
    3. MISMATCH: first-author surnames differ (a swapped/wrong lead author).
    4. MATCH (CONFIRMED): the sentinel-stripped surname lists are EQUAL, OR one
       side is a *leading prefix* of the other AND the shorter side carried an
       explicit elision sentinel ("and others"/"et al"). The author claim is then
       positively confirmed (exactly, or as an explicitly-truncated head).
    5. PARTIAL: the shorter side is an in-order subsequence (or leading prefix)
       of the longer WITHOUT a sentinel -- authors are consistent but the claim
       silently drops interior/trailing authors, so it is not a full confirmation.
    6. Otherwise score a symmetric slice (Jaccard + LCS): >= threshold is a MATCH,
       below is a MISMATCH (real conflict beyond the shared lead author).

    ``strict`` (arXiv 2026 / hallucination-leak mode): the cost is asymmetric --
    a leaked wrong/swapped author is far worse than an FP. Two relaxations are
    removed:
      * The alphabetization guard (a same-multiset record sorted A-Z is treated
        as a record-side sort artifact) is DISABLED. An order-reliable source
        with the same multiset but a different lead is a real swap.
      * The hard first-author guard no longer requires a differing multiset:
        a different lead author against an order-reliable source is a MISMATCH
        even if the multiset matches.
    Returns an :class:`AuthorMatchResult` carrying the trichotomy and a 0-1 score.
    """
    a_has_sentinel = _has_author_sentinel(entry_names)
    b_has_sentinel = _has_author_sentinel(api_names)
    a = _strip_author_sentinels(entry_names)
    b = _strip_author_sentinels(api_names)

    # If either side has no usable names, there is nothing to confirm or refute.
    if not a or not b:
        return AuthorMatchResult(MatchOutcome.NON_COMPARABLE, 1.0)

    # Hard first-author signal: a different lead author whose author multiset
    # also DIFFERS is a real mismatch (a genuinely wrong/extra lead author).
    # When the multisets are identical the lead difference is pure reordering --
    # deferred to the same-multiset block below, which decides swap vs artifact.
    # In strict mode, a same-multiset lead difference against an order-reliable
    # source is also a real swap (the alphabetization escape clause is dropped).
    if a[0] != b[0] and sorted(a) != sorted(b):
        return AuthorMatchResult(MatchOutcome.MISMATCH, 0.0)
    if strict and a[0] != b[0] and order_reliable:
        return AuthorMatchResult(MatchOutcome.MISMATCH, 0.0)

    # Exact (sentinel-stripped) equality: full positive confirmation.
    if a == b:
        return AuthorMatchResult(MatchOutcome.MATCH, 1.0)

    # Leading-prefix containment: one side is a contiguous head of the other.
    # An explicit elision sentinel on the SHORTER side means the citation
    # deliberately truncated a leading run ("first k authors, and others") --
    # that is a confirmation, not a silent drop.
    if _is_leading_prefix(a, b):  # entry is a head of the (longer) api list
        if a_has_sentinel:
            return AuthorMatchResult(MatchOutcome.MATCH, 1.0)
        return AuthorMatchResult(MatchOutcome.PARTIAL, 1.0)
    if _is_leading_prefix(b, a):  # api is a head of the (longer) entry list
        if b_has_sentinel:
            return AuthorMatchResult(MatchOutcome.MATCH, 1.0)
        return AuthorMatchResult(MatchOutcome.PARTIAL, 1.0)

    # In-order subsequence (but not a contiguous leading prefix): interior or
    # trailing authors are dropped. Consistent, never a full confirmation, even
    # with a sentinel -- the elision is not a simple leading truncation.
    if _is_ordered_subsequence(a, b) or _is_ordered_subsequence(b, a):
        return AuthorMatchResult(MatchOutcome.PARTIAL, 1.0)

    # Same author multiset, different order (a genuine reordering; requires full
    # multiset equality, not mere overlap -- a single differing author such as a
    # record-side typo 'Ren'/'Rent' is NOT a reordering and falls through to the
    # order-agnostic score below). Against an order-preserving source this is a
    # real swapped-authors defect -> MISMATCH. Two exclusions, where the order
    # carries no signal so the shared author set is a positive confirmation:
    #   * Semantic Scholar (order_reliable=False) -- flat, unordered names.
    #   * An alphabetized API order (sorted A-Z) -- a record-side sort artifact
    #     (e.g. Crossref NeurIPS/ICML proceedings deposits), not a publication
    #     swap. This was a false-positive source on valid multi-author papers.
    if sorted(a) == sorted(b):
        # ``strict`` disables the alphabetization escape: against an order-
        # reliable source a same-multiset reordering is a real swap even if
        # the record looks alphabetized (the arXiv-2026 policy treats the
        # asymmetric leak cost as far worse than the FP it introduces).
        if order_reliable and (strict or not _looks_alphabetized(b)):
            return AuthorMatchResult(MatchOutcome.MISMATCH, 0.0)
        return AuthorMatchResult(MatchOutcome.MATCH, 1.0)

    # Symmetric slice + combined (Jaccard + LCS) score.
    n = min(len(a), len(b), prefix_n)
    a_slice, b_slice = a[:n], b[:n]
    score = combined_author_score(a_slice, b_slice, jaccard_weight=0.5, sequence_weight=0.5)
    if score >= threshold:
        return AuthorMatchResult(MatchOutcome.MATCH, score)
    return AuthorMatchResult(MatchOutcome.MISMATCH, score)


# ------------- P2.5: Expanded Venue Aliases -------------

EXPANDED_VENUE_ALIASES: dict[str, set[str]] = {
    # ML/AI Conferences (from existing fact_checker.py)
    "neurips": {
        "nips",
        "advances in neural information processing systems",
        "neural information processing systems",
        "conference on neural information processing systems",
        "annual conference on neural information processing systems",
        # Numbered proceedings, e.g. "The 36th Conference on Neural ..." -- the
        # ordinal/year is stripped during normalization, leaving these forms.
        "th conference on neural information processing systems",
        "th annual conference on neural information processing systems",
    },
    "icml": {
        "international conference on machine learning",
        "proceedings of the international conference on machine learning",
        # "Proceedings of the Nth International Conference on Machine Learning":
        # the ordinal year is removed by normalization, leaving "th ...".
        "proceedings of the th international conference on machine learning",
        "th international conference on machine learning",
    },
    "iclr": {
        "international conference on learning representations",
        "proceedings of the international conference on learning representations",
    },
    "aaai": {
        "association for the advancement of artificial intelligence",
        "aaai conference on artificial intelligence",
        "proceedings of the aaai conference on artificial intelligence",
    },
    "cvpr": {
        "computer vision and pattern recognition",
        "ieee conference on computer vision and pattern recognition",
        "ieee/cvf conference on computer vision and pattern recognition",
        "conference on computer vision and pattern recognition",
    },
    "iccv": {
        "international conference on computer vision",
        "ieee international conference on computer vision",
        "ieee/cvf international conference on computer vision",
    },
    "eccv": {
        "european conference on computer vision",
    },
    "acl": {
        "association for computational linguistics",
        "annual meeting of the association for computational linguistics",
        # Findings track of ACL -- a distinct track of the SAME venue.
        "findings of acl",
        "findings of the association for computational linguistics: acl",
    },
    "emnlp": {
        "empirical methods in natural language processing",
        "conference on empirical methods in natural language processing",
        # Findings track of EMNLP. The exact-match pass resolves these before
        # the substring fallback would otherwise collapse them into "acl".
        "findings of emnlp",
        "findings of the association for computational linguistics: emnlp",
    },
    "naacl": {
        "north american chapter of the association for computational linguistics",
        "annual conference of the north american chapter of the association for computational linguistics",
        "naacl-hlt",
        "naacl hlt",
        "findings of naacl",
        "findings of the association for computational linguistics: naacl",
    },
    "kdd": {
        "knowledge discovery and data mining",
        "sigkdd",
        "acm sigkdd",
        "acm sigkdd conference on knowledge discovery and data mining",
        "acm sigkdd international conference on knowledge discovery and data mining",
    },
    "ijcai": {
        "international joint conference on artificial intelligence",
    },
    "uai": {
        "uncertainty in artificial intelligence",
        "conference on uncertainty in artificial intelligence",
    },
    "aistats": {
        "artificial intelligence and statistics",
        "international conference on artificial intelligence and statistics",
    },
    "colt": {
        "conference on learning theory",
        "annual conference on learning theory",
        "annual conference on computational learning theory",
    },
    "interspeech": {
        "conference of the international speech communication association",
        "annual conference of the international speech communication association",
    },
    "icassp": {
        "international conference on acoustics, speech and signal processing",
        "ieee international conference on acoustics, speech and signal processing",
        "ieee international conference on acoustics, speech, and signal processing",
    },
    "jmlr": {
        "journal of machine learning research",
        # ISO-4 abbreviated form. Period-stripping in
        # ``_normalize_venue_for_matching`` turns ``J. Mach. Learn. Res.`` into
        # ``j mach learn res`` before lookup.
        "j mach learn res",
        # ``jmlr workshop and conference proceedings`` is intentionally NOT here:
        # it is caught by ``_SERIES_MARKERS`` as PMLR-style umbrella series.
    },
    "tmlr": {
        "transactions on machine learning research",
        # ISO-4 abbreviated form. ``Trans. Mach. Learn. Res.`` -> ``trans mach
        # learn res`` after the period-strip in ``_normalize_venue_for_matching``.
        "trans mach learn res",
        # OpenReview / Zotero exports often say ``Accepted by TMLR``.
        "accepted by tmlr",
    },
    # Systems/DB (new)
    "sigmod": {
        "acm sigmod",
        "international conference on management of data",
        "sigmod conference",
        "proceedings of the acm sigmod international conference on management of data",
    },
    "vldb": {
        "very large data bases",
        "vldb endowment",
        "pvldb",
        "proceedings of the vldb endowment",
        "international conference on very large data bases",
    },
    "icde": {
        "international conference on data engineering",
        "ieee international conference on data engineering",
    },
    "osdi": {
        "operating systems design and implementation",
        "usenix symposium on operating systems design and implementation",
    },
    "sosp": {
        "symposium on operating systems principles",
        "acm symposium on operating systems principles",
    },
    # IR/Web (new)
    "sigir": {
        "acm sigir",
        "research and development in information retrieval",
        "international acm sigir conference on research and development in information retrieval",
    },
    "www": {
        "the web conference",
        "thewebconf",
        "world wide web",
        "international world wide web conference",
        "international conference on world wide web",
        "proceedings of the web conference",
        "acm web conference",
    },
    "wsdm": {
        "web search and data mining",
        "acm international conference on web search and data mining",
    },
    "cikm": {
        "conference on information and knowledge management",
        "acm international conference on information and knowledge management",
    },
    # HCI (new)
    "chi": {
        "acm chi",
        "human factors in computing systems",
        "acm conference on human factors in computing systems",
        "chi conference on human factors in computing systems",
    },
    # Recommender systems (new)
    "recsys": {
        "acm recsys",
        "recommender systems",
        "acm conference on recommender systems",
    },
    # Robotics/Vision (new)
    "icra": {
        "international conference on robotics and automation",
        "ieee international conference on robotics and automation",
    },
    "iros": {
        "intelligent robots and systems",
        "ieee/rsj international conference on intelligent robots and systems",
    },
    "corl": {
        "conference on robot learning",
        "proceedings of the conference on robot learning",
    },
    # NLP (extend existing)
    "coling": {
        "international conference on computational linguistics",
        "proceedings of the international conference on computational linguistics",
    },
    "eacl": {
        "european chapter of the association for computational linguistics",
        "proceedings of the european chapter of the association for computational linguistics",
        "conference of the european chapter of the association for computational linguistics",
        "findings of eacl",
        "findings of the association for computational linguistics: eacl",
    },
    "conll": {
        "conference on computational natural language learning",
        "proceedings of the conference on computational natural language learning",
    },
    # Journals (new)
    "tpami": {
        "ieee transactions on pattern analysis and machine intelligence",
        "ieee tpami",
        "transactions on pattern analysis and machine intelligence",
    },
    "ijcv": {
        "international journal of computer vision",
    },
    "tacl": {
        "transactions of the association for computational linguistics",
    },
    "nature": {
        "nature",
        # Don't add common substrings — exact match preferred for single-word journals
    },
    "science": {
        "science",
    },
    # Add specific Nature sub-journals as separate entries:
    "nature_mi": {
        "nature machine intelligence",
        "nat mach intell",
    },
    "nature_comm": {
        "nature communications",
        "nat commun",
    },
    "pnas": {
        "proceedings of the national academy of sciences",
        "proc natl acad sci",
    },
}


#: Canonical venues whose name is a common English word that is also a prefix of
#: distinct sibling journals (Nature Physics, Science Robotics). For these,
#: substring matching is unsafe, so they only match on exact equality.
GENERIC_SINGLE_WORD_VENUES: frozenset[str] = frozenset({"nature", "science", "pnas"})


#: OpenReview ``venueid`` shape: ``Acronym.cc/YYYY/<Track>``. Unique to
#: OpenReview-hosted submissions (no real venue string uses ``.cc/YYYY/``), so a
#: prefix-strip to the bare acronym is leak-safe: a hallucinated entry's venue
#: would not match this shape at all.
_OPENREVIEW_VENUEID_RE = re.compile(
    r"^([a-z]+)\.cc/\d{4}/[a-z0-9_\-/]+$",
    re.IGNORECASE,
)

#: Track / decoration tokens that ML conferences attach to a base venue string
#: (``ICLR 2023 poster``, ``NeurIPS 2022 oral``, ``ICML 2023 spotlight``,
#: ``ICLR 2023 Notable top-5%``). Every token is a generic ML-conference
#: qualifier with NO standalone venue identity, so stripping it from a
#: fabricated entry (``FakeConf 2023 poster`` -> ``fakeconf 2023``) still does
#: not canonicalize to any real venue. ``workshop`` is intentionally excluded
#: -- workshops are distinct venues from their host conference's main track and
#: must not be conflated (``ICLR 2023 Workshop on X`` should NOT match the
#: ICLR proceedings).
_TRACK_DECORATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnotable\s+top[\s\-]*\d+%?", re.IGNORECASE),
    re.compile(r"\btop[\s\-]*\d+%?", re.IGNORECASE),
    re.compile(r"\bdatasets\s+and\s+benchmarks(?:\s+track)?\b", re.IGNORECASE),
    re.compile(r"\bconference\s+track\b", re.IGNORECASE),
    re.compile(r"\bmain\s+track\b", re.IGNORECASE),
    re.compile(r"\blong\s+paper\b", re.IGNORECASE),
    re.compile(r"\bshort\s+paper\b", re.IGNORECASE),
    # NOTE: "findings" deliberately NOT stripped here -- "ACL Findings" / "EMNLP
    # Findings" are distinct sub-venues with their own canonical aliases above.
    re.compile(r"\b(poster|oral|spotlight|highlight|demo|demonstration|tutorial)\b", re.IGNORECASE),
)


def _strip_track_decorations(venue_norm: str) -> str:
    """Strip well-known track / decoration suffixes from a lowercased venue.

    Conference records (especially OpenReview) routinely tag the venue string
    with a track or decoration (``ICLR 2023 poster``, ``NeurIPS 2022 Datasets
    and Benchmarks Track``, ``ICML 2023 spotlight``, ``ICLR 2023 Notable
    top-5%``). These tokens are generic ML-conference qualifiers -- not
    standalone venues -- so stripping them lifts the bare acronym to the
    surface for alias lookup. A fabricated venue (``FakeConf 2023 poster``)
    still does not canonicalize after stripping, so no new false-MATCH path is
    opened. ``workshop`` is intentionally NOT stripped: workshops are distinct
    venues from their host conference's main proceedings.
    """
    out = venue_norm
    for pat in _TRACK_DECORATION_PATTERNS:
        out = pat.sub(" ", out)
    return " ".join(out.split()).strip()


def _normalize_venue_for_matching(venue: str) -> str:
    """Normalize venue string for matching.

    Args:
        venue: Raw venue name

    Returns:
        Normalized venue string
    """
    venue_norm = venue.lower().strip()

    # FIX A2: OpenReview venueid pre-pass -- ``ICLR.cc/2024/Conference`` ->
    # ``iclr``, ``NeurIPS.cc/2022/Datasets_and_Benchmarks_Track`` -> ``neurips``.
    # The ``.cc/YYYY/...`` shape is unique to OpenReview venueids, so leak risk
    # is zero (no real venue string uses it).
    m = _OPENREVIEW_VENUEID_RE.match(venue_norm)
    if m:
        venue_norm = m.group(1).lower()

    # FIX C1: ISO-4 abbreviated journal forms use period-separated tokens
    # (``Trans. Mach. Learn. Res.``). Drop the trailing periods so the bare
    # tokens line up with the alias map ("trans mach learn res").
    venue_norm = venue_norm.replace(".", " ")

    # Remove common prefixes
    for prefix in ["proceedings of the ", "proceedings of ", "proc. ", "proc ", "in "]:
        if venue_norm.startswith(prefix):
            venue_norm = venue_norm[len(prefix) :]

    # Remove years
    venue_norm = re.sub(r"\b\d{4}\b", "", venue_norm)
    venue_norm = " ".join(venue_norm.split()).strip()

    # FIX A3: strip track / decoration suffixes (``ICLR 2023 poster`` ->
    # ``iclr``, ``NeurIPS 2022 Datasets and Benchmarks Track`` -> ``neurips``)
    # AFTER year removal so the year-strip fires first.
    venue_norm = _strip_track_decorations(venue_norm)

    return venue_norm


def get_canonical_venue(venue: str, aliases: dict[str, set[str]] | None = None) -> str | None:
    """Map a venue name to its canonical form using the expanded alias map.

    Args:
        venue: Raw venue name from BibTeX entry
        aliases: Optional custom alias map (defaults to EXPANDED_VENUE_ALIASES)

    Returns:
        Canonical venue name, or None if not found
    """
    if aliases is None:
        aliases = EXPANDED_VENUE_ALIASES

    venue_norm = _normalize_venue_for_matching(venue)
    if not venue_norm:
        return None

    # Pass 1: exact equality against every canonical key and alias, including
    # short acronyms (len <= 3). This must run before any substring matching so
    # that a bare acronym ("ACL", "KDD", "UAI") maps to its own canonical venue
    # instead of substring-colliding with a *different* acronym that merely
    # contains it ("acl" is a substring of "naacl"). Exact match is always safe.
    for canonical, alias_set in aliases.items():
        if venue_norm == canonical or venue_norm in alias_set:
            return canonical

    # Pass 2: substring matching for spelled-out / decorated forms. Skip names
    # <= 3 chars here: short acronyms are substrings of longer venue names and of
    # each other, so substring-matching them produces false collisions (handled
    # exactly by Pass 1 above).
    for canonical, alias_set in aliases.items():
        all_names = alias_set | {canonical}
        for name in all_names:
            if len(name) <= 3:
                continue
            # Generic single-word *journal* names ("nature"/"science") are
            # prefixes of distinct sibling journals ("Nature Physics", "Science
            # Robotics"), so substring matching would wrongly collapse them and
            # mask a genuine venue mismatch -- require an exact match for those.
            # Single-token acronyms ("iclr", "neurips") are NOT generic words and
            # still substring-match so a track/poster suffix ("ICLR (Poster)")
            # canonicalizes to the same venue.
            if canonical in GENERIC_SINGLE_WORD_VENUES and " " not in name:
                continue
            # Require substantial overlap for substring matching
            shorter, longer = sorted([name, venue_norm], key=len)
            if len(shorter) / len(longer) < 0.4:
                # FIX A1: the 0.4 ratio gate rejects a clean substring match like
                # ``len('iclr') / len('iclr poster') = 0.36`` even though ``iclr``
                # appears at a word boundary. For 4-7 char single-token acronyms
                # we accept a word-boundary match instead. This is strictly more
                # conservative than the substring branch below: every word-boundary
                # match is also a substring match, but ``acl`` does NOT word-
                # boundary-match inside ``naacl`` (no boundary between ``na`` and
                # ``acl``), so the historical naacl/acl collision stays excluded.
                if (
                    4 <= len(name) <= 7
                    and " " not in name
                    and name == canonical
                    and re.search(rf"\b{re.escape(name)}\b", venue_norm)
                ):
                    return canonical
                continue  # Too different in length for substring match
            if name in venue_norm or venue_norm in name:
                return canonical

    return None


#: Substrings that mark a venue as a preprint server or a publisher *series*
#: rather than a specific published venue. An arXiv-indexed record often carries
#: a blank or preprint venue while the entry cites the real conference, so a
#: preprint venue on EITHER side must never produce a hard mismatch. "PMLR" /
#: "proceedings of machine learning research" is the umbrella series for many
#: distinct conferences (ICML, AISTATS, CoRL, ...), so it cannot pin a single
#: venue and is likewise treated as non-comparable.
#: Preprint-server markers, matched as substrings of the lowercased raw venue.
_PREPRINT_SERVER_MARKERS: tuple[str, ...] = (
    "arxiv",
    "biorxiv",
    "medrxiv",
    "chemrxiv",
    "preprint",
    "corr",  # arXiv's "Computing Research Repository" DBLP label
    # FIX C3: SSRN is a working-paper / preprint server. CrossRef returns
    # ``SSRN Electronic Journal`` for SSRN copies of papers later published at
    # JMLR / conference venues, so an SSRN-side venue must be non-comparable
    # (not a hard mismatch) and let another source confirm the venue. Unique
    # acronym -- no real journal name contains ``ssrn``.
    "ssrn",
)

#: Publisher-*series* markers (matched against the lowercased raw venue). These
#: name an umbrella series that spans many distinct conferences, so they cannot
#: pin a single venue. Matched on the RAW venue (not the prefix-stripped form)
#: so "Proceedings of Machine Learning Research" (PMLR) is caught while the
#: distinct journal "Journal of Machine Learning Research" (JMLR) is NOT.
_SERIES_MARKERS: tuple[str, ...] = (
    "proceedings of machine learning research",
    "pmlr",
    "jmlr workshop and conference proceedings",
    "w&cp",
)

#: Hosting-*platform* markers. A record whose venue is only the platform name
#: (OpenReview hosts ICLR, NeurIPS, TMLR, and many workshops) says nothing about
#: the published venue, exactly like a preprint server -- so a venue comparison
#: against it is non-comparable, not a mismatch. Not added as a venue *alias*
#: (OpenReview is not a venue), only as a non-comparable platform string.
_PLATFORM_MARKERS: tuple[str, ...] = ("openreview",)


def is_preprint_or_series_venue(venue: str) -> bool:
    """True if ``venue`` names a preprint server or non-specific publisher series.

    Such venues (arXiv/CoRR, bioRxiv, "Proceedings of Machine Learning Research",
    PMLR/JMLR W&CP, and hosting platforms like OpenReview) cannot pin a single
    published venue, so a venue comparison against them is *non-comparable*
    rather than a mismatch. The distinct journal "Journal of Machine Learning
    Research" (JMLR) is deliberately NOT matched.
    """
    if not venue:
        return False
    raw = venue.lower().strip()
    if not raw:
        return False
    return (
        any(m in raw for m in _PREPRINT_SERVER_MARKERS)
        or any(m in raw for m in _SERIES_MARKERS)
        or any(m in raw for m in _PLATFORM_MARKERS)
    )
