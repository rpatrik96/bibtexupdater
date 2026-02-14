"""Standalone matching utilities for bibliographic comparison.

This module provides advanced matching functions for:
- P2.2: Near-miss title detection (high fuzzy score but character-level differences)
- P2.3: Ordered author comparison (LCS-based sequence similarity)
- P2.5: Expanded venue aliases (comprehensive venue normalization)

These functions are standalone and can be used by fact_checker.py or other modules.
"""

from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from bibtex_updater.utils import jaccard_similarity, normalize_title_for_match

__all__ = [
    "title_edit_distance",
    "is_near_miss_title",
    "word_level_diff",
    "author_sequence_similarity",
    "combined_author_score",
    "EXPANDED_VENUE_ALIASES",
    "get_canonical_venue",
]


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


# ------------- P2.5: Expanded Venue Aliases -------------

EXPANDED_VENUE_ALIASES: dict[str, set[str]] = {
    # ML/AI Conferences (from existing fact_checker.py)
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
        "world wide web",
        "international world wide web conference",
        "international conference on world wide web",
        "proceedings of the web conference",
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


def _normalize_venue_for_matching(venue: str) -> str:
    """Normalize venue string for matching.

    Args:
        venue: Raw venue name

    Returns:
        Normalized venue string
    """
    venue_norm = venue.lower().strip()

    # Remove common prefixes
    for prefix in ["proceedings of the ", "proceedings of ", "proc. ", "in "]:
        if venue_norm.startswith(prefix):
            venue_norm = venue_norm[len(prefix) :]

    # Remove years
    venue_norm = re.sub(r"\b\d{4}\b", "", venue_norm)
    venue_norm = " ".join(venue_norm.split()).strip()

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

    for canonical, alias_set in aliases.items():
        all_names = alias_set | {canonical}
        for name in all_names:
            if len(name) <= 3:
                continue
            # Require substantial overlap for substring matching
            shorter, longer = sorted([name, venue_norm], key=len)
            if len(shorter) / len(longer) < 0.4:
                continue  # Too different in length for substring match
            if name == venue_norm or name in venue_norm or venue_norm in name:
                return canonical

    return None
