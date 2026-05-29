#!/usr/bin/env python3
"""Calibration module for fact-checker confidence scores.

This module provides calibrated confidence estimation for fact-check results,
replacing hardcoded per-status confidence values with evidence-based scoring
that incorporates:
- Actual match scores from bibliographic field comparisons
- Per-field confidence decomposition with configurable weights
- Query coverage (how many sources were queried vs. errored)
- Status-specific priors adjusted by empirical evidence

The calibration is designed to improve Expected Calibration Error (ECE)
by making confidence values reflect actual match quality rather than
just the assigned status.
"""

from __future__ import annotations

__all__ = [
    "compute_confidence_from_scores",
    "decompose_field_confidence",
    "status_aware_confidence",
    "calibrate_result",
    "DEFAULT_FIELD_WEIGHTS",
    "STATUS_BASE_CONFIDENCE",
]

# Status-specific base confidences (priors), grounded in the fact-checker's
# THREE-WAY verdict model. The number is "how confident are we that the assigned
# verdict is the right call", NOT a probability that the entry is correct.
#
# Three buckets, with a strict ordering the priors MUST respect:
#
#   CLEARLY-CORRECT  (high, ~0.85-0.90): a positive record confirms the entry.
#       verified / preprint_only / published_version_exists / *_verified.
#
#   CLEARLY-PROBLEM  (high): positive evidence the entry is wrong.
#       - STRONGEST (~0.92-0.95): the evidence is self-contained and not a fuzzy
#         field comparison -- future_date / invalid_year (arithmetic), a
#         fabricated DOI that resolves elsewhere (doi_mismatch) or an arXiv ID
#         that resolves elsewhere (arxiv_id_mismatch), and hallucinated (no real
#         paper / chimeric title stitched from several papers).
#       - SOFTER (~0.72-0.80): a single bibliographic field disagrees while the
#         rest match (title/author/year/venue/url_content mismatch, partial_match).
#         One fuzzy-compared field is weaker positive evidence than the above.
#
#   DON'T-KNOW / abstention  (LOW, near-neutral ~0.40-0.50): the tool could not
#       decide. By construction this MUST sit below BOTH confident buckets -- an
#       entry we failed to adjudicate must never report high confidence either
#       way. Covers not_found / unconfirmed / api_error / doi_not_found and the
#       book/working-paper/url "not found" variants, plus url_accessible (HTTP
#       200 with no content check) which only weakly informs the verdict.
#
# Anchors (do not creep): confident buckets >= 0.72; abstentions <= 0.50.
_CONF_CORRECT = 0.88  # CLEARLY-CORRECT anchor
_PROB_STRONG = 0.93  # CLEARLY-PROBLEM, self-contained positive evidence
_PROB_SOFT = 0.78  # CLEARLY-PROBLEM, single fuzzy-compared field disagrees
_ABSTAIN = 0.45  # DON'T-KNOW, near-neutral, strictly below both above
STATUS_BASE_CONFIDENCE = {
    # --- CLEARLY-CORRECT: a positive record confirms the entry ---
    "verified": _CONF_CORRECT,
    "preprint_only": _CONF_CORRECT,
    "published_version_exists": _CONF_CORRECT,
    "url_verified": _CONF_CORRECT,
    "book_verified": _CONF_CORRECT,
    "working_paper_verified": _CONF_CORRECT,
    # --- CLEARLY-PROBLEM (strong): self-contained positive evidence ---
    "hallucinated": _PROB_STRONG,  # no real paper / chimeric title
    "future_date": _PROB_STRONG,  # arithmetic, not a fuzzy match
    "invalid_year": _PROB_STRONG,  # arithmetic, not a fuzzy match
    "doi_mismatch": _PROB_STRONG,  # cited DOI resolves to a different paper
    "arxiv_id_mismatch": _PROB_STRONG,  # cited arXiv ID resolves elsewhere
    # --- CLEARLY-PROBLEM (soft): one disagreeing field, rest match ---
    "title_mismatch": _PROB_SOFT,
    "author_mismatch": _PROB_SOFT,
    "given_name_substitution": _PROB_SOFT,  # surnames match, a given name is a different person
    "year_mismatch": _PROB_SOFT,
    "venue_mismatch": _PROB_SOFT,
    "partial_match": _PROB_SOFT,
    "url_content_mismatch": _PROB_SOFT,
    # --- DON'T-KNOW / abstention: could not adjudicate, near-neutral ---
    "not_found": _ABSTAIN,
    "unconfirmed": _ABSTAIN,
    "api_error": _ABSTAIN,
    "doi_not_found": _ABSTAIN,
    "url_not_found": _ABSTAIN,
    "book_not_found": _ABSTAIN,
    "working_paper_not_found": _ABSTAIN,
    "url_accessible": _ABSTAIN,  # 200 only, no content check -> weak signal
    # --- not verifiable ---
    "skipped": 0.0,
}

# Abstention ("don't-know") statuses. These must stay near-neutral: their
# confidence is capped below the confident buckets and is NOT boosted by
# inverting a low match score or by counting how many sources were queried.
# (Mirrors fact_checker.ABSTAINED_STATUS_VALUES, kept local to avoid importing
# the concurrently-edited module.)
_ABSTAIN_STATUSES = frozenset(
    {
        "not_found",
        "unconfirmed",
        "api_error",
        "doi_not_found",
        "url_not_found",
        "book_not_found",
        "working_paper_not_found",
        "url_accessible",
    }
)

# Self-contained positive-evidence problems whose verdict does NOT come from a
# scored title-search candidate (arithmetic on the year, or an identifier that
# resolves to a different paper). Source coverage is irrelevant to them, so they
# must not be docked the "few sources returned hits" penalty -- otherwise these
# strongest problems would calibrate below soft single-field mismatches.
_SELF_EVIDENT_PROBLEM_STATUSES = frozenset(
    {
        "future_date",
        "invalid_year",
        "doi_mismatch",
        "arxiv_id_mismatch",
    }
)

# Default field importance weights for confidence decomposition
# Higher weight = more discriminative for determining correctness
DEFAULT_FIELD_WEIGHTS = {
    "title": 0.40,  # Most discriminative field
    "author": 0.25,
    "year": 0.15,
    "venue": 0.20,
}


def compute_confidence_from_scores(
    status: str,
    best_match_score: float | None,
    field_comparisons: dict,
) -> float:
    """Compute calibrated confidence from actual match scores.

    Instead of hardcoded confidence per status, derive confidence from:
    - The best candidate match score (0.0-1.0)
    - Individual field comparison scores
    - The status itself (as a prior)

    Args:
        status: FactCheckStatus value as string (e.g., "verified", "not_found")
        best_match_score: Overall match score for best candidate (0.0-1.0), or None
        field_comparisons: Dict mapping field names to FieldComparison-like dicts
                          with keys: similarity_score, matches

    Returns:
        Confidence value 0.0-1.0 that the entry is correctly classified.
    """
    # Get base confidence from status
    base_conf = STATUS_BASE_CONFIDENCE.get(status, 0.5)

    # Abstentions ("don't-know") return their low base prior unchanged: a verdict
    # the tool could not make must NOT be inflated by inverting a low match score.
    # (A high match score would contradict the abstention, so cap at the prior.)
    if status in _ABSTAIN_STATUSES:
        if best_match_score is not None and best_match_score > 0.5:
            # A strong candidate contradicts the abstention -> even less certain.
            return base_conf * 0.5
        return base_conf

    # Statuses that don't involve API matching (no scores to use)
    no_match_statuses = {
        "future_date",
        "invalid_year",
        # Pre-search positive-evidence check: the verdict comes from the DOI's /
        # arXiv ID's own record, not from a scored title-search candidate.
        "doi_mismatch",
        "arxiv_id_mismatch",
        "skipped",
    }

    if status in no_match_statuses or best_match_score is None:
        return base_conf

    # For statuses with matches, blend base confidence with actual match score.
    # High match scores boost confidence for positive statuses (verified);
    # low match scores boost confidence for hallucinated.

    if status == "verified":
        # Verified: high match score → high confidence
        # Use match score directly, with slight boost from base prior
        return 0.7 * best_match_score + 0.3 * base_conf

    elif status == "hallucinated":
        # Hallucinated: low match score → high confidence in classification
        # Invert the match score
        confidence_in_low_score = 1.0 - best_match_score
        return 0.7 * confidence_in_low_score + 0.3 * base_conf

    elif status in (
        "title_mismatch",
        "author_mismatch",
        "year_mismatch",
        "venue_mismatch",
        "partial_match",
        "url_content_mismatch",
    ):
        # Field-specific mismatches: moderate match score expected
        # Use base confidence adjusted by how extreme the mismatch is
        # If overall match is high despite mismatch, confidence should be lower
        if best_match_score > 0.8:
            # High overall score despite mismatch → less confident in mismatch classification
            return base_conf * 0.8
        else:
            # Moderate score confirms mismatch classification
            return base_conf

    # For other statuses (web, book, working paper), use base confidence
    return base_conf


def decompose_field_confidence(
    field_comparisons: dict,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Decompose confidence into per-field contributions.

    Args:
        field_comparisons: Dict mapping field names to FieldComparison-like dicts
                          with keys: similarity_score, matches
        weights: Optional custom field weights. If None, uses DEFAULT_FIELD_WEIGHTS.

    Returns:
        Dict mapping field names to weighted confidence contributions (0.0-1.0).
        The sum of all contributions approximates the overall confidence.
    """
    if weights is None:
        weights = DEFAULT_FIELD_WEIGHTS

    contributions = {}

    for field_name, comparison in field_comparisons.items():
        # Get similarity score (0.0-1.0)
        similarity = comparison.get("similarity_score", 0.0)

        # Get field weight (default to 0.0 for unknown fields)
        weight = weights.get(field_name, 0.0)

        # Contribution is weighted similarity
        contributions[field_name] = weight * similarity

    return contributions


def status_aware_confidence(
    status: str,
    best_match_score: float | None,
    field_comparisons: dict,
    sources_queried: list[str],
    sources_with_hits: list[str],
    errors: list[str],
) -> float:
    """Compute confidence factoring in query coverage and errors.

    Adjustments:
    - Fewer sources queried → lower confidence (less evidence)
    - More errors → lower confidence (uncertain)
    - All sources agree → higher confidence
    - NOT_FOUND with all 3 sources queried → high confidence it's fabricated
    - NOT_FOUND with only 1 source (2 errored) → low confidence

    Args:
        status: FactCheckStatus value as string
        best_match_score: Overall match score (0.0-1.0) or None
        field_comparisons: Field comparison results
        sources_queried: List of API sources that were attempted
        sources_with_hits: List of API sources that returned results
        errors: List of error messages encountered

    Returns:
        Final calibrated confidence 0.0-1.0.
    """
    # Start with base confidence from scores
    base_confidence = compute_confidence_from_scores(status, best_match_score, field_comparisons)

    # Compute coverage penalty
    num_queried = len(sources_queried)
    num_with_hits = len(sources_with_hits)
    num_errors = len(errors)

    if num_queried == 0:
        # No sources queried → very low confidence
        return base_confidence * 0.3

    # Coverage ratio: what fraction of queried sources returned results?
    coverage = num_with_hits / num_queried if num_queried > 0 else 0.0

    # Error penalty: each error reduces confidence
    error_penalty = 1.0 - (0.15 * min(num_errors, 3))  # Cap at 3 errors for penalty

    # Coverage boost/penalty depends on status
    if status in _ABSTAIN_STATUSES:
        # Abstentions stay near-neutral. We deliberately do NOT scale confidence
        # up with the number of sources queried: "not found in N databases" is
        # still a don't-know, and querying more sources must not push an
        # abstention toward a confident verdict. Hold the low prior flat.
        coverage_factor = 1.0
    elif status in _SELF_EVIDENT_PROBLEM_STATUSES:
        # Self-contained problems (year arithmetic, mis-resolving DOI/arXiv ID):
        # the verdict isn't a scored candidate match, so source coverage is
        # irrelevant. Don't penalize them for "no hits"; hold the strong prior.
        coverage_factor = 1.0
    elif status in ("verified", "title_mismatch", "author_mismatch", "year_mismatch", "venue_mismatch"):
        # For positive matches: coverage matters less (one good match is enough)
        # But still penalize if only 1 source and it has errors
        if num_with_hits == 0:
            coverage_factor = 0.5
        else:
            coverage_factor = 0.9 + 0.1 * min(coverage, 1.0)  # Range: 0.9 to 1.0
    else:
        # For other statuses: linear coverage factor
        coverage_factor = 0.7 + 0.3 * coverage

    # Combine factors
    final_confidence = base_confidence * coverage_factor * error_penalty

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, final_confidence))


def calibrate_result(
    status: str,
    best_match_score: float | None,
    field_comparisons: dict,
    sources_queried: list[str],
    sources_with_hits: list[str],
    errors: list[str],
    field_weights: dict[str, float] | None = None,
) -> float:
    """Full calibration pipeline combining all calibration strategies.

    This is the main entry point for calibrating a fact-check result.
    It combines:
    1. Base confidence from status priors
    2. Evidence from actual match scores
    3. Per-field confidence decomposition
    4. Query coverage and error handling

    Args:
        status: FactCheckStatus value as string
        best_match_score: Overall match score (0.0-1.0) or None
        field_comparisons: Field comparison results (dict with similarity_score, matches)
        sources_queried: List of API sources attempted
        sources_with_hits: List of API sources that returned results
        errors: List of error messages
        field_weights: Optional custom field weights (uses DEFAULT_FIELD_WEIGHTS if None)

    Returns:
        Calibrated confidence value 0.0-1.0.

    Example:
        >>> calibrate_result(
        ...     status="verified",
        ...     best_match_score=0.92,
        ...     field_comparisons={
        ...         "title": {"similarity_score": 0.95, "matches": True},
        ...         "author": {"similarity_score": 0.90, "matches": True},
        ...     },
        ...     sources_queried=["crossref", "dblp", "semanticscholar"],
        ...     sources_with_hits=["crossref", "dblp"],
        ...     errors=[],
        ... )
        0.89  # High confidence due to high match score and good coverage
    """
    # Compute per-field decomposed confidence
    if field_comparisons:
        field_conf = decompose_field_confidence(field_comparisons, field_weights)
        weighted_field_score = sum(field_conf.values())
    else:
        weighted_field_score = None

    # Get status-aware confidence (uses match score + coverage)
    base_confidence = status_aware_confidence(
        status,
        best_match_score,
        field_comparisons,
        sources_queried,
        sources_with_hits,
        errors,
    )

    # Blend base confidence with field-level evidence. Only for statuses where a
    # record was found and per-field scores are meaningful (CLEARLY-CORRECT
    # `verified` and the soft single-field CLEARLY-PROBLEM mismatches). Abstentions
    # are excluded so field scores cannot inflate a don't-know.
    match_statuses = (
        "verified",
        "title_mismatch",
        "author_mismatch",
        "year_mismatch",
        "venue_mismatch",
        "partial_match",
        "url_content_mismatch",
    )
    if weighted_field_score is not None and status in match_statuses:
        # For match-based statuses, field decomposition provides additional signal
        return 0.7 * base_confidence + 0.3 * weighted_field_score

    return base_confidence
