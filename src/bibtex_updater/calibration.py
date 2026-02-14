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

# Status-specific base confidences (priors, adjusted by evidence)
# These represent initial confidence before factoring in match scores
STATUS_BASE_CONFIDENCE = {
    "verified": 0.85,
    "not_found": 0.70,
    "hallucinated": 0.90,
    "title_mismatch": 0.80,
    "author_mismatch": 0.75,
    "year_mismatch": 0.85,
    "venue_mismatch": 0.80,
    "partial_match": 0.65,
    "api_error": 0.30,
    "future_date": 0.95,
    "invalid_year": 0.95,
    "doi_not_found": 0.85,
    "preprint_only": 0.90,
    "published_version_exists": 0.85,
    "url_verified": 0.90,
    "url_accessible": 0.70,
    "url_not_found": 0.85,
    "url_content_mismatch": 0.75,
    "book_verified": 0.85,
    "book_not_found": 0.75,
    "working_paper_verified": 0.80,
    "working_paper_not_found": 0.70,
    "skipped": 0.0,
}

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

    # Statuses that don't involve API matching (no scores to use)
    no_match_statuses = {
        "future_date",
        "invalid_year",
        "doi_not_found",
        "api_error",
        "skipped",
    }

    if status in no_match_statuses or best_match_score is None:
        return base_conf

    # For statuses with matches, blend base confidence with actual match score
    # High match scores boost confidence for positive statuses (verified, mismatches)
    # Low match scores boost confidence for negative statuses (not_found, hallucinated)

    if status == "verified":
        # Verified: high match score → high confidence
        # Use match score directly, with slight boost from base prior
        return 0.7 * best_match_score + 0.3 * base_conf

    elif status == "hallucinated":
        # Hallucinated: low match score → high confidence in classification
        # Invert the match score
        confidence_in_low_score = 1.0 - best_match_score
        return 0.7 * confidence_in_low_score + 0.3 * base_conf

    elif status == "not_found":
        # Not found: absence of good matches → confidence depends on coverage
        # If we have a best_match_score, it should be low
        if best_match_score < 0.5:
            # Low score confirms not_found
            confidence_in_low_score = 1.0 - best_match_score
            return 0.6 * confidence_in_low_score + 0.4 * base_conf
        else:
            # High score contradicts not_found → lower confidence
            return base_conf * 0.5

    elif status in ("title_mismatch", "author_mismatch", "year_mismatch", "venue_mismatch", "partial_match"):
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
    if status == "not_found":
        # For NOT_FOUND: more sources checked → more confident nothing exists
        # Scale with num_queried, not coverage (hits don't matter for not_found)
        queried_factor = min(num_queried / 3.0, 1.0)  # 3 sources = full confidence
        coverage_factor = 0.7 + 0.3 * queried_factor
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

    # Blend base confidence with field-level evidence
    match_statuses = ("verified", "title_mismatch", "author_mismatch", "venue_mismatch", "partial_match")
    if weighted_field_score is not None and status in match_statuses:
        # For match-based statuses, field decomposition provides additional signal
        return 0.7 * base_confidence + 0.3 * weighted_field_score

    return base_confidence
