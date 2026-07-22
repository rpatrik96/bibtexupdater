"""Chain the resolver and the fact-checker in one process.

Rationale ("clean bib, fast"): when the resolver actively upgrades a preprint, it
pulls in a record straight from a bibliographic database (Crossref/DBLP/OpenAlex/
OpenReview) after matching the entry's own title+authors, and -- since
``Updater.update_entry`` now rebuilds the entry atomically from that record -- the
result is a clean, internally consistent published entry. Re-verifying it would be
redundant. So the chain resolves first, then fact-checks ONLY the entries the
resolver did not vouch for, sharing one ``HttpClient``/cache/limiter so the
checker's queries also hit the warm cache the resolver populated.

Entries that are NOT actively upgraded -- non-preprints, preprints the resolver
could not resolve, failures -- still go to the checker, where verification matters.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

# Entries with one of these resolver actions were upgraded to (or already carry,
# from a prior run) a clean bibliographic-database record, so re-verifying them is
# redundant. Everything else is fact-checked.
_RESOLVED_ACTIONS = frozenset({"upgraded", "skipped_resolved"})


def select_entries_to_check(results: list[Any]) -> list[dict[str, Any]]:
    """Return the updated entries that still need fact-checking.

    Entries the resolver upgraded now, or resolved in a prior run
    (``skipped_resolved``), are trusted and excluded.
    """
    return [r.updated for r in results if r is not None and r.action not in _RESOLVED_ACTIONS]


def _status_value(status: Any) -> str:
    """Normalize a FactCheckStatus enum (or plain string) to its string value."""
    return getattr(status, "value", status)


# Trust thresholds for deciding whether an upgrade is safe to skip-check. These
# are NON-COMPENSATORY (both must hold) and stricter than the resolver's own
# compensatory 0.85 accept gate -- they mirror the checker's separate title/author
# bars, so a perfect author score cannot rescue a weak title. Tunable.
TRUST_TITLE_MIN = 0.90
TRUST_AUTHOR_MIN = 0.80
# A published record may post-date the cited (preprint) year by a few years; a
# record EARLIER than the cited year, or implausibly far after, is suspicious.
TRUST_YEAR_FORWARD_MAX = 5


@dataclass
class UpgradeTrust:
    """Whether an upgrade is confidently trustworthy (safe to skip the check)."""

    trusted: bool
    title_score: float
    author_score: float
    reasons: list[str] = field(default_factory=list)


def _int_year(value: Any) -> int | None:
    import re

    if value is None:
        return None
    m = re.search(r"\d{4}", str(value))
    return int(m.group(0)) if m else None


def _norm_doi(doi: Any) -> str | None:
    if not doi:
        return None
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix) :]
    return d or None


def classify_upgrade(
    result: Any,
    *,
    title_min: float = TRUST_TITLE_MIN,
    author_min: float = TRUST_AUTHOR_MIN,
    year_forward_max: int = TRUST_YEAR_FORWARD_MAX,
) -> UpgradeTrust:
    """Classify an upgrade (original -> resolved record) as trusted or flagged.

    Cheap and network-free: it compares the ORIGINAL entry's claim against the
    upgraded metadata (which is the resolved record after the atomic rewrite).
    Non-compensatory title+author bars plus a year-corroboration check; an upgrade
    is trusted only if every check passes.
    """
    from rapidfuzz.fuzz import token_sort_ratio

    from .utils import authors_last_names, jaccard_similarity, normalize_title_for_match

    orig = result.original or {}
    upd = result.updated or {}

    title_score = (
        token_sort_ratio(
            normalize_title_for_match(orig.get("title", "")),
            normalize_title_for_match(upd.get("title", "")),
        )
        / 100.0
    )
    author_score = jaccard_similarity(
        authors_last_names(orig.get("author", ""))[:3],
        authors_last_names(upd.get("author", ""))[:3],
    )

    reasons: list[str] = []
    if title_score < title_min:
        reasons.append(f"title match {title_score:.2f} < {title_min:.2f}")
    if author_score < author_min:
        reasons.append(f"author match {author_score:.2f} < {author_min:.2f}")

    oy, ry = _int_year(orig.get("year")), _int_year(upd.get("year"))
    if oy and ry:
        if ry < oy:
            reasons.append(f"record year {ry} precedes cited year {oy}")
        elif ry - oy > year_forward_max:
            reasons.append(f"record year {ry} is {ry - oy}y after cited year {oy}")

    return UpgradeTrust(
        trusted=not reasons,
        title_score=title_score,
        author_score=author_score,
        reasons=reasons,
    )


def decide_flagged(upd_entry: dict[str, Any], check_result: Any) -> tuple[str, list[str]]:
    """Adjudicate a flagged upgrade against an independent check of the original.

    Returns ("keep"|"revert", reasons). The upgrade is reverted to the original
    entry when the independent retrieval lands on a DIFFERENT record (DOI
    disagreement) or the original verifies as problematic -- both signal the
    resolver likely matched the wrong paper.
    """
    reasons: list[str] = []
    status = _status_value(check_result.status) if check_result is not None else "not_checked"

    rec_doi = _norm_doi(upd_entry.get("doi"))
    best_match = getattr(check_result, "best_match", None) if check_result is not None else None
    bm_doi = _norm_doi(getattr(best_match, "doi", None)) if best_match is not None else None
    if rec_doi and bm_doi and rec_doi != bm_doi:
        reasons.append(f"independent retrieval disagrees (found {bm_doi}, resolver chose {rec_doi})")

    if _is_problematic(status):
        reasons.append(f"original verifies as {status}")

    return ("revert" if reasons else "keep"), reasons


def build_chain_report(
    results: list[Any],
    check_results: list[Any],
    *,
    trust_by_key: dict[str, Any] | None = None,
    decision_by_key: dict[str, tuple[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Merge resolver actions, trust classification, and checker verdicts.

    Trusted upgrades are reported as ``skipped_resolved`` (skipped the check).
    Flagged upgrades carry their flag reasons, the keep/revert decision, and the
    independent check status -- so no risky rewrite is silent. Non-upgraded entries
    carry the checker's status (or ``not_checked``).
    """
    trust_by_key = trust_by_key or {}
    decision_by_key = decision_by_key or {}
    checks = {cr.entry_key: cr for cr in check_results}
    entries: list[dict[str, Any]] = []
    counts = {"resolved_skipped": 0, "checked": 0, "flagged": 0, "reverted": 0}
    for r in results:
        if r is None:
            continue
        key = r.updated.get("ID") or r.original.get("ID")
        trust = trust_by_key.get(key)
        if r.action in _RESOLVED_ACTIONS and (trust is None or trust.trusted):
            entries.append({"key": key, "action": r.action, "trust": "trusted", "check": "skipped_resolved"})
            counts["resolved_skipped"] += 1
        elif r.action in _RESOLVED_ACTIONS:
            decision, dreasons = decision_by_key.get(key, ("keep", []))
            cr = checks.get(key)
            status = _status_value(cr.status) if cr is not None else "not_checked"
            entries.append(
                {
                    "key": key,
                    "action": r.action,
                    "trust": "flagged",
                    "decision": decision,
                    "check": status,
                    "reasons": list(getattr(trust, "reasons", [])) + list(dreasons),
                }
            )
            counts["flagged"] += 1
            if decision == "revert":
                counts["reverted"] += 1
            if cr is not None:
                counts["checked"] += 1
        else:
            cr = checks.get(key)
            status = _status_value(cr.status) if cr is not None else "not_checked"
            entries.append({"key": key, "action": r.action, "trust": "n/a", "check": status})
            if cr is not None:
                counts["checked"] += 1
    return {"entries": entries, "summary": {"total": sum(1 for r in results if r is not None), **counts}}


@dataclass
class ChainResult:
    """Outcome of a resolve->check run."""

    results: list[Any] = field(default_factory=list)
    cleaned_entries: list[dict[str, Any]] = field(default_factory=list)
    check_results: list[Any] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    # The checker processor that produced ``check_results``. The drivers need it to
    # emit the same JSON report and strict verdict as the plain checker CLI.
    processor: Any = None


def run_chain(
    entries: list[dict[str, Any]],
    resolve_args: Any,
    logger: logging.Logger,
    *,
    check_args: Any,
) -> ChainResult:
    """Resolve preprints, then fact-check only the non-upgraded entries.

    A single ``HttpClient`` (built by the checker factory) is shared by both
    pipelines, so the resolver and the checker reuse one cache and rate limiter.
    """
    from .fact_checker import build_checker_processor
    from .updater import (
        build_main_components,
        order_results_by_entries,
        process_entries_parallel,
    )

    # Build the checker first; it owns the shared HttpClient/cache/limiter.
    processor, shared_http = build_checker_processor(
        check_args,
        logger,
        strict_mode=bool(getattr(check_args, "strict", False)),
        strict_warn_cnv=bool(getattr(check_args, "strict_warn_cnv", False)),
    )

    # Resolve, reusing that http (warm cache + one rate limiter).
    components = build_main_components(resolve_args, logger, http=shared_http)
    results = process_entries_parallel(
        entries,
        components.detector,
        components.resolver,
        components.updater,
        logger,
        getattr(resolve_args, "max_workers", 4),
        force_recheck=bool(getattr(resolve_args, "force_recheck", False)),
    )
    results = order_results_by_entries(results, entries)

    # Classify each upgrade. Trusted upgrades are skipped; flagged ones are
    # independently verified (their ORIGINAL claim is fact-checked) so a risky
    # rewrite is never applied silently.
    trust_by_key: dict[str, Any] = {}
    flagged_keys: set[str] = set()
    for r in results:
        key = r.updated.get("ID") or r.original.get("ID")
        if r.action == "upgraded":
            trust = classify_upgrade(r)
            trust_by_key[key] = trust
            if not trust.trusted:
                flagged_keys.add(key)
        elif r.action == "skipped_resolved":
            # Resolved in a prior run; nothing to re-compare, treat as trusted.
            trust_by_key[key] = UpgradeTrust(trusted=True, title_score=1.0, author_score=1.0)

    # Check the non-upgraded entries (their updated form) plus the ORIGINAL claim
    # of every flagged upgrade (to adjudicate keep vs revert).
    non_upgraded = [r.updated for r in results if r.action not in _RESOLVED_ACTIONS]
    flagged_originals = [r.original for r in results if (r.updated.get("ID") or r.original.get("ID")) in flagged_keys]
    check_inputs = non_upgraded + flagged_originals

    trusted_upgrades = sum(1 for r in results if r.action in _RESOLVED_ACTIONS) - len(flagged_keys)
    logger.info(
        "Resolver upgraded %d/%d (%d trusted, %d flagged for review); fact-checking %d entries",
        sum(1 for r in results if r.action in _RESOLVED_ACTIONS),
        len(results),
        trusted_upgrades,
        len(flagged_keys),
        len(check_inputs),
    )

    check_results: list[Any] = []
    if check_inputs:
        check_results = processor.process_entries(
            check_inputs,
            jsonl_path=getattr(check_args, "jsonl", None),
            max_workers=getattr(check_args, "workers", 8),
        )
    checks_by_key = {cr.entry_key: cr for cr in check_results}

    # Adjudicate flagged upgrades and assemble the final (clean) entries: revert a
    # flagged upgrade to its original when independent verification disagrees.
    decision_by_key: dict[str, tuple[str, list[str]]] = {}
    cleaned_entries: list[dict[str, Any]] = []
    for r in results:
        key = r.updated.get("ID") or r.original.get("ID")
        if key in flagged_keys:
            decision, dreasons = decide_flagged(r.updated, checks_by_key.get(key))
            decision_by_key[key] = (decision, dreasons)
            cleaned_entries.append(r.original if decision == "revert" else r.updated)
        else:
            cleaned_entries.append(r.updated)

    report = build_chain_report(results, check_results, trust_by_key=trust_by_key, decision_by_key=decision_by_key)
    return ChainResult(
        results=results,
        cleaned_entries=cleaned_entries,
        check_results=check_results,
        report=report,
        processor=processor,
    )


# ------------- arg bridging -------------
# Each flag lives on one CLI but the chain needs both arg namespaces. We
# synthesize the missing one from its own parser's defaults (so every option is
# present) and propagate the shared knobs (cache, S2 key, rate limit, workers).


def _bridge_check_args(update_args: Any) -> Any:
    from .fact_checker import build_parser as build_check_parser

    check_args = build_check_parser().parse_args(["__chain__"])
    check_args.cache_file = getattr(update_args, "cache", check_args.cache_file)
    check_args.no_cache = bool(getattr(update_args, "no_cache", False))
    check_args.s2_api_key = getattr(update_args, "s2_api_key", None)
    check_args.rate_limit = getattr(update_args, "rate_limit", check_args.rate_limit)
    check_args.workers = getattr(update_args, "max_workers", check_args.workers)
    check_args.verbose = bool(getattr(update_args, "verbose", False))
    return check_args


def _bridge_resolve_args(check_args: Any) -> Any:
    from .updater import build_arg_parser as build_update_parser

    resolve_args = build_update_parser().parse_args(["__chain__"])
    resolve_args.cache = getattr(check_args, "cache_file", resolve_args.cache)
    resolve_args.no_cache = bool(getattr(check_args, "no_cache", False))
    resolve_args.s2_api_key = getattr(check_args, "s2_api_key", None)
    resolve_args.rate_limit = getattr(check_args, "rate_limit", resolve_args.rate_limit)
    resolve_args.max_workers = getattr(check_args, "workers", resolve_args.max_workers)
    resolve_args.verbose = bool(getattr(check_args, "verbose", False))
    return resolve_args


# ------------- bib IO + reporting -------------


def _load_entries(paths: list[str], logger: logging.Logger) -> list[dict[str, Any]]:
    import bibtexparser

    entries: list[dict[str, Any]] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                db = bibtexparser.load(f)
            entries.extend(db.entries)
            logger.info("Loaded %d entries from %s", len(db.entries), path)
        except FileNotFoundError:
            logger.error("File not found: %s", path)
        except Exception as e:  # noqa: BLE001 - surface a parse error, keep going
            logger.error("Failed to parse %s: %s", path, e)
    return entries


def _write_bib(entries: list[dict[str, Any]], path: str, logger: logging.Logger) -> None:
    import bibtexparser

    from .updater import BibWriter

    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = entries
    BibWriter().dump_to_file(db, path)
    logger.info("Wrote %d entries to %s", len(entries), path)


def _default_resolved_out(bibfiles: list[str]) -> str:
    base = bibfiles[0] if bibfiles else "references.bib"
    root, ext = os.path.splitext(base)
    return f"{root}.resolved{ext or '.bib'}"


# Statuses that are NOT positive evidence of a problem (verified or simply
# could-not-verify). Anything else is surfaced as a warning. Kept as substrings
# so the report stays robust to the checker's exact status vocabulary.
_OK_STATUS_HINTS = ("verified", "skipped_resolved", "accessible", "not_checked")
_ABSTAINED_HINTS = ("not_found", "unconfirmed", "preprint_year")


def _is_problematic(status: str) -> bool:
    s = str(status).lower()
    if any(h in s for h in _OK_STATUS_HINTS) or any(h in s for h in _ABSTAINED_HINTS):
        return False
    return s not in {"", "api_error", "skipped"}


def _print_report(report: dict[str, Any], logger: logging.Logger) -> None:
    summary = report.get("summary", {})
    logger.info("=" * 60)
    logger.info(
        "CHAIN SUMMARY: %d entries -- %d trusted-upgrade (skipped), %d fact-checked, " "%d flagged, %d reverted",
        summary.get("total", 0),
        summary.get("resolved_skipped", 0),
        summary.get("checked", 0),
        summary.get("flagged", 0),
        summary.get("reverted", 0),
    )
    # No silent rewrites: every low-confidence upgrade is surfaced with its reasons
    # and whether it was kept (for review) or reverted to the original.
    flagged = [e for e in report.get("entries", []) if e.get("trust") == "flagged"]
    if flagged:
        logger.warning("Flagged upgrades (low-confidence rewrites): %d", len(flagged))
        for e in flagged:
            verb = "REVERTED to original" if e.get("decision") == "revert" else "kept -- review"
            logger.warning("  %s -> %s: %s", e.get("key"), verb, "; ".join(e.get("reasons", [])))
    problematic = [e for e in report.get("entries", []) if _is_problematic(e.get("check", ""))]
    if problematic:
        logger.warning("Problematic entries (positive evidence of a problem): %d", len(problematic))
        for e in problematic:
            logger.warning("  %s: %s", e.get("key"), e.get("check"))


def _write_check_report(check_args: Any, result: ChainResult, logger: logging.Logger) -> None:
    """Persist the checker's JSON report, matching the plain ``bibtex-check`` schema.

    Both chain drivers return before ``fact_checker.main``'s report-writing tail, so
    a chain run has to discharge ``--report`` itself or the flag is silently dropped.
    """
    report_path = getattr(check_args, "report", None)
    if not report_path:
        return

    processor = result.processor
    doc: dict[str, Any] = (
        processor.generate_json_report(result.check_results)
        if processor is not None
        else {"summary": {}, "entries": []}
    )
    # Trusted upgrades never reach the checker, so ``entries`` covers only the
    # verified subset. The chain view is attached alongside it so the report still
    # accounts for every input entry and says which ones were skipped as upgrades.
    doc["chain"] = result.report
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.error("Failed to write JSON report %s: %s", report_path, e)
        return
    logger.info("JSON report written to %s", report_path)


def _strict_exit_code(check_args: Any, result: ChainResult, logger: logging.Logger) -> int:
    """Mirror the plain checker's strict gate: only positive-evidence problems fail.

    Abstentions (could-not-verify) never fail on their own; ``--strict-warn-cnv``
    opts into failing on them too.
    """
    if not getattr(check_args, "strict", False):
        return 0
    processor = result.processor
    if processor is None:
        return 0

    from .fact_checker import FactCheckStatus

    summary = processor.generate_summary(result.check_results)
    problem_count = summary.get("problematic_count", 0)
    if problem_count > 0:
        logger.warning("Strict mode: %d PROBLEMATIC entries (positive evidence of a problem)", problem_count)
        return 4
    cnv_warn_count = summary.get("status_counts", {}).get(FactCheckStatus.STRICT_WARN_CNV.value, 0)
    if getattr(check_args, "strict_warn_cnv", False) and cnv_warn_count > 0:
        logger.warning(
            "Strict mode (warn-cnv): %d STRICT_WARN_CNV entries (could-not-verify, opt-in fail)",
            cnv_warn_count,
        )
        return 4
    return 0


def _write_update_report(update_args: Any, result: ChainResult, logger: logging.Logger) -> None:
    """Persist the resolver's JSONL original->updated report (``bibtex-update --report``)."""
    report_path = getattr(update_args, "report", None)
    if not report_path:
        return

    from .updater import write_report_line

    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            for res in result.results:
                if res is not None:
                    write_report_line(fh, res)
    except OSError as e:
        logger.error("Failed to write JSONL report %s: %s", report_path, e)
        return
    logger.info("JSONL report written to %s", report_path)


def run_update_then_check(update_args: Any, logger: logging.Logger) -> int:
    """Driver for ``bibtex-update --then-check``: upgrade, write clean bib, verify the rest."""
    entries = _load_entries(list(getattr(update_args, "inputs", []) or []), logger)
    if not entries:
        logger.error("No entries found in input files")
        return 1

    check_args = _bridge_check_args(update_args)
    result = run_chain(entries, update_args, logger, check_args=check_args)

    out_path = getattr(update_args, "output", None)
    if not out_path and getattr(update_args, "in_place", False):
        out_path = update_args.inputs[0]
    if out_path:
        _write_bib(result.cleaned_entries, out_path, logger)
    else:
        logger.warning("No -o/--output or --in-place given; cleaned bib not written")

    _print_report(result.report, logger)
    _write_update_report(update_args, result, logger)
    return 0


def run_check_resolve_first(check_args: Any, logger: logging.Logger) -> int:
    """Driver for ``bibtex-check --resolve-first``: upgrade, always persist clean bib, verify the rest."""
    entries = _load_entries(list(getattr(check_args, "bibfiles", []) or []), logger)
    if not entries:
        logger.error("No entries found in input files")
        return 1

    resolve_args = _bridge_resolve_args(check_args)
    result = run_chain(entries, resolve_args, logger, check_args=check_args)

    out_path = getattr(check_args, "resolved_out", None) or _default_resolved_out(
        list(getattr(check_args, "bibfiles", []) or [])
    )
    _write_bib(result.cleaned_entries, out_path, logger)

    _print_report(result.report, logger)
    _write_check_report(check_args, result, logger)
    if getattr(check_args, "jsonl", None):
        logger.info("JSONL report streamed to %s (%d entries)", check_args.jsonl, len(result.check_results))
    return _strict_exit_code(check_args, result, logger)
