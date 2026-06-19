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


def build_chain_report(results: list[Any], check_results: list[Any]) -> dict[str, Any]:
    """Merge resolver actions and checker verdicts into one per-entry report.

    Resolved entries are reported as ``skipped_resolved``; the rest carry the
    checker's status (or ``not_checked`` if the checker produced no result).
    """
    checks = {cr.entry_key: cr for cr in check_results}
    entries: list[dict[str, Any]] = []
    checked = 0
    resolved_skipped = 0
    for r in results:
        if r is None:
            continue
        key = r.updated.get("ID") or r.original.get("ID")
        if r.action in _RESOLVED_ACTIONS:
            entries.append({"key": key, "action": r.action, "check": "skipped_resolved"})
            resolved_skipped += 1
        else:
            cr = checks.get(key)
            status = _status_value(cr.status) if cr is not None else "not_checked"
            entries.append({"key": key, "action": r.action, "check": status})
            if cr is not None:
                checked += 1
    return {
        "entries": entries,
        "summary": {
            "total": sum(1 for r in results if r is not None),
            "resolved_skipped": resolved_skipped,
            "checked": checked,
        },
    }


@dataclass
class ChainResult:
    """Outcome of a resolve->check run."""

    results: list[Any] = field(default_factory=list)
    cleaned_entries: list[dict[str, Any]] = field(default_factory=list)
    check_results: list[Any] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


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
    cleaned_entries = [r.updated for r in results]

    to_check = select_entries_to_check(results)
    upgraded = sum(1 for r in results if r.action in _RESOLVED_ACTIONS)
    logger.info(
        "Resolver vouched for %d/%d entries; fact-checking the remaining %d",
        upgraded,
        len(results),
        len(to_check),
    )

    check_results: list[Any] = []
    if to_check:
        check_results = processor.process_entries(
            to_check,
            jsonl_path=getattr(check_args, "jsonl", None),
            max_workers=getattr(check_args, "workers", 8),
        )

    report = build_chain_report(results, check_results)
    return ChainResult(
        results=results,
        cleaned_entries=cleaned_entries,
        check_results=check_results,
        report=report,
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
        "CHAIN SUMMARY: %d entries -- %d resolved (skipped check), %d fact-checked",
        summary.get("total", 0),
        summary.get("resolved_skipped", 0),
        summary.get("checked", 0),
    )
    problematic = [e for e in report.get("entries", []) if _is_problematic(e.get("check", ""))]
    if problematic:
        logger.warning("Problematic entries (positive evidence of a problem): %d", len(problematic))
        for e in problematic:
            logger.warning("  %s: %s", e.get("key"), e.get("check"))


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
    return 0
