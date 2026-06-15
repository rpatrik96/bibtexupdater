#!/usr/bin/env python3
"""Evaluate ``bibtex-check`` on a HALLMARK split and report detection metrics.

HALLMARK (https://github.com/rpatrik96/hallmark) is a citation-hallucination
detection benchmark. Each entry carries a gold ``label`` of ``VALID`` or
``HALLUCINATED``. This script runs ``bibtex-check`` over one split, maps the raw
``status`` field to a VALID / HALLUCINATED / abstain verdict, and computes the
detection metrics used in the benchmark report (``benchmarks/HALLMARK.md``):

    - Detection Rate (DR)  = recall on the HALLUCINATED class = TP / (TP + FN)
    - False Positive Rate  = FP / (FP + TN)   (rate of real refs wrongly flagged)
    - Precision            = TP / (TP + FP)
    - F1 (HALLUCINATED)    = harmonic mean of precision and DR
    - MCC                  = Matthews correlation coefficient
    - Coverage             = 1 - abstentions / n   (non-abstaining fraction)

Positive class is HALLUCINATED (the thing the tool should catch).

Reproducibility
---------------
The reported numbers were produced through HALLMARK's thin baseline wrapper,
which runs ``bibtex-check`` and then merges a lightweight, benchmark-side
pre-screening pass (future-date / implausible-year, placeholder authors). That
pre-screening layer lives in HALLMARK, not in bibtex-updater. This script ports
the *non-networked* pre-screening checks so it reproduces the published numbers
standalone; the optional networked DOI-resolution check is gated behind
``--doi-prescreen``. Pass ``--no-prescreen`` to score raw ``bibtex-check`` only.

Usage
-----
    export S2_API_KEY=...   # optional: lifts Semantic Scholar rate limits
    python scripts/eval_hallmark.py \
        --split /path/to/hallmark/data/v1.0/test_public.jsonl \
        --out results_test_public.json

The status -> label mapping below is the canonical bibtex-check status
vocabulary (FactCheckStatus); keep it in sync with the enum in
``src/bibtex_updater/fact_checker.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# bibtex-check status -> {HALLUCINATED, VALID} and the "abstain" set.
# Mirrors HALLMARK's baseline wrapper mapping (hallmark/baselines/bibtexupdater.py),
# which in turn follows the FactCheckStatus enum in fact_checker.py.
# ---------------------------------------------------------------------------
STATUS_TO_LABEL: dict[str, str] = {
    # Core academic verification
    "verified": "VALID",
    "not_found": "HALLUCINATED",
    "title_mismatch": "HALLUCINATED",
    "author_mismatch": "HALLUCINATED",
    "year_mismatch": "HALLUCINATED",
    "venue_mismatch": "HALLUCINATED",
    "partial_match": "HALLUCINATED",
    "hallucinated": "HALLUCINATED",
    "api_error": "VALID",  # conservative: do not flag on transient API errors
    # >=1.2.0 statuses
    "unconfirmed": "VALID",  # could-not-verify abstention -> conservative VALID
    "given_name_substitution": "HALLUCINATED",
    "arxiv_id_mismatch": "HALLUCINATED",
    "doi_mismatch": "HALLUCINATED",
    "title_near_miss": "HALLUCINATED",
    "author_truncated": "HALLUCINATED",
    "nonexistent_venue": "HALLUCINATED",
    "unpublished_at_claimed_venue": "HALLUCINATED",
    "strict_warn_preprint_year": "VALID",
    "strict_warn_cnv": "VALID",
    # Pre-API validation
    "future_date": "HALLUCINATED",
    "invalid_year": "HALLUCINATED",
    "doi_not_found": "HALLUCINATED",
    # Preprint detection
    "preprint_only": "HALLUCINATED",
    "published_version_exists": "VALID",
    # Web / book / working-paper (only with academic_only=False)
    "url_verified": "VALID",
    "url_accessible": "VALID",
    "url_not_found": "HALLUCINATED",
    "url_content_mismatch": "HALLUCINATED",
    "book_verified": "VALID",
    "book_not_found": "HALLUCINATED",
    "working_paper_verified": "VALID",
    "working_paper_not_found": "HALLUCINATED",
    "skipped": "VALID",  # conservative
}

# Statuses that count as an abstention (no academic-database confirmation) for the
# Coverage metric. This is the canonical HALLMARK v1.1.1 definition used to produce
# the committed dev/test per-entry files: an entry abstains iff no academic record
# either confirmed it (``unconfirmed``) or could be located at all (``not_found``).
# Note ``not_found`` still yields a HALLUCINATED *verdict* (no DB record is strong
# evidence of fabrication) — it abstains only for Coverage accounting, separating
# "the tool decided" from "the tool found a record". The remaining sentinels
# (``api_error``/``skipped``/``strict_warn_*``/``missing``) are transient or
# out-of-scope and are likewise non-confirmations.
#
# When bibtex-check emits an explicit ``abstained`` boolean (>=1.2.0), that flag is
# trusted directly (see ``run_bibtex_check``); this set is the fallback and the
# definition the report's Coverage column documents. Keep it in sync with the
# canonical set in the committed per-entry files: {not_found, unconfirmed}.
ABSTAIN_STATUSES: frozenset[str] = frozenset(
    {
        "unconfirmed",  # academic record found but could not confirm -> conservative VALID
        "not_found",  # no academic record located -> HALLUCINATED verdict, abstains for Coverage
        "api_error",
        "skipped",
        "strict_warn_preprint_year",
        "strict_warn_cnv",
        "missing",  # sentinel: bibtex-check produced no record for the key
    }
)

# Benchmark reference year (HALLMARK v1.x freeze). Years strictly above this are
# treated as future-dated. Override with --reference-year for other releases.
DEFAULT_REFERENCE_YEAR = 2026


# ---------------------------------------------------------------------------
# Lightweight, non-networked pre-screening (ported from HALLMARK).
# ---------------------------------------------------------------------------
_CAPITALIZED_DIGIT_TOKEN = re.compile(r"^[A-Z][a-z]*\d+$")
_INITIAL_ONLY_TOKEN = re.compile(r"^[A-Z]\.$")
_BARE_INITIAL_TOKEN = re.compile(r"^[A-Z]$")
_PURE_UPPERCASE_TOKEN = re.compile(r"^[A-Z]{2,}$")


def _author_chunks(author_field: str) -> list[list[str]]:
    chunks: list[list[str]] = []
    for chunk in re.split(r"\s+and\s+", author_field):
        tokens = [p.strip().strip(",") for p in chunk.split() if p.strip().strip(",")]
        if tokens:
            chunks.append(tokens)
    return chunks


def _is_initials_only_name(tokens: list[str]) -> bool:
    if not tokens:
        return False
    return all(_INITIAL_ONLY_TOKEN.match(t) or _BARE_INITIAL_TOKEN.match(t) for t in tokens)


def prescreen_year(fields: dict[str, str], reference_year: int) -> bool:
    """Return True if the year is future-dated or implausibly old (HALLUCINATED)."""
    year_str = fields.get("year")
    if not year_str:
        return False
    try:
        year = int(str(year_str))
    except ValueError:
        return False
    return year > reference_year or year < 1900


def prescreen_authors(fields: dict[str, str]) -> bool:
    """Return True if authors match a placeholder/synthetic pattern (HALLUCINATED).

    Ports check_author_heuristics + check_capitalized_unknown_authors from HALLMARK.
    """
    author_field = fields.get("author", "") or ""
    stripped = author_field.strip()
    if not stripped:
        return False
    if len(stripped) < 3:
        return True
    if stripped.lower() in ("et al.", "et al"):
        return True
    if re.search(r"\bAuthor\d+\b|\bAuthor[A-Z]\b", author_field):
        return True

    authors = re.split(r"\s+and\s+", author_field)
    if len(authors) > 1:
        single_letter = 0
        for author in authors:
            lastname = author.split(",")[0].strip() if "," in author else author.strip().split()[-1:]
            lastname = lastname if isinstance(lastname, str) else (lastname[0] if lastname else "")
            if len(lastname) == 1:
                single_letter += 1
        if single_letter == len(authors):
            return True

    chunks = _author_chunks(author_field)
    if not chunks:
        return False
    tokens = [t for chunk in chunks for t in chunk]
    for token in tokens:
        if _CAPITALIZED_DIGIT_TOKEN.match(token) and not re.fullmatch(r"Author\d+", token):
            return True
    if all(_is_initials_only_name(chunk) for chunk in chunks):
        return True
    pure_upper = sum(1 for t in tokens if _PURE_UPPERCASE_TOKEN.match(t))
    if len(tokens) > 1 and pure_upper / len(tokens) > 0.5:
        return True
    return False


# ---------------------------------------------------------------------------
# Data loading + BibTeX serialization
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    key: str
    btype: str
    fields: dict[str, str]
    label: str


def load_split(path: Path) -> list[Entry]:
    """Load a HALLMARK split, dropping contamination-canary entries."""
    entries: list[Entry] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # Skip benchmark contamination canaries (not scoreable entries).
        if rec.get("source") == "canary" or str(rec.get("bibtex_key", "")).startswith("__canary__"):
            continue
        label = rec["label"]
        if label not in ("VALID", "HALLUCINATED"):
            continue
        entries.append(
            Entry(
                key=rec["bibtex_key"],
                btype=rec.get("bibtex_type", "article"),
                fields={k: str(v) for k, v in (rec.get("fields") or {}).items() if v is not None},
                label=label,
            )
        )
    return entries


def _escape(value: str) -> str:
    return value.replace("{", "").replace("}", "")


def entries_to_bib(entries: list[Entry]) -> str:
    out: list[str] = []
    for e in entries:
        lines = [f"@{e.btype}{{{e.key},"]
        for fk, fv in e.fields.items():
            lines.append(f"  {fk} = {{{_escape(str(fv))}}},")
        lines.append("}")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Run bibtex-check + parse statuses
# ---------------------------------------------------------------------------
def run_bibtex_check(
    entries: list[Entry],
    rate_limit: int,
    s2_api_key: str | None,
    timeout: float,
    academic_only: bool = True,
) -> dict[str, tuple[str, bool, bool, float | None]]:
    """Run bibtex-check; return {bibtex_key: (status, abstained, coverage_incomplete, p_valid)}.

    bibtex-check writes one JSON record per entry with ``key``, ``status`` and an
    explicit ``abstained`` boolean. The temp ``.bib``/JSONL live in a fresh temp
    dir, but the SQLite fact-checker cache is created in the *current working
    directory* (``<cwd>/.cache.fact_checker.db``) — run from an isolated CWD to
    avoid contending with another bibtex-check process on the same cache.
    """
    tmpdir = Path(tempfile.mkdtemp())
    bib_path = tmpdir / "input.bib"
    jsonl_path = tmpdir / "results.jsonl"
    bib_path.write_text(entries_to_bib(entries))

    cmd = [
        "bibtex-check",
        str(bib_path),
        "--jsonl",
        str(jsonl_path),
        "--rate-limit",
        str(rate_limit),
    ]
    if academic_only:
        cmd.append("--academic-only")
    if s2_api_key:
        cmd.extend(["--s2-api-key", s2_api_key])

    print(f"[bibtex-check] {len(entries)} entries, rate-limit={rate_limit}", file=sys.stderr)
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[bibtex-check] timed out; parsing partial output", file=sys.stderr)

    results: dict[str, tuple[str, bool, bool, float | None]] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("key") or rec.get("bibtex_key") or rec.get("id")
            status = rec.get("status", "skipped")
            # Coverage is defined on the *status* (the canonical HALLMARK definition),
            # not on bibtex-check's per-version ``abstained`` flag. The flag is
            # version-dependent (e.g. some builds report not_found as abstained=False),
            # so deriving abstention from ABSTAIN_STATUSES keeps Coverage identical
            # across tool versions and across all splits in the report.
            abstained = status in ABSTAIN_STATUSES
            # >=1.3.0 output contract: ``coverage_incomplete`` marks abstentions
            # reached under source errors/throttling; ``p_valid`` is the explicit
            # P(entry-as-cited is genuine). Both absent on older builds.
            coverage_incomplete = bool(rec.get("coverage_incomplete", False))
            p_valid = rec.get("p_valid")
            if key is not None:
                results[key] = (status, abstained, coverage_incomplete, p_valid)
    return results


# ---------------------------------------------------------------------------
# Verdict + metrics
# ---------------------------------------------------------------------------
def verdict(
    entry: Entry,
    status: str,
    abstained: bool,
    prescreen: bool,
    reference_year: int,
    coverage_incomplete: bool = False,
) -> tuple[str, bool]:
    """Return (label, abstained) for one entry.

    Raw bibtex-check status drives the verdict; pre-screening can only upgrade a
    VALID verdict to HALLUCINATED (never the reverse), matching HALLMARK's merge.
    A pre-screening override resolves the abstention (the verdict is now decided).

    Mirrors the HALLMARK wrapper for the >=1.3.0 contract: a ``not_found``
    produced while sources were erroring/throttled (``coverage_incomplete``) is
    an abstention, not evidence of fabrication -> conservative VALID.
    """
    if status == "not_found" and coverage_incomplete:
        tool_label = "VALID"
    else:
        tool_label = STATUS_TO_LABEL.get(status, "VALID")
    if prescreen and tool_label == "VALID":
        if prescreen_year(entry.fields, reference_year) or prescreen_authors(entry.fields):
            return "HALLUCINATED", False
    return tool_label, abstained


def compute_metrics(pairs: list[tuple[str, str]], abstained: int, n_total: int) -> dict:
    tp = fp = tn = fn = 0
    for gold, pred in pairs:
        if gold == "HALLUCINATED" and pred == "HALLUCINATED":
            tp += 1
        elif gold == "VALID" and pred == "HALLUCINATED":
            fp += 1
        elif gold == "VALID" and pred == "VALID":
            tn += 1
        elif gold == "HALLUCINATED" and pred == "VALID":
            fn += 1
    n_hall = tp + fn
    n_valid = tn + fp
    dr = tp / n_hall if n_hall else float("nan")
    fpr = fp / n_valid if n_valid else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * dr / (precision + dr) if precision + dr else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else float("nan")
    coverage = 1.0 - (abstained / n_total) if n_total else float("nan")

    def r(x: float) -> float | None:
        return None if isinstance(x, float) and math.isnan(x) else round(x, 4)

    return {
        "n": n_total,
        "n_hallucinated": n_hall,
        "n_valid": n_valid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate": r(dr),
        "fpr": r(fpr),
        "precision": r(precision),
        "f1": r(f1),
        "mcc": r(mcc),
        "abstained": abstained,
        "coverage": r(coverage),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, type=Path, help="Path to a HALLMARK split JSONL file.")
    ap.add_argument("--out", type=Path, default=None, help="Write the metrics JSON here.")
    ap.add_argument("--per-entry", type=Path, default=None, help="Write per-entry verdicts JSONL here.")
    ap.add_argument("--rate-limit", type=int, default=90, help="bibtex-check API requests per minute.")
    ap.add_argument("--batch-size", type=int, default=25, help="Entries per bibtex-check subprocess call.")
    ap.add_argument("--timeout", type=float, default=7200.0, help="Per-batch subprocess timeout (s).")
    ap.add_argument("--reference-year", type=int, default=DEFAULT_REFERENCE_YEAR)
    ap.add_argument("--no-prescreen", action="store_true", help="Score raw bibtex-check only.")
    ap.add_argument(
        "--doi-prescreen",
        action="store_true",
        help="(Networked, slow) Additionally HTTP-resolve DOIs in pre-screening. Off by default.",
    )
    ap.add_argument("--s2-api-key", default=None, help="Semantic Scholar API key (else $S2_API_KEY).")
    args = ap.parse_args()

    import os

    s2_key = args.s2_api_key or os.environ.get("S2_API_KEY")
    prescreen = not args.no_prescreen

    entries = load_split(args.split)
    print(f"[load] {args.split.name}: {len(entries)} scoreable entries", file=sys.stderr)

    all_status: dict[str, tuple[str, bool]] = {}
    for bi in range(0, len(entries), args.batch_size):
        batch = entries[bi : bi + args.batch_size]
        t0 = time.time()
        all_status.update(run_bibtex_check(batch, args.rate_limit, s2_key, args.timeout))
        print(
            f"[batch {bi // args.batch_size}] {len(batch)} entries in {time.time() - t0:.0f}s",
            file=sys.stderr,
        )

    pairs: list[tuple[str, str]] = []
    abstained = 0
    per_entry: list[dict] = []
    for e in entries:
        status, raw_abstained, cov_inc, p_valid = all_status.get(e.key, ("missing", True, False, None))
        if args.doi_prescreen and prescreen:
            status = _maybe_doi_prescreen(e, status)
        pred, ab = verdict(e, status, raw_abstained, prescreen, args.reference_year, coverage_incomplete=cov_inc)
        if ab:
            abstained += 1
        pairs.append((e.label, pred))
        per_entry.append(
            {
                "bibtex_key": e.key,
                "gold_label": e.label,
                "pred_label": pred,
                "btu_status": status,
                "abstained": ab,
                "coverage_incomplete": cov_inc,
                "p_valid": p_valid,
            }
        )

    metrics = compute_metrics(pairs, abstained, len(entries))
    metrics["split"] = args.split.name
    metrics["prescreen"] = prescreen
    metrics["reference_year"] = args.reference_year
    print(json.dumps(metrics, indent=2))

    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"[wrote] {args.out}", file=sys.stderr)
    if args.per_entry:
        args.per_entry.write_text("\n".join(json.dumps(r) for r in per_entry) + "\n")
        print(f"[wrote] {args.per_entry}", file=sys.stderr)


_ARXIV_DATACITE_DOI = re.compile(r"^(10\.48550/arxiv\.\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)


def _normalize_doi_for_resolution(doi: str) -> str | None:
    """Mirror bibtex-updater's DOI normalization for the resolution check.

    Strips a ``https://doi.org/`` / ``dx.doi.org`` prefix, then drops the trailing
    ``vN`` version suffix from arXiv DataCite DOIs (``10.48550/arXiv.<id>vN``): the
    versioned form 404s at doi.org while the unversioned one resolves. Matches the
    HALLMARK pre-screening fix so ``--doi-prescreen`` no longer flags valid
    versioned-arXiv preprints. Returns None if no DOI core is found.
    """
    s = doi.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE).strip()
    m = re.search(r"10\.\d+/[^\s]+", s)
    if not m:
        return None
    core = m.group(0)
    av = _ARXIV_DATACITE_DOI.match(core)
    if av:
        return av.group(1)
    return core


def _maybe_doi_prescreen(entry: Entry, status: str) -> str:
    """Optionally upgrade VALID->HALLUCINATED via a networked DOI HEAD check.

    Only a definitive 404/410 from doi.org flags; transient errors / bot-blocks /
    redirect-target 404s do not (FPR-safe), mirroring the HALLMARK pre-screening
    layer and bibtex-updater's own DOI handling.
    """
    if STATUS_TO_LABEL.get(status, "VALID") != "VALID":
        return status
    doi = entry.fields.get("doi")
    if not doi:
        return status
    normalized = _normalize_doi_for_resolution(doi)
    if not normalized:
        return status
    try:
        import httpx

        resp = httpx.head(f"https://doi.org/{normalized}", timeout=10.0, follow_redirects=True)
        # Only doi.org's own definitive verdict (no successful redirect first) is
        # evidence the DOI does not exist; a 404 from a publisher landing page
        # reached after a redirect is not.
        if resp.status_code in (404, 410) and not resp.history:
            return "doi_not_found"
    except Exception:
        return status
    return status


if __name__ == "__main__":
    main()
