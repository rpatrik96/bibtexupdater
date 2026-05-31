#!/usr/bin/env python3
"""Render the committed HALLMARK benchmark table for ``bibtex-check`` (btu v1.2.0).

This is the *reporting* companion to ``scripts/eval_hallmark.py``: it consumes the
per-entry verdict JSONL files produced by an evaluation run and emits the
detection-metric grid that lands in ``benchmarks/HALLMARK.md``. Keeping the
metric math in one place guarantees the Markdown table, the JSON summary, and the
paper cross-check all agree.

Why a separate renderer:

* Detection metrics (DR / FPR / Precision / F1 / MCC) are computed from the
  ``gold_label`` / ``pred_label`` columns and are unaffected by abstention.
* **Coverage** is re-derived here from ``btu_status`` using the single canonical
  HALLMARK v1.1.1 abstain set (``{not_found, unconfirmed}`` plus transient
  sentinels), rather than trusting any per-file ``btu_abstained`` flag. Different
  pipelines (the dev/test cascade vs. the live stress/crossdomain runner) wrote
  that boolean under slightly different rules; re-deriving from status makes the
  Coverage column consistent across every split.

Usage
-----
    python scripts/render_hallmark_report.py \
        --dev   /path/to/bibtexupdater_dev_public_per_entry.jsonl \
        --test  /path/to/bibtexupdater_test_public_per_entry.jsonl \
        --stress /path/to/per_entry_stress_test.jsonl \
        --crossdomain /path/to/per_entry_test_crossdomain.jsonl \
        --json-out benchmarks/hallmark_v1_2_0.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Canonical HALLMARK v1.1.1 abstention set for the Coverage metric. An entry
# abstains iff no academic record confirmed it (``unconfirmed``) or could be
# located at all (``not_found``); the transient sentinels are non-confirmations
# too. ``not_found`` still yields a HALLUCINATED verdict — it abstains only for
# Coverage accounting. Keep in sync with eval_hallmark.ABSTAIN_STATUSES.
ABSTAIN_STATUSES: frozenset[str] = frozenset(
    {
        "unconfirmed",
        "not_found",
        "api_error",
        "skipped",
        "strict_warn_preprint_year",
        "strict_warn_cnv",
        "missing",
    }
)


def _is_abstained(row: dict) -> bool:
    """Re-derive abstention from the recorded status (canonical, version-stable).

    Rows that carry no ``btu_status`` were resolved by HALLMARK's non-networked
    pre-screening before bibtex-check ran, so they are confident verdicts, not
    abstentions.
    """
    status = row.get("btu_status")
    if status is None:
        return False
    return status in ABSTAIN_STATUSES


def score(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        gold, pred = r["gold_label"], r["pred_label"]
        if gold == "HALLUCINATED" and pred == "HALLUCINATED":
            tp += 1
        elif gold == "VALID" and pred == "HALLUCINATED":
            fp += 1
        elif gold == "VALID" and pred == "VALID":
            tn += 1
        elif gold == "HALLUCINATED" and pred == "VALID":
            fn += 1
    n = len(rows)
    n_hall = tp + fn
    n_valid = tn + fp
    abstained = sum(1 for r in rows if _is_abstained(r))

    def _ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    dr = _ratio(tp, n_hall)
    fpr = _ratio(fp, n_valid)
    precision = _ratio(tp, tp + fp)
    f1: float | None = None
    if precision is not None and dr is not None and (precision + dr) > 0:
        f1 = round(2 * precision * dr / (precision + dr), 4)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = round(((tp * tn) - (fp * fn)) / denom, 4) if denom else None
    coverage = _ratio(n - abstained, n)
    return {
        "n": n,
        "n_hallucinated": n_hall,
        "n_valid": n_valid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate": dr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "abstained": abstained,
        "coverage": coverage,
    }


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def render_table(results: dict[str, dict]) -> str:
    header = (
        "| Split | n | Valid | Hall | DR | FPR | Precision | F1 | MCC | Coverage |\n"
        "|-------|---:|------:|-----:|----:|----:|----------:|---:|----:|---------:|"
    )
    lines = [header]
    for label, m in results.items():
        lines.append(
            f"| {label} | {m['n']} | {m['n_valid']} | {m['n_hallucinated']} "
            f"| {_fmt(m['detection_rate'], pct=True)} | {_fmt(m['fpr'], pct=True)} "
            f"| {_fmt(m['precision'], pct=True)} | {_fmt(m['f1'])} | {_fmt(m['mcc'])} "
            f"| {_fmt(m['coverage'], pct=True)} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--stress", type=Path, required=True)
    ap.add_argument("--crossdomain", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    split_paths = {
        "dev_public": args.dev,
        "test_public": args.test,
        "stress_test": args.stress,
        "test_crossdomain": args.crossdomain,
    }
    results = {label: score(load_rows(p)) for label, p in split_paths.items()}

    print(render_table(results))
    if args.json_out:
        payload = {
            "btu_version": "1.2.0",
            "label_version": "v1.1.1",
            "abstain_definition": sorted(ABSTAIN_STATUSES),
            "splits": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
