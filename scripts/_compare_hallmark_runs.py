#!/usr/bin/env python3
"""Compare a fresh eval_hallmark.py run against the committed btu v1.2.0 results.

Joins the new per-entry verdicts with the HALLMARK split (gold labels +
hallucination types) and the committed v1.2.0 per-entry file, then reports:

  * headline metrics new vs old (DR / FPR / precision / F1 / MCC / coverage)
  * per-hallucination-type detection rate, new vs old, with deltas
  * status distribution of the new run's false negatives / false positives
  * coverage_incomplete counts (new output contract)
  * ECE of the new ``p_valid`` (10 equal-width bins, gold VALID = 1)

Usage:
    python3 scripts/_compare_hallmark_runs.py \
        --new eval_runs/dev_public/per_entry_dev_public.jsonl \
        --old ~/Documents/GitHub/hallmark/results/relabel_delta/btu_v1_2_0/bibtexupdater_dev_public_per_entry.jsonl \
        --split ~/Documents/GitHub/hallmark/data/v1.0/dev_public.jsonl \
        [--json-out eval_runs/dev_public/compare.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def tool_label(row: dict) -> str:
    """Re-derive the verdict from ``btu_status`` ALONE (no pre-screening).

    Isolates the tool's own behaviour: the committed v1.2.0 per-entry file folds
    in HALLMARK's (networked, since-fixed) pre-screening overrides, so a raw
    ``pred_label`` comparison conflates the tool change with a pre-screening
    change. Scoring both sides from status here is the apples-to-apples view of
    what the merged code did. Mirrors STATUS_TO_LABEL in eval_hallmark.py.
    """
    from eval_hallmark import STATUS_TO_LABEL  # local import; same dir on sys.path

    return STATUS_TO_LABEL.get(row.get("btu_status"), "VALID")


def headline(pairs: list[tuple[str, str]]) -> dict:
    tp = fp = tn = fn = 0
    for gold, pred in pairs:
        if gold == "HALLUCINATED" and pred == "HALLUCINATED":
            tp += 1
        elif gold == "VALID" and pred == "HALLUCINATED":
            fp += 1
        elif gold == "VALID" and pred == "VALID":
            tn += 1
        else:
            fn += 1
    dr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * dr / (prec + dr) if prec + dr else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "dr": dr, "fpr": fpr, "precision": prec, "f1": f1, "mcc": mcc}


def fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    split_rows = load_jsonl(args.split)
    gold_by_key = {}
    type_by_key = {}
    for r in split_rows:
        k = r.get("bibtex_key", "")
        if r.get("source") == "canary" or str(k).startswith("__canary__"):
            continue
        if r.get("label") in ("VALID", "HALLUCINATED"):
            gold_by_key[k] = r["label"]
            type_by_key[k] = r.get("hallucination_type") or "valid"

    new_rows = {r["bibtex_key"]: r for r in load_jsonl(args.new)}
    old_rows = {r["bibtex_key"]: r for r in load_jsonl(args.old)}

    keys = [k for k in gold_by_key if k in new_rows]
    missing_new = [k for k in gold_by_key if k not in new_rows]
    only_old = [k for k in gold_by_key if k in old_rows and k not in new_rows]
    print(f"# Comparison on {args.split.name}: {len(keys)} joined entries "
          f"({len(missing_new)} missing from new run, {len(only_old)} of those in old)\n")

    new_pairs = [(gold_by_key[k], new_rows[k]["pred_label"]) for k in keys]
    old_keys = [k for k in keys if k in old_rows]
    old_pairs = [(gold_by_key[k], old_rows[k]["pred_label"]) for k in old_keys]

    h_new, h_old = headline(new_pairs), headline(old_pairs)
    print("## Headline — full stack (pred_label as recorded; positive class = HALLUCINATED)\n")
    print("NOTE: the committed v1.2.0 file folds in HALLMARK's old (networked, since-fixed)")
    print("pre-screening; this run uses eval_hallmark.py's non-networked pre-screening, so this")
    print("row mixes the tool change with a pre-screening change. See the tool-only table below.\n")
    print("| metric | v1.2.0 (committed) | new run | delta |")
    print("|---|---|---|---|")
    for m in ("dr", "fpr", "precision", "f1", "mcc"):
        d = h_new[m] - h_old[m] if (h_new[m] == h_new[m] and h_old[m] == h_old[m]) else float("nan")
        sign = "+" if d == d and d >= 0 else ""
        print(f"| {m.upper()} | {fmt(h_old[m])} | {fmt(h_new[m])} | {sign}{fmt(d)} |")
    print(f"\ncounts new: TP {h_new['tp']} FP {h_new['fp']} TN {h_new['tn']} FN {h_new['fn']}  |  "
          f"old: TP {h_old['tp']} FP {h_old['fp']} TN {h_old['tn']} FN {h_old['fn']}\n")

    # Tool-only: re-derive BOTH sides' labels from btu_status (no pre-screening).
    tnew_pairs = [(gold_by_key[k], tool_label(new_rows[k])) for k in old_keys]
    told_pairs = [(gold_by_key[k], tool_label(old_rows[k])) for k in old_keys]
    t_new, t_old = headline(tnew_pairs), headline(told_pairs)
    print("## Headline — TOOL-ONLY (status->label, no pre-screening either side; apples-to-apples)\n")
    print("| metric | v1.2.0 tool | new tool | delta |")
    print("|---|---|---|---|")
    for m in ("dr", "fpr", "precision", "f1", "mcc"):
        d = t_new[m] - t_old[m] if (t_new[m] == t_new[m] and t_old[m] == t_old[m]) else float("nan")
        sign = "+" if d == d and d >= 0 else ""
        print(f"| {m.upper()} | {fmt(t_old[m])} | {fmt(t_new[m])} | {sign}{fmt(d)} |")
    print(f"\ncounts new: TP {t_new['tp']} FP {t_new['fp']} TN {t_new['tn']} FN {t_new['fn']}  |  "
          f"old: TP {t_old['tp']} FP {t_old['fp']} TN {t_old['tn']} FN {t_old['fn']}\n")

    # Per-type detection rate (TOOL-ONLY, the apples-to-apples view).
    per_type: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"new": [], "old": []})
    for k in keys:
        if gold_by_key[k] != "HALLUCINATED":
            continue
        t = type_by_key[k]
        per_type[t]["new"].append(tool_label(new_rows[k]) == "HALLUCINATED")
        if k in old_rows:
            per_type[t]["old"].append(tool_label(old_rows[k]) == "HALLUCINATED")
    print("## Per-hallucination-type detection rate (TOOL-ONLY)\n")
    print("| type | n | v1.2.0 tool | new tool | delta |")
    print("|---|---|---|---|---|")
    for t in sorted(per_type, key=lambda t: sum(per_type[t]["new"]) / max(len(per_type[t]["new"]), 1)):
        nw, od = per_type[t]["new"], per_type[t]["old"]
        drn = sum(nw) / len(nw) if nw else float("nan")
        dro = sum(od) / len(od) if od else float("nan")
        d = drn - dro if drn == drn and dro == dro else float("nan")
        sign = "+" if d == d and d >= 0 else ""
        print(f"| {t} | {len(nw)} | {fmt(dro)} | {fmt(drn)} | {sign}{fmt(d)} |")

    # FN/FP status distributions (new run).
    fn_status = Counter(new_rows[k].get("btu_status") for k in keys
                        if gold_by_key[k] == "HALLUCINATED" and new_rows[k]["pred_label"] == "VALID")
    fp_status = Counter(new_rows[k].get("btu_status") for k in keys
                        if gold_by_key[k] == "VALID" and new_rows[k]["pred_label"] == "HALLUCINATED")
    print("\n## New-run misses by tool status\n")
    print("FN (hallucinated -> VALID):", dict(fn_status.most_common()))
    print("FP (valid -> HALLUCINATED):", dict(fp_status.most_common()))

    cov_inc = sum(1 for k in keys if new_rows[k].get("coverage_incomplete"))
    print(f"\ncoverage_incomplete entries (new): {cov_inc}")

    # ECE of p_valid (gold VALID = 1).
    scored = [(float(new_rows[k]["p_valid"]), 1.0 if gold_by_key[k] == "VALID" else 0.0)
              for k in keys if new_rows[k].get("p_valid") is not None]
    ece = float("nan")
    if scored:
        bins: list[list[tuple[float, float]]] = [[] for _ in range(10)]
        for p, y in scored:
            bins[min(int(p * 10), 9)].append((p, y))
        ece = sum(
            abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) * len(b)
            for b in bins if b
        ) / len(scored)
    print(f"ECE of p_valid as P(valid) (10 bins, n={len(scored)}): {fmt(ece)}  "
          f"[v1.2.0 wrapper-confidence ECE reference: 0.383 dev / 0.399 test]")

    if args.json_out:
        out = {
            "split": args.split.name,
            "n_joined": len(keys),
            "headline_new": h_new,
            "headline_old": h_old,
            "per_type": {
                t: {
                    "n": len(v["new"]),
                    "dr_new": (sum(v["new"]) / len(v["new"])) if v["new"] else None,
                    "dr_old": (sum(v["old"]) / len(v["old"])) if v["old"] else None,
                }
                for t, v in per_type.items()
            },
            "fn_status_new": dict(fn_status),
            "fp_status_new": dict(fp_status),
            "coverage_incomplete_new": cov_inc,
            "ece_p_valid_new": None if ece != ece else round(ece, 4),
        }
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
