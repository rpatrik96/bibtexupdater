# `bibtex-check` on HALLMARK v1.1.1

This page reports `bibtex-check` (the `bibtex-updater` fact-checker, **version
1.2.0**) on the [HALLMARK](https://github.com/rpatrik96/hallmark) citation-
hallucination benchmark, **dataset release v1.1.1** (corrected gold labels). It
is an independent re-score of the same per-entry verdicts the HALLMARK paper
reports for the co-designed `bibtexupdater` row, plus two splits the paper's main
table does not break out (`stress_test`, `test_crossdomain`).

## What is being measured

HALLMARK frames citation verification as binary detection. Each reference carries
a gold `label` of `VALID` (a real, correctly-cited paper) or `HALLUCINATED` (a
fabricated or corrupted reference). The positive class is `HALLUCINATED` — the
thing the tool should catch.

| Metric | Definition |
|--------|------------|
| **DR** (Detection Rate) | recall on `HALLUCINATED` = TP / (TP + FN) |
| **FPR** (False Positive Rate) | FP / (FP + TN) — real references wrongly flagged |
| **Precision** | TP / (TP + FP) |
| **F1** | harmonic mean of Precision and DR (positive class = `HALLUCINATED`) |
| **MCC** | Matthews correlation coefficient (balanced across the confusion matrix) |
| **Coverage** | 1 − abstentions / n — the fraction of entries the tool did *not* leave in the could-not-confirm bucket |

`bibtex-check`'s raw `status` is mapped to a verdict by the same table HALLMARK's
baseline wrapper uses (`STATUS_TO_LABEL` in `scripts/eval_hallmark.py`):
mismatch/integrity statuses (`*_mismatch`, `doi_mismatch`, `arxiv_id_mismatch`,
`future_date`, `partial_match`, `not_found`, …) → `HALLUCINATED`; `verified` and
the conservative could-not-verify statuses (`unconfirmed`) → `VALID`.

**Coverage / abstention.** An entry abstains for Coverage iff no academic record
either confirmed it (`unconfirmed`) or could be located at all (`not_found`).
`not_found` still yields a `HALLUCINATED` *verdict* — a missing database record is
strong evidence of fabrication — but it abstains for Coverage accounting, which
separates "the tool decided" from "the tool found a record". This is the canonical
HALLMARK v1.1.1 definition and is re-derived from the recorded `status` (not from a
per-version `abstained` flag) so the column is consistent across every split.

## Results — btu v1.2.0 on HALLMARK v1.1.1

| Split | n | Valid | Hall | DR | FPR | Precision | F1 | MCC | Coverage |
|-------|---:|------:|-----:|----:|----:|----------:|---:|----:|---------:|
| dev_public | 1119 | 513 | 606 | 86.5% | 9.2% | 91.8% | 0.890 | 0.771 | 82.2% |
| test_public | 831 | 312 | 519 | 87.7% | 11.5% | 92.7% | 0.901 | 0.750 | 79.4% |
| stress_test | 121 | 0 | 121 | 64.5% | n/a | 100.0% | 0.784 | n/a | 62.8% |
| test_crossdomain | 500 | 200 | 300 | 89.0% | 37.5% | 78.1% | 0.832 | 0.543 | 73.6% |

All four rows are final. `dev_public` / `test_public` are re-scored offline from
the committed HALLMARK per-entry verdicts and reproduce the paper's co-designed
`bibtexupdater` row exactly (see the cross-check below). `stress_test` /
`test_crossdomain` were produced by a live `bibtex-check` run (btu v1.2.0,
Semantic-Scholar-rate-limited, isolated fact-checker cache) and rendered with the
same `scripts/render_hallmark_report.py` over the four per-entry files, so the
whole grid is reproducible deterministically.

Splits: `dev_public` / `test_public` are the public ML-conference splits; `stress_test`
is the hardest tier (near-miss titles, version mismatches, plausible fabrications,
almost entirely `HALLUCINATED`); `test_crossdomain` draws valid and perturbed
references from outside the ML-conference distribution (PubMed/biomedical, etc.).

Notes on the per-split numbers:

* **`stress_test`** is ~99% `HALLUCINATED` by construction, so FPR has a degenerate
  denominator (no real-reference negatives to falsely flag) and is reported as `n/a`;
  read this split as a pure catch-rate (DR) stress test.
* **`test_crossdomain`** carries genuine `VALID` negatives, so FPR and MCC are
  meaningful there and probe out-of-distribution generalization of the verdict gate.
  The result is the sharpest caveat in this report: FPR rises to **37.5%** (vs
  9–12% in-distribution), so `bibtex-check`'s low false-positive rate does **not**
  transfer to non-ML references. It cannot locate many biomedical/other-domain
  papers through its academic-database queries and flags the unfound ones — the
  same conservative gate that is precise on ML-conference citations becomes a
  liability out-of-distribution. Read the in-distribution FPR (~0.09) as a
  property of the ML-citation regime, not a universal guarantee.

## Cross-check against the HALLMARK paper

The paper's co-designed `bibtexupdater` row reports **dev DR .865 / FPR .092** and
**test DR .877 / FPR .115**. Re-scoring the committed per-entry verdicts here
reproduces exactly:

| Split | DR (here) | FPR (here) | DR (paper) | FPR (paper) | Match |
|-------|----------:|-----------:|-----------:|------------:|:-----:|
| dev_public | 0.865 | 0.092 | 0.865 | 0.092 | ✅ |
| test_public | 0.877 | 0.115 | 0.877 | 0.115 | ✅ |

**PASS** — no wrapper/mapping discrepancy. A mismatch here would indicate a
status→verdict mapping bug between the tool and the benchmark wrapper.

## Co-design transparency

`bibtex-check` is the tool under test; **HALLMARK is co-designed with it** — the
benchmark's hallucination taxonomy and several wrapper fixes were developed
alongside this fact-checker. These numbers are therefore best read as an honest
*upper-bound-ish* characterization of the tool on a benchmark it helped shape, not
as a blind third-party evaluation. The dataset/tool versions are pinned (HALLMARK
v1.1.1 labels, btu v1.2.0) and the eval is fully reproducible (below) so the
co-design does not hide behind unstated assumptions. For the residual leaks that
survive the default verdict gate, see [`../docs/KNOWN_LEAKS.md`](../docs/KNOWN_LEAKS.md).

## Reproducing

Requires a HALLMARK checkout for the split JSONL files and (optionally) a Semantic
Scholar API key to lift rate limits.

```bash
export S2_API_KEY=...   # optional but recommended

# Score one split end-to-end (runs bibtex-check, then computes the metrics):
python scripts/eval_hallmark.py \
    --split /path/to/hallmark/data/v1.0/test_public.jsonl \
    --out results_test_public.json \
    --per-entry per_entry_test_public.jsonl

# Re-render this table from per-entry verdict files of all four splits:
python scripts/render_hallmark_report.py \
    --dev per_entry_dev_public.jsonl \
    --test per_entry_test_public.jsonl \
    --stress per_entry_stress_test.jsonl \
    --crossdomain per_entry_test_crossdomain.jsonl \
    --json-out benchmarks/hallmark_v1_2_0.json
```

`eval_hallmark.py` runs `bibtex-check` per entry and merges HALLMARK's non-networked
pre-screening (future-date / placeholder-author) to match the published pipeline;
`render_hallmark_report.py` recomputes the metric grid from the per-entry verdicts.
The dev/test rows were produced offline from the committed HALLMARK per-entry
verdicts (no refetch); stress/crossdomain were run live against an isolated
fact-checker cache.

_Generated from `benchmarks/hallmark_v1_2_0.json`._
