# `bibtex-check` on HALLMARK v1.1.1

This page reports `bibtex-check` (the `bibtex-updater` fact-checker, **version
1.2.0**) on the [HALLMARK](https://github.com/rpatrik96/hallmark) citation-
hallucination benchmark, **dataset release v1.1.1** (corrected gold labels). It
is an independent re-score of the same per-entry verdicts the HALLMARK paper
reports for the co-designed `bibtexupdater` row, plus two splits the paper's main
table does not break out (`stress_test`, `test_crossdomain`).

> **Newer build:** for the post-1.3.0 reliability overhaul (Unreleased / PR #45),
> see [Results — post-1.3.0 reliability overhaul](#results--post-130-reliability-overhaul-unreleased-commit-c758f7e)
> below: tool-only **dev DR 84.8% / FPR 4.7%**, **test DR 88.4% / FPR 4.2%**, with
> large gains on the previously-abstaining weak types.

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

## Results — post-1.3.0 reliability overhaul (Unreleased, commit `c758f7e`)

The reliability/throughput overhaul (PR [#45](https://github.com/rpatrik96/bibtexupdater/pull/45):
default-mode author-truncation flagging, identifier-based venue consensus + a
`nonexistent_venue` registry check, identifier-less preprint-as-published
detection, the author-FP fixes, the `p_valid`/`coverage_incomplete` contract) was
re-evaluated live with `scripts/eval_hallmark.py` on the same two public splits,
**with a Semantic Scholar API key** (`S2_API_KEY`). The headline below is
**tool-only** — both the new run and the committed v1.2.0 per-entry file are
scored from the raw `btu_status` with no pre-screening, so the comparison isolates
the *tool* change. (The committed v1.2.0 file folds in HALLMARK's old networked
DOI pre-screen, since fixed in [hallmark#14](https://github.com/rpatrik96/hallmark/pull/14);
comparing raw `pred_label` would conflate the tool change with that pre-screening
change. See `eval_runs/*/compare.json` for the full-stack rows too.)

| Split | DR | FPR | Precision | F1 | MCC | DR Δ | FPR Δ | F1 Δ |
|-------|----:|----:|----------:|---:|----:|-----:|------:|-----:|
| dev_public  | **84.8%** | **4.7%** | 95.5% | **0.899** | **0.799** | +2.8pp | −0.4pp | +0.019 |
| test_public | **88.4%** | **4.2%** | 97.2% | **0.926** | **0.824** | +4.6pp | −3.8pp | +0.038 |

(v1.2.0 tool-only baseline: dev 82.0% / 5.1% / 0.880 / 0.768; test 83.8% / 8.0% /
0.889 / 0.738.) Both splits improve on **every** headline metric: detection rate up
~3–5pp, false-positive rate cut (test FPR roughly halved), and MCC up 3–8pp.

**Where the detection gains come from** (tool-only per-type detection rate; the
five types the overhaul targeted, all of which previously abstained as
`unconfirmed`):

| Hallucination type | dev v1.2.0 → new | test v1.2.0 → new |
|--------------------|:----------------:|:-----------------:|
| `partial_author_list`     | 0.219 → **0.438** (+21.9pp) | 0.129 → **0.419** (+29.0pp) |
| `preprint_as_published`   | 0.677 → 0.645 (−3.2pp)      | 0.690 → **0.862** (+17.2pp) |
| `wrong_venue`             | 0.447 → **0.489** (+4.3pp)  | 0.735 → **0.853** (+11.8pp) |
| `nonexistent_venue`       | 0.436 → **0.564** (+12.8pp) | 0.459 → **0.568** (+10.8pp) |
| `arxiv_version_mismatch`  | 0.612 → **0.694** (+8.2pp)  | 0.756 → **0.778** (+2.2pp)  |

The remaining always-caught types (`fabricated_doi`, `future_date`,
`placeholder_authors`, `chimeric_title`, `hybrid_fabrication`,
`plausible_fabrication`, `merged_citation`) stay at 1.000 on both splits;
`swapped_authors` holds (dev 0.985, test 0.968). The new positive-evidence
statuses fire as designed (dev: `author_truncated` ×9, `nonexistent_venue` ×2,
`preprint_only` ×1; test similar).

**Calibration.** The new explicit `p_valid` (probability the entry as cited is
genuine) has **ECE 0.252 (dev) / 0.278 (test)** as a P(valid) estimate — well
below the v1.2.0 verdict-confidence ECE the HALLMARK paper reports (0.383 / 0.399),
i.e. the new score is materially better calibrated for thresholding/ranking.

**Honest caveats.**
* *Default-mode near-miss leaks persist.* Five entries (dev) verify despite a
  Levenshtein-1 title perturbation (e.g. "Meta-**Learnings**" vs "Meta-Learning");
  these are the documented `KNOWN_LEAKS.md` cases that `--strict`'s `TITLE_NEAR_MISS`
  catches. The author-FP fixes removed an *incidental* `author_mismatch` detection
  on such title-perturbed entries (their authors are real), which is why default-mode
  `near_miss_title` is flat-to-slightly-down rather than up.
* *One small real regression:* dev `preprint_as_published` −3.2pp (one entry, now an
  abstention) — a side effect of the more conservative venue gate that drives the
  FPR win; on test the same type is +17.2pp.
* *Run hygiene.* The live run used the default `--rate-limit 90` and hit
  intermittent throttling on the keyless sources (DBLP/OpenReview); 20 `test_public`
  entries were dropped by subprocess slowdowns and **re-run and merged** (see
  `eval_runs/test_public_fill/`) so the grid reflects all 831 entries. `dev_public`
  completed in one pass. Numbers are reproducible but, being a live API run, are not
  bit-identical across runs the way the committed v1.2.0 re-score is.

This block was produced by `scripts/eval_hallmark.py` + `scripts/_compare_hallmark_runs.py`
over `eval_runs/{dev,test}_public/`; `stress_test` / `test_crossdomain` were not
re-run for this build and their v1.2.0 rows above stand.

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
