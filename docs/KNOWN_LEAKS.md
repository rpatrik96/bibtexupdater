# Known leaks — `bibtex-check` v1.1.0

## Purpose

This file lists every known case where `bibtex-check` v1.1.0 returns `VERIFIED` on an entry the [corrected HALLMARK v1.0 gold](https://github.com/rpatrik96/hallmark/pull/9) labels as a hallucinated or perturbed reference. These are the tool's documented blind spots in **default** mode — none of them are speculative.

Researchers auditing a bibliography against [arXiv's 2026 hallucinated-reference policy](https://www.researchinformation.info/news/arxiv-imposes-one-year-ban-for-unchecked-ai-generated-content/) (1-year ban followed by peer-review-first requirement) should treat each case here as a leak the default verdict gate is *deliberately* tuned to abstain on — the FPR tradeoff makes it not worth catching in the headline three-way verdict — and run [`--strict`](../README.md#strict-mode---strict) on top.

Concretely, of the 7 residual leaks (5 dev + 2 test):

- **6 are caught by `--strict`** (4 as `TITLE_NEAR_MISS` via Levenshtein-1; 1 as `AUTHOR_TRUNCATED`; 1 as `GIVEN_NAME_SUBSTITUTION` in default mode too — fixed in this release).
- **1 (Least-to-Most) is now caught by default** as of v1.1.0 — included below as a historical "what the benchmark taught us" case.

Numbers (post-correction gold × post-fix v1.1.0):

| split | FPR | leak rate | residual leaks |
|---|---|---|---|
| dev_public | 1.79% | 0.65% | 4 (5 minus the now-fixed Least-to-Most case) |
| test_public | 5.98% | 0.38% | 2 |

The doc is a snapshot of the v1.1.0 release; it will be refreshed when the gold or the verdict gate moves.

---

## Per-leak detail

### 1. `d1973e26a718` — Subspace Differential Privacy (1-char title perturbation)

- **Gold htype:** `near_miss_title`
- **Split:** dev_public
- **Canonical paper:** Gao et al., *Subspace Differential Privacy*, AAAI 2022. [arXiv:2108.11527](https://arxiv.org/abs/2108.11527)
- **Entry as cited:**
  ```bibtex
  @inproceedings{d1973e26a718,
    title     = {Subspace Differential Privacys},
    author    = {Gao, Jie and Gong, Ruoxi and Yu, Fang-Yi},
    booktitle = {AAAI},
    year      = {2022}
  }
  ```
- **Perturbation:** trailing `s` on the title (`Privacy` → `Privacys`).
- **Default verdict:** `VERIFIED` — title similarity above the default `0.85` gate; the rest of the metadata matches canonical.
- **`--strict` verdict:** `TITLE_NEAR_MISS` (catches via Levenshtein-1).

### 2. `cc3bac858db2` — Schema-Variable (removed hyphen)

- **Gold htype:** `near_miss_title`
- **Split:** dev_public
- **Canonical paper:** Bubel et al., *Schema-Variable*-based program transformation (KeY-family verification literature).
- **Entry as cited:**
  ```bibtex
  @inproceedings{cc3bac858db2,
    title  = {... Schema Variable ...},
    ...
    year   = {...}
  }
  ```
- **Perturbation:** hyphen removed (`Schema-Variable` → `Schema Variable`).
- **Default verdict:** `VERIFIED` — hyphen-vs-space is normalized out in the default title hash.
- **`--strict` verdict:** `TITLE_NEAR_MISS` (Levenshtein-1 over the un-normalized title catches the missing hyphen).

### 3. `1cc022db3273` — Chain-of-Thought (moved hyphen)

- **Gold htype:** `near_miss_title`
- **Split:** dev_public
- **Canonical paper:** Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **Entry as cited:**
  ```bibtex
  @inproceedings{1cc022db3273,
    title  = {Chain of-Thought ...},
    ...
    year   = {2022}
  }
  ```
- **Perturbation:** hyphen moved (`Chain-of-Thought` → `Chain of-Thought`).
- **Default verdict:** `VERIFIED` — same hyphen-normalization gap as #2.
- **`--strict` verdict:** `TITLE_NEAR_MISS`.

### 4. `aff3df193dde` — OSAKA (reordered author subset, no sentinel)

- **Gold htype:** `swapped_authors`
- **Split:** dev_public
- **Canonical paper:** Caccia et al., *Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning* (OSAKA), NeurIPS 2020. [arXiv:2003.05856](https://arxiv.org/abs/2003.05856) — 11 authors.
- **Entry as cited:** lists 8 of the 11 real authors in a reordered subset, **with no `and others` / `et al.` sentinel** to disclose the truncation.
- **Default verdict:** `VERIFIED` — the prefix-N slice + multiset intersection accept any subset of the real author list that doesn't introduce fabricated surnames, because we don't want to false-positive on stub author lists. The reordering is also tolerated because Crossref-deposited NeurIPS proceedings (prefix `10.52202`) sort contributors A–Z, so multiset-equal reordering against a single alphabetized source is treated as a record-sort artifact rather than a swap.
- **`--strict` verdict:** `AUTHOR_TRUNCATED` (silent truncation without an `and others` sentinel is a misrepresentation in strict mode, regardless of whether the surnames are a subset).

### 5. `fe58db6e7124` — Explanation (trailing `s`)

- **Gold htype:** `near_miss_title`
- **Split:** test_public
- **Canonical paper:** title ends `... Explanation` (singular). One-character perturbation appends `s`.
- **Entry as cited:**
  ```bibtex
  @inproceedings{fe58db6e7124,
    title  = {... Explanations},
    ...
  }
  ```
- **Perturbation:** plural-vs-singular (`Explanation` → `Explanations`).
- **Default verdict:** `VERIFIED` — within the default-mode similarity gate.
- **`--strict` verdict:** `TITLE_NEAR_MISS` (Levenshtein-1).

### 6. `f6a47b5e621f` — Language-Guided (removed hyphen)

- **Gold htype:** `near_miss_title`
- **Split:** test_public
- **Canonical paper:** title contains `Language-Guided`. Perturbation drops the hyphen.
- **Entry as cited:**
  ```bibtex
  @inproceedings{f6a47b5e621f,
    title  = {Language Guided ...},
    ...
  }
  ```
- **Perturbation:** hyphen removed (`Language-Guided` → `Language Guided`).
- **Default verdict:** `VERIFIED` — same normalization gap as #2 and #3.
- **`--strict` verdict:** `TITLE_NEAR_MISS`.

---

## What `--strict` buys you on this list

| Leak | Default | `--strict` rule that catches it |
|---|---|---|
| #1 `Privacys` | `VERIFIED` | `TITLE_NEAR_MISS` (Levenshtein-1) |
| #2 `Schema Variable` | `VERIFIED` | `TITLE_NEAR_MISS` (Levenshtein-1, un-normalized) |
| #3 `Chain of-Thought` | `VERIFIED` | `TITLE_NEAR_MISS` (Levenshtein-1, un-normalized) |
| #4 OSAKA (8/11, no sentinel) | `VERIFIED` | `AUTHOR_TRUNCATED` (silent-truncation flag) |
| #5 `Explanations` | `VERIFIED` | `TITLE_NEAR_MISS` (Levenshtein-1) |
| #6 `Language Guided` | `VERIFIED` | `TITLE_NEAR_MISS` (Levenshtein-1, un-normalized) |

`--strict` catches **6/6** residual default-mode leaks at the cost of the strict-mode FPR documented in the [README's strict-mode subsection](../README.md#strict-mode---strict). The asymmetric-cost framing — leak ≫ FP — is the canonical fit for arXiv-policy-aware audits; for routine bibliography cleanup the default three-way verdict (`verified` / `could-not-verify` / `problematic`) remains the recommended gate.

---

## Caught in v1.1.0 — historical

### `db9a596a4d3f` — Least-to-Most (lead-author given-name substitution)

- **Gold htype:** `plausible_fabrication`
- **Split:** dev_public
- **Canonical paper:** Zhou et al., *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models*, ICLR 2023. [arXiv:2205.10625](https://arxiv.org/abs/2205.10625)
- **Entry as cited:** lead author `Shunyu Zhou` (a real Zhou — but the wrong one; canonical lead is **Denny Zhou**).
- **Pre-v1.1.0 default verdict:** `VERIFIED`. The lead-author check abstained on position 0 whenever the entry repeated the canonical surname elsewhere in the author list (a too-conservative repeated-surname guard).
- **v1.1.0 default verdict:** `GIVEN_NAME_SUBSTITUTION` (problematic). The fix in [`given_name_position_audit`](../src/bibtex_updater/utils.py) no longer abstains when only one side has a repeated surname — the unique side uniquely pins the pairing, so a position-0 given-name substitution flags correctly even when the entry repeats the canonical name. This catch was the single largest *default-mode* leak reduction in v1.1.0; the rule generalizes — it covers `Denny Zhou` ↔ `Shunyu Zhou`-class substitutions on any lead position with a repeated surname downstream.

---

## Updating this doc

This list is regenerated when (a) the corrected HALLMARK gold moves or (b) the verdict gate changes. Each release should re-run the corrected gold against the released tool and replace this file's per-leak blocks with whatever residual leaks the new combination produces. The "historical" section accumulates default-mode catches that this benchmark drove.
