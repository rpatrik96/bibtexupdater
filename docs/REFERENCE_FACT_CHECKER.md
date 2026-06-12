# Reference Fact-Checker

Validate that bibliographic entries in BibTeX files exist in external databases and have correct metadata. Useful for detecting hallucinated or incorrectly cited references before submission.

## Installation

```bash
pip install bibtex-updater
# or run without installing
uv run --with bibtex-updater bibtex-check references.bib
```

## Quick Start

```bash
# Basic validation
bibtex-check references.bib

# Generate detailed JSON report
bibtex-check references.bib --report report.json

# Stream per-entry results to JSONL
bibtex-check references.bib --jsonl results.jsonl

# CI/CD mode: fail if not-found / hallucinated entries
bibtex-check references.bib --strict
```

## How It Works

The fact-checker runs a multi-stage verification pipeline.

### Pre-API validation (zero cost)

1. **Year validation** — flags future dates (`year > current_year`), implausible years (`< 1800`), and non-numeric/missing years before making any API call (`--no-check-years` disables).
2. **DOI resolution** — checks the entry's DOI resolves; only HTTP 404/410 from `doi.org` counts as a fabricated DOI. Publisher blocks (403, 418 IEEE bot-detection, 429) are *not* treated as invalid (`--no-check-dois` disables).

### Identity-anchored integrity checks

These run before the title/author search and fire only on *positive evidence*, so they keep the false-positive rate low (they abstain whenever the determination is uncertain — e.g. an IEEE bot-block or a DOI Crossref doesn't index).

3. **DOI-target consistency** — fetches the Crossref record the entry's DOI actually resolves to. If that record's title clearly differs from the entry's title, the DOI points to a *different paper* → `doi_mismatch`. A copy-paste DOI that would otherwise be silently verified against the entry's real paper is caught here.
4. **arXiv-ID consistency** — the same check for the entry's cited arXiv ID; a mismatch → `arxiv_id_mismatch`. arXiv DOIs are version-normalized (`…v2` stripped) so a versioned/unversioned mismatch is not falsely flagged.
5. **ID-anchored author fabrication** — when a valid DOI/arXiv ID *does* resolve to the cited paper (title confirms) but the author list is swapped or contains placeholder names, the entry is flagged `author_mismatch`: the identifier is the entry's own, so a real author divergence on it is positive evidence of fabrication.
6. **Identifier fast paths** — when the entry's own DOI/arXiv ID passed the consistency checks *and* the single authoritative record behind that identifier fully confirms every claimed field at the full thresholds, the multi-source cascade is skipped. The fast paths can only short-circuit a clean `verified` — anything less falls through to the normal cascade. Disable with `--no-fast-path`; automatically inert in `--strict` mode.

### Cascading source verification

Title/author search runs a single cascade — there is no parallel "query every source" mode. Sources are queried in order and the cascade short-circuits as soon as one returns a candidate at or above the high-confidence threshold (`0.95`):

| Step | Source | Role | Rate (default) |
|------|--------|------|------|
| 1 | **CrossRef** | DOI-backed literature; fielded `query.title` + `query.author` | ~300/min |
| 2 | **OpenAlex** | high-rate aggregator (polite pool), broad coverage; fielded `title.search` | ~150/min |
| 3 | **DBLP** | authoritative CS/ML-conference index (token-AND search) | ~30/min |
| 4 | **OpenReview** | authoritative ICLR/NeurIPS/TMLR submission registry (API v1 with an API v2 fallback for migrated venues, e.g. ICLR 2024+) | ~30/min |
| 5 | **Semantic Scholar** | preprint coverage; slowest without a key | ~10/min keyless |

The order is throughput-aware: fast, broad sources first so the slow keyless Semantic Scholar fallback is only reached on hard entries. OpenReview is consulted before Semantic Scholar because it owns the submission record for most ML conferences and can *positively confirm* ICLR/NeurIPS/TMLR papers that the DOI/CS-index sources can only leave unconfirmed. Set a Semantic Scholar API key (`--s2-api-key` or `S2_API_KEY`) to lift S2 to ~60 req/min — with a key, a single-best-title `/paper/search/match` step additionally runs right after CrossRef and the final S2 relevance-search step is skipped whenever it contributed, so per-entry S2 spend stays at one call.

**Retrieval** uses fielded title search (CrossRef `query.title`, OpenAlex `title.search`) against the raw, author-free title rather than a free-text `title + surname` blob — the blob returned unrelated papers for DOI-less ML-conference titles. Each step retrieves `--top-k` candidates (default 3, max 10) and re-ranks them by title similarity.

### Scoring and verdict

For each entry the tool:

1. Runs pre-API checks and identity-anchored integrity checks (cheap, high-precision).
2. Walks the cascade, scoring candidates with fuzzy title matching plus author Jaccard similarity.
3. Detects **chimeric titles** (a citation whose title splices two real papers) before picking a best match.
4. Compares fields against the best match with alias-aware venue matching.
5. Runs a **cross-source author intersection** — authors confirmed by ≥2 sources earn a multi-source bonus; authors no source confirms are flagged suspect.
6. Assigns a three-way verdict (below).

Author handling: sources return authors in as-published order, so author-order differences are real citation errors and are flagged, not smoothed over. Surname comparison uses each source's structured `family` field where available (Crossref, OpenAlex, OpenReview `~Given_Family` handles), so family-first/CJK names like "Chen Xing" ↔ "Xing Chen" are not falsely flagged. When the matched source has only flat names (Semantic Scholar, DBLP), a Crossref structured-name lookup vets a potential author mismatch before it is reported.

## Verdicts: verified vs. could-not-verify vs. problematic

`VERIFIED` requires *every* claimed field to be **positively confirmed** against the matched record — not merely "not contradicted".

- **Verified** (`verified`) — clean pass; all claimed fields confirmed.
- **Could-not-verify** (`unconfirmed`, `not_found`) — **abstention**, *not* a clean pass. A record was found and nothing was contradicted, but at least one claimed field could not be positively confirmed (e.g. a published venue backed only by a preprint, or a consistent-but-incomplete author list), or no matching record was found at all. These entries warrant review. The per-entry `coverage_incomplete` flag distinguishes a *clean* exhaustive miss from an abstention reached while sources errored or were throttled (re-run the latter after a cooldown).
- **Problematic** — positive evidence of a defect: `title_mismatch`, `author_mismatch`, `year_mismatch`, `venue_mismatch`, `nonexistent_venue`, `partial_match`, `doi_mismatch`, `arxiv_id_mismatch`, `author_truncated`, `hallucinated` (chimeric title, fabricated DOI, future/invalid year, ID misattribution).

`hallucinated` is reserved for positive-evidence signals; a merely weak title-search match **abstains** as `not_found` rather than asserting fabrication.

## Status Codes

| Status | Bucket | Description |
|--------|--------|-------------|
| `verified` | verified | Every claimed field positively confirmed |
| `unconfirmed` | could-not-verify | Record found, a claimed field unconfirmable (abstention) |
| `not_found` | could-not-verify | No matching record found |
| `hallucinated` | problematic | Positive evidence of fabrication (chimeric/fabricated DOI/etc.) |
| `title_mismatch` | problematic | Title differs significantly from best match |
| `author_mismatch` | problematic | Author list differs (incl. ID-anchored fabrication) |
| `year_mismatch` | problematic | Publication year differs beyond tolerance (exact-year for same-conference citations) |
| `venue_mismatch` | problematic | Journal/venue differs |
| `nonexistent_venue` | problematic | Claimed venue unknown to the DBLP/OpenAlex venue registries and reported by no source for this (otherwise real) paper |
| `author_truncated` | problematic | Author list silently truncated (co-authors dropped without `et al.`/`and others`); corroborated by ≥ 2 sources |
| `partial_match` | problematic | Multiple fields differ |
| `doi_mismatch` | problematic | Cited DOI resolves to a different paper |
| `arxiv_id_mismatch` | problematic | Cited arXiv ID resolves to a different paper |
| `future_date` / `invalid_year` | problematic | Year in the future / missing / implausible |
| `doi_not_found` | problematic | DOI returns HTTP 404/410 |
| `api_error` | — | Errors occurred during API queries |
| `skipped` | — | Entry type not verifiable |

Web references (`url_*`), books (`book_*`), and working papers (`working_paper_*`) have their own status families; see `--skip-web`, `--skip-books`, `--skip-working-papers`.

## Numeric confidence score

The JSONL output carries an additive 0–100 `confidence_score` summarizing per-field similarity with explicit penalty/bonus contributions (constants from CheckIfExist, not auto-fit):

- Multi-source bonus: `+10` when ≥2 sources confirm the same authors
- Penalties: title-mismatch `−20`, author-mismatch `−20`, journal/venue-mismatch `−15`, fabricated-author `−10` each (capped at `−20`)
- Asymmetric formula for the high-title / low-author chimeric case: `confidence = S_title − 0.5 × (100 − S_author)`

## Command Line Options

```
usage: bibtex-check [-h] [--report FILE] [--jsonl FILE] [--strict]
                    [--strict-warn-cnv] [--verbose]
                    [--title-threshold FLOAT] [--author-threshold FLOAT]
                    [--year-tolerance INT] [--venue-threshold FLOAT]
                    [--cache-file FILE] [--rate-limit INT] [--s2-api-key KEY]
                    [--mailto EMAIL] [--no-cache] [--no-check-dois]
                    [--no-check-years] [--no-check-venue-existence]
                    [--no-fast-path] [--workers N] [--skip-web] [--skip-books]
                    [--skip-working-papers] [--academic-only]
                    [--verify-url-content] [--url-timeout FLOAT]
                    [--google-books-api-key KEY] [--no-google-books]
                    [--top-k N] [--openalex-mailto EMAIL] [--non-generative]
                    bibfiles [bibfiles ...]
```

| Option | Default | Description |
|--------|---------|-------------|
| `bibfiles` | — | One or more BibTeX files to check |
| `--report`, `-r FILE` | — | Write full JSON report to FILE |
| `--jsonl FILE` | — | Write one JSON object per line (streamed) |
| `--strict` | off | Exit code 4 if not-found / hallucinated entries found |
| `--verbose`, `-v` | off | Enable debug logging |

**Thresholds:** `--title-threshold` (0.90), `--author-threshold` (0.80), `--year-tolerance` (1), `--venue-threshold` (0.70).

**API options:** `--cache-file` (`.cache.fact_checker.json`), `--rate-limit` (45 req/min, scales per-service limits), `--s2-api-key KEY` (or `S2_API_KEY` env var), `--mailto EMAIL` (or `BIBTEX_CHECK_MAILTO`; polite-pool contact for Crossref/OpenAlex, feeds the User-Agent and the `--openalex-mailto` default), `--no-cache`, `--no-check-dois`, `--no-check-years`, `--no-check-venue-existence` (disable the DBLP/OpenAlex venue-registry existence check behind `nonexistent_venue`), `--no-fast-path` (always run the full cascade; disables the DOI/arXiv identifier-anchored fast paths), `--workers N` (8).

**Cascade (CheckIfExist):** `--top-k N` (3, max 10) candidates per source; `--openalex-mailto EMAIL` for the OpenAlex polite pool.

**Entry-type filtering:** `--skip-web`, `--skip-books`, `--skip-working-papers`, `--academic-only`.

**Web/book options:** `--verify-url-content`, `--url-timeout` (10s), `--google-books-api-key KEY`, `--no-google-books`.

**Policy:** `--non-generative` (or `BIBTEX_CHECK_NON_GENERATIVE=1`) refuses to load any LLM backend at runtime, for ACL ARR / ICML 2026 LLM-in-review policy compliance. The package ships no LLM backends today, so this is a forward-compat guard plus a startup banner.

## Output Formats

### JSON Report (`--report`)

Full structured report: a `summary` block (totals, status counts, verified/abstained/problematic counts, `coverage_incomplete_count`, timestamp) plus per-entry records with field comparisons, the best-matching record, the sources consulted/confirmed, and the same `p_valid`/`coverage_incomplete` contract fields as the JSONL output.

### JSONL Report (`--jsonl`)

One JSON object per line, streamed as entries complete — useful for large bibliographies and incremental processing:

```jsonl
{"key": "smith2020", "category": "academic", "status": "verified", "abstained": false, "coverage_incomplete": false, "confidence": 0.89, "p_valid": 0.945, "confidence_score": 96.0, "mismatched_fields": [], "api_sources": ["crossref", "dblp"], "errors": []}
{"key": "fake2099", "category": "academic", "status": "hallucinated", "abstained": false, "coverage_incomplete": false, "confidence": 0.93, "p_valid": 0.035, "confidence_score": 12.0, "mismatched_fields": ["title", "author"], "api_sources": [], "errors": []}
```

Per-line fields:

| Field | Meaning |
|-------|---------|
| `key`, `category`, `status` | BibTeX key, entry category, and the status code (table above) |
| `abstained` | `true` for could-not-verify verdicts — abstentions must never be read as confirmed hallucinations |
| `coverage_incomplete` | `true` when an abstention (or `api_error`) was reached while ≥ 1 source errored / was throttled / circuit-broken. A `not_found` with this flag is **not** a clean exhaustive miss — re-run after a cooldown before treating it as evidence of fabrication |
| `confidence` | 0–1 confidence that the **assigned status is the right call**. Direction-free: a confident `hallucinated` and a confident `verified` both carry high `confidence` |
| `p_valid` | 0–1 probability that the entry **as cited** refers to a real publication with correct metadata — **threshold/rank on this**, not on `confidence`. Verified-polarity statuses map above 0.5, problem-polarity below 0.5, abstentions sit at 0.5, and a clean exhaustive `not_found` at 0.35. Note `preprint_only` is problem-polarity for `p_valid` (the claimed venue is contradicted) even though the verdict itself is confident |
| `confidence_score` | additive 0–100 numeric confidence (next section) |
| `mismatched_fields`, `api_sources`, `errors` | non-confirmed field names, sources with hits, and per-source error strings |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (or non-strict mode) |
| 1 | Input error (file not found, parse error) |
| 4 | Strict mode: not-found or hallucinated entries found |

## CI/CD Integration

### GitHub Actions

```yaml
- name: Validate references
  run: |
    pip install bibtex-updater
    bibtex-check references.bib --strict --report report.json
- name: Upload report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: fact-check-report
    path: report.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-references
        name: Validate BibTeX references
        entry: bibtex-check --strict
        language: python
        files: \.bib$
        additional_dependencies: [bibtex-updater]
```

## Caching and Rate Limiting

API responses are cached to `.cache.fact_checker.json` by default (SQLite-backed, WAL mode, thread-safe). This speeds up repeated runs and reduces rate-limit pressure. Clear it by deleting the file or pointing `--cache-file` elsewhere; `--no-cache` disables caching entirely.

Rate limits are enforced **per service** (Crossref, OpenAlex, DBLP, OpenReview, Semantic Scholar, …), scaled by `--rate-limit` (default 45 = scale 1.0) with per-service caps below each service's documented ceiling; arXiv stays flat at 20 req/min per its politeness ask. The limiter is **adaptive**: `Retry-After` and rate-limit response headers (including on retryable 429/5xx responses) drive automatic backoff. Set `--mailto` (or `BIBTEX_CHECK_MAILTO`) so Crossref/OpenAlex serve you from their polite pools. With `--workers` concurrent entries and the cascade short-circuiting on easy entries (~1.4 API calls/entry), most bibliographies finish quickly; a Semantic Scholar key further lifts the slowest source.

## Comparison with `bibtex-update`

| Feature | `bibtex-update` | `bibtex-check` |
|---------|-----------------|----------------|
| Purpose | Transform preprints to published | Validate entries exist & match |
| Modifies files | Yes | No (read-only) |
| Target entries | Preprints | All entries |
| Output | Updated `.bib` file | Validation report (JSON/JSONL) |
| Use case | Bibliography cleanup | Quality assurance / CI |
