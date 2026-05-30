# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-30

Carries the v1.0.0 false-positive work to held-out (HALLMARK v1.0 `test_public`), adds an opt-in `--strict` evaluation mode aligned with [arXiv's 2026 hallucinated-reference policy](https://www.nature.com/articles/d41586-026-01595-5), cuts the could-not-verify bucket on real refs by ~70%, and corrects 30 HALLMARK mislabels upstream via [hallmark#9](https://github.com/rpatrik96/hallmark/pull/9).

### bibtex-check accuracy (HALLMARK v1.0 corrected gold, apples-to-apples)

| | Pre-fix | Post-fix | Δ |
|---|---|---|---|
| dev_public FPR | 2.59% | **1.79%** | −31% |
| test_public FPR (held-out) | 8.97% | **5.98%** | −33% |
| dev_public leak | 0.49% | 0.65% (3 fpr-tradeoff title typos + 1 reordered subset) | +1 |
| test_public leak | 0.38% | 0.38% (2 fpr-tradeoff title typos) | 0 |

### Added

- **`--strict` mode** (`BIBTEX_CHECK_STRICT=1`) for high-stakes submissions where leak ≫ FP. Tightens title (Levenshtein-1), year (tolerance 0), author-set (single-source single-extra), author-order (no alphabetization escape), and silent-truncation checks. New statuses `TITLE_NEAR_MISS`, `AUTHOR_TRUNCATED`, `STRICT_WARN_PREPRINT_YEAR`.
- **`--strict-warn-cnv`** (requires `--strict`) promotes `unconfirmed`/`not_found` to a fourth visible category `STRICT_WARN_CNV`, distinct from `PROBLEMATIC`.
- **Cross-source author-fabrication detection** (`fact_checker.py:_detect_author_fabrication`): when the entry has ≥2 surnames absent from every order-reliable candidate's full author set (≥2 sources contributing, no `and others` sentinel), the author outcome downgrades to `AUTHOR_MISMATCH`. Catches fabricated trailing authors that slip past the prefix-N slice.
- **`_PLATFORM_MARKERS`** (`matching.py`): OpenReview / `OpenReview.net` treated as a hosting-platform venue (NON_COMPARABLE), like arXiv/PMLR. Also adds `ssrn` to preprint-server markers.
- **`_looks_alphabetized`** (`matching.py`) detects A–Z-sorted record author lists.
- **`_strip_track_decorations`** (`matching.py`) normalizes OpenReview venue strings like `"ICLR 2023 poster"` / `"NeurIPS 2022 oral"` to bare acronyms before alias lookup. Deliberately leaves `findings` and `workshop` intact — those are distinct sub-venues.

### Changed

- **`latex_to_plain`** (`utils.py`) now `html.unescape`s before LaTeX-stripping — DBLP-scraped `&apos;`/`&amp;` no longer survive normalization (cleared the d'Amore / Ch'ng / D'Hondt FP cluster).
- **`symmetric_author_match`** (`matching.py`) no longer fires order-`MISMATCH` when the API record is alphabetized (record-sort artifact common in Crossref proceedings deposits, prefix `10.52202` NeurIPS). Preserves swap detection against non-alphabetized sources. The previous d302fb5 multiset rule was net-negative on HALLMARK v1.0 (every "swapped_authors" leak turned out to be a benchmark mislabel); this change keeps the defense-in-depth for real swaps while eliminating the alphabetization-driven FPs.
- **`_compare_all_fields`** (`fact_checker.py`) routes a preprint/series record's year to `NON_COMPARABLE` (mirroring the existing venue logic): a preprint can't refute a published year.
- **`given_name_position_audit`** (`utils.py`) no longer abstains when only one side has a repeated surname; the unique side uniquely pins the pairing, so a position-0 substitution still flags correctly (catches lead-author swaps like `"Shunyu Zhou"` / `"Denny Zhou"` even when the entry repeats the canonical name elsewhere).
- **`get_canonical_venue`** (`matching.py`) accepts word-boundary substring matches for single-token venue acronyms (4–7 chars), so OpenReview venues like `"ICLR 2023 poster"` canonicalize to `iclr` even when the alias-to-full ratio is below 0.4. Word-boundary anchoring prevents `acl`-inside-`naacl` collisions.
- **`_normalize_venue_for_matching`** (`matching.py`) strips OpenReview `venueid` patterns (`ICLR.cc/2024/Conference`), drops trailing periods, and collapses `". "` → `" "` before alias lookup, so dotted ISO-4 forms like `"Trans. Mach. Learn. Res."` canonicalize.
- **`EXPANDED_VENUE_ALIASES`** (`matching.py`) gains dotted ISO-4 forms and bare acronyms for `tmlr` (incl. `"accepted by tmlr"`) and `jmlr`.
- **`OpenReviewClient.search`** (`sources.py`): paperhash now preserves Unicode surname spellings (Müller stays `müller`) because OpenReview's index keys on the literal form. When paperhash misses, a `/notes/search?term=<title>` fallback runs — gated to require both title and `first_author`, so author-less searches still return `[]`.
- **DBLP cascade query** (`fact_checker.py:_query_cascade`) LaTeX-strips the title and Unicode-folds the first-author surname before forming the query, so titles with `{B}race {G}roups` and accented surnames (`György`) hit DBLP's token-AND matcher.

### Fixed

- 5 previously-flagged `author_mismatch` FPs cleared by HTML-entity decoding + alphabetization gate (`d'Amore`, `Ch'ng`, KL-Gaussian, CARE, RLHF).
- Preprint-year FPs cleared by the `NON_COMPARABLE` routing.
- Lead-author given-name substitution (Least-to-Most: `"Shunyu Zhou"` → canonical `"Denny Zhou"`) now flags as `GIVEN_NAME_SUBSTITUTION` (the repeated-surname guard previously skipped position 0 when the entry repeated the canonical name).

### Transparency

- **`docs/KNOWN_LEAKS.md`**: enumerates every residual `VERIFIED`-on-a-real-leak case against the corrected HALLMARK v1.0 gold (5 dev + 2 test), with the BibTeX-style entry, the canonical paper, the precise perturbation, and the `--strict` rule that catches it. `--strict` catches 6/6 default-mode leaks (4 `TITLE_NEAR_MISS`, 1 `AUTHOR_TRUNCATED`, plus the Least-to-Most case now caught in default mode too). Linked from the README's "Verdicts" subsection.

### Upstream HALLMARK dataset

[hallmark#9](https://github.com/rpatrik96/hallmark/pull/9) corrects 30 entries the v1.0 auto-labeller flagged as fabricated but are in fact real, correctly-cited papers (3 batches; includes FlashAttention, DDPM, Imagen, SimCLR, Performers, ViT-vs-CNN, Chain-of-Thought (Wei), Zero-Shot Reasoner (Kojima), MERLOT, SimSiam, AdaFed, …). Failure mode: arXiv DOIs register with DataCite, not CrossRef, so the auto-labeller's CrossRef-resolution check returned "no resolve" for legitimate arXiv-published papers. Three of the corrections override prior-audit rejections (DDPM-Dhariwal, FlashAttention, Imagen) on independent arXiv-grounded evidence; full provenance + conflict notes live in `scripts/patch_mislabels.py`.

### Test suite

1064 → 1088 passing tests (+24 strict-mode tests; +8 cross-source author-fab + lead-given-name tests; +3 regression tests for FIX 1/2/3/5).

## [0.10.0] - 2026-05-28

### Changed
- **Comparison surnames now have a single source of truth (`PublishedRecord.surname_keys`)**: the recurring "subtle wrong verdict" bugs all shared one root cause — *comparison asymmetry*, where the BibTeX-entry side and the API-record side reduced the same surname through *different* normalization (e.g. entry `"Aaron van den Oord"` → `oord` vs. record `family` kept raw as `van den oord` → Jaccard `0` → false `AUTHOR_MISMATCH` / `HALLUCINATED`). The fix had been a helper called at ~7 separate sites, which is exactly how it drifted. `PublishedRecord` now exposes `surname_keys(limit)` — the one place that turns a record author into a comparison key, routing each `family` through the same `last_name_from_person` the entry side uses via `authors_last_names`. All seven comparison sites (5 in `updater.py`, 2 in `fact_checker.py`, plus `FieldFiller._find_best_crossref_match` and `WorkingPaperVerifier._score` which were still keying the record side *raw*) now consume it, so the two sides are symmetric by construction and cannot drift again. The now-redundant module-level `_record_surnames` helper was removed. No thresholds, weights, or verdict logic changed.

### Added
- **`PublishedRecord.canonical_venue`**: a single record-side accessor for the canonical venue (wrapping `get_canonical_venue`), mirroring `surname_keys` for the venue dimension.

### Fixed
- **DBLP homonym-disambiguation suffix is now stripped at the surname-key level too**: a record family carrying a `NNNN` homonym suffix (`"Yu Sun 0020"`, `"Chuan Guo 0001"`) reduced to the *number* (`0020`) as its comparison key, falsely mismatching the same author. `last_name_from_person` now drops a trailing 4-digit token before reducing to the final surname token (guarded so an all-digits name is never emptied). This is defense-in-depth alongside the existing strip in `dblp_hit_to_record`, so the key is robust even when a suffix reaches the comparison layer from any source.

### Tests
- **+54 tests across two new self-checking oracles** that would have caught the asymmetry class before it shipped, instead of after:
  - `tests/test_record_roundtrip.py` — a record turned into an entry (via the production `Updater.update_entry` path) must verify against *itself* with zero field mismatches and clear the resolver `MATCH_THRESHOLD`; covers particle surnames, multi-word particles, diacritics, conference venues, and many-author records.
  - `tests/test_metamorphic_symmetry.py` — states each past bug as an *invariance*: a true match's verdict must survive citation-style transforms (`"Given Family"` ↔ `"Family, Given"`, diacritics, DBLP suffix, particle placement), `combined_author_score` must be symmetric, and `canonical_venue` must not collapse sibling journals.
- Full suite: 839 passed, 1 skipped (was 785 + 54 new; zero regressions).

## [0.9.2] - 2026-05-27

### Fixed
- **Misattributed arXiv IDs silently survived (and could rewrite an entry into an unrelated paper)**: an entry's cited arXiv ID was trusted without checking that the ID's *actual* paper matches the entry, so a wrong ID (copy-paste / lookup error) survived because title/author search VERIFIED the entry against the real paper from Crossref/DBLP/S2 — and the resolver's arXiv-ID-keyed stages would rewrite the entry into the unrelated paper at `confidence = 1.0` (real cases: `onebench2024`'s correct `2412.07689` "ONEBench…" replaced by `2412.06745` "RoboTron-Drive…"). `bibtex-check` now runs a pre-search `_check_arxiv_id_consistency()` that fetches the entry's own arXiv ID and reports `FactCheckStatus.ARXIV_ID_MISMATCH` when the fetched title diverges (gated by `FactCheckerConfig.check_arxiv_consistency`, default on, with `arxiv_consistency_min_title=0.50`). `bibtex-update`'s `Resolver._verify_arxiv_match()` gates Stage 1 / 1b on `MATCH_THRESHOLD` like the search-based stages, so a wrong cited ID falls through to title-based resolution instead of corrupting the entry.
- **DBLP homonym disambiguation suffix produced false `author_mismatch`**: DBLP appends a 4-digit suffix to homonymous author names (`"Yu Sun 0020"`, `"Chuan Guo 0001"`); `dblp_hit_to_record` took the number as the family name. The suffix is now stripped so the surname is the real family name (the digits move into the given name).
- **Particle surnames produced false `AUTHOR_MISMATCH` / false `HALLUCINATED` and false resolver rejections**: author comparison was asymmetric. The BibTeX entry side reduced a name to its last token, while the API-record side used the raw multi-word `family` field verbatim (`"van den Oord"` → `van den oord`), so Jaccard against the entry's `oord` was `0` and a correctly-cited paper by an author with a nobiliary particle (von, van der, de la, dos, …) was flagged as an author mismatch — depressing the score enough to risk a false `HALLUCINATED` in `bibtex-check`, and (combined with the new arXiv-match gate, below) falsely *rejecting* a correctly-resolved record in `bibtex-update`. `last_name_from_person` now reduces any family — including particles — to its final, most distinctive token, and every comparison site runs **both** the entry side and the API-record side through it: `FactChecker._compare_all_fields` / `_score_candidate`, and the resolver's five match-score sites via a new `_record_surnames` helper. Both citation styles now produce the same key symmetrically.
- **`extract_arxiv_id_from_text` mistook numbers that merely look like a modern arXiv ID**: any `NNNN.NNNNN` substring matched, so a DOI fragment (`10.1234/5678.9012` → `5678.9012`) or an arbitrary number was treated as an arXiv identifier, driving false preprint detection in `bibtex-update` and bogus arXiv lookups in `bibtex-check` / Zotero sync. A new `is_valid_arxiv_id` rejects modern IDs whose `YYMM` month is not `01–12`, and extraction now scans all candidates (`finditer`) returning the first *valid* one. `FactChecker._arxiv_id_from_entry` applies the same month check to `eprint`.
- **Legacy / `.pdf` arXiv URLs were truncated**: `ARXIV_HOST_RE` stopped at the first `/`, so `arxiv.org/abs/hep-th/9901001` yielded `hep-th` and `arxiv.org/pdf/2602.01031v2.pdf` leaked the `v2.pdf` suffix. The host pattern now captures the full path segment, and extraction strips a trailing `.pdf` and `vN` version and re-normalizes through `ARXIV_ID_RE`, so legacy category IDs survive intact and modern IDs are returned bare.
- **Generic single-word venue aliases collapsed distinct sibling journals**: `get_canonical_venue("Nature Physics")` substring-matched the alias `"nature"` and returned canonical `"nature"`, conflating *Nature* with *Nature Physics* / *Science* with *Science Robotics* / *PNAS* with *PNAS Nexus* and masking a genuine `VENUE_MISMATCH`. Only the generic single-word journal names (`nature`, `science`, `pnas`) now require an exact match; acronym venues keep substring matching, so a track/suffix form (`"NeurIPS Track"` → `neurips`, `"ICML 2021"` → `icml`) and multi-word siblings (`"Nature Communications"` → `nature_comm`) still canonicalize correctly.

### Internal
- De-duplicated the arXiv-record lookup: `_check_arxiv_id_consistency` and `_query_arxiv_by_id` now share a memoized `FactChecker._arxiv_record(id)` instead of fetching + parsing the same arXiv Atom feed twice per entry. Factored the record-surname extraction (`updater._record_surnames`) and the arXiv-ID normalization (`extract_arxiv_id_from_text._normalize`) out of duplicated call sites.

### Tests
- +22 tests in `tests/test_subtle_failures.py` (particle-surname symmetry on both the fact-checker and resolver paths, impossible-month arXiv-ID rejection, legacy/`.pdf` URL extraction, generic-journal vs acronym venue matching); plus arXiv-ID↔title consistency, resolver `_verify_arxiv_match`, and DBLP-suffix/`CoRR`-venue regression tests. Full suite: 785 passed, 1 skipped (zero regressions).

## [0.9.1] - 2026-05-26

### Added
- **Authoritative arXiv-by-ID verification in `bibtex-check`**: the academic verifier previously matched entries purely by title/author text search against Crossref/DBLP/Semantic Scholar. Brand-new arXiv-only preprints are not yet indexed by those aggregators, so the search returned unrelated records and the entry was reported `HALLUCINATED` (or `NOT_FOUND`) with a nonsense best match — e.g. all five real Feb–Apr 2026 hallucination-benchmark preprints (`HalluHard` arXiv:2602.01031, `HalluCitation` arXiv:2601.18724, …) were flagged as fabricated. `FactChecker` now extracts the arXiv ID from the `eprint`/`archivePrefix` fields or an `arxiv.org/abs/<id>` URL and fetches the authoritative record from the arXiv export API (new `ArxivClient` + `arxiv_atom_to_record` Atom parser), adding it as a scored verification candidate before the empty/chimeric checks. Valid preprints now verify instead of false-flagging; entries without an arXiv identifier never hit the arXiv client. The lookup is wired through `UnifiedFactChecker`; the `FactChecker(..., arxiv=...)` parameter is optional, so existing callers and tests are unaffected. +7 tests in `tests/test_arxiv_id_lookup.py`.

### Fixed
- **S2 arXiv lookups that only tag a preprint with a published venue**: Semantic Scholar's arXiv-keyed resolver can return a record that mixes the *published* venue (e.g. `publicationVenue.name="International Conference on Learning Representations"`) with the arXiv preprint's `year` and `externalIds.DOI` (`10.48550/arXiv...`, `journal.name="ArXiv"`). Building a `PublishedRecord` from such a payload produced an internally inconsistent upgrade (ICLR venue + arXiv 2024 year + arXiv DOI) — e.g. `Identifiable Exchangeable Mechanisms...` (arXiv:2406.14302, published ICLR 2025) was upgraded to `journal={International Conference on Learning Representations}` but kept `year={2024}` and `doi={10.48550/arxiv.2406.14302}`. All three S2-from-arXiv builders (`Resolver.s2_from_arxiv`, `Resolver.s2_batch_lookup`, `AsyncResolver.s2_from_arxiv`) now reject a resolved record when the resolved DOI is a preprint DOI (arXiv `10.48550/arxiv` **or** bioRxiv/medRxiv `10.1101`, mirroring `Detector.detect`) **or** S2's `journal.name` is a known preprint host (reusing `PREPRINT_HOSTS`), so resolution falls through to a source carrying the real published record (DBLP → the correct ICLR-2025 / OpenReview entry, no arXiv DOI). Genuine S2 records with a real publisher DOI + non-preprint journal are unaffected. Also fixes the related `--force-recheck` non-idempotency, where the retained arXiv DOI re-triggered preprint detection on every run.
- **Silent data loss on non-standard BibTeX entry types**: `BibLoader` constructed `BibTexParser(common_strings=True)` without `ignore_nonstandard_types=False`. Because bibtexparser defaults `ignore_nonstandard_types=True`, biblatex entry types (`@online`, `@software`, `@dataset`, `@patent`, `@electronic`, `@thesis`, …) were silently dropped at parse time — never entering the database, never resolved, written, or reported (a real run lost `@online{nanda2025pragmatic}`, 177→176 entries). The parser now passes `ignore_nonstandard_types=False`, so these entries are retained and round-trip through `BibWriter` unchanged. (Accepted trade-off: this also retains typo'd/unknown entry types such as `@junk` instead of dropping them — a visible, fixable artifact is preferable to silent data loss.)
- **Dropped-entry audit trail**: added a defense-in-depth safety net for entries the parser still skips silently (genuinely malformed input). A new pure helper `detect_dropped_keys(raw_text, parsed_ids)` compares declared `@key` markers against parsed IDs; `load_databases` now logs a `WARNING` naming each dropped citation key and its file, and — when `--report` is set — emits one JSONL row per dropped key via the new `write_dropped_report_line` using the existing report schema with `action="dropped"`. Existing report rows and their ordering are unchanged.

## [0.9.0] - 2026-05-08

### Added
- **Cascading source verification** (`--cascade`): explicit CrossRef → Semantic Scholar → OpenAlex order with high-confidence short-circuit. Adds OpenAlex as a fourth source via the new `OpenAlexClient` in `bibtex_updater.sources`. Inspired by [CheckIfExist (Abbonato 2026)](https://arxiv.org/abs/2602.15871) Algorithm 1.
- **Top-K candidate retrieval** (`--top-k N`): per-source top-K results re-ranked by RapidFuzz Levenshtein title similarity before expensive cross-checks (default 3, capped at 10).
- **Cross-source author intersection**: `cross_source_author_intersection()` cross-validates author family names across sources. `confirmed = ∩` of normalized names; `suspect = union \ confirmed`. Multi-source bonus `β_ms ∈ [0, 10]` when ≥2 sources confirm the same authors. Catches `swapped_authors` / chimeric citation hallucinations.
- **Numeric `confidence_score`** (0–100): additive in the JSONL output. Two formulas — Case A asymmetric for the high-title-low-author chimeric case (`S_title − 0.5 × (100 − S_author)`), Case B average + bonus otherwise — with explicit penalty constants (`PENALTY_TITLE_MISMATCH`, `PENALTY_AUTHOR_MISMATCH`, etc.) at module level for override without auto-fitting.
- **Rich `VerificationResult`**: per-entry struct with `similarity_breakdown`, `confirmed_authors`, `suspect_authors`, `sources_consulted`, `sources_confirmed`, `issues`, `matched_metadata`. Built via `build_verification_result()` from a classic `FactCheckResult` — purely additive, no schema break.
- **Non-generative-AI mode** (`--non-generative` CLI flag and `BIBTEX_CHECK_NON_GENERATIVE=1` env var): refuses to load any LLM backend at runtime. Forward-compat guard for [ACL ARR](https://aclrollingreview.org/reviewerguidelines#q-can-i-use-generative-ai) and [ICML 2026](https://icml.cc/Conferences/2026/LLM-Policy) LLM-in-review policy compliance. Inspired by [HalluCiteChecker (Sakai et al. 2026)](https://arxiv.org/abs/2604.26835).
- **CLI flags**: `--cascade`, `--top-k N`, `--openalex-mailto EMAIL`, `--non-generative`.
- **New module** `bibtex_updater.sources` exposing `OpenAlexClient`, `select_top_k_by_title_similarity`, `cross_source_author_intersection`, and the cascade tuning constants.

### Changed
- `FactCheckerConfig` gains `cascade_mode`, `top_k`, `cascade_low_confidence`, `cascade_high_confidence`, `openalex_mailto`. All default to legacy parallel-search behavior unless `--cascade` is set.
- `FactChecker.API_SOURCES` now includes `openalex` (used in cascade mode only).

### Backward compatibility
- All existing JSONL output keys retained. New `confidence_score` is additive.
- Default behavior unchanged unless `--cascade` is set.
- All 673 pre-existing tests pass; +35 new tests in `tests/test_cascade_sources.py`.

## [0.7.0] - 2026-02-12

### Added
- **Pre-API year validation**: Entries with future years (`year > current_year`) flagged as `future_date`, implausible years (`< 1800`) or non-numeric years as `invalid_year` — zero API cost
- **DOI resolution check**: HEAD request to `doi.org` catches fabricated DOIs (`doi_not_found` status) before expensive API lookups
- **Alias-aware venue matching**: 17 ML/AI venue aliases (NeurIPS/NIPS, ICML, ICLR, CVPR, ICCV, etc.) with canonical name resolution; known-different venues always flagged as mismatches
- **Preprint-vs-published detection**: Queries Semantic Scholar to detect entries claiming a venue (e.g., "NeurIPS") when only an arXiv preprint exists (`preprint_only` status)
- **Streaming JSONL output**: Results flushed to `--jsonl` file after each entry; partial results survive timeouts, crashes, and Ctrl+C
- **Semantic Scholar API key support** for `bibtex-check`: `--s2-api-key` flag and `S2_API_KEY` env var for authenticated rate limits (1 req/s vs shared pool)
- **New CLI flags**: `--no-cache`, `--no-check-dois`, `--no-check-years`
- **New status codes**: `future_date`, `invalid_year`, `doi_not_found`, `preprint_only`, `published_version_exists`

### Changed
- Venue comparison now uses `venues_match()` with alias map instead of raw fuzzy score — eliminates false matches between similar-named but distinct conferences (e.g., CVPR vs ICCV)
- `process_entries()` accepts optional `jsonl_path` for streaming output
- `FactCheckerConfig` gains `check_years` and `check_dois` boolean flags (both default `True`)

## [0.6.1] - 2026-02-10

### Documentation
- Added animated GIFs to README for visual feature demonstration
  - Hero pipeline animation showing all 9 resolution stages
  - Before/after preprint-to-published transformation
  - Zotero sync integration and AI collection organization
  - Obsidian AI auto-keywording with knowledge graph
  - Reference fact-checker with hallucination detection
- GIF generation scripts in `scripts/` for reproducibility

## [0.6.0] - 2026-02-10

### Added
- **OpenAlex as resolution source** (Stage 1b) for preprint-to-published version tracking
  - 250M+ works across all disciplines with best-in-class version deduplication
  - Looks up by arXiv ID first, falls back to DOI
  - `openalex_work_to_record()` converter validates `publishedVersion` in locations, rejects preprint types/venues
  - Rate limiting at 100 req/sec (OpenAlex polite pool)
  - Both sync (`Resolver`) and async (`AsyncResolver`) implementations
- **Europe PMC as resolution source** (Stage 1c) for bioRxiv/medRxiv preprints
  - Life science specialist with preprint-to-published linking
  - Only activates for bioRxiv/medRxiv entries (gated by `10.1101/` DOI prefix or journal name)
  - Title+author search with `SRC:MED` filter and fuzzy match validation
  - `europepmc_result_to_record()` converter with PPR (preprint) source rejection
  - Rate limiting at 20 req/sec
  - Both sync and async implementations
- **Ecosystem landscape documentation** (`docs/LANDSCAPE.md`)
  - Integrated databases with per-source details
  - Evaluated-but-not-integrated databases with reasoning
  - Competing tools (rebiber, reffix, PreprintResolver, bibcure, PreprintMatch, PaperMemory)
  - Citation hallucination checkers and BibTeX cleanup tools
- 31 new tests for OpenAlex and Europe PMC integrations (583 total)

## [0.5.1] - 2026-02-08

### Documentation
- Updated README with all features from v0.2.0–v0.5.0
  - Added `bibtex-zotero-organize` and `bibtex-obsidian-keywords` to CLI commands table
  - Added ACL Anthology to multi-source resolution lists and feature descriptions
  - Added Zotero Organizer and Obsidian Keywords feature sections
  - Added resolution tracking (`--mark-resolved`) to feature list
- Updated `docs/BIBTEX_UPDATER.md` with new CLI flags (`--no-cache`, `--clear-cache`, `--s2-api-key`, `--user-agent`, `--mark-resolved`, `--force-recheck`, `--resolution-cache`, `--resolution-cache-ttl`), ACL Anthology pipeline stage, and per-service rate limits
- Updated `docs/ZOTERO_UPDATER.md` with ACL Anthology in resolution pipeline

## [0.5.0] - 2026-02-06

### Added
- **ACL Anthology as resolution source** for computational linguistics papers
  - Automatically resolves papers from ACL, EMNLP, NAACL, EACL, AACL, CoNLL, COLING, and Findings venues
  - New stage 3b in the resolution pipeline (between DBLP and Semantic Scholar)
  - Zero overhead for non-NLP papers: only triggers when ACL DOI prefix (`10.18653/v1/`) or aclanthology.org URL is present
  - `extract_acl_anthology_id()` utility for extracting Anthology IDs from DOIs and URLs
  - `acl_anthology_bib_to_record()` for converting ACL Anthology BibTeX to `PublishedRecord`
  - Rate limiting at 30 req/min for aclanthology.org
  - Both sync (`Resolver`) and async (`AsyncResolver`) implementations
- 9 new NLP venue patterns added to `KNOWN_CONFERENCE_VENUES` for credibility validation

### Fixed
- BibTeX field parser now correctly handles nested braces (e.g., `{BERT}`, `Guzm{\'a}n`)
  - Replaced non-greedy regex with brace-depth-counting parser
- Handle `None` values for aliases/keywords in Obsidian frontmatter

## [0.4.1] - 2026-02-01

### Fixed
- AI backends now return atomic keywords instead of compound phrases like "Causal Inference in Machine Learning"
  - Updated classification prompts to instruct AI to return separate topics for multi-concept papers
  - Compound keywords using " - " separator are still properly split by existing logic

## [0.4.0] - 2026-02-01

### Added
- **Auto-Keywording for Obsidian Notes** (`bibtex-obsidian-keywords` command)
  - AI-powered keyword generation for paper notes using Claude, OpenAI, or local embeddings
  - Generates `[[wikilinks]]` for better Obsidian knowledge graph connectivity
  - `--dry-run` to preview changes without modifying files
  - `--limit N` to process exactly N enrichable notes (skipped notes don't count)
  - `--min-keywords N` to skip notes that already have enough keywords (saves API calls)
  - `--backend` to choose AI provider (claude, openai, embedding)
  - `--topics-file` to provide existing topics for consistent tagging
- **Zotero tags → wikilinks** in paper template
  - Existing Zotero tags automatically convert to `[[wikilinks]]` on import
- **Templater enrichment script** (`zotero-enrich-keywords.md`)
  - Post-import AI keyword enrichment directly in Obsidian

### Changed
- Improved abstract extraction regex to handle blank lines after callout headers

## [0.3.0] - 2026-02-01

### Added
- **Obsidian Zotero Sync Templates** (`examples/obsidian-zotero-sync/`)
  - Templater scripts for automating Zotero → Obsidian annotation extraction
  - `zotero-sync.md`: Interactive single-paper sync (update current paper or import by citekey)
  - `zotero-bulk-sync.md`: Bulk import new papers by comparing CSL-JSON export against existing notes
  - `zotero-paper-template.md`: Example paper template with color-coded annotation callouts
  - Comprehensive README with setup instructions for hotkeys and optional startup automation

### Documentation
- Added examples directory with reusable Obsidian templates
- Color-coded annotation system: Summary (yellow), Important (red), Notation (green), Technical (blue), Contribution (purple), Literature (pink), Assumption (orange), Wrong (gray)

## [0.2.0] - 2026-02-01

### Added
- **Zotero Sync Integration** (`--zotero` flag for `bibtex-update`)
  - Simultaneously sync upgraded entries to Zotero library when updating .bib files
  - Matches bib entries to Zotero items by arXiv ID, DOI, or fuzzy title+author
  - `--zotero-dry-run` to preview Zotero changes without applying
  - `--zotero-collection` to limit sync to specific Zotero collection
  - `--zotero-library-type` for user or group libraries
- **Zotero Library Organizer** (`bibtex-zotero-organize` command)
  - Automatically organize Zotero items into hierarchical collections based on research taxonomy
  - Multiple classification backends: Claude, OpenAI, local embeddings
  - Caching for classification results to reduce API calls
  - Dry-run mode and batch processing with configurable limits
- **Tag-based chunking for Zotero updates** (#25)
  - `preprint-upgraded`, `preprint-checked`, `preprint-error` tags for tracking
  - `--recheck` mode to retry previously checked items
  - `--force` mode to reprocess all items regardless of tags

### Changed
- Extended `ProcessResult` dataclass with `arxiv_id` and `record` fields for Zotero sync

### Dependencies
- Added `pyyaml>=6.0` for taxonomy configuration (organizer)
- Added optional `sentence-transformers>=2.2.0` for local embedding backend

## [0.1.4] - 2026-01-30

### Added
- `--user-agent` CLI argument for custom API request headers (#23)
  - Also reads `BIBTEX_UPDATER_USER_AGENT` environment variable
  - Helps avoid Semantic Scholar 429 rate limit errors by identifying requests
- `--mark-resolved` flag to tag updated entries with `_resolved_from` field (#24)
  - Stores original preprint identifier (e.g., `arXiv:2401.00001`)
  - Entries with this field are automatically skipped on subsequent runs
- `--force-recheck` flag to ignore `_resolved_from` markers and reprocess all entries (#24)
- `skipped_resolved` count in summary output when entries are skipped

### Fixed
- Semantic Scholar API 400 Bad Request errors (#23)
  - Removed deprecated `doi` field from API requests
  - DOI is now correctly extracted from `externalIds.DOI`

## [0.1.3] - 2026-01-30

### Added
- `--no-cache` flag to disable caching entirely for fresh lookups
- `--clear-cache` flag to clear existing cache files before running
- `--s2-api-key` flag for Semantic Scholar API authentication (also reads `S2_API_KEY` env var)
- `MATCH_THRESHOLD` constant (0.85) in `Resolver` and `AsyncResolver` for consistent matching

### Fixed
- arXiv ID extraction from `journal` and `howpublished` fields (major bug fix)
  - Previously, entries like `journal={arXiv preprint arXiv:2310.15213}` would not extract
    the arXiv ID, causing cache key collisions and missed API lookups
  - Now correctly extracts arXiv IDs from all relevant fields
- Match threshold consistency: lowered from hardcoded 0.9 to 0.85 across all search stages,
  aligned with `FieldFiller.MATCH_THRESHOLD`

### Improved
- Real-world test shows 117% improvement in upgrade rate (12→26 papers) and 31% reduction
  in failures (45→31) on a 162-entry bibliography

## [0.1.2] - 2026-01-30

### Fixed
- Conference papers now properly accepted in `process_entry` functions (#19)
  - v0.1.1 fix was incomplete: `_credible_journal_article` was updated but
    `process_entry` and `process_entry_with_preload` still had hardcoded checks
  - "Attention Is All You Need" and similar NeurIPS/ICML papers now resolve correctly

## [0.1.1] - 2026-01-29

### Added
- UV-enabled wrapper script (`scripts/bibtex-x`) for running without venv management (#16)
- Conference paper support: NeurIPS, ICML, ICLR, AAAI, CVPR, etc. now resolve correctly (#19)

### Fixed
- Cross-device link error when output file is on different filesystem (#17)
- Field ordering now consistent (author, title, booktitle, journal, year, etc.) (#18)
- DBLP conference papers (`proceedings-article`) now accepted instead of filtered out (#19)
- Semantic Scholar type normalization (`Conference` → `proceedings-article`) (#19)
- Filter out CoRR (arXiv journal name in DBLP) from results (#19)
- Google Scholar conference venue detection for ML conferences (#19)

## [0.1.0+] - Unreleased improvements

### Added
- Per-service rate limiting via `RateLimiterRegistry` for optimized API throughput
- Semantic resolution caching via `ResolutionCache` with configurable TTL
- Negative result caching to avoid re-querying known failures
- Batch API support for faster bulk lookups:
  - arXiv: up to 100 IDs per request
  - Semantic Scholar: up to 500 papers via batch endpoint
  - Crossref: filter queries for multiple DOIs
- Async HTTP client (`AsyncHttpClient`) for parallel operations
- Parallel bibliographic search via `AsyncResolver`
- Entry prioritization (`prioritize_entries`) for confidence-based ordering
- Adaptive rate limiting based on API response headers
- New CLI arguments: `--resolution-cache`, `--resolution-cache-ttl`

### Changed
- Default max workers increased from 4 to 8 for better I/O overlap
- Refactored `_resolve_uncached` into 6 modular stage methods (CC: 85 → 8)
- Refactored `main()` into composable helper functions (CC: 74 → 4)

### Improved
- Test coverage: 341 → 360 tests (+19 new tests for stage methods)
- CI/CD: Added Codecov coverage reporting

## [0.1.0] - 2025-01-25

### Added
- Initial release as PyPI package (`pip install bibtex-updater`)
- Proper Python package with src-layout structure
- CLI entry points: `bibtex-update`, `bibtex-check`, `bibtex-filter`, `bibtex-zotero`
- `bibtex-update` command for upgrading preprints to published versions
  - Resolution via arXiv API, Crossref, DBLP, Semantic Scholar
  - Optional Google Scholar support via scholarly package
  - Confidence scoring with configurable thresholds
  - Diff output and JSONL report generation
- `bibtex-check` command for validating bibliography references
  - Multi-source validation (Crossref, DBLP, Semantic Scholar)
  - Detection of hallucinated or fabricated references
  - Detailed mismatch categorization (title, author, year, venue)
  - Support for books, web references, working papers
- `bibtex-filter` command for filtering to cited references
  - Standalone stdlib-only implementation (works on Overleaf)
  - Support for multiple .tex files and bib files
  - LaTeX comment stripping
- `bibtex-zotero` command for updating Zotero libraries
  - Batch update preprints in Zotero to published versions
  - Preserves notes, tags, and attachments
  - Dry-run mode for previewing changes

### Features
- Thread-safe rate limiting for API requests
- On-disk JSON caching for API responses
- Comprehensive test suite with pytest fixtures
- MIT License

[Unreleased]: https://github.com/rpatrik96/bibtexupdater/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/rpatrik96/bibtexupdater/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/rpatrik96/bibtexupdater/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.7.0...v0.9.0
[0.7.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/rpatrik96/bibtexupdater/compare/v0.6.0...v0.6.1
[0.5.1]: https://github.com/rpatrik96/bibtexupdater/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/rpatrik96/bibtexupdater/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rpatrik96/bibtexupdater/releases/tag/v0.1.0
