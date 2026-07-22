# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **LaTeX accent macros destroyed accented surnames.** `latex_to_plain` stripped a LaTeX command and substituted a *space*, but accent macros whose command is a letter (`\H`, `\c`, `\k`, `\v`, `\u`, `\r`, `\d`, `\b`) are separated from their base character by whitespace — so `Heged{\H u}s` became `Heged us`, whose surname key is `us`, and `Erd{\H o}s` reduced to `os`. Every author with a Hungarian, Polish, Turkish, Romanian, Czech or Scandinavian name was unmatchable against their own record: an author list that was character-for-character correct scored 0.0 similarity and was reported AUTHOR_MISMATCH. Accent and glyph macros now decode to precomposed Unicode, verified at 100% agreement against `pylatexenc` over a 1003-case corpus.
- **`howpublished` was never read as a venue.** Venue extraction was `journal or booktitle`, so `@misc` front matter (editorials, magazine columns) that names its journal in `howpublished` reported "No venue claimed" — the venue then matched vacuously and the journal name never constrained retrieval. `series` is used as a further fallback; a URL-valued `howpublished` stays a web reference, not a venue claim.
- **`editor` did not back `author`.** `@proceedings` and `@book` name editors rather than authors, so every volume-level entry was compared with an *empty* author list.
- **Springer/IFIP publisher series were treated as venues.** Crossref and OpenAlex return the series as container-title for proceedings published in them, so an entry citing the real conference was contradicted by `Lecture Notes in Computer Science`. The series roster covered only PMLR/JMLR W&CP; it now includes LNCS/LNNS/LNEE/LNBIP/LNICST, CCIS, Studies in Computational Intelligence, AISC, IFIP AICT and SpringerBriefs, which abstain instead of mismatching.
- **Shortened venue names counted as different venues.** An index storing `2021 IFIP/IEEE International Symposium on Integrated Network Management (IM)` as `Integrated Network Management`, or dropping a subtitle, produced a venue mismatch. One name containing the other is now the same venue — unless the longer one adds a satellite-event marker (workshop, companion, tutorial, doctoral consortium), which stays a genuine mismatch. This also closes a pre-existing leak where alias lookup resolved `ICML Workshop on X` to `ICML`, contradicting the documented intent that workshops are distinct venues.
- **`@proceedings` titles were compared as paper titles.** A volume title *is* the conference name, so it carries a leading year, an ordinal, a `Proceedings of the` prefix and a trailing `(ACRONYM YEAR)`. Comparing it verbatim flagged entries at similarities as high as 0.97, where `(CNSM 2024)` vs `(CNSM)` tripped the near-miss rule. Volume titles are now normalized as venue names, and the near-miss rule — which looks for deliberate tampering — is suppressed for them.
- **Theses were verdicted against unrelated papers.** `@phdthesis`/`@mastersthesis` are absent from Crossref/OpenAlex/DBLP/S2 by construction, so the best candidate was whatever paper shared a surname; one Hungarian dissertation was reported as mismatching a paper on the American labor movement. Without a confirmed title a thesis now abstains (could-not-verify) instead of asserting a problem. Positive evidence still flags: DOI, year and arXiv-ID validation all run earlier.

## [1.5.1] - 2026-07-22

Report persistence and the strict exit code now survive a chained run.

### Fixed

- **`--report` silently dropped in chained runs** (#52): `bibtex-check --resolve-first --report X.json` wrote the resolved bib and a console summary but never wrote `X.json` — `main()` returns from inside the chain driver before the report-writing tail, so every persistence obligation the parent CLI had already parsed was discarded. `bibtex-update --then-check --report` lost its JSONL the same way. Each driver now writes its own report: the checker path emits the same `{summary, entries}` schema as the plain path (so existing parsers are unaffected), with the chain view attached under a new `chain` key — trusted upgrades never reach the checker, so `entries` alone under-reports the input set, and the `chain` block is what accounts for every entry and records which were skipped as upgrades.
- **`--strict` always exited 0 under `--resolve-first`** (#52): a CI gate on the strict exit code passed regardless of how many problematic entries were found. The chain path now applies the same gate as the plain checker — only positive-evidence problems fail, with `--strict-warn-cnv` opting into failing on abstentions.
- **`BIBTEX_CHECK_STRICT=1` ignored when chaining** (#52): the env-var form of `--strict` never reached the chain, which reads strictness off the checker arg namespace.

## [1.5.0] - 2026-07-20

Resolve→check chaining with a non-compensatory trust gate, OpenAlex premium API-key support, and a thread-safety fix in the adaptive rate limiter.

### Added

- **Chained resolve→check** (`bibtex-check --resolve-first`, #49): runs the preprint resolver first and fact-checks only the entries it did **not** upgrade — upgraded entries are clean database records by construction, so re-checking them is wasted spend. Shares one cache/HTTP client across both stages and always writes the cleaned bib (`--resolved-out`, default `<input>.resolved.bib`). Entries the resolver already upgraded in a previous run are skipped on re-entry.
- **Non-compensatory trust gate on upgrades** (#49): a resolver upgrade must clear every trust criterion individually (no averaging away a failed one); risky upgrades are flagged and reverted atomically, and upgraded entries are rebuilt from the resolved record rather than field-patched, so a partial upgrade can no longer leave a hybrid entry.
- **OpenAlex premium API key** (`--openalex-api-key` / `OPENALEX_API_KEY`, #50): authenticated OpenAlex requests for deployments with a premium key — required at practical throughput since OpenAlex began enforcing keys (Feb 2026).

### Fixed

- **Adaptive-limiter lock race** (#51): `AdaptiveRateLimiterRegistry` mutated `_limits` outside the lock guarding the paired `_limiters` swap in three places (header-driven slowdown, 429 backoff, `reset_limit`); concurrent `adapt` calls could tear the limit/limiter pair. The read-modify-write now runs under the lock. Extracted from #43, whose broader rate-limit policy work was superseded by the v1.4.0 circuit breaker and per-service limits.

### Docs

- arXiv-scale deployment bottleneck analysis (what dominates cost/latency when checking bibliographies at archive scale).

## [1.4.0] - 2026-06-15

`bibtex-check` release branch: a clearer output contract for downstream consumers (`p_valid`, `coverage_incomplete`), new positive-evidence detectors for fabricated venues / silent author truncation / preprint-as-published / wrong-year citations, a batch of false-positive fixes for author checks and DOI-anchored verification, and substantial throughput work (identifier fast paths, key-gated Semantic Scholar steps, realistic adaptive rate limits).

### Added

- **`p_valid` — explicit P(valid) on every per-entry record** (`calibration.p_valid_from_result`; new JSONL/JSON-report key `p_valid`, also on `FactCheckResult`/`VerificationResult`). The existing `confidence` (`overall_confidence`) means "confidence the *assigned status* is the right call", which inverts direction across statuses (a confident `hallucinated` means the entry is almost certainly fake). `p_valid` is the value to threshold/rank on: the probability that the entry **as cited** refers to a real publication with correct metadata. Computed from a documented status-polarity map — VALID-polarity statuses map to `0.5 + 0.5·confidence`, PROBLEM-polarity to `0.5 − 0.5·confidence`, abstentions stay at the neutral `0.5`. A *clean* exhaustive `not_found` (see `coverage_incomplete` below) prices at `0.35` — in well-indexed domains, "no source knows this paper" is weak evidence of fabrication. `preprint_only`/`unpublished_at_claimed_venue` are PROBLEM-polarity for `p_valid` even though the verdict-confidence prior is high: the claim as cited ("published at venue X") is contradicted.
- **`coverage_incomplete` — abstentions reached under source errors/throttling are now distinguishable from clean exhaustive misses** (new JSONL/JSON-report key next to `abstained`; summary counter `coverage_incomplete_count`; surfaced in the CLI summary next to the could-not-verify bucket with a re-run-after-cooldown hint). A `not_found`/`unconfirmed` produced while sources were erroring or circuit-broken was previously indistinguishable from "every source answered and none knows this paper", so downstream consumers flagged throttled lookups as hallucinations. True when the verdict is an abstention (incl. the `--strict-warn-cnv` promotion) AND ≥ 1 source errored for the entry; always true for `api_error`; always false for verified/problem verdicts (a positive verdict stands on its evidence).
- **Silent author-list truncation flagged in default mode** (`author_truncated`, previously `--strict`-only): an in-order author prefix with co-authors silently dropped (no `et al.`/`and others`/`...` disclosure) escalates from the `unconfirmed` abstention to a problem verdict — under tight gates (structured, order-reliable best record; ≥ 2 dropped or ≥ ⅓ of the canonical list; ≥ 2 order-reliable sources independently corroborate the longer list).
- **Identifier-based venue identity for cross-source venue consensus**: records now carry ISSN, OpenAlex source id, and DBLP stream key, and the cross-source wrong-venue check groups sources by these identifiers (union-find) in addition to canonical-name equality — venues outside the hand-curated alias map (non-ML journals, regional venues) can now form the ≥ 2-source consensus needed to flag a wrong venue.
- **`nonexistent_venue` status + venue-registry existence check** (`--no-check-venue-existence` to disable): a fabricated venue has no record to contradict it, so it abstained as `unconfirmed` forever. On the abstention path only, when the claimed venue is unknown to the alias map and no source reports it for the (otherwise real) paper, the DBLP venue registry and OpenAlex `/sources` are probed; only when **both** answer successfully with zero plausible matches does the verdict escalate to this positive-problem status. Any lookup error keeps the abstention.
- **Same-conference exact-year rule**: the ±1-year tolerance (meant for preprint/online-first drift) no longer excuses a wrong year when entry and record canonicalize to the *same conference*, the record is not a preprint, and ≥ 2 order-reliable sources agree on the record year — conference proceedings years are exact. Journals keep the tolerance.
- **Identifier-less preprint-as-published detection**: entries citing an arXiv-only paper as published *without* carrying a DOI/eprint (the common offender shape) could never reach the preprint check. Two new signals close the gap: an arXiv-id pivot through the matched record (its `arxiv_id`/DataCite DOI), and a DBLP signal that fires when every strong-title DBLP hit is the CoRR (arXiv) stream and no source reports the claimed venue. Both are gated off whenever any source positively confirms the venue.
- **OpenReview API v2 fallback**: venues that migrated to api2.openreview.net (ICLR 2024+, NeurIPS 2023+) were invisible to the v1-only client, silently dropping recent years from the authoritative ICLR/NeurIPS registry. The search now falls back to the v2 `/notes/search` endpoint when both v1 strategies miss.
- **Key-gated Semantic Scholar `/paper/search/match` cascade step**: with an S2 API key, the single-best-title-match endpoint is consulted right after Crossref (authenticated S2 is fast), and the final S2 relevance-search step is skipped whenever the match step contributed — per-entry S2 spend stays at one call. Keyless cascades are byte-for-byte unchanged.
- **Semantic Scholar `/paper/batch` bulk prefetch** (API-key + cache deployments): one POST per ≤ 500 entry identifiers primes the exact cache entries the per-entry `get_paper` lookups read. Best-effort — any failure falls back to per-entry fetches.
- **DOI- and arXiv-anchored fast paths** (`--no-fast-path` to disable; automatically inert in `--strict`): after the entry's own identifier passed the consistency check, the cascade is skipped when the single authoritative record behind that identifier *fully confirms every claimed field* at the full thresholds (the arXiv path additionally demands exact author-sequence equality and no venue/DOI claim). The fast paths can only short-circuit a clean `verified`; anything less falls through to the normal cascade.
- **`--mailto` / `BIBTEX_CHECK_MAILTO`**: polite-pool contact identity for Crossref/OpenAlex (feeds the User-Agent and the `--openalex-mailto` default). When unset, the historical placeholder is kept and a one-time warning recommends a real address.

### Changed

- **Per-service rate limits are now realistic and adaptive**: Crossref 300/min (was 50/min against a ~50 req/*s* polite-pool ceiling), OpenAlex 150/min (previously not scaled by `--rate-limit` at all), Semantic Scholar 60/min with an API key, and arXiv *lowered* to a flat 20/min (the old 30/min default exceeded arXiv's ~1 req/3 s politeness ask). The adaptive rate-limiter registry is now actually wired into the HTTP layer: every real transport response — including retryable 429/5xx — feeds `Retry-After`/header-driven backoff, which was previously dead code.
- **Cascade stop-condition memoized per invocation**: the all-fields-confirmed check no longer recompares the same candidate records after every source step (pure CPU saving; verdicts identical).

### Fixed

- **Crossref subtitles are joined into record titles**: ACM/IEEE deposits split colon-titles into `title` + `subtitle`, so the DOI-consistency check saw only the head ("NeRF" vs "NeRF: Representing Scenes…") and flagged *correct* DOIs as `doi_mismatch`.
- **Crossref record year prefers publication dates over the DOI deposit date**: `created` (deposit) is now consulted strictly last (`published-print` > `published-online` > `issued` > `published` > `created`); deposit dates run years off for backfilled archives and minted false `year_mismatch` findings.
- **Batch DOI pre-validation and the Crossref `/works` cache warm-up now reach the CLI path**: both optimizations were computed and then discarded because the CLI wraps the checker in `UnifiedFactChecker`, which neither forwarded the pre-validated DOIs nor exposed the shared clients. No duplicate per-entry `doi.org` HEADs, no throwaway HTTP client.
- **HTTP response cache stores non-JSON payloads**: arXiv Atom XML responses were never cached (every arXiv lookup re-hit the network); they are now stored in a versioned envelope that preserves the content type on replay. Legacy cache files remain readable.
- **LaTeX is stripped from retrieval queries on all sources** (Crossref `query.title`, OpenAlex `title.search`, S2 free-text, OpenReview, the relaxed-author fallback, the structured author recheck — previously DBLP only): `{B}race {G}roups`-style markup degraded ranking on every external index. Scoring/comparison still uses the original strings.
- **`WebVerifier` uses the shared HTTP client** for URL HEAD/content checks (same pool, rate limiters, and error mapping as every other source); the unused `requests` runtime dependency is dropped.
- **Given-name/author-swap false positives from alphabetized records and mixed-initial names**: shared-surname order conclusions are no longer anchored on alphabetized records (Crossref NeurIPS-proceedings deposits sort contributors); the given-name audit grades the entry against *every* same-surname record author and abstains when any candidate explains it benignly; a single-letter first given token ("J. Westerborn" vs "Johan") compares as an initial, never as a full-name substitution; and a same-multiset reorder against a display-alphabetized record softens to the `unconfirmed` abstention instead of a positive flag. True positives (given-name substitution, real swaps against publication-order records) keep firing.
- **2-author same-multiset swaps require cross-source corroboration**: with two authors, alphabetical order coincides with publication order half the time, so a lone alphabetizing source cannot distinguish a swap from a sort artifact. The mismatch now stands only when a second order-reliable source independently shows the same non-entry order; otherwise the entry abstains (never `verified`). 3+-author swap detection and `--strict` behavior are unchanged.
- **Generational suffixes are stripped from surname keys**: `last_name_from_person`/`_normalize_surname_key` dropped trailing 4-digit DBLP homonym suffixes and single-letter initials, but not generational suffixes — so "John Smith Jr." reduced to the surname key `jr` and spuriously mismatched the suffix-less "John Smith" of the same author. A shared `_reduce_trailing_to_surname` helper now drops `jr/sr/ii/iii/iv` on both the entry and the authoritative-record side (kept symmetric). (#47)
- **Glued separator-less initials are no longer mis-graded as a given-name substitution**: PubMed/biomedical given names written as concatenated initial runs ("ME" for *Maria Elisabetta*, "RMF" for *Robin Maria Francisca*) were read as a full given token and escalated to a false `given_name_substitution` author flag. A glued all-caps initial run is now expanded to spaced initials before grading, routing it through the existing initials-compatible path (never a substitution); genuine substitutions and real short given names are unaffected. Removes a cluster of out-of-domain false positives (HALLMARK `test_crossdomain` FPR −11pp). (#48)

## [1.3.0] - 2026-06-11

### Added

- **OpenReview resolution stage** (`bibtex-update`): the resolver now upgrades **accepted** OpenReview submissions (ICLR/NeurIPS/TMLR) to their published `@inproceedings` (venue + `openreview.net/forum?id=…` URL) as stage 3c — after ACL Anthology, before Semantic Scholar — a throttle-resilient fallback for when DBLP is rate-limited. Rejected, withdrawn, under-review, and CoRR notes are never resolved (`openreview_acceptance`). For these DOI-less ML venues the venue + forum-URL record is the canonical published form, so the stage is default-on. Wired into both the synchronous `Resolver` and the async `AsyncResolver`. (OpenReview was already a step in the `bibtex-check` verification cascade.)

### Fixed

- **Author-fabrication check no longer flags authors present in the best-matched record** (`bibtex-check`): "absent from every candidate" was computed over order-reliable sources only, excluding the best-matched record when its source is order-unreliable (arXiv, Semantic Scholar). A recent paper whose full author list lives on arXiv while Crossref/OpenAlex have indexed only a lead-author stub had its non-lead authors flagged "likely fabricated" even though the best match (similarity 1.0) confirmed them. Order-reliability gates author *order*, not *presence*; the best-matched record's surnames now veto the flag. OSAKA/OrdinalCLIP trailing-author leak detection is unchanged.
- **`GIVEN_NAME_SUBSTITUTION` no longer swallows gross author-set mismatches** (`bibtex-check`): a lone author mismatch carrying an incidental substitution finding (frequent family names colliding with a fabricated roster) was routed to `GIVEN_NAME_SUBSTITUTION` and dropped from the PROBLEMATIC bucket, waving fabrications through when the entry lacked an arXiv-ID anchor. The route now requires the audit's matching-surnames escalation note.

## [1.2.0] - 2026-05-30

Catch-rate release. Four new behavioral capabilities targeted at the `could-not-verify` HALLUCINATED bucket and one residual `wrong_venue` leak class. Converted ~110 abstentions to caught problematics across dev+test, caught the SCoRe wrong-venue leak (the v1.1.0 `cheap_fix` target), and kept the held-out FPR steady.

### bibtex-check accuracy (HALLMARK v1.0 corrected gold)

| | v1.1.0 | v1.2.0 | Δ |
|---|---|---|---|
| dev_public FPR | 1.59% | **1.99%** | +2 FPs (X2/X4 trade-offs documented) |
| test_public FPR (held-out) | 2.32% | **2.32%** | unchanged |
| **dev_public caught-on-hallucinated** | 60.4% | **75.2%** | **+14.8pp** |
| **test_public caught-on-hallucinated** | 58.0% | **73.7%** | **+15.7pp** |
| dev_public leak | 0.65% (4) | 0.65% (4) | unchanged |
| test_public leak | 0.76% (4) | **0.57% (3)** | **−1 leak (SCoRe caught)** |

The +14.8pp / +15.7pp catch-rate increases came from the X3 ID-anchored field-mismatch helper + the X4 relaxed-author retrieval fallback unblocking entries the v1.1.0 cascade abstained on. The SCoRe leak (entry claimed NeurIPS, real venue is ICLR 2021) was caught by the new cross-source venue verification (X1).

### Known minor regressions

The catch-rate work introduces a small, characterized FP set on dev_public (test FPR is unchanged):

- `ed071a6dfa34` (Improving Robustness using Generated Data): `verified` → `arxiv_id_mismatch`. X2 extracted the arXiv ID from the DataCite DOI, and `_check_arxiv_id_consistency` flagged a divergence — flagged for investigation as a v1.2.1 target.
- `e59d381d98e6` (2026-synthetic VALID — Self-Supervised Learning via Flow-Guided Neural Operator): `verified` → `given_name_substitution`. Narrow X2+`1e37f7c` interaction class: 2026 arXiv-DOI VALID entries where the structured-record competitor disagrees with the arXiv record on author given-name spelling. Documented as a known v1.2.1 target.
- `f185501f556e` (Beyond log2(T) regret — dev) and `d07ee00b0c0f` (𝒩-WL Graph Neural Networks — test): both `not_found` → `partial_match` via X4's relaxed-author fallback retrieving a wrong-paper candidate. This is the documented X4 trade-off (`not_found → partial_match` on DOI-less entries when the relaxed retrieval surfaces a near-title match with different authors).

Offsetting clears (v1.1.0 FPs that v1.2.0 verifies correctly): `adf6c58262bf` (Community Concealment from Unsupervised Graph Learning) and `e51140f8b514` (RLang — Rodríguez-Sánchez). Net: dev +2 FPs, test 0.

### Added

- **`_detect_cross_source_venue_mismatch`** (`fact_checker.py`) — the venue analogue of `_detect_author_fabrication`. When ≥2 order-reliable sources contributed candidate records and agree on a canonical venue that differs from the entry's canonical venue (and the entry's venue is a recognized published venue, not blank/preprint), downgrade the venue outcome to `MISMATCH` and route the status to `VENUE_MISMATCH`. Catches the SCoRe-shape residual leak (entry claims NeurIPS, every authoritative source agrees on ICLR).
- **arXiv DataCite DOI extraction** in `_arxiv_id_from_entry` (`fact_checker.py`). Mines `entry["doi"]` for `10.48550/arXiv.<id>` (case-insensitive, version-stripping). The rest of the arXiv-ID-anchored machinery is unchanged: `_check_arxiv_id_consistency` fetches the arXiv record, `_id_anchored_author_mismatch` fires when the fetched authors disagree. Unblocks the HALLUCINATED + arXiv-DataCite-DOI cluster (HALLMARK's 2026-synthetic batch).
- **`_id_anchored_field_mismatch`** (`fact_checker.py`) — the field analogue of `_id_anchored_author_mismatch`. Fires when (a) entry DOI resolves via Crossref, (b) the DOI record's title confirms the entry, AND (c) `compare_venue` returns a hard MISMATCH (not NON_COMPARABLE) OR `compare_year` returns a hard MISMATCH beyond tolerance (gated against preprint-twin records via the existing `_doi_is_preprint` helper). Emits `VENUE_MISMATCH` / `YEAR_MISMATCH` on DOI-confirmed entries.
- **Relaxed-author retrieval fallback** in `_query_cascade` (`fact_checker.py`). When the standard cascade returns zero candidates (or all candidates fall below `abstention_below`), retry Crossref and OpenAlex with the raw title (no first-author constraint). Tagged `from_fallback=True` so downstream scoring reflects the weaker retrieval signal. The transition is **never** `not_found → VERIFIED`; the realistic transition is `not_found → AUTHOR_MISMATCH` (the cascade now finds a wrong-paper candidate whose authors disagree).
- **Order-reliable structured-record preference in selection** (`fact_checker.py:_select_best_candidate`, new constant `_ORDER_RELIABLE_PREFERENCE_BAND = 0.02`). Inside a 0.02 score sub-band, an order-reliable candidate (DBLP / OpenReview / Crossref structured) wins selection over an order-unreliable one (e.g. an arXiv-API record). Without this, the arXiv DataCite DOI extraction would let arXiv records (preprint venue → NON_COMPARABLE → fewer hard mismatches in the confirmation-key tiebreak) win selection over structured records and skip the given-name audit. The narrow 0.02 sub-band preserves arXiv-API wins when it's clearly the better match.

### Changed

- Status taxonomy gains `VENUE_MISMATCH` and `YEAR_MISMATCH` as DOI-anchored findings (previously these statuses were only emitted by the unconditional field comparison; now also from the `_id_anchored_field_mismatch` helper on DOI-confirmed records).

### Tests

1088 → 1122 passing (+30 fix tests + 4 regression-fix tests). New modules: `tests/test_cross_source_venue.py`, `tests/test_arxiv_datacite_doi.py`, `tests/test_id_anchored_field_mismatch.py`, `tests/test_relaxed_author_fallback.py`, plus 4 new tests in `tests/test_fact_checker.py::TestOrderReliablePreferenceOverArxivOnly`.

### Migration notes

This is a **minor** release (semver MINOR), not a patch — the four new fixes add new behavioral capabilities (new finding sources, new candidate pool entries, new fallback cascade step). Existing entries that were `verified` in v1.1.0 stay `verified` in v1.2.0 (verified against the corrected gold spot-check); new catches come from the previously-abstaining could-not-verify pool.

## [1.1.0] - 2026-05-30

Carries the v1.0.0 false-positive work to held-out (HALLMARK v1.0 `test_public`), adds an opt-in `--strict` evaluation mode aligned with [arXiv's 2026 hallucinated-reference policy](https://www.nature.com/articles/d41586-026-01595-5), cuts the could-not-verify bucket on real refs by ~70%, and corrects 30 HALLMARK mislabels upstream via [hallmark#9](https://github.com/rpatrik96/hallmark/pull/9).

### bibtex-check accuracy (HALLMARK v1.0 corrected gold, apples-to-apples)

| | Pre-fix | Post-fix | Δ |
|---|---|---|---|
| dev_public FPR | 2.58% | **1.59%** | −38.5% |
| test_public FPR (held-out) | 8.94% | **2.32%** | **−74.1%** |
| dev_public leak (raw / policy-adjusted) | 0.49% | 0.65% / **0.32%** | +1 raw |
| test_public leak (raw / policy-adjusted) | 0.38% | 0.76% / **0.57%** | +2 raw |

The test_public FPR drop is driven primarily by the v1.1.0 CNV venue/retrieval refinements (`ea63b7d`) — held-out FPR reduced by **−74%** from the v1.0.0 baseline.

Policy-adjusted leak rates exclude hyphen-only title differences — see [`docs/KNOWN_LEAKS.md`](docs/KNOWN_LEAKS.md). Of the 8 residual `verified`-on-a-real-leak cases in default mode, 3 are hyphen-only (`Schema Variable`/`Schema-Variable`, `Chain of-Thought`/`Chain-of-Thought`, `Language Guided`/`Language-Guided`) — hyphenation is bibliographic noise that varies across DBLP / Crossref / publisher records, and flagging it would generate FPs on most legit refs. `--strict` (Levenshtein-1) still catches every hyphen difference for arXiv-style high-stakes audits. The remaining 5 policy-adjusted residual leaks are 3 letter-add title perturbations (`Privacys`, `Explanations`, `Models`), 1 author-list truncation (OSAKA), and 1 wrong-venue substitution (SCoRe claims NeurIPS, real venue is ICLR 2021 — flagged as a v1.1.1 `cheap_fix` target via cross-source venue verification).

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
