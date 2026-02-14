# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-02-14

### Performance
- **Migrated from DiskCache to SqliteCache** — eliminated O(N²) bottleneck in cache operations
- **Added rate limiter jitter** — prevents burst-stall pattern by randomizing request timing (±10% variance)
- **Increased HTTP connection pool limits** — 50 max connections, 20 keepalive (up from 10/10 defaults)
- **Reuse shared HTTP client for DOI validation** — eliminates redundant connection overhead

### Fixed
- **Per-service rate limiting with RateLimiterRegistry** — consistent rate limiting across all modules (updater, fact checker, filter, Zotero)
- **Correct service names in updater rate limiter configuration** — fixed mismatched service identifiers
- **Added missing service parameter in crossref search** — ensures proper rate limit application

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

[Unreleased]: https://github.com/rpatrik96/bibtexupdater/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/rpatrik96/bibtexupdater/compare/v0.7.0...v0.8.0
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
