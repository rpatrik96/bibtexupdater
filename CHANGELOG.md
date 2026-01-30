# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rpatrik96/bibtexupdater/releases/tag/v0.1.0
