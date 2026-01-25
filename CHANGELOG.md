# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/rpatrik96/bibtexupdater/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rpatrik96/bibtexupdater/releases/tag/v0.1.0
