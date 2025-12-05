# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BibTeX Updater is a Python CLI tool that replaces preprint BibTeX entries (arXiv, bioRxiv, medRxiv) with their published journal versions. It also includes a bibliography filtering tool for extracting only cited references from LaTeX documents.

## Commands

### Development Setup
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing and linting
```

### Running Tests
```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_updater.py -v     # Run a specific test file
pytest -k "test_name" -v            # Run tests matching a pattern
```

### Code Quality
```bash
pre-commit run --all-files          # Run all linters (black, ruff)
black bibtex_updater.py filter_bibliography.py
ruff check --fix .
mypy bibtex_updater.py --ignore-missing-imports
```

### Running the Tools
```bash
# Update preprints in a bib file
python bibtex_updater.py input.bib -o output.bib

# Filter bibliography to cited entries only
python filter_bibliography.py paper.tex references.bib -o filtered.bib
```

## Architecture

### Main Scripts
- **bibtex_updater.py**: Core preprint-to-published resolver using a 6-stage resolution pipeline:
  1. arXiv API → Crossref DOI lookup
  2. Crossref `is-preprint-of` relations
  3. DBLP bibliographic search
  4. Semantic Scholar API
  5. Crossref bibliographic search
  6. Google Scholar (opt-in via `--use-scholarly`)

- **filter_bibliography.py**: Extracts citation keys from LaTeX files and filters .bib to only cited entries

### Key Classes (bibtex_updater.py)
- `PreprintDetector`: Identifies preprint entries via arXiv IDs, preprint DOIs, or journal name patterns
- `MatchCandidate`: Represents a potential published version with confidence scoring
- `RateLimiter`: Thread-safe rate limiting for API calls
- `ResolutionCache`: On-disk JSON cache for API responses

### Matching Logic
Candidates are validated using:
- Title matching: token-sort ratio (fuzzy) ≥ 90%
- Author matching: Jaccard similarity on last names
- Combined score: `0.7 × title_score + 0.3 × author_score ≥ 0.9`

### GitHub Actions Workflows
- `ci.yml`: Runs linting and tests on push/PR
- `reusable-bib-update.yml`: Reusable workflow for other repos to update bibliographies
- `update-bibliography.yml` / `update-and-filter-bibliography.yml`: Example workflows for Overleaf integration

## Code Style
- Line length: 120 characters (configured in pyproject.toml)
- Formatting: Black + Ruff
- Python version: 3.9+
