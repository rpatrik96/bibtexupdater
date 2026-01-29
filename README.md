# BibTeX Updater

Tools for managing BibTeX bibliographies: automatically update preprints to published versions, validate references against external databases, and filter to only cited references.

## Installation

### From PyPI (Recommended)

```bash
pip install bibtex-updater

# With Google Scholar support
pip install bibtex-updater[scholarly]

# With Zotero support
pip install bibtex-updater[zotero]

# All optional dependencies
pip install bibtex-updater[all]
```

### From Source

```bash
git clone https://github.com/rpatrik96/bibtexupdater.git
cd bibtexupdater
pip install -e ".[dev]"
```

### Using uv (No Installation)

Run directly without managing virtual environments using [uv](https://docs.astral.sh/uv/):

```bash
# Run any command directly
uv run --with "bibtex-updater[all]" bibtex-update references.bib -o updated.bib

# Or use the provided wrapper script
./scripts/bibtex-x update references.bib -o updated.bib
./scripts/bibtex-x check references.bib
./scripts/bibtex-x filter paper.tex -b references.bib -o filtered.bib
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `bibtex-update` | Replace preprints with published versions |
| `bibtex-check` | Validate references exist with correct metadata |
| `bibtex-filter` | Filter to only cited entries |
| `bibtex-zotero` | Update preprints in Zotero library |

## Quick Start

### Update Preprints

```bash
# Update preprints to published versions
bibtex-update references.bib -o updated.bib

# Preview changes (dry run)
bibtex-update references.bib --dry-run --verbose
```

### Validate References (Fact-Check)

```bash
# Check if references exist and have correct metadata
bibtex-check references.bib --report report.json

# Strict mode: exit with error if hallucinated/not-found entries
bibtex-check references.bib --strict
```

### Filter Bibliography

```bash
# Filter to only cited entries
bibtex-filter paper.tex -b references.bib -o filtered.bib

# Multiple tex files
bibtex-filter *.tex -b references.bib -o filtered.bib
```

### Update Zotero Library

```bash
# Set credentials (get from zotero.org/settings/keys)
export ZOTERO_LIBRARY_ID="your_user_id"
export ZOTERO_API_KEY="your_api_key"

# Preview changes
bibtex-zotero --dry-run

# Apply updates
bibtex-zotero
```

## Standalone Scripts

For environments without pip (e.g., Overleaf), `filter_bibliography.py` can be used directly as it has no dependencies:

```bash
# Copy the script and run directly
python filter_bibliography.py paper.tex -b references.bib -o filtered.bib
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/BIBTEX_UPDATER.md](docs/BIBTEX_UPDATER.md) | Full BibTeX updater documentation |
| [docs/REFERENCE_FACT_CHECKER.md](docs/REFERENCE_FACT_CHECKER.md) | Full reference fact-checker documentation |
| [docs/ZOTERO_UPDATER.md](docs/ZOTERO_UPDATER.md) | Full Zotero updater documentation |
| [docs/FILTER_BIBLIOGRAPHY.md](docs/FILTER_BIBLIOGRAPHY.md) | Full filter documentation |
| [examples/](examples/) | Example workflows and configuration files |

## Overleaf Integration

Both tools integrate with Overleaf via GitHub Actions or latexmkrc.

### GitHub Actions (Recommended)

1. Enable GitHub sync in Overleaf (Menu -> Sync -> GitHub)
2. Copy a workflow from [examples/workflows/](examples/workflows/) to `.github/workflows/`
3. Changes synced from Overleaf automatically trigger updates

### latexmkrc (Direct Overleaf)

For `filter_bibliography.py` only (no dependencies required):

1. Upload `filter_bibliography.py` to your Overleaf project
2. Create `.latexmkrc` based on [examples/latexmkrc](examples/latexmkrc)
3. Recompile - filtered bibliography appears in your file list

## Features

### BibTeX Updater (`bibtex-update`)

- **Multi-source resolution**: arXiv, Crossref, DBLP, Semantic Scholar, Google Scholar
- **High accuracy**: Title and author fuzzy matching with confidence thresholds
- **Batch processing**: Multiple files with concurrent workers (default: 8)
- **Deduplication**: Merge duplicates by DOI or normalized title+authors
- **Smart caching**: On-disk cache + semantic resolution cache with TTL
- **Per-service rate limiting**: Optimized rate limits per API (Crossref, S2, DBLP, arXiv)
- **Batch API support**: Faster bulk lookups via arXiv/S2/Crossref batch endpoints

### Zotero Updater (`bibtex-zotero`)

- **Direct Zotero integration**: Fetches and updates items via Zotero API
- **Same resolution pipeline**: Uses the same multi-source resolution
- **Preserves metadata**: Keeps notes, tags, and attachments intact
- **Idempotent**: Already-published papers are automatically skipped
- **Dry-run mode**: Preview changes before applying

### Reference Fact-Checker (`bibtex-check`)

- **Multi-source validation**: Crossref, DBLP, Semantic Scholar
- **Detailed mismatch detection**: Title, author, year, venue comparisons
- **Hallucination detection**: Identifies likely fabricated references
- **Structured reports**: JSON and JSONL output formats
- **CI/CD integration**: Strict mode with exit codes for automation

### Filter Bibliography (`bibtex-filter`)

- **Zero dependencies**: Uses only Python standard library
- **Works on Overleaf**: No pip install needed
- **Multiple bib files**: Merge and filter from multiple sources
- **Citation detection**: Supports natbib, biblatex, and standard LaTeX citations

## Python API

```python
from bibtex_updater import Detector, Resolver, Updater, HttpClient, RateLimiter, DiskCache

# Create HTTP client with rate limiting and caching
rate_limiter = RateLimiter(req_per_min=30)
cache = DiskCache(".cache.json")
http_client = HttpClient(
    timeout=30.0,
    user_agent="bibtex-updater/0.1.0",
    rate_limiter=rate_limiter,
    cache=cache
)

# Detect preprints
detector = Detector()
detection = detector.detect(entry)

if detection.is_preprint:
    # Resolve to published version
    resolver = Resolver(http_client)
    candidate = resolver.resolve(detection)

    if candidate and candidate.confidence >= 0.9:
        # Update the entry
        updater = Updater()
        updated_entry = updater.update_entry(entry, candidate.record, detection)
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/rpatrik96/bibtexupdater.git
cd bibtexupdater
pip install -e ".[dev,all]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=bibtex_updater --cov-report=term-missing

# Code quality
pre-commit run --all-files

# Build package
python -m build

# Check package
twine check dist/*
```

## License

MIT License - see [LICENSE](LICENSE) for details.
