# BibTeX Updater

Tools for managing BibTeX bibliographies: automatically update preprints to published versions and filter to only cited references.

## Tools

| Tool | Description | Dependencies |
|------|-------------|--------------|
| `bibtex_updater.py` | Replace preprints with published versions | pip install required |
| `zotero_updater.py` | Update preprints in Zotero library | pip install required |
| `filter_bibliography.py` | Filter to only cited entries | **None** (stdlib only) |

## Quick Start

### Update Preprints

```bash
# Install dependencies
pip install bibtexparser requests crossref-commons httpx rapidfuzz

# Update preprints to published versions
python bibtex_updater.py references.bib -o updated.bib
```

### Update Zotero Library

```bash
# Install dependencies
pip install pyzotero

# Set credentials (get from zotero.org/settings/keys)
export ZOTERO_LIBRARY_ID="your_user_id"
export ZOTERO_API_KEY="your_api_key"

# Preview changes
python zotero_updater.py --dry-run

# Apply updates
python zotero_updater.py
```

### Filter Bibliography

```bash
# No installation needed - uses only Python standard library
python filter_bibliography.py paper.tex -b references.bib -o filtered.bib
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/BIBTEX_UPDATER.md](docs/BIBTEX_UPDATER.md) | Full BibTeX updater documentation |
| [docs/ZOTERO_UPDATER.md](docs/ZOTERO_UPDATER.md) | Full Zotero updater documentation |
| [docs/FILTER_BIBLIOGRAPHY.md](docs/FILTER_BIBLIOGRAPHY.md) | Full filter documentation |
| [examples/](examples/) | Example workflows and configuration files |

## Overleaf Integration

Both tools integrate with Overleaf via GitHub Actions or latexmkrc.

### GitHub Actions (Recommended)

1. Enable GitHub sync in Overleaf (Menu → Sync → GitHub)
2. Copy a workflow from [examples/workflows/](examples/workflows/) to `.github/workflows/`
3. Changes synced from Overleaf automatically trigger updates

### latexmkrc (Direct Overleaf)

For `filter_bibliography.py` only (no dependencies required):

1. Upload `filter_bibliography.py` to your Overleaf project
2. Create `.latexmkrc` based on [examples/latexmkrc](examples/latexmkrc)
3. Recompile - filtered bibliography appears in your file list

## Features

### BibTeX Updater

- **Multi-source resolution**: arXiv, Crossref, DBLP, Semantic Scholar, Google Scholar
- **High accuracy**: Title and author fuzzy matching with confidence thresholds
- **Batch processing**: Multiple files with concurrent workers
- **Deduplication**: Merge duplicates by DOI or normalized title+authors
- **Caching**: On-disk cache to avoid repeated API calls

### Zotero Updater

- **Direct Zotero integration**: Fetches and updates items via Zotero API
- **Same resolution pipeline**: Uses bibtex_updater's multi-source resolution
- **Preserves metadata**: Keeps notes, tags, and attachments intact
- **Idempotent**: Already-published papers are automatically skipped
- **Dry-run mode**: Preview changes before applying

### Filter Bibliography

- **Zero dependencies**: Uses only Python standard library
- **Works on Overleaf**: No pip install needed
- **Multiple bib files**: Merge and filter from multiple sources
- **Citation detection**: Supports natbib, biblatex, and standard LaTeX citations

## Development

```bash
# Clone and install
git clone https://github.com/rpatrik96/bibtexupdater.git
cd bibtexupdater
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest -v

# Code quality
pre-commit run --all-files
```

## License

MIT License - see [LICENSE](LICENSE) for details.
