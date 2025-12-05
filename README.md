# BibTeX Updater

Automatically replace preprint BibTeX entries (arXiv, bioRxiv, medRxiv, etc.) with their published journal versions.

## Features

- **Multi-source resolution**: Queries arXiv, Crossref, DBLP, Semantic Scholar, and optionally Google Scholar
- **High accuracy**: Uses title and author matching with configurable confidence thresholds
- **Batch processing**: Process multiple .bib files with concurrent workers
- **Deduplication**: Merge duplicate entries by DOI or normalized title+authors
- **Flexible output**: In-place editing, merged output, or dry-run preview
- **Caching**: On-disk cache to avoid repeated API calls
- **Detailed reporting**: JSONL report of all changes made

## Installation

```bash
# Clone the repository
git clone https://github.com/rpatrik96/bibtexupdater.git
cd bibtexupdater

# Install dependencies
pip install -r requirements.txt

# Optional: Install scholarly for Google Scholar fallback
pip install scholarly>=1.7.0
```

### Dependencies

- `bibtexparser` - BibTeX parsing
- `requests` - HTTP client
- `httpx` - Async HTTP client
- `crossref-commons` - Crossref API
- `rapidfuzz` - Fuzzy string matching
- `scholarly` (optional) - Google Scholar access

## Quick Start

```bash
# Basic usage - process a .bib file and write to output
python bibtex_updater.py references.bib -o updated_references.bib

# Edit files in place
python bibtex_updater.py references.bib --in-place

# Preview changes without writing (dry run)
python bibtex_updater.py references.bib --dry-run

# Process multiple files and merge
python bibtex_updater.py file1.bib file2.bib -o merged.bib --dedupe
```

## Usage

```
usage: replace_preprints.py [-h] [-o OUTPUT | --in-place] [--keep-preprint-note]
                            [--rekey] [--dedupe] [--dry-run] [--report REPORT]
                            [--cache CACHE] [--rate-limit RATE_LIMIT]
                            [--max-workers MAX_WORKERS] [--timeout TIMEOUT]
                            [--verbose] [--use-scholarly]
                            [--scholarly-proxy {tor,free,none}]
                            [--scholarly-delay SCHOLARLY_DELAY]
                            inputs [inputs ...]

Replace preprint BibTeX entries with published versions when available.

positional arguments:
  inputs                Input .bib files

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output merged .bib (when not using --in-place)
  --in-place            Edit files in place
  --keep-preprint-note  Keep a note pointing to arXiv id
  --rekey               Regenerate BibTeX keys as authorYearTitle
  --dedupe              Merge duplicates by DOI or normalized title+authors
  --dry-run             Preview changes without writing files
  --report REPORT       Write JSONL report mapping original→updated
  --cache CACHE         On-disk cache file (default: .cache.replace_preprints.json)
  --rate-limit RATE_LIMIT
                        Requests per minute (default 45)
  --max-workers MAX_WORKERS
                        Max concurrent workers (default 4)
  --timeout TIMEOUT     HTTP timeout seconds (default 20.0)
  --verbose             Verbose logging

Google Scholar options:
  --use-scholarly       Enable Google Scholar fallback (requires scholarly package)
  --scholarly-proxy {tor,free,none}
                        Proxy for Google Scholar requests (default: none)
  --scholarly-delay SCHOLARLY_DELAY
                        Delay between Google Scholar requests in seconds (default: 5.0)
```

## How It Works

The tool uses a 6-stage resolution pipeline to find published versions of preprints:

1. **arXiv → Crossref**: For arXiv preprints, query the arXiv API for DOI, then look up in Crossref
2. **Crossref Relations**: Check Crossref's `is-preprint-of` relation links
3. **DBLP Search**: Search DBLP by title and author
4. **Semantic Scholar**: Query Semantic Scholar's paper database
5. **Crossref Search**: Bibliographic search in Crossref by title/author
6. **Google Scholar** (opt-in): Fallback search via scholarly package

Each candidate is validated using:
- **Title matching**: Token-sort ratio (fuzzy matching) ≥ 90%
- **Author matching**: Jaccard similarity on last names
- **Combined score**: `0.7 × title_score + 0.3 × author_score ≥ 0.9`
- **Metadata credibility**: Must have journal, volume/number/pages, or URL

## Examples

### Basic Update

```bash
# Input: references.bib with arXiv preprints
python bibtex_updater.py references.bib -o updated.bib --verbose
```

**Before:**
```bibtex
@article{kingma2014adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}
```

**After:**
```bibtex
@article{kingma2014adam,
  title={Adam: A Method for Stochastic Optimization},
  author={Diederik P Kingma and Jimmy Ba},
  journal={International Conference on Learning Representations},
  year={2015},
  doi={10.48550/arXiv.1412.6980}
}
```

### Keep Preprint Reference

```bash
python bibtex_updater.py refs.bib -o updated.bib --keep-preprint-note
```

Adds a note field: `note = {arXiv:1412.6980}`

### With Google Scholar Fallback

```bash
# Enable Google Scholar (slower, but catches more)
python bibtex_updater.py refs.bib -o updated.bib --use-scholarly

# With Tor proxy for heavy usage
python bibtex_updater.py refs.bib -o updated.bib --use-scholarly --scholarly-proxy tor
```

### Generate Report

```bash
python bibtex_updater.py refs.bib -o updated.bib --report changes.jsonl
```

The JSONL report contains one JSON object per entry with fields:
- `key_old`, `key_new` - BibTeX keys
- `doi_old`, `doi_new` - DOIs
- `action` - "upgraded", "unchanged", or "failed"
- `method` - Resolution method used (e.g., "arXiv->Crossref(works)")
- `confidence` - Match confidence score

## Preprint Detection

The tool detects preprints based on:

- **arXiv identifiers**: In `eprint`, `url`, or `note` fields
- **Preprint DOIs**: `10.48550/arxiv.*` or `10.1101/*` (bioRxiv/medRxiv)
- **Journal names**: Containing "arxiv", "biorxiv", "medrxiv", "preprint", etc.
- **Entry types**: `@unpublished` or `@misc` with preprint indicators

## Development

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_scholarly.py -v

# Run with coverage
pytest --cov=bibtex_updater
```

### Code Quality

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit run --all-files

# Format code
black bibtex_updater.py
ruff check --fix bibtex_updater.py
```

## Rate Limiting

The tool respects API rate limits:
- **Crossref**: 45 requests/minute (polite pool)
- **DBLP**: Built-in delays
- **Semantic Scholar**: Automatic backoff
- **Google Scholar**: 5 second delay (configurable)

Use `--rate-limit` to adjust the global rate limit.

## Troubleshooting

### "No reliable published match found"

The preprint may not have a published version yet, or the published version has different title/authors. Try:
- `--use-scholarly` for broader search
- Check if the paper was published under a different title

### Google Scholar blocking

Google Scholar aggressively blocks automated requests. Solutions:
- Use `--scholarly-proxy tor` with Tor installed
- Increase delay with `--scholarly-delay 10`
- Use `--scholarly-proxy free` for rotating free proxies

### Cache issues

Delete the cache file to force fresh lookups:
```bash
rm .cache.replace_preprints.json
```

## Overleaf Integration

Automatically update preprint references in your Overleaf projects using GitHub Actions.

### How It Works

```
Edit .bib in Overleaf → Sync to GitHub → Action runs → Updated .bib committed → Overleaf pulls
```

### Setup (One-Time)

1. **Enable GitHub sync in Overleaf**
   - Open your project in Overleaf
   - Go to Menu → Sync → GitHub
   - Link to a new or existing repository

2. **Add the workflow file** to your repository

   Create `.github/workflows/update-bibliography.yml`:

   ```yaml
   name: Update Bibliography
   on:
     push:
       paths: ['**.bib']

   jobs:
     update:
       uses: rpatrik96/bibtexupdater/.github/workflows/reusable-bib-update.yml@main
       with:
         bib_files: 'references.bib'
         dedupe: true
   ```

3. **Done!** Any .bib changes synced from Overleaf will automatically trigger updates.

### Reusable Workflow Options

```yaml
jobs:
  update:
    uses: rpatrik96/bibtexupdater/.github/workflows/reusable-bib-update.yml@main
    with:
      bib_files: '*.bib'           # Files to process (default: *.bib)
      dedupe: true                  # Remove duplicates (default: true)
      keep_preprint_note: false     # Keep arXiv ID in note (default: false)
      verbose: false                # Verbose logging (default: false)
```

### Manual Trigger

You can also manually trigger the workflow from the GitHub Actions tab:
1. Go to your repository → Actions
2. Select "Update Bibliography"
3. Click "Run workflow"
4. Optionally specify a specific .bib file

## Bibliography Filtering

Filter your bibliography to include only cited references. Uses only Python standard library (no pip dependencies), making it ideal for Overleaf and restricted environments.

```bash
python filter_bibliography.py paper.tex -b references.bib -o filtered.bib
```

For detailed documentation including Overleaf integration and local latexmkrc setup, see [docs/FILTER_BIBLIOGRAPHY.md](docs/FILTER_BIBLIOGRAPHY.md).

## License

MIT License - see [LICENSE](LICENSE) for details.
