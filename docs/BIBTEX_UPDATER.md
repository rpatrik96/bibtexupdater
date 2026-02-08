# BibTeX Updater

Automatically replace preprint BibTeX entries (arXiv, bioRxiv, medRxiv, etc.) with their published journal versions.

## Features

- **Multi-source resolution**: Queries arXiv, Crossref, DBLP, ACL Anthology, Semantic Scholar, and optionally Google Scholar
- **High accuracy**: Uses title and author matching with configurable confidence thresholds
- **Batch processing**: Process multiple .bib files with concurrent workers
- **Deduplication**: Merge duplicate entries by DOI or normalized title+authors
- **Flexible output**: In-place editing, merged output, or dry-run preview
- **Caching**: On-disk cache to avoid repeated API calls
- **Detailed reporting**: JSONL report of all changes made

## Installation

```bash
pip install bibtexparser requests crossref-commons httpx rapidfuzz

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

## Command Reference

```
usage: bibtex_updater.py [-h] [-o OUTPUT | --in-place] [--keep-preprint-note]
                         [--rekey] [--dedupe] [--dry-run] [--report REPORT]
                         [--cache CACHE] [--rate-limit RATE_LIMIT]
                         [--max-workers MAX_WORKERS] [--timeout TIMEOUT]
                         [--verbose] [--use-scholarly]
                         [--scholarly-proxy {tor,free,none}]
                         [--scholarly-delay SCHOLARLY_DELAY]
                         inputs [inputs ...]
```

| Option | Description |
|--------|-------------|
| `inputs` | Input .bib files |
| `-o, --output` | Output merged .bib (when not using --in-place) |
| `--in-place` | Edit files in place |
| `--keep-preprint-note` | Keep a note pointing to arXiv id |
| `--rekey` | Regenerate BibTeX keys as authorYearTitle |
| `--dedupe` | Merge duplicates by DOI or normalized title+authors |
| `--dry-run` | Preview changes without writing files |
| `--report` | Write JSONL report mapping original→updated |
| `--cache` | On-disk cache file (default: .cache.replace_preprints.json) |
| `--rate-limit` | Requests per minute (default 45) |
| `--no-cache` | Disable caching entirely for fresh lookups |
| `--clear-cache` | Clear existing cache files before running |
| `--s2-api-key` | Semantic Scholar API key (or set `S2_API_KEY`) |
| `--user-agent` | Custom User-Agent for API requests (or set `BIBTEX_UPDATER_USER_AGENT`) |
| `--mark-resolved` | Tag updated entries with `_resolved_from` field to skip on re-runs |
| `--force-recheck` | Ignore `_resolved_from` markers and reprocess all entries |
| `--resolution-cache` | Resolution cache file for semantic caching |
| `--resolution-cache-ttl` | TTL for resolution cache entries |
| `--max-workers` | Max concurrent workers (default 8) |
| `--timeout` | HTTP timeout seconds (default 20.0) |
| `--verbose` | Verbose logging |

### Google Scholar Options

| Option | Description |
|--------|-------------|
| `--use-scholarly` | Enable Google Scholar fallback (requires scholarly package) |
| `--scholarly-proxy` | Proxy for Google Scholar requests: `tor`, `free`, or `none` (default) |
| `--scholarly-delay` | Delay between Google Scholar requests in seconds (default: 5.0) |

## How It Works

The tool uses a multi-stage resolution pipeline to find published versions of preprints:

1. **arXiv → Crossref**: For arXiv preprints, query the arXiv API for DOI, then look up in Crossref
2. **Crossref Relations**: Check Crossref's `is-preprint-of` relation links
3. **DBLP Search**: Search DBLP by title and author
3b. **ACL Anthology**: For NLP papers with ACL DOI prefix (`10.18653/v1/`) or aclanthology.org URLs
4. **Semantic Scholar**: Query Semantic Scholar's paper database
5. **Crossref Search**: Bibliographic search in Crossref by title/author
6. **Google Scholar** (opt-in): Fallback search via scholarly package

### Matching Logic

Each candidate is validated using:
- **Title matching**: Token-sort ratio (fuzzy matching) ≥ 90%
- **Author matching**: Jaccard similarity on last names
- **Combined score**: `0.7 × title_score + 0.3 × author_score ≥ 0.9`
- **Metadata credibility**: Must have journal, volume/number/pages, or URL

### Preprint Detection

The tool detects preprints based on:

- **arXiv identifiers**: In `eprint`, `url`, or `note` fields
- **Preprint DOIs**: `10.48550/arxiv.*` or `10.1101/*` (bioRxiv/medRxiv)
- **Journal names**: Containing "arxiv", "biorxiv", "medrxiv", "preprint", etc.
- **Entry types**: `@unpublished` or `@misc` with preprint indicators

## Examples

### Basic Update

```bash
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

## Overleaf Integration

Automatically update preprint references in your Overleaf projects using GitHub Actions.

### How It Works

```
Edit .bib in Overleaf → Sync to GitHub → Action runs → Updated .bib committed → Overleaf pulls
```

### Setup

1. **Enable GitHub sync in Overleaf**
   - Open your project in Overleaf
   - Go to Menu → Sync → GitHub
   - Link to a new or existing repository

2. **Add the workflow file** to your repository

   Copy [examples/workflows/update-bibliography.yml](../examples/workflows/update-bibliography.yml) to `.github/workflows/`:

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

## Rate Limiting

The tool respects API rate limits with per-service rate limiting:
- **Crossref**: 45 requests/minute (polite pool)
- **DBLP**: Built-in delays
- **ACL Anthology**: 30 requests/minute
- **Semantic Scholar**: Automatic backoff (supports `--s2-api-key` for higher limits)
- **arXiv**: Adaptive rate limiting based on response headers
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
