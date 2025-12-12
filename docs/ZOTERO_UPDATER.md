# Zotero Updater

Automatically update preprint entries in your Zotero library to their published journal versions.

## Features

- **Direct Zotero integration**: Fetches and updates items via Zotero API
- **Multi-source resolution**: Uses the same resolution pipeline as bibtex_updater (arXiv, Crossref, DBLP, Semantic Scholar)
- **Preserves metadata**: Keeps notes, tags, and attachments intact
- **Dry-run mode**: Preview changes before applying
- **Filtering**: Process specific collections or tagged items only
- **Idempotent**: Already-published papers are automatically skipped

## Installation

```bash
pip install pyzotero bibtexparser requests crossref-commons httpx rapidfuzz
```

### Dependencies

- `pyzotero` - Zotero API client
- All dependencies from `bibtex_updater.py` (see [BIBTEX_UPDATER.md](BIBTEX_UPDATER.md))

## Setup

### Get Zotero Credentials

1. Go to [zotero.org/settings/keys](https://www.zotero.org/settings/keys)
2. Note your **User ID** (numeric, shown at the top)
3. Click **Create new private key**
   - Give it a description (e.g., "preprint-updater")
   - Enable **Allow library access**
   - Enable **Allow write access**
4. Copy the generated API key

### Configure Environment

Set environment variables:

```bash
export ZOTERO_LIBRARY_ID="your_user_id"
export ZOTERO_API_KEY="your_api_key"
```

Or pass them as command-line arguments (see below).

## Quick Start

```bash
# Preview changes (dry run)
python zotero_updater.py --dry-run

# Apply updates
python zotero_updater.py

# Process only items with a specific tag
python zotero_updater.py --tag "to-update"

# Process only a specific collection
python zotero_updater.py --collection ABCD1234
```

## Command Reference

```
usage: zotero_updater.py [-h] [--dry-run] [--collection COLLECTION]
                          [--tag TAG] [--limit LIMIT] [--verbose]
                          [--library-id LIBRARY_ID] [--api-key API_KEY]
                          [--library-type {user,group}]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview changes without applying |
| `--collection` | Only process items in this collection (key) |
| `--tag` | Only process items with this tag |
| `--limit` | Maximum items to process (default: 100) |
| `--verbose, -v` | Verbose output |
| `--library-id` | Zotero library ID (or set `ZOTERO_LIBRARY_ID`) |
| `--api-key` | Zotero API key (or set `ZOTERO_API_KEY`) |
| `--library-type` | Library type: `user` (default) or `group` |

## How It Works

### Preprint Detection

The tool identifies preprints in your Zotero library based on:

- **URLs**: Contains `arxiv.org`, `biorxiv.org`, or `medrxiv.org`
- **Journal names**: Contains "arxiv", "biorxiv", or "medrxiv"
- **DOI patterns**: `10.48550/arxiv.*` or `10.1101/*`
- **Extra field**: Contains `arXiv:XXXX.XXXXX`

### Resolution Pipeline

For each detected preprint, the tool uses the bibtex_updater resolution pipeline:

1. **arXiv → Crossref**: Query arXiv API for DOI, look up in Crossref
2. **Crossref Relations**: Check `is-preprint-of` links
3. **DBLP Search**: Search by title and author
4. **Semantic Scholar**: Query paper database
5. **Crossref Search**: Bibliographic search by title/author

### Update Process

When a published version is found:

1. **Updates metadata**: Title, authors, journal, DOI, URL, volume, issue, pages, date
2. **Changes item type**: Upgrades to `journalArticle`
3. **Preserves arXiv reference**: Adds original arXiv ID to the Extra field
4. **Adds tag**: Marks item with `preprint-upgraded` tag
5. **Preserves**: Notes, tags, collections, and attachments are not modified

### Idempotency

**Published papers are automatically skipped.** The tool checks if an item is already published by verifying:

- Journal name is not a preprint server
- DOI is not a preprint DOI pattern
- URL is not a preprint server URL

This ensures running the tool multiple times is safe and won't overwrite already-published items.

## Examples

### Basic Usage

```bash
# Set credentials
export ZOTERO_LIBRARY_ID="12345678"
export ZOTERO_API_KEY="abcdefghijklmnop"

# Preview what would be updated
python zotero_updater.py --dry-run --verbose
```

**Output:**
```
INFO: Found 15 preprint(s) out of 150 items
INFO: [1/15] Processing: Attention Is All You Need...
INFO:   ✓ would_update: arXiv preprint → Advances in Neural Information Processing Systems (SemanticScholar(arXiv))
INFO: [2/15] Processing: BERT: Pre-training of Deep Bidirectional...
INFO:   ✓ would_update: arXiv preprint → NAACL-HLT (DBLP)
...

============================================================
SUMMARY
============================================================
Total processed:  15
Updated:          12
Not found:        3
Errors:           0
```

### Process Tagged Items

Create a workflow where you tag items you want to check:

```bash
# In Zotero: Add tag "check-published" to items you want to update
# Then run:
python zotero_updater.py --tag "check-published"
```

### Process a Collection

```bash
# Get collection key from Zotero URL or API
python zotero_updater.py --collection ABCD1234 --dry-run
```

### Group Libraries

```bash
python zotero_updater.py --library-type group --library-id 98765
```

## Output

### Update Results

Each processed item returns one of these actions:

| Action | Description |
|--------|-------------|
| `updated` | Successfully updated to published version |
| `would_update` | Would update (dry-run mode) |
| `not_found` | No published version found (may still be preprint-only) |
| `skipped` | Not detected as preprint (already published) |
| `error` | Error during processing |

### Summary Report

After processing, a summary is printed:

```
============================================================
SUMMARY
============================================================
Total processed:  15
Updated:          12
Not found:        3
Errors:           0

--- Updated/Would Update ---
  [ABC123] Attention Is All You Need
    arXiv preprint → Advances in Neural Information Processing Systems
    DOI: 10.5555/3295222.3295349, Method: SemanticScholar(arXiv), Conf: 0.95

--- Not Found (may still be preprint-only) ---
  [XYZ789] My Recent Preprint That Hasn't Been Published Yet
```

## Caching

The tool uses a local cache (`.cache.zotero_updater.json`) to avoid repeated API calls. Delete this file to force fresh lookups:

```bash
rm .cache.zotero_updater.json
```

## Troubleshooting

### "ZOTERO_LIBRARY_ID and ZOTERO_API_KEY required"

Set the environment variables or pass `--library-id` and `--api-key`:

```bash
export ZOTERO_LIBRARY_ID="your_user_id"
export ZOTERO_API_KEY="your_api_key"
```

### "pyzotero not installed"

Install the required dependency:

```bash
pip install pyzotero
```

### Items not being detected as preprints

The tool looks for specific patterns. Ensure your preprint items have:
- arXiv/bioRxiv/medRxiv in the URL or journal field
- Or a preprint DOI pattern

### "No published version found"

The paper may not have been published yet, or was published with a different title. The tool only updates when it finds a high-confidence match.

## Comparison with bibtex_updater.py

| Feature | bibtex_updater.py | zotero_updater.py |
|---------|-------------------|-------------------|
| Input | BibTeX files | Zotero library |
| Output | BibTeX files | Zotero library |
| Resolution | Same pipeline | Same pipeline |
| Google Scholar | Optional | Not available |
| Batch files | Yes | N/A |
| Deduplication | Yes | N/A |

Both tools share the same resolution pipeline, so they produce consistent results.
