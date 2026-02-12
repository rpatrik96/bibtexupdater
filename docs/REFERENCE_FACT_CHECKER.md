# Reference Fact-Checker

Validate that bibliographic entries in BibTeX files exist in external databases and have correct metadata. Useful for detecting hallucinated or incorrectly cited references.

## Installation

```bash
pip install bibtexparser httpx rapidfuzz
```

## Quick Start

```bash
# Basic validation
python reference_fact_checker.py references.bib

# Generate detailed JSON report
python reference_fact_checker.py references.bib --report report.json

# CI/CD mode: fail if problematic entries found
python reference_fact_checker.py references.bib --strict
```

## How It Works

The fact-checker runs a multi-stage verification pipeline:

### Pre-API Validation (zero cost)
1. **Year validation** — Flags future dates (`year > current_year`), implausible years (`< 1800`), and non-numeric years before making any API calls
2. **DOI resolution** — HEAD request to `doi.org` catches fabricated DOIs (HTTP 404)

### API Verification
3. **Crossref** — Primary source for journal articles with DOIs
4. **DBLP** — Computer science publications
5. **Semantic Scholar** — Broad coverage across disciplines

### Post-Match Analysis
6. **Venue verification** — Alias-aware matching for 17 ML/AI venues (NeurIPS/NIPS, ICML, ICLR, CVPR, etc.); known-different venues always flagged
7. **Preprint detection** — Queries S2 to detect entries claiming a venue when only an arXiv preprint exists

For each entry, it:
1. Runs pre-API checks (year, DOI) to catch obvious issues cheaply
2. Searches all sources using title + first author
3. Scores candidates using fuzzy title matching (70%) + author Jaccard similarity (30%)
4. Compares fields against the best match with alias-aware venue matching
5. Checks preprint-vs-published status via Semantic Scholar
6. Assigns a status based on match quality

## Status Codes

| Status | Description |
|--------|-------------|
| `verified` | Entry matches an external record within thresholds |
| `not_found` | No matching record found in any database |
| `hallucinated` | Very low match score (<50%), likely fabricated |
| `title_mismatch` | Title differs significantly from best match |
| `author_mismatch` | Author list differs from best match |
| `year_mismatch` | Publication year differs beyond tolerance |
| `venue_mismatch` | Journal/venue differs from best match |
| `partial_match` | Multiple fields differ from best match |
| `api_error` | Errors occurred during API queries |

## Command Line Options

```
usage: reference_fact_checker.py [-h] [--report FILE] [--jsonl FILE]
                                  [--strict] [--verbose]
                                  [--title-threshold FLOAT]
                                  [--author-threshold FLOAT]
                                  [--year-tolerance INT]
                                  [--venue-threshold FLOAT]
                                  [--cache-file FILE]
                                  [--rate-limit INT]
                                  bibfiles [bibfiles ...]

positional arguments:
  bibfiles              BibTeX files to check

options:
  --report, -r FILE     Write JSON report to FILE
  --jsonl FILE          Write JSONL report to FILE
  --strict              Exit with code 4 if NOT_FOUND or HALLUCINATED found
  --verbose, -v         Enable debug logging

thresholds:
  --title-threshold     Title similarity threshold (default: 0.90)
  --author-threshold    Author similarity threshold (default: 0.80)
  --year-tolerance      Year tolerance in years (default: 1)
  --venue-threshold     Venue similarity threshold (default: 0.70)

API options:
  --cache-file          Cache file path (default: .cache.fact_checker.json)
  --rate-limit          Requests per minute limit (default: 45)
```

## Output Formats

### JSON Report (`--report`)

Full structured report with all details:

```json
{
  "summary": {
    "total": 10,
    "status_counts": {
      "verified": 8,
      "not_found": 1,
      "hallucinated": 1
    },
    "verified_rate": 0.8,
    "problematic_count": 2,
    "timestamp": "2024-01-15T10:30:00"
  },
  "entries": [
    {
      "key": "smith2020",
      "type": "article",
      "status": "verified",
      "confidence": 0.95,
      "field_comparisons": {
        "title": {
          "entry_value": "Deep Learning for NLP",
          "api_value": "Deep Learning for NLP",
          "similarity_score": 1.0,
          "matches": true
        }
      },
      "best_match": {
        "doi": "10.1234/jml.2020.001",
        "title": "Deep Learning for NLP",
        "journal": "Journal of ML",
        "year": 2020
      }
    }
  ]
}
```

### JSONL Report (`--jsonl`)

One JSON object per line, useful for streaming/processing:

```jsonl
{"key": "smith2020", "status": "verified", "confidence": 0.95, "mismatched_fields": [], "api_sources": ["crossref"]}
{"key": "fake2099", "status": "hallucinated", "confidence": 0.3, "mismatched_fields": ["title", "author"], "api_sources": []}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (or non-strict mode) |
| 1 | Input error (file not found, parse error) |
| 4 | Strict mode: NOT_FOUND or HALLUCINATED entries found |

## Thresholds

The fact-checker uses configurable similarity thresholds:

| Field | Default | Description |
|-------|---------|-------------|
| Title | 0.90 | Token-sort ratio (fuzzy string matching) |
| Author | 0.80 | Jaccard similarity on last names |
| Year | ±1 | Absolute difference tolerance |
| Venue | 0.70 | Token-sort ratio on journal/booktitle |
| Hallucination | 0.50 | Below this combined score = likely fake |

## CI/CD Integration

### GitHub Actions

```yaml
- name: Validate references
  run: |
    pip install bibtexparser httpx rapidfuzz
    python reference_fact_checker.py references.bib --strict --report report.json

- name: Upload report
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: fact-check-report
    path: report.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-references
        name: Validate BibTeX references
        entry: python reference_fact_checker.py --strict
        language: python
        files: \.bib$
        additional_dependencies: [bibtexparser, httpx, rapidfuzz]
```

## Caching

API responses are cached to `.cache.fact_checker.json` by default. This:
- Speeds up repeated runs
- Reduces API rate limit issues
- Persists across sessions

To clear the cache, delete the file or use a different `--cache-file`.

## Rate Limiting

The tool respects API rate limits (default: 45 requests/minute). For large bibliographies:
- The cache helps avoid redundant requests
- Progress is logged for each entry
- Consider running overnight for very large files

## Comparison with bibtex_updater.py

| Feature | bibtex_updater.py | reference_fact_checker.py |
|---------|-------------------|---------------------------|
| Purpose | Transform preprints to published | Validate entries exist |
| Modifies files | Yes | No (read-only) |
| Target entries | Preprints only | All entries |
| Output | Updated .bib file | Validation report |
| Use case | Bibliography cleanup | Quality assurance |
