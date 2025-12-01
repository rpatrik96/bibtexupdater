# Bibliography Filtering

Filter your bibliography to include only cited references from your LaTeX documents.

## Overview

`filter_bibliography.py` scans LaTeX source files for citation commands and creates a filtered bibliography file containing only the entries that are actually cited in your document. This helps:

- Reduce .bib file size for submissions
- Keep bibliographies clean and focused
- Identify missing citations

## Installation

The script requires only `bibtexparser`:

```bash
pip install bibtexparser
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## CLI Usage

### Basic Usage

```bash
# Filter to only cited entries
python filter_bibliography.py paper.tex references.bib -o filtered.bib

# Process multiple .tex files
python filter_bibliography.py *.tex references.bib -o filtered.bib

# Recursively scan a directory
python filter_bibliography.py ./chapters/ references.bib -o filtered.bib --recursive
```

### Command Reference

```
usage: filter_bibliography.py [-h] [-o OUTPUT] [-r] [--case-sensitive]
                              [-n] [-v] [--list-citations]
                              [--warn-missing | --no-warn-missing]
                              tex_sources [tex_sources ...] bib_file
```

| Option | Description |
|--------|-------------|
| `tex_sources` | One or more .tex files or directories to scan |
| `bib_file` | Input bibliography file |
| `-o, --output` | Output file (default: `<input>_filtered.bib`) |
| `-r, --recursive` | Recursively search directories for .tex files |
| `--case-sensitive` | Use case-sensitive key matching (default: case-insensitive) |
| `-n, --dry-run` | Preview changes without writing |
| `-v, --verbose` | Enable debug logging |
| `--list-citations` | List all found citations and exit |
| `--warn-missing` | Warn about missing citations (default: True) |
| `--no-warn-missing` | Suppress missing citation warnings |

### Examples

```bash
# Preview what would be filtered (dry run)
python filter_bibliography.py paper.tex refs.bib --dry-run

# List all citations found in a document
python filter_bibliography.py paper.tex refs.bib --list-citations

# Process entire project with subdirectories
python filter_bibliography.py . references.bib -o filtered.bib -r

# Suppress warnings about citations not in .bib
python filter_bibliography.py paper.tex refs.bib -o out.bib --no-warn-missing
```

## Supported Citation Commands

The script detects citations from all major LaTeX bibliography packages:

### Standard LaTeX / natbib
- `\cite`, `\cite*`
- `\citep`, `\citep*`, `\citet`, `\citet*`
- `\citealt`, `\citealp`
- `\citeauthor`, `\citeyear`, `\citeyearpar`
- `\nocite`

### BibLaTeX
- `\parencite`, `\Parencite`
- `\textcite`, `\Textcite`
- `\autocite`, `\Autocite`
- `\smartcite`, `\Smartcite`
- `\supercite`
- `\fullcite`
- `\footcite`, `\footcitetext`

### Optional Arguments
The script correctly handles optional arguments:
- `\cite[p. 5]{key}`
- `\citep[see][p. 10]{key1,key2}`

## Overleaf Integration

### Why latexmkrc Doesn't Work on Overleaf

You **cannot** run `filter_bibliography.py` directly on Overleaf via latexmkrc because:

1. **Missing dependencies**: Overleaf doesn't have `bibtexparser` installed
2. **No pip access**: You cannot install Python packages on Overleaf
3. **Limited shell access**: While Overleaf has shell-escape enabled, it only works for pre-installed tools

### Recommended: GitHub Actions Workflow

The best way to use bibliography filtering with Overleaf is through the GitHub Actions workflow, which runs in GitHub's environment with full Python support.

#### Setup

1. **Enable GitHub sync in Overleaf**
   - Open your project in Overleaf
   - Go to Menu → Sync → GitHub
   - Link to a new or existing repository

2. **Add the workflow file** to your repository

   Create `.github/workflows/update-and-filter-bibliography.yml`:

   ```yaml
   name: Update and Filter Bibliography
   on:
     push:
       paths:
         - '**.bib'
         - '**.tex'

   jobs:
     update:
       runs-on: ubuntu-latest
       if: github.actor != 'github-actions[bot]'

       steps:
         - uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.11'

         - name: Install dependencies
           run: pip install bibtexparser

         - name: Download filter script
           run: |
             curl -O https://raw.githubusercontent.com/rpatrik96/bibtexupdater/main/filter_bibliography.py

         - name: Filter bibliography
           run: |
             python filter_bibliography.py . references.bib -o references.bib -r --no-warn-missing

         - name: Commit changes
           run: |
             git config user.name "github-actions[bot]"
             git config user.email "github-actions[bot]@users.noreply.github.com"
             git add -A
             git diff --staged --quiet || git commit -m "Filter bibliography to cited entries"
             git push
   ```

3. **How it works**
   ```
   Edit in Overleaf → Sync to GitHub → Action runs →
   Filters .bib → Commits → Overleaf pulls updated .bib
   ```

#### Using the Reusable Workflow

Alternatively, use the project's reusable workflow for both preprint updates AND filtering:

```yaml
name: Update and Filter Bibliography
on:
  push:
    paths: ['**.bib', '**.tex']

jobs:
  update:
    uses: rpatrik96/bibtexupdater/.github/workflows/reusable-bib-update.yml@main
    with:
      bib_files: 'references.bib'
      dedupe: true
```

## Local latexmkrc Integration

For local compilation (not Overleaf), you can integrate `filter_bibliography.py` with latexmk.

### Basic Setup

1. Copy `latexmkrc.example` to your project as `latexmkrc` (or `.latexmkrc`)
2. Adjust paths to match your project
3. Run `latexmk -pdf main.tex`

### Example latexmkrc

```perl
# Filter bibliography before bibtex runs
$bibtex = 'python /path/to/filter_bibliography.py %R.tex references.bib -o references.bib --no-warn-missing && bibtex %O %B';
```

See `latexmkrc.example` in the repository for a complete example with comments.

### Notes

- The filter runs every time bibtex is invoked
- Use `--no-warn-missing` to avoid noise during compilation
- Works with both bibtex and biber backends
