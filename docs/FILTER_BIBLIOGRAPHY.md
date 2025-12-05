# Bibliography Filtering

Filter your bibliography to include only cited references from your LaTeX documents.

## Overview

Two versions are available:

| Script | Dependencies | Use Case |
|--------|--------------|----------|
| `filter_bibliography.py` | `bibtexparser` | Local development |
| `filter_bibliography_minimal.py` | **None** (stdlib only) | **Overleaf**, CI/CD, restricted environments |

Both scripts scan LaTeX source files for citation commands and create a filtered bibliography file containing only the entries that are actually cited in your document. This helps:

- Reduce .bib file size for submissions
- Keep bibliographies clean and focused
- Identify missing citations

## Installation

### Full Version (filter_bibliography.py)

```bash
pip install bibtexparser
```

### Minimal Version (filter_bibliography_minimal.py)

No installation needed! Uses only Python standard library. Just download the script:

```bash
curl -O https://raw.githubusercontent.com/rpatrik96/bibtexupdater/main/filter_bibliography_minimal.py
```

## CLI Usage

### Basic Usage

```bash
# Filter to only cited entries
python filter_bibliography.py paper.tex -b references.bib -o filtered.bib

# Process multiple .tex files
python filter_bibliography.py *.tex -b references.bib -o filtered.bib

# Recursively scan a directory
python filter_bibliography.py ./chapters/ -b references.bib -o filtered.bib --recursive

# Multiple bib files (merged; errors on duplicates)
python filter_bibliography.py paper.tex -b refs1.bib refs2.bib -o filtered.bib
```

### Command Reference

```
usage: filter_bibliography.py [-h] -b BIB [BIB ...] [-o OUTPUT] [-r]
                              [--case-sensitive] [-n] [-v] [--list-citations]
                              [--warn-missing | --no-warn-missing]
                              tex_sources [tex_sources ...]
```

| Option | Description |
|--------|-------------|
| `tex_sources` | One or more .tex files or directories to scan |
| `-b, --bib` | Input bibliography file(s). Multiple files are merged; duplicates cause an error. |
| `-o, --output` | Output file (default: `<first_input>_filtered.bib`) |
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
python filter_bibliography.py paper.tex -b refs.bib --dry-run

# List all citations found in a document
python filter_bibliography.py paper.tex -b refs.bib --list-citations

# Process entire project with subdirectories
python filter_bibliography.py . -b references.bib -o filtered.bib -r

# Suppress warnings about citations not in .bib
python filter_bibliography.py paper.tex -b refs.bib -o out.bib --no-warn-missing

# Merge multiple bib files
python filter_bibliography.py paper.tex -b main.bib extra.bib -o filtered.bib
```

## Supported Citation Commands

Both versions detect citations from major LaTeX bibliography packages:

### Common Commands (both versions)
- `\cite`, `\cite*`, `\citep`, `\citep*`, `\citet`, `\citet*`
- `\nocite`
- `\parencite`, `\Parencite`
- `\textcite`, `\Textcite`
- `\autocite`, `\Autocite`

### Additional Commands (full version only)
- `\citealt`, `\citealp`
- `\citeauthor`, `\citeyear`, `\citeyearpar`
- `\smartcite`, `\Smartcite`, `\supercite`
- `\fullcite`, `\footcite`, `\footcitetext`

### Optional Arguments
Both versions correctly handle optional arguments:
- `\cite[p. 5]{key}`
- `\citep[see][p. 10]{key1,key2}`

## Overleaf Integration

### Option 1: END Block (Recommended for pdfLaTeX)

The minimal version `filter_bibliography_minimal.py` works directly on Overleaf with no dependencies!

This approach runs the filter **after** compilation completes. The filtered `.bib` file appears in your Overleaf file list for download. Keep your original `\bibliography{references}` unchanged.

**Setup:**

1. Download `filter_bibliography_minimal.py` from the repository
2. Upload it to your Overleaf project root
3. Create a file named `.latexmkrc` in your project root with:

**Single bib file:**
```perl
END {
  system("python3 filter_bibliography_minimal.py . -b references.bib -o refs_filtered.bib -r --no-warn-missing");
}
```

**Multiple bib files:**
```perl
END {
  system("python3 filter_bibliography_minimal.py . -b refs1.bib refs2.bib -o refs_filtered.bib -r --no-warn-missing");
}
```

4. Recompile - you'll see `refs_filtered.bib` appear in your Overleaf file list!

### Option 2: Pre-bibtex Hook (Alternative)

If you want to **use** the filtered bibliography during compilation (instead of just generating it as an artifact), use the bibtex/biber hook approach:

**For natbib** (`\usepackage{natbib}`):
```perl
$bibtex = "python3 filter_bibliography_minimal.py %R.tex -b references.bib -o refs_filtered.bib --no-warn-missing && bibtex %O %B";
```

**For biblatex** (`\usepackage{biblatex}`):
```perl
$biber = "python3 filter_bibliography_minimal.py %R.tex -b references.bib -o refs_filtered.bib --no-warn-missing && biber %O %S";
```

**Important:** Update your .tex file to use the filtered bibliography:
- natbib: `\bibliography{refs_filtered}` (without .bib extension)
- biblatex: `\addbibresource{refs_filtered.bib}`

### Option 3: GitHub Actions Workflow

Alternatively, use GitHub Actions for both preprint updates AND filtering. This is useful if you also want to auto-update preprints to published versions.

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
             python filter_bibliography.py . -b references.bib -o references.bib -r --no-warn-missing

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

## Local .latexmkrc Integration

For local compilation (not Overleaf), you can integrate `filter_bibliography.py` with latexmk.

### Basic Setup

1. Copy `.latexmkrc.example` to your project as `.latexmkrc`
2. Adjust paths to match your project
3. Run `latexmk -pdf main.tex`

### Example .latexmkrc

**Recommended: END block** (generates filtered file as artifact):
```perl
END {
  system('python3 /path/to/filter_bibliography.py . -b references.bib -o refs_filtered.bib -r --no-warn-missing');
}
```

**Alternative: bibtex/biber hook** (uses filtered file during compilation):

For natbib:
```perl
$bibtex = 'python3 /path/to/filter_bibliography.py %R.tex -b references.bib -o refs_filtered.bib --no-warn-missing && bibtex %O %B';
```

For biblatex:
```perl
$biber = 'python3 /path/to/filter_bibliography.py %R.tex -b references.bib -o refs_filtered.bib --no-warn-missing && biber %O %S';
```

**Multiple bib files:**
```perl
END {
  system('python3 /path/to/filter_bibliography.py . -b refs1.bib refs2.bib -o refs_filtered.bib -r --no-warn-missing');
}
```

**Note**: When using the bibtex/biber hook, update your .tex file to use `refs_filtered` instead of `references`.

See `.latexmkrc.example` in the repository for a complete example with comments.

### Notes

- The END block runs after compilation; the hook runs before bibtex/biber
- Use `--no-warn-missing` to avoid noise during compilation
- Multiple bib files are merged; duplicate keys cause an error
