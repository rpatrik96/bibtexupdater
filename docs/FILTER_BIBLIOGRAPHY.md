# Bibliography Filtering

Filter your bibliography to include only cited references from your LaTeX documents.

## Overview

`filter_bibliography.py` scans LaTeX source files for citation commands and creates a filtered bibliography file containing only the entries that are actually cited in your document. This helps:

- Reduce .bib file size for submissions
- Keep bibliographies clean and focused
- Identify missing citations

**No dependencies required** - uses only Python standard library, making it ideal for Overleaf and restricted environments.

## Installation

No installation needed! Just download the script:

```bash
curl -O https://raw.githubusercontent.com/rpatrik96/bibtexupdater/main/filter_bibliography.py
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
```

## Supported Citation Commands

The script detects citations from major LaTeX bibliography packages:

- `\cite`, `\cite*`, `\citep`, `\citep*`, `\citet`, `\citet*`
- `\nocite`
- `\parencite`, `\Parencite`
- `\textcite`, `\Textcite`
- `\autocite`, `\Autocite`

Optional arguments are correctly handled:
- `\cite[p. 5]{key}`
- `\citep[see][p. 10]{key1,key2}`

## Overleaf Integration

### END Block (Recommended)

This approach runs the filter **after** compilation completes. The filtered `.bib` file appears in your Overleaf file list for download.

**Setup:**

1. Upload `filter_bibliography.py` to your Overleaf project root
2. Create a file named `.latexmkrc` in your project root with:

**Single bib file:**
```perl
END {
  system("python3 filter_bibliography.py . -b references.bib -o refs_filtered.bib -r --no-warn-missing");
}
```

**Multiple bib files:**
```perl
END {
  system("python3 filter_bibliography.py . -b refs1.bib refs2.bib -o refs_filtered.bib -r --no-warn-missing");
}
```

3. Recompile - you'll see `refs_filtered.bib` appear in your Overleaf file list!

### Pre-bibtex Hook (Alternative)

If you want to **use** the filtered bibliography during compilation:

**For natbib** (`\usepackage{natbib}`):
```perl
$bibtex = "python3 filter_bibliography.py %R.tex -b references.bib -o refs_filtered.bib --no-warn-missing && bibtex %O %B";
```

**For biblatex** (`\usepackage{biblatex}`):
```perl
$biber = "python3 filter_bibliography.py %R.tex -b references.bib -o refs_filtered.bib --no-warn-missing && biber %O %S";
```

**Important:** Update your .tex file to use the filtered bibliography:
- natbib: `\bibliography{refs_filtered}` (without .bib extension)
- biblatex: `\addbibresource{refs_filtered.bib}`

## Local .latexmkrc Integration

For local compilation, integrate with latexmk:

1. Copy [examples/latexmkrc](../examples/latexmkrc) to your project as `.latexmkrc`
2. Adjust paths to match your project
3. Run `latexmk -pdf main.tex`

See [examples/](../examples/) for complete configuration examples.

**Notes:**
- The END block runs after compilation; the hook runs before bibtex/biber
- Use `--no-warn-missing` to avoid noise during compilation
- Multiple bib files are merged; duplicate keys cause an error
