# Examples

This directory contains example configuration files for integrating bibtexupdater into your workflow.

## Files

### `latexmkrc`

Example latexmk configuration for running `filter_bibliography.py` during LaTeX compilation.

**Usage:**
1. Copy to your project as `.latexmkrc`
2. Adjust the `$BIB_FILES` variable to match your bibliography file(s)
3. Run `latexmk -pdf main.tex`

Works on Overleaf (with pdfLaTeX compiler) - just upload both `filter_bibliography.py` and `.latexmkrc` to your project.

### `workflows/`

Example GitHub Actions workflow files for automating bibliography management.

| Workflow | Description |
|----------|-------------|
| `update-bibliography.yml` | Update preprints to published versions |
| `filter-bibliography.yml` | Filter to only cited entries |
| `update-and-filter-bibliography.yml` | Both update AND filter |

**Usage:**
1. Copy the desired workflow to your repository's `.github/workflows/` directory
2. Adjust file paths in the workflow to match your project
3. Commit and push - the workflow runs automatically on .bib/.tex changes

## Integration with Overleaf

For Overleaf projects synced to GitHub:

1. Enable GitHub sync in Overleaf (Menu → Sync → GitHub)
2. Add a workflow file to `.github/workflows/`
3. Changes synced from Overleaf trigger the workflow
4. Updated .bib is committed back and synced to Overleaf

See the [documentation](../docs/) for detailed setup instructions.
