#!/usr/bin/env python3
"""CLI entry point for bibtex-zotero command.

Updates Zotero library preprints to published versions.
"""

import sys


def main() -> None:
    """Entry point for bibtex-zotero command."""
    from bibtex_updater.zotero import main as zotero_main

    sys.exit(zotero_main())


if __name__ == "__main__":
    main()
