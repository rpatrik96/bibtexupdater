#!/usr/bin/env python3
"""CLI entry point for bibtex-update command.

Updates preprint BibTeX entries to their published versions.
"""

import sys


def main() -> None:
    """Entry point for bibtex-update command."""
    from bibtex_updater.updater import main as updater_main

    sys.exit(updater_main())


if __name__ == "__main__":
    main()
