#!/usr/bin/env python3
"""CLI entry point for bibtex-filter command.

Filters bibliography to only cited references.
"""

import sys


def main() -> None:
    """Entry point for bibtex-filter command."""
    from bibtex_updater.filter import main as filter_main

    sys.exit(filter_main())


if __name__ == "__main__":
    main()
