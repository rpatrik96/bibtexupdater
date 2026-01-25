#!/usr/bin/env python3
"""CLI entry point for bibtex-check command.

Validates bibliography references against external APIs.
"""

import sys


def main() -> None:
    """Entry point for bibtex-check command."""
    from bibtex_updater.fact_checker import main as fact_checker_main

    sys.exit(fact_checker_main())


if __name__ == "__main__":
    main()
