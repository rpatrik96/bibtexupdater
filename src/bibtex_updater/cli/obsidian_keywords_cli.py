"""CLI for AI-powered Obsidian keyword generation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate AI-powered keywords for Obsidian paper notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single note
  bibtex-obsidian-keywords ~/Notes/Papers/@smith2024.md

  # Process all papers in a folder (dry run)
  bibtex-obsidian-keywords ~/Notes/Papers/ --dry-run

  # Process with specific backend
  bibtex-obsidian-keywords ~/Notes/Papers/ --backend openai

  # Output as JSON for Templater integration
  bibtex-obsidian-keywords ~/Notes/Papers/@smith2024.md --json
""",
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a note file or folder of notes",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "openai", "embedding"],
        default="claude",
        help="AI backend for classification (default: claude)",
    )
    parser.add_argument(
        "--model",
        help="Model name override",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=5,
        help="Maximum keywords to generate per note (default: 5)",
    )
    parser.add_argument(
        "--pattern",
        default="@*.md",
        help="Glob pattern for note files (default: @*.md)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum notes to enrich (skipped notes don't count towards limit)",
    )
    parser.add_argument(
        "--min-keywords",
        type=int,
        default=0,
        help="Skip notes that already have this many keywords (default: 0, process all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        help="File with existing topic names (one per line)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    if args.json_output:
        level = logging.WARNING  # Suppress info logs for JSON output
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Validate path first
    path: Path = args.path
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        return 1

    # Load existing topics if provided
    existing_topics = []
    if args.topics_file and args.topics_file.exists():
        existing_topics = [
            line.strip()
            for line in args.topics_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    # Initialize generator
    from bibtex_updater.obsidian_keywords import ObsidianKeywordGenerator

    generator = ObsidianKeywordGenerator(
        backend=args.backend,
        model=args.model,
        existing_topics=existing_topics,
        max_keywords=args.max_keywords,
    )

    # Process
    if path.is_file():
        results = [generator.process_note(path, dry_run=args.dry_run)]
    else:
        # path.is_dir() - we already validated existence above
        results = generator.process_folder(
            path,
            pattern=args.pattern,
            dry_run=args.dry_run,
            limit=args.limit,
            min_existing_keywords=args.min_keywords,
        )

    # Output
    if args.json_output:
        output = [
            {
                "file": str(r.file_path),
                "action": r.action,
                "existing_keywords": r.existing_keywords,
                "generated_keywords": r.generated_keywords,
                "message": r.message,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        # Summary
        enriched = sum(1 for r in results if r.action == "enriched")
        dry_run = sum(1 for r in results if r.action == "dry_run")
        skipped = sum(1 for r in results if r.action == "skipped")
        errors = sum(1 for r in results if r.action == "error")

        print(f"\nProcessed {len(results)} notes:")
        if enriched:
            print(f"  Enriched: {enriched}")
        if dry_run:
            print(f"  Would enrich: {dry_run}")
        if skipped:
            print(f"  Skipped: {skipped}")
        if errors:
            print(f"  Errors: {errors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
