#!/usr/bin/env python3
"""CLI for the Zotero Paper Organizer.

Automatically organize Zotero papers into collections using AI classification.

Usage:
    bibtex-zotero-organize --tag organize --backend claude
    bibtex-zotero-organize --collection ABC123 --backend embedding
    bibtex-zotero-organize --dry-run --estimate-cost
    bibtex-zotero-organize --config organizer.yaml --taxonomy taxonomy.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from bibtex_updater.organizer.classifier import BACKENDS, ClassifierFactory
from bibtex_updater.organizer.config import ClassifierConfig, OrganizerConfig
from bibtex_updater.organizer.main import OrganizeResult, ZoteroPaperOrganizer


def load_config_file(path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Config dictionary
    """
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML required for config files. Install: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_summary(results: list[OrganizeResult], dry_run: bool) -> None:
    """Print summary of results.

    Args:
        results: List of OrganizeResult objects
        dry_run: Whether this was a dry run
    """
    organized = [r for r in results if r.action == "organized"]
    would_organize = [r for r in results if r.action == "dry_run"]
    skipped = [r for r in results if r.action == "skipped"]
    errors = [r for r in results if r.action == "error"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed:  {len(results)}")

    if dry_run:
        print(f"Would organize:   {len(would_organize)}")
    else:
        print(f"Organized:        {len(organized)}")

    print(f"Skipped:          {len(skipped)}")
    print(f"Errors:           {len(errors)}")

    # Collections created
    all_created = []
    for r in results:
        all_created.extend(r.collections_created)
    if all_created:
        print(f"Collections created: {len(set(all_created))}")

    # Detailed results
    if organized or would_organize:
        action_label = "Would Organize" if dry_run else "Organized"
        print(f"\n--- {action_label} ---")
        for r in would_organize if dry_run else organized:
            print(f"  [{r.item_key}] {r.title}")
            print(f"    Topics: {', '.join(r.topics)}")
            print(f"    Confidence: {r.confidence:.2f}" + (" (cached)" if r.cached else ""))

    if errors:
        print("\n--- Errors ---")
        for r in errors:
            print(f"  [{r.item_key}] {r.title}")
            print(f"    Error: {r.message}")


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Organize Zotero papers into collections using AI classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process items with a specific tag
  bibtex-zotero-organize --tag organize --backend claude

  # Process a specific collection
  bibtex-zotero-organize --collection ABC123 --backend openai

  # Use local embeddings (free, no API key needed)
  bibtex-zotero-organize --tag organize --backend embedding

  # Dry run with cost estimate
  bibtex-zotero-organize --tag organize --dry-run --estimate-cost

  # Use configuration files
  bibtex-zotero-organize --config organizer.yaml --taxonomy taxonomy.yaml

Environment Variables:
  ZOTERO_LIBRARY_ID   Your Zotero user/library ID
  ZOTERO_API_KEY      Zotero API key with write permissions
  ANTHROPIC_API_KEY   Claude API key (for --backend claude)
  OPENAI_API_KEY      OpenAI API key (for --backend openai)
""",
    )

    # Selection options
    selection = parser.add_argument_group("Item Selection")
    selection.add_argument("--tag", help="Process items with this tag")
    selection.add_argument("--collection", dest="collection_key", help="Process items in this collection")
    selection.add_argument("--item", dest="item_keys", action="append", help="Process specific item key(s)")
    selection.add_argument("--limit", type=int, default=100, help="Maximum items to process (default: 100)")
    selection.add_argument("--offset", type=int, default=0, help="Skip first N items")

    # Classification options
    classification = parser.add_argument_group("Classification")
    classification.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="claude",
        help="AI backend for classification (default: claude)",
    )
    classification.add_argument("--model", help="Model name for the backend")
    classification.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for classification (default: 0.7)",
    )
    classification.add_argument("--taxonomy", dest="taxonomy_file", help="Path to taxonomy YAML file")

    # Collection options
    collections = parser.add_argument_group("Collections")
    collections.add_argument(
        "--no-create",
        dest="create_collections",
        action="store_false",
        help="Don't create new collections",
    )

    # Output options
    output = parser.add_argument_group("Output")
    output.add_argument("--dry-run", action="store_true", help="Preview changes without modifying Zotero")
    output.add_argument("--estimate-cost", action="store_true", help="Estimate API cost before processing")
    output.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Config options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", dest="config_file", help="Path to YAML config file")
    config_group.add_argument("--library-id", help="Zotero library ID (or set ZOTERO_LIBRARY_ID)")
    config_group.add_argument("--api-key", dest="zotero_api_key", help="Zotero API key (or set ZOTERO_API_KEY)")
    config_group.add_argument(
        "--library-type",
        default="user",
        choices=["user", "group"],
        help="Library type (default: user)",
    )
    config_group.add_argument("--classifier-api-key", help="API key for classifier backend")

    # Utility options
    utility = parser.add_argument_group("Utility")
    utility.add_argument("--check-backends", action="store_true", help="Check available backends and exit")

    args = parser.parse_args(argv)

    # Check backends utility
    if args.check_backends:
        print("Available backends:")
        for backend in BACKENDS:
            available, message = ClassifierFactory.check_backend(backend)
            status = "OK" if available else "UNAVAILABLE"
            print(f"  {backend}: [{status}] {message}")
        return 0

    # Load config file if provided
    file_config: dict[str, Any] = {}
    if args.config_file:
        file_config = load_config_file(args.config_file)

    # Build configuration (CLI args override file config)
    library_id = args.library_id or file_config.get("library_id") or os.environ.get("ZOTERO_LIBRARY_ID")
    zotero_api_key = args.zotero_api_key or file_config.get("api_key") or os.environ.get("ZOTERO_API_KEY")

    if not library_id or not zotero_api_key:
        print("Error: ZOTERO_LIBRARY_ID and ZOTERO_API_KEY required", file=sys.stderr)
        print("  Set environment variables or use --library-id and --api-key", file=sys.stderr)
        print("  Get credentials at: https://www.zotero.org/settings/keys", file=sys.stderr)
        return 1

    # Build classifier config
    classifier_config = ClassifierConfig(
        backend=args.backend,
        model=args.model or file_config.get("classifier", {}).get("model"),
        api_key=args.classifier_api_key,
        confidence_threshold=args.confidence_threshold,
    )

    # Build organizer config
    config = OrganizerConfig(
        classifier=classifier_config,
        library_id=library_id,
        api_key=zotero_api_key,
        library_type=args.library_type or file_config.get("library_type", "user"),
        taxonomy_file=args.taxonomy_file or file_config.get("taxonomy_file"),
        create_collections=args.create_collections,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Check if selection criteria provided
    if not any([args.tag, args.collection_key, args.item_keys]):
        print("Error: Must specify --tag, --collection, or --item", file=sys.stderr)
        return 1

    try:
        organizer = ZoteroPaperOrganizer(config)
    except Exception as e:
        print(f"Error initializing organizer: {e}", file=sys.stderr)
        return 1

    # Estimate cost if requested
    if args.estimate_cost:
        # Fetch items to get count
        items = organizer.fetch_items_to_process(
            tag=args.tag,
            collection_key=args.collection_key,
            item_keys=args.item_keys,
            limit=args.limit,
            offset=args.offset,
        )
        num_items = len(items)
        cost = organizer.estimate_cost(num_items)
        print("\nCost Estimate:")
        print(f"  Items to process: {num_items}")
        print(f"  Backend: {args.backend}")
        print(f"  Estimated cost: ${cost:.4f}")

        if args.dry_run:
            # Continue with dry run
            pass
        else:
            # Ask for confirmation
            print("\nProceed with processing? [y/N] ", end="")
            response = input().strip().lower()
            if response != "y":
                print("Aborted.")
                return 0

    # Run the organizer
    results = organizer.run(
        tag=args.tag,
        collection_key=args.collection_key,
        item_keys=args.item_keys,
        limit=args.limit,
        offset=args.offset,
    )

    print_summary(results, args.dry_run)

    # Return error code if there were errors
    errors = [r for r in results if r.action == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
