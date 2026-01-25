#!/usr/bin/env python3
"""
filter_bibliography.py - Standalone bibliography filtering for LaTeX projects.

Uses only Python standard library (no pip dependencies). Designed for use on
Overleaf and other restricted environments.

Usage:
    python filter_bibliography.py paper.tex -b references.bib -o filtered.bib
    python filter_bibliography.py *.tex -b references.bib -o filtered.bib
    python filter_bibliography.py ./chapters/ -b references.bib -o filtered.bib -r

    # Multiple bib files (merged, errors on duplicates):
    python filter_bibliography.py paper.tex -b refs1.bib refs2.bib -o filtered.bib

For latexmkrc END block integration (works on Overleaf with pdfLaTeX):
    END {
      system("python3 filter_bibliography.py . -b refs.bib -o refs_filtered.bib -r --no-warn-missing");
    }

Requirements: Python 3.6+ (standard library only - no pip dependencies)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


# =============================================================================
# Citation Extraction
# =============================================================================

# Simplified citation pattern - common commands only
# Supports: \cite, \citep, \citet, \nocite, \parencite, \textcite, \autocite
# With optional star (*) and optional arguments ([...])
CITE_KEYS_PATTERN = re.compile(
    r"\\("
    r"(?:no)?cite[tp]?"  # \cite, \citep, \citet, \nocite
    r"|parencite|Parencite"  # biblatex \parencite
    r"|textcite|Textcite"  # biblatex \textcite
    r"|autocite|Autocite"  # biblatex \autocite
    r")\*?"  # Optional star
    r"(?:\s*\[[^\]]*\])*"  # Optional arguments in square brackets
    r"\s*\{([^}]+)\}"  # Citation keys in braces
)


def strip_comments(text: str) -> str:
    """
    Remove LaTeX comments from text, handling escaped percent signs.

    Correctly handles:
    - % This is a comment -> removed
    - 50\\% improvement -> kept (escaped %)
    """
    lines = []
    for line in text.split("\n"):
        pos = 0
        while pos < len(line):
            idx = line.find("%", pos)
            if idx == -1:
                lines.append(line)
                break
            elif idx == 0 or line[idx - 1] != "\\":
                # Found unescaped %, remove from here to end
                lines.append(line[:idx])
                break
            else:
                # Found escaped \%, continue searching
                pos = idx + 1
        else:
            lines.append(line)
    return "\n".join(lines)


def extract_citations_from_text(text: str) -> set[str]:
    """Extract citation keys from LaTeX text."""
    citations: set[str] = set()
    for match in CITE_KEYS_PATTERN.finditer(text):
        keys_str = match.group(2)
        for key in keys_str.split(","):
            key = key.strip()
            if key:
                citations.add(key)
    return citations


def extract_citations_from_file(filepath: str) -> set[str]:
    """Extract citations from a .tex file, ignoring comments."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        LOG.error(f"Could not read file {filepath}: {e}")
        return set()

    cleaned_text = strip_comments(text)
    return extract_citations_from_text(cleaned_text)


# =============================================================================
# BibTeX Parser (brace-counting algorithm)
# =============================================================================


class BibEntry:
    """Represents a single BibTeX entry with its raw content."""

    def __init__(self, entry_type: str, key: str, raw_content: str):
        self.entry_type = entry_type  # e.g., "article", "inproceedings"
        self.key = key  # Citation key
        self.raw_content = raw_content  # Complete raw entry text

    def __str__(self) -> str:
        return self.raw_content

    def __repr__(self) -> str:
        return f"BibEntry({self.entry_type!r}, {self.key!r})"


# Pattern to find entry starts: @type{
ENTRY_START_PATTERN = re.compile(r"@(\w+)\s*\{", re.IGNORECASE)


def find_matching_brace(text: str, start_pos: int) -> int:
    """
    Find the position of the closing brace that matches the opening brace at start_pos.

    Uses brace-counting algorithm to handle nested braces like {A {GPU} Implementation}.
    Correctly ignores braces inside quoted strings.

    Args:
        text: Full text to search
        start_pos: Position of opening '{'

    Returns:
        Position of matching '}', or -1 if unmatched
    """
    if start_pos >= len(text) or text[start_pos] != "{":
        return -1

    level = 0
    pos = start_pos
    in_quotes = False

    while pos < len(text):
        char = text[pos]

        # Handle quoted strings (braces inside quotes don't count)
        if char == '"' and (pos == 0 or text[pos - 1] != "\\"):
            in_quotes = not in_quotes
        elif not in_quotes:
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    return pos

        pos += 1

    return -1  # Unmatched brace


def extract_entry_key(entry_body: str) -> str | None:
    """
    Extract the citation key from an entry body.

    Entry body starts after '@type{', so the key is everything
    up to the first comma.
    """
    # Skip leading whitespace
    body = entry_body.lstrip()

    # Find the comma that ends the key
    comma_pos = body.find(",")
    if comma_pos == -1:
        return None

    key = body[:comma_pos].strip()
    return key if key else None


def parse_bib_file(filepath: str) -> list[BibEntry]:
    """
    Parse a .bib file and return list of BibEntry objects.

    Uses brace-counting to correctly handle nested braces in field values.
    Skips @string, @preamble, and @comment entries.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        LOG.error(f"Could not read bib file {filepath}: {e}")
        return []

    return parse_bib_string(content)


def parse_bib_string(content: str) -> list[BibEntry]:
    """Parse BibTeX content from a string."""
    entries: list[BibEntry] = []

    for match in ENTRY_START_PATTERN.finditer(content):
        entry_type = match.group(1).lower()

        # Skip special BibTeX commands
        if entry_type in ("comment", "preamble", "string"):
            continue

        # Find the opening brace position
        brace_start = match.end() - 1  # Position of '{'

        # Find matching close brace
        brace_end = find_matching_brace(content, brace_start)
        if brace_end == -1:
            LOG.warning(f"Unmatched brace in entry starting at position {match.start()}")
            continue

        # Extract the full entry text
        raw_content = content[match.start() : brace_end + 1]

        # Extract the key from inside the braces
        entry_body = content[brace_start + 1 : brace_end]
        key = extract_entry_key(entry_body)

        if key:
            entries.append(BibEntry(entry_type, key, raw_content))
        else:
            LOG.warning(f"Could not extract key from entry: {raw_content[:50]}...")

    return entries


def parse_bib_files(filepaths: list[str]) -> tuple[list[BibEntry], dict[str, list[str]]]:
    """
    Parse multiple .bib files and merge entries.

    Args:
        filepaths: List of paths to .bib files

    Returns:
        Tuple of (list of all BibEntry objects, dict of duplicate keys to their files)
    """
    all_entries: list[BibEntry] = []
    key_to_file: dict[str, str] = {}  # key (lowercase) -> first file it appeared in
    duplicates: dict[str, list[str]] = {}  # key (lowercase) -> list of files with this key

    for filepath in filepaths:
        entries = parse_bib_file(filepath)
        for entry in entries:
            key_lower = entry.key.lower()
            if key_lower in key_to_file:
                # Track duplicate but don't error yet
                if key_lower not in duplicates:
                    duplicates[key_lower] = [key_to_file[key_lower]]
                duplicates[key_lower].append(filepath)
                LOG.debug(
                    f"Duplicate key '{entry.key}' found in '{filepath}' (first seen in '{key_to_file[key_lower]}')"
                )
            else:
                key_to_file[key_lower] = filepath
            all_entries.append(entry)

    return all_entries, duplicates


# =============================================================================
# Bibliography Filtering
# =============================================================================


def filter_entries(
    entries: list[BibEntry],
    cited_keys: set[str],
    case_sensitive: bool = False,
) -> tuple[list[BibEntry], set[str], set[str]]:
    """
    Filter entries to only include those that are cited.

    Args:
        entries: List of all BibEntry objects
        cited_keys: Set of citation keys found in .tex files
        case_sensitive: Whether matching is case-sensitive

    Returns:
        Tuple of (filtered entries, found keys, missing keys)
    """
    # Build lookup map
    if case_sensitive:
        key_map = {entry.key: entry for entry in entries}
        lookup_keys = cited_keys
    else:
        key_map = {entry.key.lower(): entry for entry in entries}
        lookup_keys = {k.lower() for k in cited_keys}

    filtered: list[BibEntry] = []
    found_keys: set[str] = set()

    for key in lookup_keys:
        if key in key_map:
            filtered.append(key_map[key])
            found_keys.add(key)

    missing_keys = lookup_keys - found_keys

    return filtered, found_keys, missing_keys


def write_bib_file(entries: list[BibEntry], filepath: str) -> None:
    """
    Write filtered entries to a .bib file atomically.

    Uses atomic write pattern: write to temp file, then rename.
    """
    # Create temp file in same directory for atomic rename
    dir_path = os.path.dirname(filepath) or "."

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".bib",
        prefix=".tmp_bib_",
        dir=dir_path,
        delete=False,
    ) as tmp:
        tmp_path = tmp.name

        # Write each entry with blank line separator
        for i, entry in enumerate(entries):
            if i > 0:
                tmp.write("\n")
            tmp.write(str(entry))
            tmp.write("\n")

        tmp.flush()
        os.fsync(tmp.fileno())

    # Atomic rename
    os.replace(tmp_path, filepath)


# =============================================================================
# File Discovery
# =============================================================================


def find_tex_files(path: str, recursive: bool = True) -> list[str]:
    """Find all .tex files in a path."""
    p = Path(path)

    if p.is_file():
        return [str(p)] if p.suffix.lower() == ".tex" else []

    pattern = "**/*.tex" if recursive else "*.tex"
    return [str(f) for f in p.glob(pattern)]


def extract_citations_from_project(
    paths: list[str],
    recursive: bool = True,
) -> tuple[set[str], list[str]]:
    """Extract all citations from multiple .tex files or directories."""
    all_citations: set[str] = set()
    all_files: list[str] = []

    for path in paths:
        if os.path.isfile(path):
            tex_files = [path]
        else:
            tex_files = find_tex_files(path, recursive=recursive)

        for tex_file in tex_files:
            LOG.debug(f"Processing: {tex_file}")
            citations = extract_citations_from_file(tex_file)
            all_citations.update(citations)
            all_files.append(tex_file)

    return all_citations, all_files


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter a bibliography file to include only cited references. "
        "Standalone version using only Python standard library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.tex -b references.bib -o filtered.bib
  %(prog)s *.tex -b references.bib -o filtered.bib
  %(prog)s ./chapters/ -b references.bib -o filtered.bib --recursive
  %(prog)s paper.tex -b refs.bib --dry-run

  # Multiple bib files:
  %(prog)s paper.tex -b refs1.bib refs2.bib refs3.bib -o filtered.bib

latexmkrc END block (works on Overleaf with pdfLaTeX):
  # Add to your .latexmkrc file:
  END {
    system("python3 %(prog)s . -b refs.bib -o refs_filtered.bib -r --no-warn-missing");
  }

  # Multiple bib files:
  END {
    system("python3 %(prog)s . -b refs1.bib refs2.bib -o refs_filtered.bib -r --no-warn-missing");
  }
        """,
    )

    parser.add_argument(
        "tex_sources",
        nargs="+",
        help="LaTeX file(s) or directory(ies) to scan for citations",
    )
    parser.add_argument(
        "bib_file",
        nargs="?",
        help="Input bibliography file (.bib) - for backward compatibility",
    )
    parser.add_argument(
        "-b",
        "--bib",
        dest="bib_files",
        nargs="+",
        metavar="BIB",
        help="Input bibliography file(s) (.bib). Multiple files are merged; duplicates cause an error.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: <input>_filtered.bib)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search directories for .tex files",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive key matching",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        help="Write log output to a file (useful for Overleaf debugging)",
    )
    parser.add_argument(
        "--list-citations",
        action="store_true",
        help="List all found citations and exit",
    )
    parser.add_argument(
        "--warn-missing",
        action="store_true",
        default=True,
        help="Warn about missing citations (default: True)",
    )
    parser.add_argument(
        "--no-warn-missing",
        action="store_false",
        dest="warn_missing",
        help="Suppress warnings about missing citations",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.log_file:
        # Add file handler for logging to file
        file_handler = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Extract citations
    LOG.info(f"Scanning {len(args.tex_sources)} source(s) for citations...")
    citations, processed_files = extract_citations_from_project(args.tex_sources, recursive=args.recursive)

    LOG.info(f"Processed {len(processed_files)} .tex file(s)")
    LOG.info(f"Found {len(citations)} unique citation(s)")

    if args.list_citations:
        print("\nCitations found:")
        for key in sorted(citations):
            print(f"  {key}")
        return 0

    if not citations:
        LOG.warning("No citations found in the input files")
        return 0

    # Determine bib files to use (support both old positional and new -b flag syntax)
    # For backward compatibility: if no -b flag, check if any tex_source is actually a .bib file
    if args.bib_files:
        bib_files = args.bib_files
    elif args.bib_file:
        bib_files = [args.bib_file]
    else:
        # Check if any "tex_sources" are actually .bib files (backward compatibility)
        bib_in_sources = [s for s in args.tex_sources if s.endswith(".bib")]
        if bib_in_sources:
            bib_files = bib_in_sources
            # Remove .bib files from tex_sources
            args.tex_sources = [s for s in args.tex_sources if not s.endswith(".bib")]
            if not args.tex_sources:
                LOG.error("No LaTeX sources specified (only .bib files found)")
                return 1
            # Re-extract citations with corrected tex_sources
            LOG.info(f"Scanning {len(args.tex_sources)} source(s) for citations...")
            citations, processed_files = extract_citations_from_project(args.tex_sources, recursive=args.recursive)
            LOG.info(f"Processed {len(processed_files)} .tex file(s)")
            LOG.info(f"Found {len(citations)} unique citation(s)")
            if not citations:
                LOG.warning("No citations found in the input files")
                return 0
        else:
            LOG.error("No bibliography file specified. Use -b flag or provide as positional argument.")
            return 1

    # Parse bibliography file(s)
    LOG.info(f"Loading bibliography from {len(bib_files)} file(s)...")
    entries, duplicates = parse_bib_files(bib_files)
    LOG.info(f"Loaded {len(entries)} entries from {len(bib_files)} bibliography file(s)")

    if duplicates:
        LOG.debug(f"Found {len(duplicates)} duplicate key(s) in bibliography (will error only if cited)")

    # Filter
    filtered, found, missing = filter_entries(entries, citations, case_sensitive=args.case_sensitive)

    # Check if any cited keys have duplicates
    cited_duplicates = []
    for key in found:
        key_lower = key.lower() if not args.case_sensitive else key
        if key_lower in duplicates:
            cited_duplicates.append((key, duplicates[key_lower]))

    if cited_duplicates:
        LOG.error("The following cited keys have duplicates in the bibliography:")
        for key, files in cited_duplicates:
            LOG.error(f"  - '{key}' appears in: {', '.join(files)}")
        LOG.error("Please remove duplicate entries to avoid ambiguity.")
        return 1

    LOG.info(f"Matched {len(found)} citation(s) to bibliography entries")

    if args.warn_missing and missing:
        LOG.warning(f"{len(missing)} citation(s) not found in bibliography:")
        for key in sorted(missing):
            LOG.warning(f"  - {key}")

    # Output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(bib_files[0])[0]
        output_path = f"{base}_filtered.bib"

    # Write output
    if args.dry_run:
        LOG.info(f"[DRY RUN] Would write {len(filtered)} entries to {output_path}")
        print("\nEntries that would be included:")
        for entry in filtered:
            print(f"  {entry.key}")
    else:
        write_bib_file(filtered, output_path)
        LOG.info(f"Wrote {len(filtered)} entries to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
