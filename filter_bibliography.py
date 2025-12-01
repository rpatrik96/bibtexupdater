#!/usr/bin/env python3
"""
Filter Bibliography: Extract cited references from .tex files and create a filtered .bib file.

This script scans LaTeX files for citation commands and creates a new bibliography file
containing only the entries that are actually cited in the document.
"""

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


# ------------- Citation Extraction -------------

# Regex pattern to match LaTeX citation commands
# Matches: \cite, \citep, \citet, \citeauthor, \citeyear, \citealt, \citealp,
#          \parencite, \textcite, \autocite, \fullcite, \footcite, \nocite, etc.
# Also handles optional arguments like \cite[p. 5]{key} or \citep[see][p. 10]{key1,key2}
CITE_PATTERN = re.compile(
    r"\\(?:no)?cite[tp]?\*?"  # Standard \cite, \citep, \citet, \nocite (with optional *)
    r"|\\(?:cite(?:author|year|yearpar|title|date|url|field|name|list|num))\*?"  # biblatex variants
    r"|\\(?:parencite|textcite|smartcite|supercite)\*?"  # biblatex commands
    r"|\\(?:autocite|autocites)\*?"  # autocite
    r"|\\(?:fullcite|footcite|footcitetext)\*?"  # full/foot citations
    r"|\\(?:citealt|citealp)\*?"  # natbib alt forms
    r"|\\(?:citep?(?:g|s)?)\*?"  # additional variants
)

# Pattern to extract citation keys from the braces following a citation command
# This handles: \cite{key}, \cite{key1,key2}, \cite[opt]{key}, \cite[pre][post]{key}
# Also handles biblatex commands like \textcite, \parencite, \autocite, etc.
CITE_KEYS_PATTERN = re.compile(
    r"\\("
    r"(?:no)?cite[a-z]*"  # \cite, \citep, \citet, \nocite, \citeauthor, etc.
    r"|parencite|Parencite"  # \parencite
    r"|textcite|Textcite"  # \textcite
    r"|smartcite|Smartcite"  # \smartcite
    r"|supercite"  # \supercite
    r"|autocite|Autocite"  # \autocite
    r"|fullcite"  # \fullcite
    r"|footcite|footcitetext"  # \footcite
    r"|blockcquote"  # \blockcquote (csquotes + biblatex)
    r")\*?"  # Optional star
    r"(?:\s*\[[^\]]*\])*"  # Optional arguments in square brackets
    r"\s*\{([^}]+)\}"  # Required argument with citation keys
)


def extract_citations_from_text(text: str) -> Set[str]:
    """
    Extract all citation keys from LaTeX text.

    Args:
        text: LaTeX source text

    Returns:
        Set of unique citation keys found in the text
    """
    citations: Set[str] = set()

    # Find all citation commands with their keys
    for match in CITE_KEYS_PATTERN.finditer(text):
        keys_str = match.group(2)
        # Split by comma and strip whitespace
        keys = [k.strip() for k in keys_str.split(",")]
        citations.update(k for k in keys if k)

    return citations


def extract_citations_from_file(filepath: str) -> Set[str]:
    """
    Extract all citation keys from a LaTeX file.

    Args:
        filepath: Path to the .tex file

    Returns:
        Set of unique citation keys found in the file
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except IOError as e:
        LOG.error(f"Could not read file {filepath}: {e}")
        return set()

    # Remove comments (lines starting with %)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Find the position of % that's not escaped
        pos = 0
        while pos < len(line):
            idx = line.find("%", pos)
            if idx == -1:
                cleaned_lines.append(line)
                break
            elif idx == 0 or line[idx - 1] != "\\":
                cleaned_lines.append(line[:idx])
                break
            else:
                pos = idx + 1
        else:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    return extract_citations_from_text(cleaned_text)


def find_tex_files(path: str, recursive: bool = True) -> List[str]:
    """
    Find all .tex files in a directory.

    Args:
        path: Directory path to search
        recursive: Whether to search subdirectories

    Returns:
        List of paths to .tex files
    """
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() == ".tex" else []

    pattern = "**/*.tex" if recursive else "*.tex"
    return [str(f) for f in p.glob(pattern)]


def extract_citations_from_project(paths: List[str], recursive: bool = True) -> Tuple[Set[str], List[str]]:
    """
    Extract all citation keys from multiple .tex files or directories.

    Args:
        paths: List of file or directory paths
        recursive: Whether to search directories recursively

    Returns:
        Tuple of (set of citation keys, list of processed files)
    """
    all_citations: Set[str] = set()
    all_files: List[str] = []

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


# ------------- BibTeX IO -------------


class BibLoader:
    """Load BibTeX files using bibtexparser."""

    def __init__(self) -> None:
        self.parser = BibTexParser(common_strings=True)
        self.parser.customization = None

    def load_file(self, path: str) -> bibtexparser.bibdatabase.BibDatabase:
        """Load a .bib file and return a BibDatabase."""
        with open(path, "r", encoding="utf-8") as f:
            return bibtexparser.load(f, parser=self.parser)

    def loads(self, text: str) -> bibtexparser.bibdatabase.BibDatabase:
        """Load a BibDatabase from a string."""
        return bibtexparser.loads(text, parser=self.parser)


class BibWriter:
    """Write BibTeX databases to files with safe atomic operations."""

    def __init__(self) -> None:
        self.writer = BibTexWriter()
        self.writer.indent = "  "
        self.writer.order_entries_by = None
        self.writer.comma_first = False

    def dumps(self, db: bibtexparser.bibdatabase.BibDatabase) -> str:
        """Serialize a BibDatabase to a string."""
        return bibtexparser.dumps(db, writer=self.writer)

    def dump_to_file(self, db: bibtexparser.bibdatabase.BibDatabase, path: str) -> None:
        """Write a BibDatabase to a file atomically."""
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".bib", prefix=".tmp_bib_")
        try:
            tmp.write(self.dumps(db))
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, path)


# ------------- Bibliography Filtering -------------


def filter_bibliography(
    bib_db: bibtexparser.bibdatabase.BibDatabase, cited_keys: Set[str], case_sensitive: bool = False
) -> Tuple[bibtexparser.bibdatabase.BibDatabase, Set[str], Set[str]]:
    """
    Filter a bibliography database to only include cited entries.

    Args:
        bib_db: Source BibDatabase
        cited_keys: Set of citation keys to keep
        case_sensitive: Whether key matching is case-sensitive

    Returns:
        Tuple of (filtered BibDatabase, found keys, missing keys)
    """
    filtered_db = bibtexparser.bibdatabase.BibDatabase()
    found_keys: Set[str] = set()

    # Create lookup map
    if case_sensitive:
        key_map = {entry["ID"]: entry for entry in bib_db.entries}
        lookup_keys = cited_keys
    else:
        key_map = {entry["ID"].lower(): entry for entry in bib_db.entries}
        lookup_keys = {k.lower() for k in cited_keys}

    # Filter entries
    for key in lookup_keys:
        if key in key_map:
            filtered_db.entries.append(key_map[key])
            found_keys.add(key)

    # Determine missing keys
    missing_keys = lookup_keys - found_keys

    return filtered_db, found_keys, missing_keys


# ------------- CLI -------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter a bibliography file to include only cited references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.tex references.bib -o filtered.bib
  %(prog)s *.tex references.bib -o filtered.bib
  %(prog)s ./chapters/ references.bib -o filtered.bib --recursive
  %(prog)s paper.tex refs.bib --dry-run
        """,
    )

    parser.add_argument("tex_sources", nargs="+", help="LaTeX file(s) or directory(ies) to scan for citations")

    parser.add_argument("bib_file", help="Input bibliography file (.bib)")

    parser.add_argument("-o", "--output", help="Output bibliography file (default: <input>_filtered.bib)")

    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively search directories for .tex files")

    parser.add_argument(
        "--case-sensitive", action="store_true", help="Use case-sensitive key matching (default: case-insensitive)"
    )

    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done without writing files")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("--list-citations", action="store_true", help="List all found citations and exit")

    parser.add_argument(
        "--warn-missing",
        action="store_true",
        default=True,
        help="Warn about citations not found in bibliography (default: True)",
    )

    parser.add_argument(
        "--no-warn-missing", action="store_false", dest="warn_missing", help="Suppress warnings about missing citations"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Extract citations from tex files
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

    # Load bibliography
    LOG.info(f"Loading bibliography from {args.bib_file}...")
    loader = BibLoader()
    try:
        bib_db = loader.load_file(args.bib_file)
    except Exception as e:
        LOG.error(f"Failed to load bibliography: {e}")
        return 1

    LOG.info(f"Loaded {len(bib_db.entries)} entries from bibliography")

    # Filter bibliography
    filtered_db, found_keys, missing_keys = filter_bibliography(bib_db, citations, case_sensitive=args.case_sensitive)

    LOG.info(f"Matched {len(found_keys)} citation(s) to bibliography entries")

    # Warn about missing citations
    if args.warn_missing and missing_keys:
        LOG.warning(f"{len(missing_keys)} citation(s) not found in bibliography:")
        for key in sorted(missing_keys):
            LOG.warning(f"  - {key}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.bib_file)[0]
        output_path = f"{base}_filtered.bib"

    # Write output
    if args.dry_run:
        LOG.info(f"[DRY RUN] Would write {len(filtered_db.entries)} entries to {output_path}")
        print("\nEntries that would be included:")
        for entry in filtered_db.entries:
            print(f"  {entry['ID']}")
    else:
        writer = BibWriter()
        writer.dump_to_file(filtered_db, output_path)
        LOG.info(f"Wrote {len(filtered_db.entries)} entries to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
