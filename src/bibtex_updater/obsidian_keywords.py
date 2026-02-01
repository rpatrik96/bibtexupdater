"""AI-powered keyword generator for Obsidian paper notes.

This module provides functionality to analyze paper notes and generate
topic keywords as [[wikilinks]] for better knowledge graph connectivity.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    """Result of keyword generation for a note.

    Attributes:
        file_path: Path to the processed file
        existing_keywords: Keywords already present in the note
        generated_keywords: AI-generated keywords
        action: Action taken ("enriched", "skipped", "error")
        message: Human-readable status message
    """

    file_path: Path
    existing_keywords: list[str] = field(default_factory=list)
    generated_keywords: list[str] = field(default_factory=list)
    action: str = "skipped"
    message: str = ""


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Tuple of (frontmatter dict, body content)
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2]
        return frontmatter, body
    except yaml.YAMLError:
        return {}, content


def extract_abstract(content: str) -> str:
    """Extract abstract from paper note content.

    Args:
        content: Markdown content

    Returns:
        Abstract text or empty string
    """
    # Try multiple patterns for abstract extraction

    # Pattern 1: Callout with optional blank line, then content
    # > [!abstract] Abstract
    # (optional blank line or empty quote line)
    # > Content here
    match = re.search(
        r">\s*\[!abstract\][^\n]*\n(?:\s*\n|>\s*\n)*((?:>\s*[^\n]+\n?)+)",
        content,
        re.IGNORECASE,
    )
    if match:
        abstract = match.group(1).strip()
        # Clean up markdown quote prefixes
        abstract = re.sub(r"^>\s*", "", abstract, flags=re.MULTILINE)
        return abstract.strip()

    # Pattern 2: Abstract content immediately follows (no blank lines)
    match = re.search(
        r">\s*\[!abstract\][^\n]*\n>\s*(.+?)(?=\n\s*\n|\n>?\s*\[!|\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        abstract = match.group(1).strip()
        abstract = re.sub(r"^>\s*", "", abstract, flags=re.MULTILINE)
        return abstract.strip()

    return ""


def extract_title(frontmatter: dict[str, Any], content: str) -> str:
    """Extract paper title from frontmatter or content.

    Args:
        frontmatter: Parsed frontmatter dict
        content: Markdown content

    Returns:
        Paper title
    """
    # Try aliases first (common pattern: second alias is title)
    aliases = frontmatter.get("aliases", [])
    if len(aliases) >= 2:
        return aliases[1]

    # Try H1 heading
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return ""


def extract_existing_keywords(frontmatter: dict[str, Any]) -> list[str]:
    """Extract existing keywords from frontmatter.

    Args:
        frontmatter: Parsed frontmatter dict

    Returns:
        List of keyword strings (without [[ ]] wrappers)
    """
    keywords = frontmatter.get("keywords", [])
    if not keywords:
        return []

    result = []
    for kw in keywords:
        if isinstance(kw, str):
            # Remove [[wikilink]] syntax
            clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", kw)
            result.append(clean.strip().strip('"'))
    return result


def split_compound_keywords(keywords: list[str]) -> list[str]:
    """Split compound keywords into compositional parts.

    AI backends may generate compound keywords like "machine learning - in context learning".
    This function splits them into atomic parts for better knowledge graph connectivity.

    Args:
        keywords: List of keywords (possibly compound)

    Returns:
        List of atomic keywords with compounds split apart
    """
    # Separators that indicate compound keywords (ordered by specificity)
    separators = [
        " - ",  # Common dash separator
        " – ",  # En-dash
        " — ",  # Em-dash
        " / ",  # Slash separator
    ]

    result = []
    for kw in keywords:
        parts = [kw]
        # Try each separator
        for sep in separators:
            new_parts = []
            for part in parts:
                if sep in part:
                    new_parts.extend(part.split(sep))
                else:
                    new_parts.append(part)
            parts = new_parts

        # Clean up and add non-empty parts
        for part in parts:
            cleaned = part.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)

    return result


def format_keywords_yaml(keywords: list[str]) -> str:
    """Format keywords as YAML list with [[wikilinks]].

    Args:
        keywords: List of keyword strings

    Returns:
        YAML-formatted keyword list
    """
    if not keywords:
        return "keywords:"

    lines = ["keywords:"]
    for kw in keywords:
        lines.append(f'  - "[[{kw}]]"')
    return "\n".join(lines)


def update_note_keywords(file_path: Path, new_keywords: list[str], merge: bool = True) -> bool:
    """Update keywords in an Obsidian note file.

    Args:
        file_path: Path to the markdown file
        new_keywords: Keywords to add
        merge: If True, merge with existing; if False, replace

    Returns:
        True if file was modified
    """
    content = file_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    if merge:
        existing = set(extract_existing_keywords(frontmatter))
        all_keywords = list(existing | set(new_keywords))
    else:
        all_keywords = new_keywords

    # Sort for consistency
    all_keywords.sort()

    # Update frontmatter
    frontmatter["keywords"] = [f"[[{kw}]]" for kw in all_keywords]

    # Rebuild content
    new_content = "---\n" + yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True) + "---" + body

    file_path.write_text(new_content, encoding="utf-8")
    return True


class ObsidianKeywordGenerator:
    """Generates AI-powered keywords for Obsidian paper notes."""

    def __init__(
        self,
        backend: str = "claude",
        model: str | None = None,
        existing_topics: list[str] | None = None,
        max_keywords: int = 5,
    ) -> None:
        """Initialize the keyword generator.

        Args:
            backend: AI backend ("claude", "openai", or "embedding")
            model: Model name override
            existing_topics: List of existing topic names to prefer
            max_keywords: Maximum keywords to generate per paper
        """
        from bibtex_updater.organizer.classifier import ClassifierFactory
        from bibtex_updater.organizer.config import ClassifierConfig

        self.backend = backend
        self.max_keywords = max_keywords
        self.existing_topics = existing_topics or []

        config = ClassifierConfig(
            backend=backend,
            model=model,
            confidence_threshold=0.5,  # Lower threshold for keyword generation
        )
        self.classifier = ClassifierFactory.create(config)

    def generate_keywords(self, title: str, abstract: str) -> list[str]:
        """Generate keywords for a paper.

        Keywords are made compositional - compound keywords like
        "machine learning - in context learning" are split into
        atomic parts ["machine learning", "in context learning"].

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            List of generated keywords (compositional/atomic)
        """
        if not abstract and not title:
            return []

        result = self.classifier.classify(
            title=title,
            abstract=abstract,
            existing_topics=self.existing_topics,
        )

        keywords = []
        for topic in result.all_topics[: self.max_keywords]:
            keywords.append(topic.topic_name)

        # Split compound keywords into compositional parts
        keywords = split_compound_keywords(keywords)

        return keywords

    def process_note(self, file_path: Path, dry_run: bool = False) -> KeywordResult:
        """Process a single Obsidian note.

        Args:
            file_path: Path to the markdown file
            dry_run: If True, don't modify the file

        Returns:
            KeywordResult with processing outcome
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)

            # Extract existing data
            existing_keywords = extract_existing_keywords(frontmatter)
            title = extract_title(frontmatter, content)
            abstract = extract_abstract(content)

            if not abstract:
                return KeywordResult(
                    file_path=file_path,
                    existing_keywords=existing_keywords,
                    action="skipped",
                    message="No abstract found",
                )

            # Generate new keywords
            generated = self.generate_keywords(title, abstract)

            # Filter out already existing keywords
            new_keywords = [kw for kw in generated if kw not in existing_keywords]

            if not new_keywords:
                return KeywordResult(
                    file_path=file_path,
                    existing_keywords=existing_keywords,
                    action="skipped",
                    message="No new keywords to add",
                )

            if not dry_run:
                update_note_keywords(file_path, new_keywords, merge=True)

            return KeywordResult(
                file_path=file_path,
                existing_keywords=existing_keywords,
                generated_keywords=new_keywords,
                action="dry_run" if dry_run else "enriched",
                message=f"Added {len(new_keywords)} keywords",
            )

        except Exception as e:
            logger.exception(f"Error processing {file_path}: {e}")
            return KeywordResult(
                file_path=file_path,
                action="error",
                message=str(e),
            )

    def process_folder(
        self,
        folder: Path,
        pattern: str = "@*.md",
        dry_run: bool = False,
        limit: int | None = None,
        min_existing_keywords: int = 0,
    ) -> list[KeywordResult]:
        """Process all matching notes in a folder.

        Args:
            folder: Folder to process
            pattern: Glob pattern for note files
            dry_run: If True, don't modify files
            limit: Maximum notes to successfully enrich (skipped notes don't count)
            min_existing_keywords: Skip notes that already have this many keywords

        Returns:
            List of KeywordResult objects
        """
        files = list(folder.glob(pattern))
        total_files = len(files)

        results = []
        enriched_count = 0
        processed = 0

        for file_path in files:
            processed += 1

            # Pre-check: skip notes with enough existing keywords (avoids API call)
            if min_existing_keywords > 0:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    frontmatter, _ = parse_frontmatter(content)
                    existing = extract_existing_keywords(frontmatter)
                    if len(existing) >= min_existing_keywords:
                        logger.info(
                            f"[{processed}/{total_files}] Skipping: {file_path.name} (has {len(existing)} keywords)"
                        )
                        results.append(
                            KeywordResult(
                                file_path=file_path,
                                existing_keywords=existing,
                                action="skipped",
                                message=f"Already has {len(existing)} keywords",
                            )
                        )
                        continue
                except Exception:
                    pass  # Continue with normal processing if pre-check fails

            logger.info(f"[{processed}/{total_files}] Processing: {file_path.name}")
            result = self.process_note(file_path, dry_run=dry_run)
            results.append(result)

            if result.action == "enriched":
                logger.info(f"  + Added: {', '.join(result.generated_keywords)}")
                enriched_count += 1
            elif result.action == "dry_run":
                logger.info(f"  ~ Would add: {', '.join(result.generated_keywords)}")
                enriched_count += 1  # Count dry_run as "would enrich"
            elif result.action == "skipped":
                logger.info(f"  - Skipped: {result.message}")
            elif result.action == "error":
                logger.error(f"  ! Error: {result.message}")

            # Check if we've reached the enrichment limit
            if limit and enriched_count >= limit:
                logger.info(f"Reached limit of {limit} enriched notes")
                break

        return results
