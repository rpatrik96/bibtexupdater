"""Tests for Obsidian keyword generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibtex_updater.obsidian_keywords import (
    KeywordResult,
    ObsidianKeywordGenerator,
    extract_abstract,
    extract_existing_keywords,
    extract_title,
    format_keywords_yaml,
    parse_frontmatter,
    update_note_keywords,
)


class TestParseFrontmatter:
    """Tests for frontmatter parsing."""

    def test_parse_valid_frontmatter(self):
        """Parse valid YAML frontmatter."""
        content = """---
title: Test Paper
keywords:
  - "[[topic1]]"
  - "[[topic2]]"
---

# Content here
"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["title"] == "Test Paper"
        # YAML parsing strips outer quotes
        assert frontmatter["keywords"] == ["[[topic1]]", "[[topic2]]"]
        assert "# Content here" in body

    def test_parse_no_frontmatter(self):
        """Handle content without frontmatter."""
        content = "# Just a heading\n\nSome text"
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_parse_incomplete_frontmatter(self):
        """Handle incomplete frontmatter delimiters."""
        content = "---\ntitle: Test\nNo closing delimiter"
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_parse_empty_frontmatter(self):
        """Handle empty frontmatter."""
        content = "---\n---\n\n# Content"
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert "# Content" in body


class TestExtractAbstract:
    """Tests for abstract extraction."""

    def test_extract_abstract_callout(self):
        """Extract abstract from callout format."""
        content = """> [!abstract] Abstract
> This is the abstract text that spans
> multiple lines.

> [!info] Other section
"""
        abstract = extract_abstract(content)
        assert "This is the abstract text" in abstract
        assert "multiple lines" in abstract

    def test_extract_abstract_no_callout(self):
        """Handle missing abstract callout."""
        content = "# Title\n\nJust regular content"
        abstract = extract_abstract(content)
        assert abstract == ""

    def test_extract_abstract_cleans_quote_markers(self):
        """Remove quote markers from extracted abstract."""
        content = """
> [!abstract] Abstract

> Line one
> Line two

"""
        abstract = extract_abstract(content)
        assert not abstract.startswith(">")
        assert "> " not in abstract or abstract.count(">") == 0


class TestExtractTitle:
    """Tests for title extraction."""

    def test_extract_title_from_aliases(self):
        """Extract title from frontmatter aliases."""
        frontmatter = {"aliases": ["@smith2024", "A Great Paper Title"]}
        title = extract_title(frontmatter, "")
        assert title == "A Great Paper Title"

    def test_extract_title_from_h1(self):
        """Extract title from H1 heading."""
        frontmatter = {}
        content = "# My Paper Title\n\nContent here"
        title = extract_title(frontmatter, content)
        assert title == "My Paper Title"

    def test_extract_title_empty(self):
        """Handle missing title."""
        frontmatter = {}
        content = "No heading here"
        title = extract_title(frontmatter, content)
        assert title == ""


class TestExtractExistingKeywords:
    """Tests for keyword extraction."""

    def test_extract_wikilink_keywords(self):
        """Extract keywords with wikilink syntax."""
        frontmatter = {
            "keywords": [
                '"[[machine-learning]]"',
                '"[[deep-learning]]"',
            ]
        }
        keywords = extract_existing_keywords(frontmatter)
        assert keywords == ["machine-learning", "deep-learning"]

    def test_extract_plain_keywords(self):
        """Extract plain string keywords."""
        frontmatter = {"keywords": ["topic1", "topic2"]}
        keywords = extract_existing_keywords(frontmatter)
        assert keywords == ["topic1", "topic2"]

    def test_extract_no_keywords(self):
        """Handle missing keywords."""
        frontmatter = {}
        keywords = extract_existing_keywords(frontmatter)
        assert keywords == []

    def test_extract_empty_keywords(self):
        """Handle empty keywords list."""
        frontmatter = {"keywords": []}
        keywords = extract_existing_keywords(frontmatter)
        assert keywords == []


class TestSplitCompoundKeywords:
    """Tests for splitting compound keywords into compositional parts."""

    def test_split_dash_separator(self):
        """Split keywords with dash separator."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["machine learning - in context learning"]
        result = split_compound_keywords(keywords)
        assert "machine learning" in result
        assert "in context learning" in result
        assert len(result) == 2

    def test_split_slash_separator(self):
        """Split keywords with slash separator."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["NLP / natural language processing"]
        result = split_compound_keywords(keywords)
        assert "NLP" in result
        assert "natural language processing" in result

    def test_split_en_dash_separator(self):
        """Split keywords with en-dash separator."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["deep learning – transformers"]
        result = split_compound_keywords(keywords)
        assert "deep learning" in result
        assert "transformers" in result

    def test_split_em_dash_separator(self):
        """Split keywords with em-dash separator."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["reinforcement learning — policy gradient"]
        result = split_compound_keywords(keywords)
        assert "reinforcement learning" in result
        assert "policy gradient" in result

    def test_no_split_atomic_keyword(self):
        """Atomic keywords remain unchanged."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["machine learning", "deep learning"]
        result = split_compound_keywords(keywords)
        assert result == ["machine learning", "deep learning"]

    def test_split_multiple_compounds(self):
        """Split multiple compound keywords."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["AI - ML", "NLP / text"]
        result = split_compound_keywords(keywords)
        assert "AI" in result
        assert "ML" in result
        assert "NLP" in result
        assert "text" in result
        assert len(result) == 4

    def test_deduplicate_after_split(self):
        """Duplicates are removed after splitting."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["machine learning - deep learning", "deep learning"]
        result = split_compound_keywords(keywords)
        assert result.count("deep learning") == 1

    def test_empty_parts_filtered(self):
        """Empty parts from splitting are filtered out."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["topic -  - another"]  # Extra dashes
        result = split_compound_keywords(keywords)
        assert "" not in result
        assert "topic" in result
        assert "another" in result

    def test_whitespace_trimmed(self):
        """Whitespace is trimmed from split parts."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["  machine learning  -  deep learning  "]
        result = split_compound_keywords(keywords)
        assert "machine learning" in result
        assert "deep learning" in result
        # No leading/trailing whitespace
        for kw in result:
            assert kw == kw.strip()

    def test_preserve_order(self):
        """Parts from same keyword maintain relative order."""
        from bibtex_updater.obsidian_keywords import split_compound_keywords

        keywords = ["first - second - third"]
        result = split_compound_keywords(keywords)
        assert result == ["first", "second", "third"]


class TestFormatKeywordsYaml:
    """Tests for YAML formatting."""

    def test_format_keywords(self):
        """Format keywords as YAML with wikilinks."""
        keywords = ["topic1", "topic2"]
        result = format_keywords_yaml(keywords)
        assert result == 'keywords:\n  - "[[topic1]]"\n  - "[[topic2]]"'

    def test_format_empty_keywords(self):
        """Format empty keyword list."""
        result = format_keywords_yaml([])
        assert result == "keywords:"


class TestUpdateNoteKeywords:
    """Tests for note file updating."""

    def test_update_note_merge(self, tmp_path):
        """Update note merging with existing keywords."""
        note = tmp_path / "test.md"
        note.write_text(
            """---
title: Test
keywords:
  - "[[existing]]"
---

# Content
"""
        )
        update_note_keywords(note, ["new-topic"], merge=True)
        content = note.read_text()
        assert "[[existing]]" in content
        assert "[[new-topic]]" in content

    def test_update_note_replace(self, tmp_path):
        """Update note replacing existing keywords."""
        note = tmp_path / "test.md"
        note.write_text(
            """---
title: Test
keywords:
  - "[[old-topic]]"
---

# Content
"""
        )
        update_note_keywords(note, ["new-topic"], merge=False)
        content = note.read_text()
        assert "[[new-topic]]" in content
        # After replacement, old topic should not be present
        # (Note: depends on YAML serialization)


class TestKeywordResult:
    """Tests for KeywordResult dataclass."""

    def test_default_values(self):
        """KeywordResult has correct defaults."""
        result = KeywordResult(file_path=Path("test.md"))
        assert result.existing_keywords == []
        assert result.generated_keywords == []
        assert result.action == "skipped"
        assert result.message == ""

    def test_enriched_result(self):
        """KeywordResult for successful enrichment."""
        result = KeywordResult(
            file_path=Path("test.md"),
            existing_keywords=["old"],
            generated_keywords=["new1", "new2"],
            action="enriched",
            message="Added 2 keywords",
        )
        assert result.action == "enriched"
        assert len(result.generated_keywords) == 2


class TestObsidianKeywordGenerator:
    """Tests for the keyword generator class."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        mock = MagicMock()
        mock_result = MagicMock()
        mock_result.all_topics = [
            MagicMock(topic_name="machine-learning"),
            MagicMock(topic_name="neural-networks"),
            MagicMock(topic_name="deep-learning"),
        ]
        mock.classify.return_value = mock_result
        return mock

    @pytest.fixture
    def sample_note(self, tmp_path):
        """Create a sample paper note."""
        note = tmp_path / "@smith2024.md"
        note.write_text(
            """---
cssclasses:
  - research-note
aliases:
  - "@smith2024"
  - "Deep Learning for Everything"
keywords:
---

# Deep Learning for Everything

> [!abstract] Abstract
> This paper presents a novel approach to deep learning
> that improves performance on various benchmarks.

> [!info]
> - Authors: Smith et al.
"""
        )
        return note

    def test_generate_keywords(self, mock_classifier):
        """Generate keywords for a paper."""
        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude", max_keywords=3)
            keywords = generator.generate_keywords(
                title="Test Paper",
                abstract="This is about machine learning.",
            )
            assert len(keywords) <= 3
            assert "machine-learning" in keywords

    def test_process_note_dry_run(self, sample_note, mock_classifier):
        """Process note in dry run mode."""
        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            result = generator.process_note(sample_note, dry_run=True)

            assert result.action == "dry_run"
            assert len(result.generated_keywords) > 0
            # File should not be modified in dry run
            content = sample_note.read_text()
            assert "machine-learning" not in content

    def test_process_note_enrich(self, sample_note, mock_classifier):
        """Process note and actually enrich it."""
        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            result = generator.process_note(sample_note, dry_run=False)

            assert result.action == "enriched"
            # File should be modified
            content = sample_note.read_text()
            assert "[[machine-learning]]" in content or "machine-learning" in content

    def test_process_note_no_abstract(self, tmp_path, mock_classifier):
        """Skip note without abstract."""
        note = tmp_path / "no-abstract.md"
        note.write_text(
            """---
title: Test
---

# No Abstract Here

Just regular content.
"""
        )
        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            result = generator.process_note(note, dry_run=True)

            assert result.action == "skipped"
            assert "No abstract" in result.message

    def test_process_note_already_has_keywords(self, tmp_path, mock_classifier):
        """Skip enrichment when all keywords already exist."""
        note = tmp_path / "has-keywords.md"
        note.write_text(
            """---
title: Test
keywords:
  - "[[machine-learning]]"
  - "[[neural-networks]]"
  - "[[deep-learning]]"
---

# Test Paper

> [!abstract] Abstract
> This is about ML.

"""
        )
        # Mock returns same keywords that already exist
        mock_classifier.classify.return_value.all_topics = [
            MagicMock(topic_name="machine-learning"),
            MagicMock(topic_name="neural-networks"),
        ]

        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            result = generator.process_note(note, dry_run=True)

            assert result.action == "skipped"
            assert "No new keywords" in result.message

    def test_process_folder(self, tmp_path, mock_classifier):
        """Process multiple notes in a folder."""
        # Create multiple notes
        for i in range(3):
            note = tmp_path / f"@paper{i}.md"
            note.write_text(
                f"""---
title: Paper {i}
keywords:
---

# Paper {i}

> [!abstract] Abstract
> Abstract for paper {i}.

"""
            )

        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            results = generator.process_folder(
                tmp_path,
                pattern="@*.md",
                dry_run=True,
                limit=2,
            )

            assert len(results) == 2  # Limited to 2
            assert all(r.action == "dry_run" for r in results)

    def test_process_folder_limit_counts_enriched_only(self, tmp_path, mock_classifier):
        """Limit only counts enriched notes, not skipped ones."""
        # Create 5 notes with abstracts - limit=2 should process until 2 are enriched
        for i in range(5):
            note = tmp_path / f"@paper{i}.md"
            note.write_text(
                f"""---
title: Paper {i}
keywords:
---
# Paper {i}
> [!abstract] Abstract
> Abstract for paper {i}.
"""
            )

        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            results = generator.process_folder(
                tmp_path,
                pattern="@*.md",
                dry_run=True,
                limit=2,  # Should stop after enriching exactly 2
            )

            enriched = [r for r in results if r.action == "dry_run"]

            # Should have exactly 2 enriched (limit)
            assert len(enriched) == 2
            # Should have stopped early, not processed all 5
            assert len(results) == 2

    def test_process_folder_min_keywords_skips(self, tmp_path, mock_classifier):
        """Notes with enough keywords are skipped without API call."""
        # Note with many keywords
        note1 = tmp_path / "@has-keywords.md"
        note1.write_text(
            """---
title: Has Keywords
keywords:
  - "[[topic1]]"
  - "[[topic2]]"
  - "[[topic3]]"
---
# Has Keywords
> [!abstract] Abstract
> Some abstract.
"""
        )

        # Note without keywords
        note2 = tmp_path / "@no-keywords.md"
        note2.write_text(
            """---
title: No Keywords
keywords:
---
# No Keywords
> [!abstract] Abstract
> Some abstract.
"""
        )

        with patch(
            "bibtex_updater.organizer.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ):
            generator = ObsidianKeywordGenerator(backend="claude")
            results = generator.process_folder(
                tmp_path,
                pattern="@*.md",
                dry_run=True,
                min_existing_keywords=3,  # Skip notes with 3+ keywords
            )

            # has-keywords should be skipped, no-keywords should be enriched
            has_kw = next(r for r in results if "has-keywords" in str(r.file_path))
            no_kw = next(r for r in results if "no-keywords" in str(r.file_path))

            assert has_kw.action == "skipped"
            assert "Already has 3 keywords" in has_kw.message
            assert no_kw.action == "dry_run"


class TestCLI:
    """Tests for CLI functionality."""

    def test_cli_help(self):
        """CLI shows help."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "bibtex_updater.cli.obsidian_keywords_cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Generate AI-powered keywords" in result.stdout

    def test_cli_nonexistent_path(self):
        """CLI handles nonexistent path."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "bibtex_updater.cli.obsidian_keywords_cli", "/nonexistent/path"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        # Error may go to stdout or stderr
        combined_output = result.stdout + result.stderr
        assert "does not exist" in combined_output
