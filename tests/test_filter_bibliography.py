"""
Tests for filter_bibliography.py

Run with: pytest tests/test_filter_bibliography.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from filter_bibliography import (
    BibEntry,
    extract_citations_from_file,
    extract_citations_from_project,
    extract_citations_from_text,
    extract_entry_key,
    filter_entries,
    find_matching_brace,
    find_tex_files,
    parse_bib_file,
    parse_bib_files,
    parse_bib_string,
    strip_comments,
    write_bib_file,
)

# ------------- Fixtures -------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_bib_path():
    """Path to sample bibliography file."""
    return str(FIXTURES_DIR / "sample.bib")


@pytest.fixture
def main_tex_path():
    """Path to main.tex test file."""
    return str(FIXTURES_DIR / "main.tex")


@pytest.fixture
def fixtures_dir():
    """Path to fixtures directory."""
    return str(FIXTURES_DIR)


# ------------- Brace Matching Tests -------------


class TestFindMatchingBrace:
    """Tests for the brace-counting algorithm."""

    def test_simple_braces(self):
        """Test simple brace matching."""
        assert find_matching_brace("{abc}", 0) == 4

    def test_nested_braces(self):
        """Test nested braces like {A {GPU} Implementation}."""
        assert find_matching_brace("{a{b}c}", 0) == 6

    def test_deeply_nested(self):
        """Test deeply nested braces."""
        assert find_matching_brace("{a{b{c}d}e}", 0) == 10

    def test_quoted_braces_ignored(self):
        """Braces inside quotes should not count."""
        # The outer braces contain a quoted string with braces
        assert find_matching_brace('{"{}"}', 0) == 5

    def test_unmatched_returns_minus_one(self):
        """Unmatched brace returns -1."""
        assert find_matching_brace("{abc", 0) == -1

    def test_empty_braces(self):
        """Test empty braces."""
        assert find_matching_brace("{}", 0) == 1

    def test_not_starting_at_brace(self):
        """If start position is not a brace, return -1."""
        assert find_matching_brace("abc{}", 0) == -1

    def test_real_bibtex_entry(self):
        """Test with a real BibTeX entry structure."""
        content = "@article{key, title = {A {GPU} Implementation}}"
        # Find the opening brace after @article
        start = content.index("{")
        end = find_matching_brace(content, start)
        assert content[end] == "}"
        assert content[start : end + 1] == "{key, title = {A {GPU} Implementation}}"


# ------------- Entry Key Extraction Tests -------------


class TestExtractEntryKey:
    """Tests for extract_entry_key function."""

    def test_simple_key(self):
        """Test simple key extraction."""
        assert extract_entry_key("test2020, title = {Test}") == "test2020"

    def test_key_with_underscore(self):
        """Test key with underscore."""
        assert extract_entry_key("my_key_2020, title = {Test}") == "my_key_2020"

    def test_key_with_dash(self):
        """Test key with dash."""
        assert extract_entry_key("my-key-2020, title = {Test}") == "my-key-2020"

    def test_key_with_leading_whitespace(self):
        """Test key with leading whitespace."""
        assert extract_entry_key("  test2020, title = {Test}") == "test2020"

    def test_no_comma(self):
        """Test when there's no comma (invalid entry)."""
        assert extract_entry_key("test2020") is None


# ------------- BibTeX Parsing Tests -------------


class TestParseBibString:
    """Tests for parse_bib_string function."""

    def test_simple_entry(self):
        """Test parsing a simple entry."""
        content = "@article{test2020, title = {Test}}"
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert entries[0].key == "test2020"
        assert entries[0].entry_type == "article"

    def test_nested_braces_in_title(self):
        """Test nested braces are preserved."""
        content = "@article{key, title = {A {GPU} Implementation}}"
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert "{GPU}" in entries[0].raw_content

    def test_multiple_entries(self):
        """Test parsing multiple entries."""
        content = """
        @article{a, title = {A}}
        @book{b, title = {B}}
        @inproceedings{c, title = {C}}
        """
        entries = parse_bib_string(content)
        assert len(entries) == 3
        assert {e.key for e in entries} == {"a", "b", "c"}

    def test_skip_string_preamble_comment(self):
        """Test that @string, @preamble, @comment are skipped."""
        content = """
        @string{jan = "January"}
        @preamble{"Some preamble"}
        @comment{This is a comment}
        @article{real, title = {Real}}
        """
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert entries[0].key == "real"

    def test_quoted_string_values(self):
        """Test entries with quoted string values."""
        content = '@article{key, author = "Smith, John", title = {Test}}'
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert entries[0].key == "key"

    def test_multiline_entry(self):
        """Test multiline entry parsing."""
        content = """@article{multiline2020,
            title = {This is a
            multiline title},
            author = {Smith, John}
        }"""
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert entries[0].key == "multiline2020"

    def test_entry_type_case_insensitive(self):
        """Test that entry types are case-insensitive."""
        content = "@ARTICLE{upper, title = {Upper}} @Article{mixed, title = {Mixed}}"
        entries = parse_bib_string(content)
        assert len(entries) == 2
        assert all(e.entry_type == "article" for e in entries)


# ------------- Citation Extraction Tests -------------


class TestExtractCitationsFromText:
    """Tests for extract_citations_from_text function (common commands only)."""

    def test_simple_cite(self):
        """Test basic \\cite command."""
        text = r"Modern optimizers \cite{kingma2015adam} improve training."
        citations = extract_citations_from_text(text)
        assert citations == {"kingma2015adam"}

    def test_citep(self):
        """Test \\citep command."""
        text = r"Self-attention revolutionized NLP \citep{vaswani2017attention}."
        citations = extract_citations_from_text(text)
        assert citations == {"vaswani2017attention"}

    def test_citet(self):
        """Test \\citet command."""
        text = r"\citet{he2016resnet} introduced residual connections."
        citations = extract_citations_from_text(text)
        assert citations == {"he2016resnet"}

    def test_nocite(self):
        """Test \\nocite command."""
        text = r"\nocite{goodfellow2016deeplearning}"
        citations = extract_citations_from_text(text)
        assert citations == {"goodfellow2016deeplearning"}

    def test_textcite(self):
        """Test biblatex \\textcite command."""
        text = r"\textcite{devlin2019bert} showed bidirectional pretraining works."
        citations = extract_citations_from_text(text)
        assert citations == {"devlin2019bert"}

    def test_parencite(self):
        """Test biblatex \\parencite command."""
        text = r"Vision Transformers work well \parencite{dosovitskiy2021vit}."
        citations = extract_citations_from_text(text)
        assert citations == {"dosovitskiy2021vit"}

    def test_autocite(self):
        """Test biblatex \\autocite command."""
        text = r"Contrastive learning is effective \autocite{chen2020simclr}."
        citations = extract_citations_from_text(text)
        assert citations == {"chen2020simclr"}

    def test_capitalized_variants(self):
        """Test capitalized variants like \\Textcite, \\Parencite."""
        text = r"\Textcite{a} and \Parencite{b} and \Autocite{c}"
        citations = extract_citations_from_text(text)
        assert citations == {"a", "b", "c"}

    def test_multiple_citations_single_command(self):
        """Test multiple citations in one command."""
        text = r"See \cite{kingma2015adam, vaswani2017attention, he2016resnet}."
        citations = extract_citations_from_text(text)
        assert citations == {"kingma2015adam", "vaswani2017attention", "he2016resnet"}

    def test_citation_with_optional_arg(self):
        """Test citation with optional argument."""
        text = r"According to \cite[Chapter 9]{goodfellow2016deeplearning}, CNNs are key."
        citations = extract_citations_from_text(text)
        assert citations == {"goodfellow2016deeplearning"}

    def test_citation_with_two_optional_args(self):
        """Test citation with pre and post optional arguments."""
        text = r"As noted \citep[see][Section 3]{vaswani2017attention}."
        citations = extract_citations_from_text(text)
        assert citations == {"vaswani2017attention"}

    def test_starred_cite(self):
        """Test starred citation commands."""
        text = r"\cite*{brown2020gpt3} and \citep*{touvron2023llama}"
        citations = extract_citations_from_text(text)
        assert citations == {"brown2020gpt3", "touvron2023llama"}

    def test_no_citations(self):
        """Test text with no citations."""
        text = "This is just plain text without any citations."
        citations = extract_citations_from_text(text)
        assert citations == set()

    def test_empty_text(self):
        """Test empty text."""
        citations = extract_citations_from_text("")
        assert citations == set()


# ------------- Comment Stripping Tests -------------


class TestStripComments:
    """Tests for strip_comments function."""

    def test_simple_comment(self):
        """Test removing simple comment."""
        text = "text % comment"
        assert strip_comments(text) == "text "

    def test_escaped_percent(self):
        """Test that escaped percent is preserved."""
        text = r"50\% improvement"
        assert strip_comments(text) == r"50\% improvement"

    def test_mixed(self):
        """Test mixed escaped and unescaped percent."""
        text = r"50\% improvement % this is a comment"
        assert strip_comments(text) == r"50\% improvement "

    def test_multiline(self):
        """Test multiline comment stripping."""
        text = "line1 % comment1\nline2 % comment2\nline3"
        result = strip_comments(text)
        assert "comment1" not in result
        assert "comment2" not in result
        assert "line3" in result


# ------------- File Operations Tests -------------


class TestExtractCitationsFromFile:
    """Tests for extract_citations_from_file function."""

    def test_main_tex_file(self, main_tex_path):
        """Test extraction from main.tex fixture."""
        citations = extract_citations_from_file(main_tex_path)
        assert "goodfellow2016deeplearning" in citations
        assert "kingma2015adam" in citations
        assert "vaswani2017attention" in citations
        # Comments should be ignored
        assert "should_be_ignored" not in citations

    def test_nonexistent_file(self):
        """Test handling of non-existent file."""
        citations = extract_citations_from_file("/nonexistent/path.tex")
        assert citations == set()


class TestFindTexFiles:
    """Tests for find_tex_files function."""

    def test_single_file(self, main_tex_path):
        """Test with single file path."""
        files = find_tex_files(main_tex_path)
        assert len(files) == 1
        assert main_tex_path in files[0]

    def test_directory_recursive(self, fixtures_dir):
        """Test recursive directory search."""
        files = find_tex_files(fixtures_dir, recursive=True)
        assert len(files) >= 1
        assert any("main.tex" in f for f in files)

    def test_non_tex_file(self, sample_bib_path):
        """Test that non-.tex files return empty list."""
        files = find_tex_files(sample_bib_path)
        assert files == []


class TestParseBibFile:
    """Tests for parse_bib_file function."""

    def test_sample_bib(self, sample_bib_path):
        """Test parsing sample.bib fixture."""
        entries = parse_bib_file(sample_bib_path)
        assert len(entries) > 0
        # Check for expected entries
        keys = {e.key for e in entries}
        assert "kingma2015adam" in keys
        assert "vaswani2017attention" in keys

    def test_nonexistent_file(self):
        """Test handling of non-existent file."""
        entries = parse_bib_file("/nonexistent/path.bib")
        assert entries == []


# ------------- Filtering Tests -------------


class TestFilterEntries:
    """Tests for filter_entries function."""

    def test_basic_filtering(self):
        """Test basic entry filtering."""
        entries = [
            BibEntry("article", "a", "@article{a, title={A}}"),
            BibEntry("article", "b", "@article{b, title={B}}"),
            BibEntry("article", "c", "@article{c, title={C}}"),
        ]
        cited = {"a", "c"}
        filtered, found, missing = filter_entries(entries, cited)
        assert len(filtered) == 2
        assert {e.key for e in filtered} == {"a", "c"}
        assert found == {"a", "c"}
        assert missing == set()

    def test_missing_citations(self):
        """Test detection of missing citations."""
        entries = [BibEntry("article", "a", "@article{a, title={A}}")]
        cited = {"a", "b", "c"}
        filtered, found, missing = filter_entries(entries, cited)
        assert len(filtered) == 1
        assert found == {"a"}
        assert missing == {"b", "c"}

    def test_case_insensitive(self):
        """Test case-insensitive matching (default)."""
        entries = [BibEntry("article", "MyKey", "@article{MyKey, title={A}}")]
        cited = {"mykey"}
        filtered, found, missing = filter_entries(entries, cited, case_sensitive=False)
        assert len(filtered) == 1

    def test_case_sensitive(self):
        """Test case-sensitive matching."""
        entries = [BibEntry("article", "MyKey", "@article{MyKey, title={A}}")]
        cited = {"mykey"}
        filtered, found, missing = filter_entries(entries, cited, case_sensitive=True)
        assert len(filtered) == 0
        assert missing == {"mykey"}

    def test_empty_citations(self):
        """Test with no citations."""
        entries = [BibEntry("article", "a", "@article{a, title={A}}")]
        filtered, found, missing = filter_entries(entries, set())
        assert len(filtered) == 0


# ------------- Write Tests -------------


class TestWriteBibFile:
    """Tests for write_bib_file function."""

    def test_write_entries(self):
        """Test writing entries to file."""
        entries = [
            BibEntry("article", "a", "@article{a,\n  title={A}\n}"),
            BibEntry("book", "b", "@book{b,\n  title={B}\n}"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            tmp_path = f.name

        try:
            write_bib_file(entries, tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
            assert "@article{a," in content
            assert "@book{b," in content
        finally:
            os.unlink(tmp_path)


# ------------- Integration Tests -------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, main_tex_path, sample_bib_path):
        """Test complete workflow: extract citations, parse bib, filter."""
        # Extract citations from .tex
        citations = extract_citations_from_file(main_tex_path)
        assert len(citations) > 0

        # Parse bibliography
        entries = parse_bib_file(sample_bib_path)
        assert len(entries) > 0

        # Filter
        filtered, found, missing = filter_entries(entries, citations)
        assert len(filtered) > 0
        assert len(found) > 0

    def test_project_extraction(self, fixtures_dir, sample_bib_path):
        """Test extracting citations from entire project."""
        citations, files = extract_citations_from_project([fixtures_dir], recursive=True)
        assert len(citations) > 0
        assert len(files) > 0

        entries = parse_bib_file(sample_bib_path)
        filtered, found, missing = filter_entries(entries, citations)
        assert len(filtered) > 0

    def test_round_trip(self, sample_bib_path):
        """Test that parsing and writing preserves content."""
        entries = parse_bib_file(sample_bib_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            tmp_path = f.name

        try:
            write_bib_file(entries, tmp_path)

            # Re-parse the written file
            reparsed = parse_bib_file(tmp_path)
            assert len(reparsed) == len(entries)
            assert {e.key for e in reparsed} == {e.key for e in entries}
        finally:
            os.unlink(tmp_path)


# ------------- Edge Cases -------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_key_with_special_chars(self):
        """Test citation keys with underscores and hyphens."""
        text = r"\cite{my_key_2020, another-key-2021}"
        citations = extract_citations_from_text(text)
        assert citations == {"my_key_2020", "another-key-2021"}

    def test_entry_with_url_braces(self):
        """Test entry with URL containing special characters."""
        content = r"@article{key, url = {https://example.com/path?a=1&b=2}}"
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert "https://example.com" in entries[0].raw_content

    def test_entry_with_math_braces(self):
        """Test entry with math notation containing braces."""
        content = r"@article{key, title = {The $O(n^{2})$ Algorithm}}"
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert "O(n^{2})" in entries[0].raw_content

    def test_consecutive_entries_no_newline(self):
        """Test parsing entries without blank lines between them."""
        content = "@article{a, title={A}}@book{b, title={B}}"
        entries = parse_bib_string(content)
        assert len(entries) == 2

    def test_unicode_in_entry(self):
        """Test entry with Unicode characters."""
        content = "@article{key, author = {Schölkopf, Bernhard}}"
        entries = parse_bib_string(content)
        assert len(entries) == 1
        assert "Schölkopf" in entries[0].raw_content


# ------------- Multi-Bib File Tests -------------


class TestParseBibFiles:
    """Tests for parse_bib_files function (multi-file support)."""

    def test_single_file(self, sample_bib_path):
        """Test parsing single file (backward compatibility)."""
        entries, duplicates = parse_bib_files([sample_bib_path])
        assert len(entries) > 0
        keys = {e.key for e in entries}
        assert "kingma2015adam" in keys

    def test_multiple_files(self):
        """Test parsing multiple files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f1:
            f1.write("@article{key1, title={Title 1}}\n")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f2:
            f2.write("@article{key2, title={Title 2}}\n")
            path2 = f2.name

        try:
            entries, duplicates = parse_bib_files([path1, path2])
            assert len(entries) == 2
            keys = {e.key for e in entries}
            assert keys == {"key1", "key2"}
            assert len(duplicates) == 0
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_duplicate_keys_detected(self):
        """Test that duplicate keys across files are detected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f1:
            f1.write("@article{duplicate, title={Title 1}}\n")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f2:
            f2.write("@article{duplicate, title={Title 2}}\n")
            path2 = f2.name

        try:
            entries, duplicates = parse_bib_files([path1, path2])
            # Both entries are returned
            assert len(entries) == 2
            # Duplicate is tracked
            assert "duplicate" in duplicates
            assert len(duplicates["duplicate"]) == 2
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_duplicate_keys_case_insensitive(self):
        """Test that duplicate detection is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f1:
            f1.write("@article{MyKey, title={Title 1}}\n")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f2:
            f2.write("@article{mykey, title={Title 2}}\n")
            path2 = f2.name

        try:
            entries, duplicates = parse_bib_files([path1, path2])
            # Both entries are returned
            assert len(entries) == 2
            # Duplicate is tracked (case-insensitive)
            assert "mykey" in duplicates
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_three_files_merged(self):
        """Test merging three files."""
        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
                f.write(f"@article{{key{i}, title={{Title {i}}}}}\n")
                paths.append(f.name)

        try:
            entries, duplicates = parse_bib_files(paths)
            assert len(entries) == 3
            keys = {e.key for e in entries}
            assert keys == {"key0", "key1", "key2"}
        finally:
            for path in paths:
                os.unlink(path)

    def test_empty_file_in_list(self):
        """Test handling of empty file in the list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f1:
            f1.write("@article{key1, title={Title 1}}\n")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f2:
            # Empty file
            path2 = f2.name

        try:
            entries, duplicates = parse_bib_files([path1, path2])
            assert len(entries) == 1
            assert entries[0].key == "key1"
        finally:
            os.unlink(path1)
            os.unlink(path2)
