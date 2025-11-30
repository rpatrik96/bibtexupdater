"""
Tests for filter_bibliography.py

Run with: pytest tests/test_filter_bibliography.py -v
"""

import os

# Add parent directory to path for imports
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from filter_bibliography import (
    BibLoader,
    BibWriter,
    extract_citations_from_file,
    extract_citations_from_project,
    extract_citations_from_text,
    filter_bibliography,
    find_tex_files,
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


@pytest.fixture
def sample_bib_db(sample_bib_path):
    """Load sample bibliography database."""
    loader = BibLoader()
    return loader.load_file(sample_bib_path)


# ------------- Citation Extraction Tests -------------


class TestExtractCitationsFromText:
    """Tests for extract_citations_from_text function."""

    def test_simple_cite(self):
        """Test basic \\cite command."""
        text = r"As shown by \cite{smith2020}, this works."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}

    def test_citep(self):
        """Test \\citep command."""
        text = r"This is important \citep{jones2021}."
        citations = extract_citations_from_text(text)
        assert citations == {"jones2021"}

    def test_citet(self):
        """Test \\citet command."""
        text = r"\citet{brown2019} showed this."
        citations = extract_citations_from_text(text)
        assert citations == {"brown2019"}

    def test_multiple_citations_single_command(self):
        """Test multiple citations in one command."""
        text = r"See \cite{smith2020, jones2021, brown2019}."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020", "jones2021", "brown2019"}

    def test_multiple_citations_with_spaces(self):
        """Test multiple citations with varying whitespace."""
        text = r"See \cite{smith2020,jones2021 , brown2019}."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020", "jones2021", "brown2019"}

    def test_citation_with_optional_arg(self):
        """Test citation with optional argument."""
        text = r"According to \cite[p.~5]{smith2020}, this is true."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}

    def test_citation_with_two_optional_args(self):
        """Test citation with pre and post optional arguments."""
        text = r"As noted \citep[see][Chapter 3]{jones2021}."
        citations = extract_citations_from_text(text)
        assert citations == {"jones2021"}

    def test_nocite(self):
        """Test \\nocite command."""
        text = r"\nocite{textbook2018}"
        citations = extract_citations_from_text(text)
        assert citations == {"textbook2018"}

    def test_textcite(self):
        """Test biblatex \\textcite command."""
        text = r"\textcite{smith2020} argues that..."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}

    def test_parencite(self):
        """Test biblatex \\parencite command."""
        text = r"This is shown \parencite{jones2021}."
        citations = extract_citations_from_text(text)
        assert citations == {"jones2021"}

    def test_autocite(self):
        """Test biblatex \\autocite command."""
        text = r"Examples abound \autocite{brown2019}."
        citations = extract_citations_from_text(text)
        assert citations == {"brown2019"}

    def test_fullcite(self):
        """Test \\fullcite command."""
        text = r"\fullcite{textbook2018}"
        citations = extract_citations_from_text(text)
        assert citations == {"textbook2018"}

    def test_footcite(self):
        """Test \\footcite command."""
        text = r"See footnote\footcite{smith2020}."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}

    def test_starred_cite(self):
        """Test starred citation commands."""
        text = r"\cite*{smith2020} and \citep*{jones2021}"
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020", "jones2021"}

    def test_citeauthor(self):
        """Test \\citeauthor command."""
        text = r"\citeauthor{smith2020} wrote this."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}

    def test_citeyear(self):
        """Test \\citeyear command."""
        text = r"Published in \citeyear{jones2021}."
        citations = extract_citations_from_text(text)
        assert citations == {"jones2021"}

    def test_no_citations(self):
        """Test text with no citations."""
        text = "This is just plain text without any citations."
        citations = extract_citations_from_text(text)
        assert citations == set()

    def test_empty_text(self):
        """Test empty text."""
        citations = extract_citations_from_text("")
        assert citations == set()

    def test_multiple_cite_commands(self):
        """Test multiple citation commands in text."""
        text = r"""
        First \cite{smith2020}, then \citep{jones2021}.
        Also \citet{brown2019} and \textcite{textbook2018}.
        """
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020", "jones2021", "brown2019", "textbook2018"}


class TestExtractCitationsFromFile:
    """Tests for extract_citations_from_file function."""

    def test_main_tex_file(self, main_tex_path):
        """Test extraction from main.tex fixture."""
        citations = extract_citations_from_file(main_tex_path)
        # Should find: smith2020, jones2021, brown2019, preprint2022, textbook2018
        # Should NOT find: should_be_ignored, also_ignored (in comments)
        assert "smith2020" in citations
        assert "jones2021" in citations
        assert "brown2019" in citations
        assert "preprint2022" in citations
        assert "textbook2018" in citations
        assert "should_be_ignored" not in citations
        assert "also_ignored" not in citations

    def test_comments_ignored(self, main_tex_path):
        """Test that citations in comments are ignored."""
        citations = extract_citations_from_file(main_tex_path)
        assert "should_be_ignored" not in citations
        assert "also_ignored" not in citations

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        citations = extract_citations_from_file("/nonexistent/path/file.tex")
        assert citations == set()


class TestFindTexFiles:
    """Tests for find_tex_files function."""

    def test_find_in_directory(self, fixtures_dir):
        """Test finding .tex files in directory."""
        files = find_tex_files(fixtures_dir, recursive=False)
        assert len(files) == 1
        assert any("main.tex" in f for f in files)

    def test_find_recursive(self, fixtures_dir):
        """Test recursive search."""
        files = find_tex_files(fixtures_dir, recursive=True)
        assert len(files) == 2
        assert any("main.tex" in f for f in files)
        assert any("appendix.tex" in f for f in files)

    def test_single_file(self, main_tex_path):
        """Test with single file path."""
        files = find_tex_files(main_tex_path)
        assert len(files) == 1
        assert main_tex_path in files

    def test_non_tex_file(self, sample_bib_path):
        """Test with non-.tex file."""
        files = find_tex_files(sample_bib_path)
        assert files == []


class TestExtractCitationsFromProject:
    """Tests for extract_citations_from_project function."""

    def test_single_file(self, main_tex_path):
        """Test extraction from single file."""
        citations, files = extract_citations_from_project([main_tex_path])
        assert len(files) == 1
        assert "smith2020" in citations
        assert "jones2021" in citations

    def test_directory_recursive(self, fixtures_dir):
        """Test extraction from directory recursively."""
        citations, files = extract_citations_from_project([fixtures_dir], recursive=True)
        assert len(files) == 2
        # Citations from main.tex
        assert "smith2020" in citations
        assert "preprint2022" in citations
        # Citations from subdir/appendix.tex
        assert "nonexistent_paper_2099" in citations

    def test_directory_non_recursive(self, fixtures_dir):
        """Test extraction from directory non-recursively."""
        citations, files = extract_citations_from_project([fixtures_dir], recursive=False)
        assert len(files) == 1
        # Should not include appendix.tex citations
        assert "nonexistent_paper_2099" not in citations


# ------------- Bibliography Filtering Tests -------------


class TestFilterBibliography:
    """Tests for filter_bibliography function."""

    def test_filter_basic(self, sample_bib_db):
        """Test basic filtering."""
        cited_keys = {"smith2020", "jones2021"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 2
        assert found == {"smith2020", "jones2021"}
        assert missing == set()

    def test_filter_with_missing(self, sample_bib_db):
        """Test filtering with missing citations."""
        cited_keys = {"smith2020", "nonexistent_key"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 1
        assert "smith2020" in found
        assert "nonexistent_key" in missing

    def test_filter_case_insensitive(self, sample_bib_db):
        """Test case-insensitive matching (default)."""
        cited_keys = {"SMITH2020", "Jones2021"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys, case_sensitive=False)

        assert len(filtered_db.entries) == 2

    def test_filter_case_sensitive(self, sample_bib_db):
        """Test case-sensitive matching."""
        cited_keys = {"SMITH2020", "jones2021"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys, case_sensitive=True)

        # SMITH2020 won't match smith2020
        assert len(filtered_db.entries) == 1
        assert "SMITH2020" in missing

    def test_filter_all_entries(self, sample_bib_db):
        """Test filtering with all entries cited."""
        cited_keys = {"smith2020", "jones2021", "brown2019", "preprint2022", "textbook2018"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 5
        assert missing == set()

    def test_filter_empty_citations(self, sample_bib_db):
        """Test filtering with no citations."""
        cited_keys = set()
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 0
        assert found == set()
        assert missing == set()


# ------------- BibTeX IO Tests -------------


class TestBibLoader:
    """Tests for BibLoader class."""

    def test_load_file(self, sample_bib_path):
        """Test loading a .bib file."""
        loader = BibLoader()
        db = loader.load_file(sample_bib_path)

        assert len(db.entries) == 5
        keys = {entry["ID"] for entry in db.entries}
        assert "smith2020" in keys
        assert "jones2021" in keys

    def test_loads_string(self):
        """Test loading from string."""
        bib_string = """
        @article{test2020,
            title = {Test Article},
            author = {Test Author},
            year = {2020}
        }
        """
        loader = BibLoader()
        db = loader.loads(bib_string)

        assert len(db.entries) == 1
        assert db.entries[0]["ID"] == "test2020"


class TestBibWriter:
    """Tests for BibWriter class."""

    def test_dumps(self, sample_bib_db):
        """Test serializing to string."""
        writer = BibWriter()
        output = writer.dumps(sample_bib_db)

        assert "@article{smith2020" in output
        assert "@inproceedings{jones2021" in output

    def test_dump_to_file(self, sample_bib_db):
        """Test writing to file."""
        writer = BibWriter()

        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False) as f:
            temp_path = f.name

        try:
            writer.dump_to_file(sample_bib_db, temp_path)

            # Verify file was written
            assert os.path.exists(temp_path)

            # Verify content can be read back
            loader = BibLoader()
            loaded_db = loader.load_file(temp_path)
            assert len(loaded_db.entries) == len(sample_bib_db.entries)
        finally:
            os.unlink(temp_path)


# ------------- Integration Tests -------------


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, fixtures_dir, sample_bib_path):
        """Test complete workflow: extract citations -> filter bibliography."""
        # Extract citations
        citations, files = extract_citations_from_project([fixtures_dir], recursive=True)

        # Load bibliography
        loader = BibLoader()
        bib_db = loader.load_file(sample_bib_path)

        # Filter bibliography
        filtered_db, found, missing = filter_bibliography(bib_db, citations)

        # Verify results
        assert len(files) == 2  # main.tex and appendix.tex
        assert len(found) == 5  # All 5 entries should be found
        assert "nonexistent_paper_2099" in missing  # Missing entry from appendix.tex

    def test_write_filtered_bibliography(self, fixtures_dir, sample_bib_path):
        """Test writing filtered bibliography to file."""
        # Extract and filter
        citations, _ = extract_citations_from_project([fixtures_dir], recursive=True)
        loader = BibLoader()
        bib_db = loader.load_file(sample_bib_path)
        filtered_db, _, _ = filter_bibliography(bib_db, citations)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False) as f:
            temp_path = f.name

        try:
            writer = BibWriter()
            writer.dump_to_file(filtered_db, temp_path)

            # Read back and verify
            loaded_db = loader.load_file(temp_path)
            # Compare unique entry IDs
            filtered_ids = {e["ID"] for e in filtered_db.entries}
            loaded_ids = {e["ID"] for e in loaded_db.entries}
            assert filtered_ids == loaded_ids
        finally:
            os.unlink(temp_path)


# ------------- Edge Case Tests -------------


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_escaped_percent_in_tex(self):
        """Test handling of escaped percent signs."""
        text = r"50\% of cases \cite{smith2020}"
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}

    def test_citation_key_with_special_chars(self):
        """Test citation keys with underscores and numbers."""
        text = r"\cite{gresele_incomplete_2019}"
        citations = extract_citations_from_text(text)
        assert citations == {"gresele_incomplete_2019"}

    def test_citation_key_with_hyphen(self):
        """Test citation keys with hyphens."""
        text = r"\cite{zheng_identifiability_2022-1}"
        citations = extract_citations_from_text(text)
        assert citations == {"zheng_identifiability_2022-1"}

    def test_multiline_cite(self):
        """Test citation command split across lines."""
        text = r"""
        \cite{
            smith2020
        }
        """
        citations = extract_citations_from_text(text)
        assert "smith2020" in citations

    def test_cite_in_math_mode(self):
        """Test citation within text containing math."""
        text = r"Given $x = y$ as shown in \cite{smith2020}."
        citations = extract_citations_from_text(text)
        assert citations == {"smith2020"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
