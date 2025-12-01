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
        """Test basic \\cite command with Adam optimizer paper."""
        text = r"Modern optimizers \cite{kingma2015adam} improve training."
        citations = extract_citations_from_text(text)
        assert citations == {"kingma2015adam"}

    def test_citep(self):
        """Test \\citep command with Transformer paper."""
        text = r"Self-attention revolutionized NLP \citep{vaswani2017attention}."
        citations = extract_citations_from_text(text)
        assert citations == {"vaswani2017attention"}

    def test_citet(self):
        """Test \\citet command with ResNet paper."""
        text = r"\citet{he2016resnet} introduced residual connections."
        citations = extract_citations_from_text(text)
        assert citations == {"he2016resnet"}

    def test_multiple_citations_single_command(self):
        """Test multiple citations with foundational ML papers."""
        text = r"See \cite{kingma2015adam, vaswani2017attention, he2016resnet}."
        citations = extract_citations_from_text(text)
        assert citations == {"kingma2015adam", "vaswani2017attention", "he2016resnet"}

    def test_multiple_citations_with_spaces(self):
        """Test multiple citations with varying whitespace."""
        text = r"See \cite{goodfellow2014gan,ho2020ddpm , chen2020simclr}."
        citations = extract_citations_from_text(text)
        assert citations == {"goodfellow2014gan", "ho2020ddpm", "chen2020simclr"}

    def test_citation_with_optional_arg(self):
        """Test citation with optional argument using Deep Learning book."""
        text = r"According to \cite[Chapter 9]{goodfellow2016deeplearning}, CNNs are key."
        citations = extract_citations_from_text(text)
        assert citations == {"goodfellow2016deeplearning"}

    def test_citation_with_two_optional_args(self):
        """Test citation with pre and post optional arguments."""
        text = r"As noted \citep[see][Section 3]{vaswani2017attention}."
        citations = extract_citations_from_text(text)
        assert citations == {"vaswani2017attention"}

    def test_nocite(self):
        """Test \\nocite command with Deep Learning book."""
        text = r"\nocite{goodfellow2016deeplearning}"
        citations = extract_citations_from_text(text)
        assert citations == {"goodfellow2016deeplearning"}

    def test_textcite(self):
        """Test biblatex \\textcite command with BERT paper."""
        text = r"\textcite{devlin2019bert} showed bidirectional pretraining works."
        citations = extract_citations_from_text(text)
        assert citations == {"devlin2019bert"}

    def test_parencite(self):
        """Test biblatex \\parencite command with ViT paper."""
        text = r"Vision Transformers work well \parencite{dosovitskiy2021vit}."
        citations = extract_citations_from_text(text)
        assert citations == {"dosovitskiy2021vit"}

    def test_autocite(self):
        """Test biblatex \\autocite command with SimCLR paper."""
        text = r"Contrastive learning is effective \autocite{chen2020simclr}."
        citations = extract_citations_from_text(text)
        assert citations == {"chen2020simclr"}

    def test_fullcite(self):
        """Test \\fullcite command with CLIP paper."""
        text = r"\fullcite{radford2021clip}"
        citations = extract_citations_from_text(text)
        assert citations == {"radford2021clip"}

    def test_footcite(self):
        """Test \\footcite command with LeNet/CNN paper."""
        text = r"See footnote\footcite{lecun1998cnn}."
        citations = extract_citations_from_text(text)
        assert citations == {"lecun1998cnn"}

    def test_starred_cite(self):
        """Test starred citation commands with LLM papers."""
        text = r"\cite*{brown2020gpt3} and \citep*{touvron2023llama}"
        citations = extract_citations_from_text(text)
        assert citations == {"brown2020gpt3", "touvron2023llama"}

    def test_citeauthor(self):
        """Test \\citeauthor command with dropout paper."""
        text = r"\citeauthor{srivastava2014dropout} introduced dropout."
        citations = extract_citations_from_text(text)
        assert citations == {"srivastava2014dropout"}

    def test_citeyear(self):
        """Test \\citeyear command with batch normalization paper."""
        text = r"Published in \citeyear{ioffe2015batchnorm}."
        citations = extract_citations_from_text(text)
        assert citations == {"ioffe2015batchnorm"}

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
        """Test multiple citation commands with various ML papers."""
        text = r"""
        First \cite{kingma2015adam}, then \citep{vaswani2017attention}.
        Also \citet{he2016resnet} and \textcite{goodfellow2016deeplearning}.
        """
        citations = extract_citations_from_text(text)
        assert citations == {"kingma2015adam", "vaswani2017attention", "he2016resnet", "goodfellow2016deeplearning"}


class TestExtractCitationsFromFile:
    """Tests for extract_citations_from_file function."""

    def test_main_tex_file(self, main_tex_path):
        """Test extraction from main.tex fixture with real ML papers."""
        citations = extract_citations_from_file(main_tex_path)
        # Should find these ML papers from main.tex:
        # - goodfellow2016deeplearning (Deep Learning book)
        # - kingma2015adam (Adam optimizer)
        # - vaswani2017attention (Transformer)
        # - he2016resnet (ResNet)
        # - ioffe2015batchnorm (Batch Normalization)
        # - goodfellow2014gan (GANs)
        # - ho2020ddpm (Diffusion models)
        # - brown2020gpt3 (GPT-3)
        # - touvron2023llama (LLaMA)
        # - srivastava2014dropout (Dropout)
        # Should NOT find: should_be_ignored, also_ignored (in comments)
        assert "goodfellow2016deeplearning" in citations
        assert "kingma2015adam" in citations
        assert "vaswani2017attention" in citations
        assert "he2016resnet" in citations
        assert "ioffe2015batchnorm" in citations
        assert "goodfellow2014gan" in citations
        assert "ho2020ddpm" in citations
        assert "brown2020gpt3" in citations
        assert "touvron2023llama" in citations
        assert "srivastava2014dropout" in citations
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
        """Test extraction from single file with ML papers."""
        citations, files = extract_citations_from_project([main_tex_path])
        assert len(files) == 1
        assert "kingma2015adam" in citations
        assert "vaswani2017attention" in citations

    def test_directory_recursive(self, fixtures_dir):
        """Test extraction from directory recursively."""
        citations, files = extract_citations_from_project([fixtures_dir], recursive=True)
        assert len(files) == 2
        # Citations from main.tex
        assert "kingma2015adam" in citations
        assert "goodfellow2014gan" in citations
        # Citations from subdir/appendix.tex
        assert "devlin2019bert" in citations
        assert "dosovitskiy2021vit" in citations
        assert "nonexistent_future_paper_2099" in citations

    def test_directory_non_recursive(self, fixtures_dir):
        """Test extraction from directory non-recursively."""
        citations, files = extract_citations_from_project([fixtures_dir], recursive=False)
        assert len(files) == 1
        # Should not include appendix.tex citations
        assert "nonexistent_future_paper_2099" not in citations
        assert "devlin2019bert" not in citations


# ------------- Bibliography Filtering Tests -------------


class TestFilterBibliography:
    """Tests for filter_bibliography function."""

    def test_filter_basic(self, sample_bib_db):
        """Test basic filtering with Adam and Transformer papers."""
        cited_keys = {"kingma2015adam", "vaswani2017attention"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 2
        assert found == {"kingma2015adam", "vaswani2017attention"}
        assert missing == set()

    def test_filter_with_missing(self, sample_bib_db):
        """Test filtering with missing citations."""
        cited_keys = {"kingma2015adam", "nonexistent_future_paper_2099"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 1
        assert "kingma2015adam" in found
        assert "nonexistent_future_paper_2099" in missing

    def test_filter_case_insensitive(self, sample_bib_db):
        """Test case-insensitive matching (default)."""
        cited_keys = {"KINGMA2015ADAM", "Vaswani2017Attention"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys, case_sensitive=False)

        assert len(filtered_db.entries) == 2

    def test_filter_case_sensitive(self, sample_bib_db):
        """Test case-sensitive matching."""
        cited_keys = {"KINGMA2015ADAM", "vaswani2017attention"}
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys, case_sensitive=True)

        # KINGMA2015ADAM won't match kingma2015adam
        assert len(filtered_db.entries) == 1
        assert "KINGMA2015ADAM" in missing

    def test_filter_all_entries(self, sample_bib_db):
        """Test filtering with all entries cited (16 ML papers)."""
        cited_keys = {
            "kingma2015adam",
            "vaswani2017attention",
            "devlin2019bert",
            "he2016resnet",
            "dosovitskiy2021vit",
            "goodfellow2014gan",
            "ho2020ddpm",
            "chen2020simclr",
            "radford2021clip",
            "ioffe2015batchnorm",
            "srivastava2014dropout",
            "lecun1998cnn",
            "goodfellow2016deeplearning",
            "brown2020gpt3",
            "touvron2023llama",
        }
        filtered_db, found, missing = filter_bibliography(sample_bib_db, cited_keys)

        assert len(filtered_db.entries) == 15
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
        """Test loading a .bib file with real ML papers."""
        loader = BibLoader()
        db = loader.load_file(sample_bib_path)

        assert len(db.entries) == 15  # 15 real ML papers
        keys = {entry["ID"] for entry in db.entries}
        assert "kingma2015adam" in keys
        assert "vaswani2017attention" in keys
        assert "he2016resnet" in keys
        assert "goodfellow2016deeplearning" in keys

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
        """Test serializing to string with real ML papers."""
        writer = BibWriter()
        output = writer.dumps(sample_bib_db)

        assert "@inproceedings{kingma2015adam" in output
        assert "@inproceedings{vaswani2017attention" in output
        assert "@book{goodfellow2016deeplearning" in output

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
        """Test complete workflow: extract citations -> filter bibliography with ML papers."""
        # Extract citations
        citations, files = extract_citations_from_project([fixtures_dir], recursive=True)

        # Load bibliography
        loader = BibLoader()
        bib_db = loader.load_file(sample_bib_path)

        # Filter bibliography
        filtered_db, found, missing = filter_bibliography(bib_db, citations)

        # Verify results
        assert len(files) == 2  # main.tex and appendix.tex
        assert len(found) == 15  # All 15 ML papers should be found
        assert "nonexistent_future_paper_2099" in missing  # Missing entry from appendix.tex

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
        text = r"Improves accuracy by 50\% \cite{kingma2015adam}"
        citations = extract_citations_from_text(text)
        assert citations == {"kingma2015adam"}

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
        """Test citation command split across lines with Transformer paper."""
        text = r"""
        \cite{
            vaswani2017attention
        }
        """
        citations = extract_citations_from_text(text)
        assert "vaswani2017attention" in citations

    def test_cite_in_math_mode(self):
        """Test citation within text containing math."""
        text = r"Given $\mathcal{L} = -\log p(x)$ as shown in \cite{ho2020ddpm}."
        citations = extract_citations_from_text(text)
        assert citations == {"ho2020ddpm"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
