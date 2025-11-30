"""Tests for the Detector class."""

from __future__ import annotations


class TestDetectorArxiv:
    """Tests for arXiv preprint detection."""

    def test_detect_arxiv_url(self, detector, make_entry):
        """Detect preprint by arXiv URL."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert detection.arxiv_id.startswith("2001.01234")
        assert "arXiv" in detection.reason

    def test_detect_arxiv_url_with_version(self, detector, make_entry):
        """Detect preprint by arXiv URL with version number."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234v3",
            journal="arXiv preprint",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert "2001.01234" in detection.arxiv_id

    def test_detect_arxiv_pdf_url(self, detector, make_entry):
        """Detect preprint by arXiv PDF URL."""
        entry = make_entry(
            url="https://arxiv.org/pdf/2001.01234.pdf",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert "2001.01234" in detection.arxiv_id

    def test_detect_arxiv_eprint_field(self, detector, make_entry):
        """Detect preprint by eprint field."""
        entry = make_entry(
            eprint="2001.01234",
            archiveprefix="arXiv",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert detection.arxiv_id == "2001.01234"

    def test_detect_arxiv_journal_field(self, detector, make_entry):
        """Detect preprint by journal field containing arXiv."""
        entry = make_entry(
            journal="arXiv preprint arXiv:2001.01234",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint

    def test_detect_arxiv_note_field(self, detector, make_entry):
        """Detect preprint by note field containing arXiv."""
        entry = make_entry(
            note="Available on arXiv:2001.01234",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert "2001.01234" in detection.arxiv_id

    def test_detect_arxiv_doi_pattern(self, detector, make_entry):
        """Detect preprint by arXiv DOI pattern."""
        entry = make_entry(
            doi="10.48550/arxiv.2001.01234",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert detection.doi == "10.48550/arxiv.2001.01234"


class TestDetectorBiorxiv:
    """Tests for bioRxiv preprint detection."""

    def test_detect_biorxiv_doi(self, detector, make_entry):
        """Detect preprint by bioRxiv DOI pattern."""
        entry = make_entry(
            doi="10.1101/2020.01.01.123456",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert detection.doi.startswith("10.1101")

    def test_detect_biorxiv_journal(self, detector, make_entry):
        """Detect preprint by journal field containing bioRxiv."""
        entry = make_entry(
            journal="bioRxiv",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint

    def test_detect_medrxiv_journal(self, detector, make_entry):
        """Detect preprint by journal field containing medRxiv."""
        entry = make_entry(
            journal="medRxiv",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint


class TestDetectorEntryType:
    """Tests for detection based on entry type."""

    def test_detect_unpublished_type(self, detector, make_entry):
        """Detect preprint by unpublished entry type."""
        entry = make_entry(ENTRYTYPE="unpublished")
        detection = detector.detect(entry)
        assert detection.is_preprint
        assert "entrytype" in detection.reason.lower()

    def test_detect_misc_type(self, detector, make_entry):
        """Detect preprint by misc entry type."""
        entry = make_entry(ENTRYTYPE="misc")
        detection = detector.detect(entry)
        assert detection.is_preprint

    def test_detect_article_with_preprint_note(self, detector, make_entry):
        """Detect preprint by article with preprint mention in note."""
        entry = make_entry(
            ENTRYTYPE="article",
            journal="Some Journal",
            note="This is a preprint",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint


class TestDetectorNonPreprint:
    """Tests for non-preprint detection."""

    def test_not_preprint_published_article(self, detector, make_entry):
        """Do not detect a properly published article as preprint."""
        entry = make_entry(
            ENTRYTYPE="article",
            journal="Journal of Machine Learning",
            volume="42",
            pages="1-20",
            doi="10.1000/jml.2021.001",
        )
        detection = detector.detect(entry)
        assert not detection.is_preprint

    def test_not_preprint_book(self, detector, make_entry):
        """Do not detect a book as preprint."""
        entry = make_entry(
            ENTRYTYPE="book",
            publisher="Example Publisher",
            title="A Great Book",
        )
        detection = detector.detect(entry)
        assert not detection.is_preprint

    def test_not_preprint_inproceedings(self, detector, make_entry):
        """Do not detect conference paper as preprint."""
        entry = make_entry(
            ENTRYTYPE="inproceedings",
            booktitle="Proceedings of the Conference",
            pages="100-110",
        )
        detection = detector.detect(entry)
        assert not detection.is_preprint


class TestDetectorEdgeCases:
    """Edge cases for detection."""

    def test_detect_empty_entry(self, detector):
        """Handle empty entry gracefully."""
        entry = {"ENTRYTYPE": "article", "ID": "empty"}
        detection = detector.detect(entry)
        # Should not crash
        assert isinstance(detection.is_preprint, bool)

    def test_detect_mixed_signals(self, detector, make_entry):
        """Handle entry with both preprint and published signals."""
        entry = make_entry(
            journal="Nature",  # Published journal
            url="https://arxiv.org/abs/2001.01234",  # But has arXiv URL
            volume="500",
            pages="1-10",
        )
        detection = detector.detect(entry)
        # arXiv URL should trigger detection
        assert detection.is_preprint

    def test_detect_case_insensitive(self, detector, make_entry):
        """Detection should be case insensitive."""
        entry = make_entry(
            journal="ARXIV PREPRINT",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint

    def test_detect_old_arxiv_format(self, detector, make_entry):
        """Detect old-style arXiv IDs (pre-2007)."""
        entry = make_entry(
            eprint="hep-th/9901001",
            archiveprefix="arXiv",
        )
        detection = detector.detect(entry)
        assert detection.is_preprint
