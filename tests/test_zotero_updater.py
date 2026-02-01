"""Tests for the Zotero updater script."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the real exception before mocking pyzotero
try:
    from pyzotero.zotero_errors import PreConditionFailedError
except ImportError:
    # Create a mock exception if pyzotero is not installed
    class PreConditionFailedError(Exception):
        pass

# Mock pyzotero before importing zotero module
with patch.dict("sys.modules", {"pyzotero": MagicMock(), "pyzotero.zotero": MagicMock()}):
    from bibtex_updater.zotero import (
        UpdateResult,
        ZoteroPrePrintUpdater,
        is_zotero_preprint,
        published_record_to_zotero_update,
        zotero_to_bibtex_entry,
    )

from bibtex_updater import PublishedRecord

# ------------- Fixtures -------------


@pytest.fixture
def make_zotero_item():
    """Factory fixture for creating Zotero items."""

    def _make_item(**kwargs) -> dict[str, Any]:
        data = {
            "key": kwargs.pop("key", "TESTKEY123"),
            "version": kwargs.pop("version", 1),
            "itemType": kwargs.pop("itemType", "journalArticle"),
            "title": kwargs.pop("title", "Test Title"),
            "creators": kwargs.pop(
                "creators",
                [
                    {"creatorType": "author", "firstName": "John", "lastName": "Doe"},
                    {"creatorType": "author", "firstName": "Jane", "lastName": "Smith"},
                ],
            ),
            "publicationTitle": kwargs.pop("publicationTitle", ""),
            "date": kwargs.pop("date", "2020"),
            "DOI": kwargs.pop("DOI", ""),
            "url": kwargs.pop("url", ""),
            "volume": kwargs.pop("volume", ""),
            "issue": kwargs.pop("issue", ""),
            "pages": kwargs.pop("pages", ""),
            "extra": kwargs.pop("extra", ""),
            "abstractNote": kwargs.pop("abstractNote", ""),
            "tags": kwargs.pop("tags", []),
        }
        data.update(kwargs)
        return {"data": data, "key": data["key"], "version": data["version"]}

    return _make_item


@pytest.fixture
def arxiv_zotero_item(make_zotero_item):
    """A typical arXiv preprint in Zotero format."""
    return make_zotero_item(
        key="ARXIV123",
        title="Deep Learning for Everything",
        publicationTitle="arXiv preprint",
        url="https://arxiv.org/abs/2001.01234",
        date="2020-01-15",
    )


@pytest.fixture
def biorxiv_zotero_item(make_zotero_item):
    """A typical bioRxiv preprint in Zotero format."""
    return make_zotero_item(
        key="BIORXIV456",
        title="Novel Gene Discovery",
        publicationTitle="bioRxiv",
        DOI="10.1101/2020.01.01.123456",
        date="2020-06-01",
    )


@pytest.fixture
def published_zotero_item(make_zotero_item):
    """A published journal article in Zotero format."""
    return make_zotero_item(
        key="PUBLISHED789",
        title="Deep Learning for Everything",
        publicationTitle="Journal of Machine Learning",
        DOI="10.1000/jml.2021.001",
        url="https://doi.org/10.1000/jml.2021.001",
        date="2021-03-01",
        volume="42",
        issue="1",
        pages="1-20",
    )


@pytest.fixture
def sample_published_record():
    """Create a sample PublishedRecord for testing."""
    return PublishedRecord(
        doi="10.1000/j.journal.2021.001",
        url="https://doi.org/10.1000/j.journal.2021.001",
        title="Deep Learning for Everything",
        authors=[
            {"given": "John", "family": "Doe"},
            {"given": "Jane", "family": "Smith"},
        ],
        journal="Journal of Machine Learning",
        publisher="Example Publisher",
        year=2021,
        volume="42",
        number="1",
        pages="1-20",
        type="journal-article",
        method="test",
        confidence=1.0,
    )


# ------------- Tests for is_zotero_preprint -------------


class TestIsZoteroPreprint:
    """Tests for the is_zotero_preprint function."""

    def test_arxiv_url_detected(self, arxiv_zotero_item):
        """Items with arXiv URLs should be detected as preprints."""
        is_preprint, arxiv_id = is_zotero_preprint(arxiv_zotero_item)
        assert is_preprint is True
        assert arxiv_id == "2001.01234"

    def test_biorxiv_journal_detected(self, biorxiv_zotero_item):
        """Items with bioRxiv journal name should be detected as preprints."""
        is_preprint, arxiv_id = is_zotero_preprint(biorxiv_zotero_item)
        assert is_preprint is True

    def test_biorxiv_doi_detected(self, make_zotero_item):
        """Items with bioRxiv DOI pattern should be detected as preprints."""
        item = make_zotero_item(DOI="10.1101/2020.05.01.123456", publicationTitle="")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True

    def test_medrxiv_url_detected(self, make_zotero_item):
        """Items with medRxiv URLs should be detected as preprints."""
        item = make_zotero_item(url="https://www.medrxiv.org/content/10.1101/2020.01.01.123456v1")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True

    def test_arxiv_in_extra_field(self, make_zotero_item):
        """Items with arXiv ID in extra field should be detected."""
        item = make_zotero_item(extra="arXiv:2001.01234", publicationTitle="", url="")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True
        assert arxiv_id == "2001.01234"

    def test_published_not_detected(self, published_zotero_item):
        """Published journal articles should NOT be detected as preprints."""
        is_preprint, arxiv_id = is_zotero_preprint(published_zotero_item)
        assert is_preprint is False
        assert arxiv_id is None

    def test_published_with_real_journal(self, make_zotero_item):
        """Items with real journal names should NOT be detected as preprints."""
        item = make_zotero_item(
            publicationTitle="Nature Machine Intelligence",
            DOI="10.1038/s42256-021-00001-1",
            url="https://doi.org/10.1038/s42256-021-00001-1",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_arxiv_doi_pattern_detected(self, make_zotero_item):
        """Items with arXiv DOI pattern should be detected as preprints."""
        item = make_zotero_item(DOI="10.48550/arxiv.2001.01234", publicationTitle="")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True


# ------------- Tests for zotero_to_bibtex_entry -------------


class TestZoteroToBibtexEntry:
    """Tests for converting Zotero items to BibTeX-like entries."""

    def test_basic_conversion(self, arxiv_zotero_item):
        """Basic fields should be converted correctly."""
        entry = zotero_to_bibtex_entry(arxiv_zotero_item)

        assert entry["ENTRYTYPE"] == "article"
        assert entry["ID"] == "ARXIV123"
        assert entry["title"] == "Deep Learning for Everything"
        assert entry["year"] == "2020"
        assert entry["url"] == "https://arxiv.org/abs/2001.01234"

    def test_author_conversion(self, arxiv_zotero_item):
        """Authors should be converted to BibTeX format."""
        entry = zotero_to_bibtex_entry(arxiv_zotero_item)

        assert "Doe, John" in entry["author"]
        assert "Smith, Jane" in entry["author"]
        assert " and " in entry["author"]

    def test_arxiv_id_extraction_from_extra(self, make_zotero_item):
        """arXiv ID should be extracted from extra field."""
        item = make_zotero_item(extra="arXiv:2001.01234 (v1)")
        entry = zotero_to_bibtex_entry(item)

        assert entry.get("eprint") == "2001.01234"
        assert entry.get("archiveprefix") == "arxiv"

    def test_doi_conversion(self, published_zotero_item):
        """DOI should be converted correctly."""
        entry = zotero_to_bibtex_entry(published_zotero_item)
        assert entry["doi"] == "10.1000/jml.2021.001"

    def test_journal_conversion(self, published_zotero_item):
        """Journal name should be converted correctly."""
        entry = zotero_to_bibtex_entry(published_zotero_item)
        assert entry["journal"] == "Journal of Machine Learning"

    def test_single_name_author(self, make_zotero_item):
        """Handle authors with single name (e.g., Madonna, Cher)."""
        item = make_zotero_item(creators=[{"creatorType": "author", "name": "Madonna"}])
        entry = zotero_to_bibtex_entry(item)
        assert "Madonna" in entry["author"]


# ------------- Tests for published_record_to_zotero_update -------------


class TestPublishedRecordToZoteroUpdate:
    """Tests for converting PublishedRecord to Zotero update payload."""

    def test_basic_update_fields(self, arxiv_zotero_item, sample_published_record):
        """Basic fields should be included in update payload."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)

        assert update["key"] == "ARXIV123"
        assert update["version"] == 1
        assert update["itemType"] == "journalArticle"
        assert update["title"] == "Deep Learning for Everything"
        assert update["publicationTitle"] == "Journal of Machine Learning"
        assert update["DOI"] == "10.1000/j.journal.2021.001"

    def test_authors_converted(self, arxiv_zotero_item, sample_published_record):
        """Authors should be converted to Zotero creator format."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)

        creators = update["creators"]
        assert len(creators) == 2
        assert creators[0]["creatorType"] == "author"
        assert creators[0]["firstName"] == "John"
        assert creators[0]["lastName"] == "Doe"

    def test_arxiv_preserved_in_extra(self, arxiv_zotero_item, sample_published_record):
        """Original arXiv ID should be preserved in extra field."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)

        assert "extra" in update
        assert "arXiv:2001.01234" in update["extra"]

    def test_url_set_to_doi_url(self, arxiv_zotero_item, sample_published_record):
        """URL should be updated to DOI URL."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)

        assert update["url"] == "https://doi.org/10.1000/j.journal.2021.001"

    def test_volume_and_pages_included(self, arxiv_zotero_item, sample_published_record):
        """Volume, issue, and pages should be included."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)

        assert update["volume"] == "42"
        assert update["issue"] == "1"
        assert update["pages"] == "1-20"

    def test_no_duplicate_arxiv_in_extra(self, make_zotero_item, sample_published_record):
        """Should not duplicate arXiv reference if already in extra."""
        item = make_zotero_item(
            url="https://arxiv.org/abs/2001.01234",
            extra="arXiv:2001.01234",
        )
        update = published_record_to_zotero_update(sample_published_record, item)

        # Should only appear once (from original extra, not duplicated)
        assert update.get("extra", "").lower().count("arxiv:2001.01234") <= 1


# ------------- Tests for idempotency - published papers should NOT be overwritten -------------


class TestPublishedPaperNotOverwritten:
    """
    Critical tests ensuring that published papers are NOT overwritten.

    The core requirement is that if a paper is already published (has a real journal,
    non-preprint DOI, etc.), the updater should NOT try to update it.
    """

    def test_published_item_not_detected_as_preprint(self, published_zotero_item):
        """A published item should not be detected as a preprint."""
        is_preprint, arxiv_id = is_zotero_preprint(published_zotero_item)
        assert is_preprint is False
        assert arxiv_id is None

    def test_item_with_real_journal_not_detected(self, make_zotero_item):
        """Item with a real journal should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="Physical Review Letters",
            DOI="10.1103/PhysRevLett.125.010501",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_item_with_nature_journal_not_detected(self, make_zotero_item):
        """Item with Nature journal should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="Nature",
            DOI="10.1038/s41586-021-00001-1",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_item_with_science_journal_not_detected(self, make_zotero_item):
        """Item with Science journal should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="Science",
            DOI="10.1126/science.abc1234",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_item_with_springer_doi_not_detected(self, make_zotero_item):
        """Item with Springer DOI should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="Machine Learning",
            DOI="10.1007/s10994-021-00001-1",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_item_with_elsevier_doi_not_detected(self, make_zotero_item):
        """Item with Elsevier DOI should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="Neural Networks",
            DOI="10.1016/j.neunet.2021.01.001",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_item_with_ieee_doi_not_detected(self, make_zotero_item):
        """Item with IEEE DOI should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="IEEE Transactions on Neural Networks",
            DOI="10.1109/TNNLS.2021.0001234",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_item_with_acm_doi_not_detected(self, make_zotero_item):
        """Item with ACM DOI should not be detected as preprint."""
        item = make_zotero_item(
            publicationTitle="Communications of the ACM",
            DOI="10.1145/3412345.3412346",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False

    def test_conference_paper_not_detected(self, make_zotero_item):
        """Conference papers with DOI should not be detected as preprints."""
        item = make_zotero_item(
            itemType="conferencePaper",
            publicationTitle="Proceedings of NeurIPS 2021",
            DOI="10.5555/3540261.3540262",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False


class TestProcessItemIdempotency:
    """Tests for process_item ensuring idempotency."""

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_process_published_item_skipped(self, mock_zotero_class, published_zotero_item):
        """Processing a published item should return 'skipped' action."""
        mock_zotero = MagicMock()
        mock_zotero_class.return_value = mock_zotero

        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
            verbose=False,
        )

        result = updater.process_item(published_zotero_item)

        assert result.action == "skipped"
        assert "Not detected as preprint" in result.message
        # Verify update_item was never called
        mock_zotero.update_item.assert_not_called()

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_published_item_not_modified(self, mock_zotero_class, make_zotero_item):
        """A published item should not be modified even if it has related arXiv entry."""
        mock_zotero = MagicMock()
        mock_zotero_class.return_value = mock_zotero

        # An item that was already upgraded - has real journal and DOI
        item = make_zotero_item(
            publicationTitle="Journal of Machine Learning Research",
            DOI="10.5555/jmlr.v22.21-001",
            extra="Originally from arXiv:2001.01234",  # Note in extra field
        )

        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=False,
            verbose=False,
        )

        result = updater.process_item(item)

        assert result.action == "skipped"
        mock_zotero.update_item.assert_not_called()


class TestUpdateResultActions:
    """Tests for the UpdateResult action types."""

    def test_update_result_skipped_action(self):
        """Verify skipped action for non-preprints."""
        result = UpdateResult(
            item_key="TEST123",
            title="Test Paper",
            action="skipped",
            message="Not detected as preprint",
            old_journal="Nature",
        )
        assert result.action == "skipped"
        assert result.new_journal is None

    def test_update_result_not_found_action(self):
        """Verify not_found action when no published version exists."""
        result = UpdateResult(
            item_key="TEST123",
            title="Test Paper",
            action="not_found",
            message="No published version found",
            old_journal="arXiv preprint",
        )
        assert result.action == "not_found"

    def test_update_result_updated_action(self):
        """Verify updated action when successfully updated."""
        result = UpdateResult(
            item_key="TEST123",
            title="Test Paper",
            action="updated",
            message="Updated to Journal of ML",
            old_journal="arXiv preprint",
            new_journal="Journal of ML",
            doi="10.1000/test",
            method="crossref",
            confidence=0.95,
        )
        assert result.action == "updated"
        assert result.new_journal == "Journal of ML"


# ------------- Edge case tests -------------


class TestEdgeCases:
    """Edge case tests for the Zotero updater."""

    def test_item_with_empty_fields(self, make_zotero_item):
        """Handle items with empty/missing fields gracefully."""
        item = make_zotero_item(
            publicationTitle="",
            DOI="",
            url="",
            extra="",
        )
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is False
        assert arxiv_id is None

    def test_item_with_none_values(self):
        """Handle items with None values in fields."""
        item = {
            "data": {
                "key": "TEST",
                "version": 1,
                "publicationTitle": None,
                "DOI": None,
                "url": None,
                "extra": None,
            }
        }
        # Should not raise an exception
        is_preprint, arxiv_id = is_zotero_preprint(item)
        # With all None/empty, should not be detected as preprint
        assert is_preprint is False

    def test_arxiv_in_journal_case_insensitive(self, make_zotero_item):
        """arXiv detection in journal name should be case insensitive."""
        item = make_zotero_item(publicationTitle="ARXIV PREPRINT")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True

    def test_biorxiv_case_insensitive(self, make_zotero_item):
        """bioRxiv detection should be case insensitive."""
        item = make_zotero_item(publicationTitle="BIORXIV")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True

    def test_partial_arxiv_url(self, make_zotero_item):
        """Handle partial/malformed arXiv URLs."""
        item = make_zotero_item(url="arxiv.org/2001.01234")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True

    def test_old_arxiv_id_format(self, make_zotero_item):
        """Handle old arXiv ID format (with category prefix)."""
        item = make_zotero_item(url="https://arxiv.org/abs/hep-th/9901001")
        is_preprint, arxiv_id = is_zotero_preprint(item)
        assert is_preprint is True


class TestPreservation:
    """Tests for preservation of metadata during updates."""

    def test_existing_extra_preserved(self, make_zotero_item, sample_published_record):
        """Existing extra field content should be preserved."""
        item = make_zotero_item(
            url="https://arxiv.org/abs/2001.01234",
            extra="Important note about this paper",
        )
        update = published_record_to_zotero_update(sample_published_record, item)

        assert "Important note about this paper" in update.get("extra", "")

    def test_version_preserved(self, arxiv_zotero_item, sample_published_record):
        """Version number should be preserved for conflict detection."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)
        assert update["version"] == arxiv_zotero_item["data"]["version"]

    def test_key_preserved(self, arxiv_zotero_item, sample_published_record):
        """Item key should be preserved."""
        update = published_record_to_zotero_update(sample_published_record, arxiv_zotero_item)
        assert update["key"] == arxiv_zotero_item["data"]["key"]


# ------------- Tests for tag-based workflow -------------


class TestTagBasedWorkflow:
    """Tests for the tag-based chunking workflow."""

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_fetch_preprints_excludes_tags(self, mock_zotero_class):
        """fetch_preprints should exclude items with specified tags."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        # Replace the zot instance with our mock
        mock_zot = MagicMock()
        mock_zot.items.return_value = []
        updater.zot = mock_zot

        updater.fetch_preprints(
            exclude_tags=["preprint-upgraded", "preprint-checked"],
            limit=50,
        )

        # Verify the tag parameter was passed correctly
        mock_zot.items.assert_called_once()
        call_kwargs = mock_zot.items.call_args[1]
        assert "tag" in call_kwargs
        assert "-preprint-upgraded" in call_kwargs["tag"]
        assert "-preprint-checked" in call_kwargs["tag"]

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_fetch_preprints_with_offset(self, mock_zotero_class):
        """fetch_preprints should support pagination via offset."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        mock_zot = MagicMock()
        mock_zot.items.return_value = []
        updater.zot = mock_zot

        updater.fetch_preprints(limit=50, offset=100)

        call_kwargs = mock_zot.items.call_args[1]
        assert call_kwargs["start"] == 100
        assert call_kwargs["limit"] == 50

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_fetch_preprints_combines_tag_and_exclusions(self, mock_zotero_class):
        """fetch_preprints should combine include tag with exclusions."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        mock_zot = MagicMock()
        mock_zot.items.return_value = []
        updater.zot = mock_zot

        updater.fetch_preprints(
            tag="my-collection",
            exclude_tags=["preprint-upgraded"],
            limit=50,
        )

        call_kwargs = mock_zot.items.call_args[1]
        # Should have both include and exclude
        assert "my-collection" in call_kwargs["tag"]
        assert "-preprint-upgraded" in call_kwargs["tag"]


class TestRunModes:
    """Tests for run() modes: normal, recheck, force."""

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_run_normal_mode_excludes_all_tags(self, mock_zotero_class):
        """Normal mode should exclude all tracking tags."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        mock_zot = MagicMock()
        mock_zot.items.return_value = []
        updater.zot = mock_zot

        updater.run(limit=10)

        call_kwargs = mock_zot.items.call_args[1]
        assert "-preprint-upgraded" in call_kwargs["tag"]
        assert "-preprint-checked" in call_kwargs["tag"]
        assert "-preprint-error" in call_kwargs["tag"]

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_run_recheck_mode(self, mock_zotero_class):
        """Recheck mode should target preprint-checked items."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        mock_zot = MagicMock()
        mock_zot.items.return_value = []
        updater.zot = mock_zot

        updater.run(limit=10, recheck=True)

        call_kwargs = mock_zot.items.call_args[1]
        # Should include preprint-checked and exclude preprint-upgraded
        assert "preprint-checked" in call_kwargs["tag"]
        assert "-preprint-upgraded" in call_kwargs["tag"]

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_run_force_mode(self, mock_zotero_class):
        """Force mode should not exclude any tags."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        mock_zot = MagicMock()
        mock_zot.items.return_value = []
        updater.zot = mock_zot

        updater.run(limit=10, force=True)

        call_kwargs = mock_zot.items.call_args[1]
        # Should not have tag exclusions (no 'tag' key or no '-' in tag)
        tag_value = call_kwargs.get("tag")
        assert tag_value is None or "-" not in str(tag_value)


class TestVersionConflictHandling:
    """Tests for version conflict retry logic."""

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_update_item_with_retry_success_first_try(self, mock_zotero_class):
        """Update should succeed on first try when no conflict."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=False,
        )

        mock_zot = MagicMock()
        updater.zot = mock_zot

        payload = {"key": "TEST123", "version": 1, "title": "Test"}
        updater._update_item_with_retry(payload)

        mock_zot.update_item.assert_called_once_with(payload)

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_update_item_with_retry_on_conflict(self, mock_zotero_class):
        """Update should retry with fresh version on conflict."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=False,
        )

        mock_zot = MagicMock()
        # Set up the mock to raise PreConditionFailedError on first call
        mock_zot.update_item.side_effect = [
            PreConditionFailedError("Version conflict"),
            None,  # Success on retry
        ]
        mock_zot.item.return_value = {"data": {"key": "TEST123", "version": 2}}
        updater.zot = mock_zot

        payload = {"key": "TEST123", "version": 1, "title": "Test"}
        updater._update_item_with_retry(payload)

        # Should have called update_item twice
        assert mock_zot.update_item.call_count == 2
        # Second call should have updated version
        second_call_payload = mock_zot.update_item.call_args_list[1][0][0]
        assert second_call_payload["version"] == 2


class TestOutcomeTagging:
    """Tests for outcome-based tagging."""

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_not_found_adds_checked_tag(self, mock_zotero_class, arxiv_zotero_item):
        """Items with no published version should get preprint-checked tag."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=False,
        )

        mock_zot = MagicMock()
        mock_zot.item.return_value = {
            "data": {"key": "ARXIV123", "version": 1, "tags": []}
        }
        updater.zot = mock_zot

        # Mock the resolver to return None (no published version found)
        updater.resolver.resolve = MagicMock(return_value=None)

        result = updater.process_item(arxiv_zotero_item)

        assert result.action == "not_found"
        # Verify _add_tag was effectively called (via update_item)
        # The tag should have been added
        assert mock_zot.update_item.called or mock_zot.item.called

    @patch("bibtex_updater.zotero.zotero.Zotero")
    def test_dry_run_does_not_add_tags(self, mock_zotero_class, arxiv_zotero_item):
        """Dry run should not add any tags."""
        updater = ZoteroPrePrintUpdater(
            library_id="123456",
            api_key="fake_api_key",
            dry_run=True,
        )

        mock_zot = MagicMock()
        updater.zot = mock_zot

        # Mock the resolver to return None
        updater.resolver.resolve = MagicMock(return_value=None)

        result = updater.process_item(arxiv_zotero_item)

        assert result.action == "not_found"
        # In dry run, update_item should not be called for tagging
        mock_zot.update_item.assert_not_called()
