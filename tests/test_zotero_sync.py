"""Tests for the Zotero sync module (bibtex-update --zotero integration)."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Create mock pyzotero module before importing zotero_sync
mock_pyzotero = MagicMock()
mock_zotero_module = MagicMock()
mock_pyzotero.zotero = mock_zotero_module
sys.modules["pyzotero"] = mock_pyzotero
sys.modules["pyzotero.zotero"] = mock_zotero_module

from bibtex_updater import PublishedRecord  # noqa: E402
from bibtex_updater.zotero_sync import (  # noqa: E402
    ZoteroSyncer,
    ZoteroSyncResult,
    print_zotero_sync_summary,
)

# ------------- Fixtures -------------


@pytest.fixture
def make_bib_entry():
    """Factory fixture for creating BibTeX entries."""

    def _make_entry(**kwargs) -> dict[str, Any]:
        entry = {
            "ENTRYTYPE": "article",
            "ID": kwargs.pop("ID", "testkey"),
            "title": "Example Title",
            "author": "Doe, Jane and Smith, John",
            "year": "2020",
        }
        entry.update(kwargs)
        return entry

    return _make_entry


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
                    {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
                    {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
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
            "tags": kwargs.pop("tags", []),
        }
        data.update(kwargs)
        return {"data": data, "key": data["key"], "version": data["version"]}

    return _make_item


@pytest.fixture
def arxiv_bib_entry(make_bib_entry):
    """A typical arXiv preprint bib entry."""
    return make_bib_entry(
        ID="arxiv2020",
        title="Deep Learning for Everything",
        author="Smith, John and Doe, Jane",
        journal="arXiv preprint arXiv:2001.01234",
        url="https://arxiv.org/abs/2001.01234",
        year="2020",
    )


@pytest.fixture
def arxiv_zotero_item(make_zotero_item):
    """A typical arXiv preprint in Zotero format."""
    return make_zotero_item(
        key="ARXIV123",
        title="Deep Learning for Everything",
        creators=[
            {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
            {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
        ],
        publicationTitle="arXiv preprint",
        url="https://arxiv.org/abs/2001.01234",
        date="2020-01-15",
    )


@pytest.fixture
def sample_published_record():
    """Create a sample PublishedRecord for testing."""
    return PublishedRecord(
        doi="10.1000/j.journal.2021.001",
        url="https://doi.org/10.1000/j.journal.2021.001",
        title="Deep Learning for Everything",
        authors=[
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
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


@pytest.fixture
def mock_zotero():
    """Create a mock Zotero client."""
    mock = MagicMock()
    mock.items.return_value = []
    mock.collection_items.return_value = []
    return mock


# ------------- Tests for ZoteroSyncResult -------------


class TestZoteroSyncResult:
    """Tests for the ZoteroSyncResult dataclass."""

    def test_updated_action(self):
        """Test updated action result."""
        result = ZoteroSyncResult(
            bib_key="smith2020",
            zotero_item_key="ZOTKEY123",
            action="updated",
            match_method="arxiv_id",
            message="Updated to Journal of ML",
        )
        assert result.action == "updated"
        assert result.match_method == "arxiv_id"

    def test_no_match_action(self):
        """Test no_match action result."""
        result = ZoteroSyncResult(
            bib_key="smith2020",
            zotero_item_key=None,
            action="no_match",
            match_method=None,
            message="No matching Zotero item found",
        )
        assert result.action == "no_match"
        assert result.zotero_item_key is None

    def test_already_published_action(self):
        """Test already_published action result."""
        result = ZoteroSyncResult(
            bib_key="smith2020",
            zotero_item_key="ZOTKEY123",
            action="already_published",
            match_method="arxiv_id",
            message="Zotero item is already published",
        )
        assert result.action == "already_published"

    def test_error_action(self):
        """Test error action result."""
        result = ZoteroSyncResult(
            bib_key="smith2020",
            zotero_item_key="ZOTKEY123",
            action="error",
            match_method="arxiv_id",
            message="API error occurred",
        )
        assert result.action == "error"


# ------------- Tests for ZoteroSyncer Matching -------------


class TestZoteroSyncerMatching:
    """Tests for ZoteroSyncer matching logic."""

    def test_match_by_arxiv_id(self, make_bib_entry, make_zotero_item, mock_zotero):
        """Matching by arXiv ID should work correctly."""
        bib_entry = make_bib_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )

        zotero_item = make_zotero_item(
            key="ZOTKEY1",
            url="https://arxiv.org/abs/2001.01234",
            publicationTitle="arXiv preprint",
        )

        mock_zotero.items.return_value = [zotero_item]

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = [zotero_item]
            zotero_item["_arxiv_id"] = "2001.01234"

            match, method = syncer.find_match(bib_entry, arxiv_id="2001.01234")

            assert match is not None
            assert match["data"]["key"] == "ZOTKEY1"
            assert method == "arxiv_id"

    def test_match_by_arxiv_id_with_version(self, make_bib_entry, make_zotero_item, mock_zotero):
        """arXiv ID matching should ignore version suffix."""
        bib_entry = make_bib_entry(
            url="https://arxiv.org/abs/2001.01234v2",
        )

        zotero_item = make_zotero_item(
            key="ZOTKEY1",
            url="https://arxiv.org/abs/2001.01234v1",
            publicationTitle="arXiv preprint",
        )

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = [zotero_item]
            zotero_item["_arxiv_id"] = "2001.01234v1"

            match, method = syncer.find_match(bib_entry, arxiv_id="2001.01234v2")

            assert match is not None
            assert method == "arxiv_id"

    def test_match_by_doi(self, make_bib_entry, make_zotero_item, mock_zotero):
        """Matching by DOI should work correctly."""
        bib_entry = make_bib_entry(
            doi="10.1101/2020.01.01.123456",
        )

        zotero_item = make_zotero_item(
            key="ZOTKEY2",
            DOI="10.1101/2020.01.01.123456",
            publicationTitle="bioRxiv",
        )

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = [zotero_item]

            match, method = syncer.find_match(bib_entry, preprint_doi="10.1101/2020.01.01.123456")

            assert match is not None
            assert match["data"]["key"] == "ZOTKEY2"
            assert method == "doi"

    def test_match_by_title_author(self, make_bib_entry, make_zotero_item, mock_zotero):
        """Matching by title+author should work correctly."""
        bib_entry = make_bib_entry(
            title="Deep Learning for Natural Language Processing",
            author="Smith, John and Doe, Jane",
        )

        zotero_item = make_zotero_item(
            key="ZOTKEY3",
            title="Deep Learning for Natural Language Processing",
            creators=[
                {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
                {"creatorType": "author", "firstName": "Jane", "lastName": "Doe"},
            ],
            publicationTitle="arXiv preprint",
        )

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = [zotero_item]

            match, method = syncer.find_match(bib_entry)

            assert match is not None
            assert match["data"]["key"] == "ZOTKEY3"
            assert method == "title_author"

    def test_no_match_when_title_different(self, make_bib_entry, make_zotero_item, mock_zotero):
        """Should not match when titles are too different."""
        bib_entry = make_bib_entry(
            title="Completely Different Title About Quantum Computing",
            author="Smith, John",
        )

        zotero_item = make_zotero_item(
            key="ZOTKEY4",
            title="Deep Learning for Natural Language Processing",
            creators=[
                {"creatorType": "author", "firstName": "John", "lastName": "Smith"},
            ],
            publicationTitle="arXiv preprint",
        )

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = [zotero_item]

            match, method = syncer.find_match(bib_entry)

            assert match is None
            assert method is None

    def test_no_match_when_no_preprints(self, make_bib_entry, mock_zotero):
        """Should return no match when Zotero has no preprints."""
        bib_entry = make_bib_entry(
            title="Some Paper",
            author="Author, Test",
        )

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = []

            match, method = syncer.find_match(bib_entry)

            assert match is None
            assert method is None


# ------------- Tests for ZoteroSyncer Sync Operations -------------


class TestZoteroSyncerSyncOperations:
    """Tests for ZoteroSyncer sync operations."""

    def test_sync_update_dry_run(
        self, arxiv_zotero_item, sample_published_record, mock_zotero
    ):
        """Dry run should not apply updates."""
        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )

            result = syncer.sync_update(
                arxiv_zotero_item,
                sample_published_record,
                bib_key="arxiv2020",
                match_method="arxiv_id",
            )

            assert result.action == "would_update"
            assert result.bib_key == "arxiv2020"
            assert result.match_method == "arxiv_id"
            mock_zotero.update_item.assert_not_called()

    def test_sync_update_applies_changes(
        self, arxiv_zotero_item, sample_published_record, mock_zotero
    ):
        """Non-dry-run should apply updates."""
        mock_zotero.item.return_value = {
            "data": {"key": "ARXIV123", "version": 1, "tags": []}
        }

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=False,
            )

            result = syncer.sync_update(
                arxiv_zotero_item,
                sample_published_record,
                bib_key="arxiv2020",
                match_method="arxiv_id",
            )

            assert result.action == "updated"
            assert mock_zotero.update_item.called

    def test_sync_batch_processes_all_entries(
        self, arxiv_bib_entry, arxiv_zotero_item, sample_published_record, mock_zotero
    ):
        """sync_batch should process all provided entries."""
        mock_zotero.items.return_value = [arxiv_zotero_item]
        arxiv_zotero_item["_arxiv_id"] = "2001.01234"

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = [arxiv_zotero_item]

            updates = [
                (arxiv_bib_entry, "2001.01234", sample_published_record),
            ]
            results = syncer.sync_batch(updates)

            assert len(results) == 1
            assert results[0].action == "would_update"

    def test_sync_batch_handles_no_match(
        self, make_bib_entry, sample_published_record, mock_zotero
    ):
        """sync_batch should handle entries with no Zotero match."""
        bib_entry = make_bib_entry(
            ID="nomatch2020",
            title="Paper Not In Zotero",
        )

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )
            syncer._preprints_cache = []

            updates = [
                (bib_entry, None, sample_published_record),
            ]
            results = syncer.sync_batch(updates)

            assert len(results) == 1
            assert results[0].action == "no_match"
            assert results[0].bib_key == "nomatch2020"


# ------------- Tests for arXiv ID Normalization -------------


class TestArxivIdNormalization:
    """Tests for arXiv ID normalization."""

    def test_normalize_removes_version(self, mock_zotero):
        """Version suffix should be removed during normalization."""
        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )

            assert syncer._normalize_arxiv_id("2001.01234v1") == "2001.01234"
            assert syncer._normalize_arxiv_id("2001.01234v2") == "2001.01234"
            assert syncer._normalize_arxiv_id("2001.01234") == "2001.01234"

    def test_normalize_case_insensitive(self, mock_zotero):
        """Normalization should be case-insensitive."""
        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )

            assert syncer._normalize_arxiv_id("2001.01234V1") == "2001.01234"


# ------------- Tests for Collection Filtering -------------


class TestCollectionFiltering:
    """Tests for Zotero collection filtering."""

    def test_fetch_preprints_with_collection(self, make_zotero_item, mock_zotero):
        """fetch_preprints should use collection_items when collection specified."""
        zotero_item = make_zotero_item(
            key="ZOTKEY1",
            url="https://arxiv.org/abs/2001.01234",
            publicationTitle="arXiv preprint",
        )
        mock_zotero.collection_items.return_value = [zotero_item]

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                collection_key="COLL123",
                dry_run=True,
            )

            preprints = syncer.fetch_preprints()

            mock_zotero.collection_items.assert_called_once()
            assert len(preprints) == 1

    def test_fetch_preprints_without_collection(self, make_zotero_item, mock_zotero):
        """fetch_preprints should use items when no collection specified."""
        zotero_item = make_zotero_item(
            key="ZOTKEY1",
            url="https://arxiv.org/abs/2001.01234",
            publicationTitle="arXiv preprint",
        )
        mock_zotero.items.return_value = [zotero_item]

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                collection_key=None,
                dry_run=True,
            )

            preprints = syncer.fetch_preprints()

            mock_zotero.items.assert_called_once()
            assert len(preprints) == 1


# ------------- Tests for Summary Output -------------


class TestSummaryOutput:
    """Tests for summary output."""

    def test_print_summary_counts(self, capsys):
        """Summary should count results correctly."""
        import logging

        logger = logging.getLogger("test_summary")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        results = [
            ZoteroSyncResult("key1", "ZOT1", "updated", "arxiv_id"),
            ZoteroSyncResult("key2", "ZOT2", "updated", "doi"),
            ZoteroSyncResult("key3", None, "no_match", None),
            ZoteroSyncResult("key4", "ZOT4", "already_published", "title_author"),
            ZoteroSyncResult("key5", "ZOT5", "error", "arxiv_id", "API error"),
        ]

        print_zotero_sync_summary(results, logger)

        captured = capsys.readouterr()
        # Output goes to stderr via logging
        output = captured.err
        assert "Matched & updated:  2" in output
        assert "No Zotero match:    1" in output
        assert "Already published:  1" in output
        assert "Errors:             1" in output


# ------------- Tests for Error Handling -------------


class TestErrorHandling:
    """Tests for error handling."""

    def test_sync_update_handles_exception(
        self, arxiv_zotero_item, sample_published_record, mock_zotero
    ):
        """sync_update should handle exceptions gracefully."""
        # Create a proper PreConditionFailed exception class
        class PreConditionFailed(Exception):
            pass

        # Set it up in the mock module BEFORE using the syncer
        mock_zotero_module.PreConditionFailed = PreConditionFailed

        # Use RuntimeError which is a proper exception class
        mock_zotero.update_item.side_effect = RuntimeError("API error")
        mock_zotero.item.return_value = {
            "data": {"key": "ARXIV123", "version": 1, "tags": []}
        }

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=False,
            )

            result = syncer.sync_update(
                arxiv_zotero_item,
                sample_published_record,
                bib_key="arxiv2020",
                match_method="arxiv_id",
            )

            assert result.action == "error"
            assert "API error" in result.message


# ------------- Tests for Preprint Cache -------------


class TestPreprintCache:
    """Tests for preprint caching."""

    def test_fetch_preprints_caches_results(self, make_zotero_item, mock_zotero):
        """fetch_preprints should cache results."""
        zotero_item = make_zotero_item(
            key="ZOTKEY1",
            url="https://arxiv.org/abs/2001.01234",
            publicationTitle="arXiv preprint",
        )
        mock_zotero.items.return_value = [zotero_item]

        with patch("pyzotero.zotero.Zotero", return_value=mock_zotero):
            syncer = ZoteroSyncer(
                library_id="123456",
                api_key="fake_key",
                dry_run=True,
            )

            # First call
            preprints1 = syncer.fetch_preprints()
            # Second call
            preprints2 = syncer.fetch_preprints()

            # Should only call API once
            assert mock_zotero.items.call_count == 1
            assert preprints1 == preprints2
