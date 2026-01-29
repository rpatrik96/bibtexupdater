"""Integration tests for the bibtex_updater pipeline."""

from __future__ import annotations

from bibtex_updater import (
    Detector,
    ProcessResult,
    PublishedRecord,
    Updater,
    process_entry,
)


class TestProcessEntry:
    """Tests for the process_entry function."""

    def test_process_non_preprint_unchanged(self, make_entry, detector, fake_resolver, updater, logger):
        """Non-preprint entries should be unchanged."""
        entry = make_entry(
            ENTRYTYPE="article",
            journal="Nature",
            volume="500",
            pages="1-10",
            doi="10.1038/nature12345",
        )
        resolver = fake_resolver(None)

        result = process_entry(entry, detector, resolver, updater, logger)

        assert result.action == "unchanged"
        assert not result.changed
        assert result.original == result.updated

    def test_process_preprint_upgraded(self, make_entry, detector, fake_resolver, updater, logger):
        """Preprint should be upgraded when resolver returns valid record."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        record = PublishedRecord(
            doi="10.1000/j.test.123",
            title="Published Title",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Real Journal",
            year=2021,
            volume="42",
            pages="1-20",
            type="journal-article",
            method="test",
            confidence=1.0,
        )
        resolver = fake_resolver(record)

        result = process_entry(entry, detector, resolver, updater, logger)

        assert result.action == "upgraded"
        assert result.changed
        assert result.updated["journal"] == "Real Journal"
        assert result.updated["doi"] == "10.1000/j.test.123"

    def test_process_preprint_failed_no_match(self, make_entry, detector, fake_resolver, updater, logger):
        """Preprint should fail when no published match found."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        resolver = fake_resolver(None)

        result = process_entry(entry, detector, resolver, updater, logger)

        assert result.action == "failed"
        assert not result.changed
        assert "No reliable published match" in result.message

    def test_process_preprint_failed_not_journal_article(self, make_entry, detector, fake_resolver, updater, logger):
        """Preprint should fail when candidate is not a journal article."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        record = PublishedRecord(
            doi="10.1000/conf.123",
            title="Conference Paper",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Conference Proceedings",
            year=2021,
            type="proceedings-article",  # Not journal-article
        )
        resolver = fake_resolver(record)

        result = process_entry(entry, detector, resolver, updater, logger)

        assert result.action == "failed"
        assert "not a journal-article" in result.message

    def test_process_preprint_failed_insufficient_metadata(self, make_entry, detector, fake_resolver, updater, logger):
        """Preprint should fail when candidate lacks sufficient metadata."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        record = PublishedRecord(
            doi="",  # No DOI
            title="Incomplete Record",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Some Journal",
            year=2021,
            type="journal-article",
            # Missing volume, number, pages, url, AND doi - fails credibility check
        )
        resolver = fake_resolver(record)

        result = process_entry(entry, detector, resolver, updater, logger)

        assert result.action == "failed"
        assert "lacks sufficient metadata" in result.message


class TestPipelineWithMockedResolver:
    """Test full pipeline with mocked components."""

    def test_pipeline_dblp_fallback(self, make_entry, logger):
        """Simulate DBLP fallback when Crossref is unavailable."""
        entry = make_entry(
            ID="dblp1",
            title="Learning Widgets from Data",
            author="Jane Doe and John Smith",
            url="https://arxiv.org/abs/2101.00001",
            journal="arXiv preprint",
        )
        detector = Detector()

        class FakeResolver:
            def __init__(self):
                self.logger = logger

            def resolve(self, entry, detection):
                return PublishedRecord(
                    doi="10.5555/1234567",
                    title=entry["title"],
                    authors=[
                        {"given": "Jane", "family": "Doe"},
                        {"given": "John", "family": "Smith"},
                    ],
                    journal="International Journal of Widgetry",
                    year=2023,
                    volume="10",
                    pages="1-15",
                    type="journal-article",
                    method="DBLP(search)",
                    confidence=0.95,
                )

        updater = Updater(keep_preprint_note=False, rekey=False)
        result = process_entry(entry, detector, FakeResolver(), updater, logger)

        assert result.action == "upgraded"
        assert result.updated.get("journal") == "International Journal of Widgetry"
        assert result.method == "DBLP(search)"
        assert result.confidence == 0.95


class TestEndToEndScenarios:
    """End-to-end test scenarios."""

    def test_arxiv_to_published(self, make_entry, detector, fake_resolver, updater, logger):
        """Complete flow from arXiv preprint to published article."""
        # Original arXiv entry
        entry = make_entry(
            ID="smith2020deep",
            title="Deep Learning for Everything",
            author="Smith, John and Doe, Jane",
            journal="arXiv preprint arXiv:2001.01234",
            year="2020",
            url="https://arxiv.org/abs/2001.01234",
            eprint="2001.01234",
            archiveprefix="arXiv",
            primaryClass="cs.LG",
        )

        # Published version
        record = PublishedRecord(
            doi="10.1000/jml.2021.001",
            title="Deep Learning for Everything",
            authors=[
                {"given": "John", "family": "Smith"},
                {"given": "Jane", "family": "Doe"},
            ],
            journal="Journal of Machine Learning",
            publisher="ML Press",
            year=2021,
            volume="15",
            number="3",
            pages="100-150",
            type="journal-article",
            method="arXiv->Crossref(works)",
            confidence=1.0,
        )
        resolver = fake_resolver(record)

        result = process_entry(entry, detector, resolver, updater, logger)

        # Verify upgrade
        assert result.action == "upgraded"
        assert result.changed

        # Verify updated entry
        updated = result.updated
        assert updated["ID"] == "smith2020deep"  # ID preserved
        assert updated["title"] == "Deep Learning for Everything"
        assert updated["journal"] == "Journal of Machine Learning"
        assert updated["year"] == "2021"
        assert updated["volume"] == "15"
        assert updated["pages"] == "100-150"
        assert updated["doi"] == "10.1000/jml.2021.001"

        # Verify preprint fields removed
        assert "eprint" not in updated
        assert "archiveprefix" not in updated
        assert "primaryClass" not in updated

        # Verify not detected as preprint anymore
        new_detection = detector.detect(updated)
        assert not new_detection.is_preprint

    def test_biorxiv_to_published(self, make_entry, detector, fake_resolver, updater, logger):
        """Complete flow from bioRxiv preprint to published article."""
        entry = make_entry(
            ID="jones2020gene",
            title="Novel Gene Discovery Method",
            author="Jones, Alice and Brown, Bob",
            journal="bioRxiv",
            year="2020",
            doi="10.1101/2020.01.01.123456",
        )

        record = PublishedRecord(
            doi="10.1038/s41586-021-12345-6",
            title="Novel Gene Discovery Method",
            authors=[
                {"given": "Alice", "family": "Jones"},
                {"given": "Bob", "family": "Brown"},
            ],
            journal="Nature",
            year=2021,
            volume="590",
            pages="100-105",
            type="journal-article",
            method="Crossref(relation)",
            confidence=1.0,
        )
        resolver = fake_resolver(record)

        result = process_entry(entry, detector, resolver, updater, logger)

        assert result.action == "upgraded"
        assert result.updated["journal"] == "Nature"
        assert result.updated["doi"] == "10.1038/s41586-021-12345-6"

    def test_keep_preprint_note_flow(self, make_entry, detector, fake_resolver, updater_with_note, logger):
        """Test that preprint note is preserved when option is enabled."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )

        record = PublishedRecord(
            doi="10.1000/j.test.123",
            title="Published Title",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Real Journal",
            year=2021,
            volume="42",
            pages="1-20",
            type="journal-article",
        )

        # Need to get detection first for the note
        detection = detector.detect(entry)
        updated = updater_with_note.update_entry(entry, record, detection)

        assert "note" in updated
        assert "arXiv:2001.01234" in updated["note"]


class TestProcessResultDataclass:
    """Tests for ProcessResult dataclass."""

    def test_process_result_unchanged(self, make_entry):
        """Test ProcessResult for unchanged entry."""
        entry = make_entry()
        result = ProcessResult(
            original=entry,
            updated=entry,
            changed=False,
            action="unchanged",
        )
        assert result.action == "unchanged"
        assert not result.changed
        assert result.method is None
        assert result.confidence is None

    def test_process_result_upgraded(self, make_entry):
        """Test ProcessResult for upgraded entry."""
        original = make_entry()
        updated = make_entry(journal="New Journal")
        result = ProcessResult(
            original=original,
            updated=updated,
            changed=True,
            action="upgraded",
            method="Crossref(works)",
            confidence=0.95,
        )
        assert result.action == "upgraded"
        assert result.changed
        assert result.method == "Crossref(works)"
        assert result.confidence == 0.95

    def test_process_result_failed(self, make_entry):
        """Test ProcessResult for failed entry."""
        entry = make_entry()
        result = ProcessResult(
            original=entry,
            updated=entry,
            changed=False,
            action="failed",
            message="No match found",
        )
        assert result.action == "failed"
        assert not result.changed
        assert result.message == "No match found"
