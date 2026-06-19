"""Tests for the Updater class."""

from __future__ import annotations

from rapidfuzz.fuzz import token_sort_ratio

from bibtex_updater import (
    PublishedRecord,
    authors_last_names,
    jaccard_similarity,
    normalize_title_for_match,
    split_authors_bibtex,
)
from bibtex_updater.updater import Updater, build_arg_parser


class TestUpdaterBasic:
    """Basic update functionality tests."""

    def test_update_entry_basic(self, updater, arxiv_entry, sample_published_record, arxiv_detection):
        """Basic entry update with all fields."""
        updated = updater.update_entry(arxiv_entry, sample_published_record, arxiv_detection)

        assert updated["title"] == sample_published_record.title
        assert updated["journal"] == sample_published_record.journal
        assert updated["year"] == str(sample_published_record.year)
        assert updated["doi"] == sample_published_record.doi
        assert updated["volume"] == sample_published_record.volume
        assert updated["pages"] == sample_published_record.pages

    def test_update_removes_preprint_fields(self, updater, make_entry, sample_published_record, arxiv_detection):
        """Update should remove preprint-only fields."""
        entry = make_entry(
            eprint="2001.01234",
            archiveprefix="arXiv",
            primaryClass="cs.LG",
            eprinttype="arxiv",
        )
        updated = updater.update_entry(entry, sample_published_record, arxiv_detection)

        assert "eprint" not in updated
        assert "archiveprefix" not in updated
        assert "primaryClass" not in updated
        assert "eprinttype" not in updated

    def test_update_preserves_id(self, updater, arxiv_entry, sample_published_record, arxiv_detection):
        """Update should preserve the original entry ID."""
        original_id = arxiv_entry["ID"]
        updated = updater.update_entry(arxiv_entry, sample_published_record, arxiv_detection)
        assert updated["ID"] == original_id

    def test_update_sets_article_type(self, updater, make_entry, sample_published_record, arxiv_detection):
        """Update should set entry type to article."""
        entry = make_entry(ENTRYTYPE="misc")
        updated = updater.update_entry(entry, sample_published_record, arxiv_detection)
        assert updated["ENTRYTYPE"] == "article"


class TestUpdaterAuthors:
    """Tests for author handling in updates."""

    def test_update_preserves_author_list(self, updater, make_entry, arxiv_detection):
        """Author list should be correctly transferred from PublishedRecord."""
        entry = make_entry(
            author="Doe, Jane and Smith, John",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[
                {"given": "Jane", "family": "Doe"},
                {"given": "John", "family": "Smith"},
            ],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        assert updated["author"] == "Jane Doe and John Smith"
        original_last_names = set(authors_last_names(entry["author"]))
        updated_last_names = set(authors_last_names(updated["author"]))
        assert original_last_names == updated_last_names

    def test_update_multiple_authors(self, updater, make_entry, arxiv_detection):
        """Multiple authors should be preserved correctly."""
        entry = make_entry(
            author="First, Alice and Second, Bob and Third, Charlie",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[
                {"given": "Alice", "family": "First"},
                {"given": "Bob", "family": "Second"},
                {"given": "Charlie", "family": "Third"},
            ],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        assert updated["author"] == "Alice First and Bob Second and Charlie Third"
        updated_last_names = authors_last_names(updated["author"], limit=10)
        assert updated_last_names == ["first", "second", "third"]

    def test_update_preserves_author_order(self, updater, make_entry, arxiv_detection):
        """Author order should be preserved after update."""
        entry = make_entry(
            author="Alpha, Ann and Beta, Bob and Gamma, Grace",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[
                {"given": "Ann", "family": "Alpha"},
                {"given": "Bob", "family": "Beta"},
                {"given": "Grace", "family": "Gamma"},
            ],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        updated_authors = split_authors_bibtex(updated["author"])
        assert len(updated_authors) == 3
        assert "Alpha" in updated_authors[0]
        assert "Beta" in updated_authors[1]
        assert "Gamma" in updated_authors[2]

    def test_update_author_consistency_jaccard(self, updater, make_entry, arxiv_detection):
        """Updated authors should have perfect Jaccard similarity with original."""
        entry = make_entry(
            author="Smith, John and Doe, Jane",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[
                {"given": "John", "family": "Smith"},
                {"given": "Jane", "family": "Doe"},
            ],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        orig_names = authors_last_names(entry["author"])
        updated_names = authors_last_names(updated["author"])
        similarity = jaccard_similarity(orig_names, updated_names)
        assert similarity == 1.0

    def test_update_author_with_only_family_name(self, updater, make_entry, arxiv_detection):
        """Handle authors with only family names."""
        entry = make_entry(author="Madonna")
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[{"given": "", "family": "Madonna"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)
        assert "Madonna" in updated["author"]


class TestUpdaterTitle:
    """Tests for title handling in updates."""

    def test_update_preserves_title(self, updater, make_entry, arxiv_detection):
        """Title should be correctly transferred from PublishedRecord."""
        entry = make_entry(title="A Study of Machine Learning")
        record = PublishedRecord(
            doi="10.1000/test",
            title="A Study of Machine Learning",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        assert updated["title"] == record.title
        assert normalize_title_for_match(updated["title"]) == normalize_title_for_match(record.title)

    def test_update_title_with_special_characters(self, updater, make_entry, arxiv_detection):
        """Titles with special characters should be handled correctly."""
        entry = make_entry(title="{Deep Learning for Schrödinger Equations}")
        record = PublishedRecord(
            doi="10.1000/test",
            title="Deep Learning for Schrödinger Equations",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        assert updated["title"] == record.title
        orig_norm = normalize_title_for_match(entry["title"])
        updated_norm = normalize_title_for_match(updated["title"])
        title_score = token_sort_ratio(orig_norm, updated_norm)
        assert title_score == 100

    def test_update_title_consistency_fuzzy_match(self, updater, make_entry, arxiv_detection):
        """Updated title should have high fuzzy match score."""
        entry = make_entry(title="Neural Networks for Image Classification")
        expected_title = "Neural Networks for Image Classification"
        record = PublishedRecord(
            doi="10.1000/test",
            title=expected_title,
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        title_score = token_sort_ratio(
            normalize_title_for_match(updated["title"]),
            normalize_title_for_match(expected_title),
        )
        assert title_score == 100


class TestUpdaterKeepPreprintNote:
    """Tests for keep_preprint_note option."""

    def test_keep_preprint_note_adds_arxiv_reference(self, updater_with_note, make_entry, arxiv_detection):
        """With keep_preprint_note=True, arXiv reference should be added to note."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater_with_note.update_entry(entry, record, arxiv_detection)

        assert "note" in updated
        assert "arXiv:2001.01234" in updated["note"]

    def test_keep_preprint_note_preserves_existing_note(self, updater_with_note, make_entry, arxiv_detection):
        """Existing note should be preserved when adding arXiv reference."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            note="Some existing note",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater_with_note.update_entry(entry, record, arxiv_detection)

        assert "Some existing note" in updated["note"]
        assert "arXiv:2001.01234" in updated["note"]

    def test_no_duplicate_arxiv_note(self, updater_with_note, make_entry, arxiv_detection):
        """Should not add duplicate arXiv reference if already present."""
        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            note="Also available as arXiv:2001.01234",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater_with_note.update_entry(entry, record, arxiv_detection)

        # Should only appear once
        assert updated["note"].count("arXiv:2001.01234") == 1


class TestUpdaterRekey:
    """Tests for rekey option."""

    def test_rekey_generates_new_key(self, updater_with_rekey, make_entry, arxiv_detection):
        """With rekey=True, a new key should be generated."""
        entry = make_entry(ID="old_key_2020")
        record = PublishedRecord(
            doi="10.1000/test",
            title="Machine Learning Study",
            authors=[{"given": "John", "family": "Smith"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater_with_rekey.update_entry(entry, record, arxiv_detection)

        assert updated["ID"] != "old_key_2020"
        assert "Smith" in updated["ID"] or "smith" in updated["ID"].lower()
        assert "2021" in updated["ID"]


class TestUpdaterIdempotent:
    """Tests for idempotent behavior."""

    def test_updated_entry_not_detected_as_preprint(self, updater, make_entry, arxiv_detection):
        """An updated entry should not be detected as a preprint again."""
        from bibtex_updater import Detector

        entry = make_entry(
            url="https://arxiv.org/abs/2001.01234",
            journal="arXiv preprint",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Real Journal",
            year=2021,
            volume="42",
            pages="1-20",
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        detector = Detector()
        new_detection = detector.detect(updated)
        assert not new_detection.is_preprint


class TestUpdaterEdgeCases:
    """Edge cases for updates."""

    def test_update_with_partial_record(self, updater, make_entry, arxiv_detection):
        """Handle PublishedRecord with missing optional fields."""
        entry = make_entry()
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test Title",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
            # No volume, number, pages, publisher
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        assert updated["doi"] == record.doi
        assert updated["title"] == record.title
        assert "volume" not in updated or updated.get("volume") == entry.get("volume")

    def test_update_drops_extra_fields_by_default(self, updater, make_entry, arxiv_detection):
        """Extra original fields are dropped by default (atomic rebuild from record).

        The upgraded entry is constructed from the resolved record so no stale
        original field can survive; ``preserve_fields`` opts specific fields back in.
        """
        entry = make_entry(
            keywords="machine learning, deep learning",
            abstract="This is an abstract.",
        )
        record = PublishedRecord(
            doi="10.1000/test",
            title="Test",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Test Journal",
            year=2021,
            type="journal-article",
        )
        updated = updater.update_entry(entry, record, arxiv_detection)

        assert "keywords" not in updated
        assert "abstract" not in updated


class TestUpdaterAtomicIntegrity:
    """Upgraded entries are built atomically from the resolved record.

    Only the citekey (and deliberate provenance: ``keep_preprint_note`` /
    ``mark_resolved``) carries over from the original entry, so no stale preprint
    field can survive the upgrade. Field preservation is opt-in via ``preserve_fields``.
    """

    @staticmethod
    def _record() -> PublishedRecord:
        return PublishedRecord(
            doi="10.1000/real",
            title="Real Published Title",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Journal of Real Results",
            year=2021,
            volume="7",
            number="2",
            pages="10-20",
            type="journal-article",
            method="test",
            confidence=1.0,
        )

    def test_drops_stale_non_record_fields(self, updater, make_entry, arxiv_detection):
        """Original fields the record does not supply must not survive the upgrade."""
        entry = make_entry(note="To appear", month="jan", abstract="Old abstract.")
        entry["mendeley-tags"] = "personal"
        updated = updater.update_entry(entry, self._record(), arxiv_detection)
        for stale in ("note", "month", "abstract", "mendeley-tags"):
            assert stale not in updated

    def test_drops_preprint_url_when_record_lacks_doi_and_url(self, updater, make_entry, arxiv_detection):
        """A published entry must never keep pointing at the preprint URL."""
        entry = make_entry(url="https://arxiv.org/abs/2001.01234")
        rec = PublishedRecord(
            doi="",
            url="",
            title="Real Published Title",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Journal of Real Results",
            year=2021,
            type="journal-article",
            method="test",
            confidence=1.0,
        )
        updated = updater.update_entry(entry, rec, arxiv_detection)
        assert updated.get("url", "") != "https://arxiv.org/abs/2001.01234"

    def test_keyset_limited_to_record_fields_and_citekey(self, updater, make_entry, arxiv_detection):
        """The upgraded entry contains only record-derived fields plus the citekey."""
        entry = make_entry(note="x", month="jan", keywords="a, b")
        updated = updater.update_entry(entry, self._record(), arxiv_detection)
        allowed = {
            "ID",
            "ENTRYTYPE",
            "title",
            "author",
            "journal",
            "booktitle",
            "publisher",
            "year",
            "volume",
            "number",
            "pages",
            "doi",
            "url",
        }
        assert set(updated) <= allowed

    def test_citekey_carries_over(self, updater, make_entry, arxiv_detection):
        """The original citekey is preserved so existing ``\\cite`` commands keep working."""
        entry = make_entry(ID="reizinger2021", note="drop me")
        updated = updater.update_entry(entry, self._record(), arxiv_detection)
        assert updated["ID"] == "reizinger2021"

    def test_preserve_fields_opt_in_keeps_only_listed(self, make_entry, arxiv_detection):
        """preserve_fields carries over exactly the named user fields, nothing else."""
        upd = Updater(keep_preprint_note=False, rekey=False, preserve_fields=("file", "keywords"))
        entry = make_entry(file="/local/p.pdf", keywords="ml, nlp", note="drop me")
        updated = upd.update_entry(entry, self._record(), arxiv_detection)
        assert updated["file"] == "/local/p.pdf"
        assert updated["keywords"] == "ml, nlp"
        assert "note" not in updated

    def test_cli_preserve_fields_flag_parses_to_list(self, make_entry, arxiv_detection):
        """The --preserve-fields CLI flag wires a comma list into Updater.preserve_fields."""
        args = build_arg_parser().parse_args(["in.bib", "--preserve-fields", "file, keywords"])
        preserve = tuple(f.strip() for f in args.preserve_fields.split(",") if f.strip())
        assert preserve == ("file", "keywords")
        upd = Updater(preserve_fields=preserve)
        updated = upd.update_entry(make_entry(file="/p.pdf", keywords="x"), self._record(), arxiv_detection)
        assert updated["file"] == "/p.pdf"
        assert updated["keywords"] == "x"

    def test_cli_preserve_fields_defaults_empty(self):
        """Without the flag, no original fields are preserved (atomic default)."""
        args = build_arg_parser().parse_args(["in.bib"])
        assert args.preserve_fields == ""
