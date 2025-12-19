"""Tests for field checking functionality."""

from __future__ import annotations

import pytest

from bibtex_updater import (
    FieldChecker,
    FieldCheckResult,
    FieldRequirementRegistry,
    MissingFieldProcessor,
    MissingFieldReport,
)

# ------------- FieldRequirementRegistry Tests -------------


class TestFieldRequirementRegistry:
    """Tests for FieldRequirementRegistry."""

    def test_article_requirements(self):
        """Article should require author, title, journal, year."""
        req = FieldRequirementRegistry.get_requirements("article")
        assert "author" in req.required
        assert "title" in req.required
        assert "journal" in req.required
        assert "year" in req.required

    def test_inproceedings_requirements(self):
        """Inproceedings should require booktitle, not journal."""
        req = FieldRequirementRegistry.get_requirements("inproceedings")
        assert "booktitle" in req.required
        assert "journal" not in req.required
        assert "author" in req.required
        assert "title" in req.required
        assert "year" in req.required

    def test_book_requirements(self):
        """Book should require title, publisher, year."""
        req = FieldRequirementRegistry.get_requirements("book")
        assert "title" in req.required
        assert "publisher" in req.required
        assert "year" in req.required
        # author or editor is in recommended
        assert "author" in req.recommended or "editor" in req.recommended

    def test_misc_requirements(self):
        """Misc should only require title."""
        req = FieldRequirementRegistry.get_requirements("misc")
        assert "title" in req.required
        assert len(req.required) == 1

    def test_unknown_type_defaults(self):
        """Unknown types should use minimal defaults."""
        req = FieldRequirementRegistry.get_requirements("unknowntype")
        assert "title" in req.required
        assert "author" in req.recommended
        assert "year" in req.recommended

    def test_case_insensitive(self):
        """Entry type lookup should be case-insensitive."""
        req_lower = FieldRequirementRegistry.get_requirements("article")
        req_upper = FieldRequirementRegistry.get_requirements("ARTICLE")
        req_mixed = FieldRequirementRegistry.get_requirements("Article")
        assert req_lower == req_upper == req_mixed

    def test_get_all_entry_types(self):
        """Should return all registered entry types."""
        types = FieldRequirementRegistry.get_all_entry_types()
        assert "article" in types
        assert "inproceedings" in types
        assert "book" in types
        assert "misc" in types


# ------------- FieldChecker Tests -------------


class TestFieldChecker:
    """Tests for FieldChecker."""

    @pytest.fixture
    def checker(self):
        """Create a FieldChecker instance."""
        return FieldChecker()

    def test_complete_article(self, checker):
        """Complete article should report no missing fields."""
        entry = {
            "ENTRYTYPE": "article",
            "ID": "test2020",
            "author": "Doe, Jane",
            "title": "Complete Article",
            "journal": "Journal of Testing",
            "year": "2020",
            "volume": "1",
            "number": "2",
            "pages": "1-10",
            "doi": "10.1000/test",
            "url": "https://example.com",
        }
        report = checker.check_entry(entry)
        assert report.missing_required == []
        assert report.missing_recommended == []

    def test_missing_required_fields(self, checker):
        """Missing required fields should be detected."""
        entry = {
            "ENTRYTYPE": "article",
            "ID": "incomplete",
            "title": "Missing Author",
            # Missing: author, journal, year
        }
        report = checker.check_entry(entry)
        assert "author" in report.missing_required
        assert "journal" in report.missing_required
        assert "year" in report.missing_required

    def test_missing_recommended_fields(self, checker):
        """Missing recommended fields should be detected."""
        entry = {
            "ENTRYTYPE": "article",
            "ID": "minimal",
            "author": "Doe, Jane",
            "title": "Minimal Article",
            "journal": "Journal",
            "year": "2020",
            # Missing recommended: volume, number, pages, doi, url
        }
        report = checker.check_entry(entry)
        assert report.missing_required == []
        assert "volume" in report.missing_recommended
        assert "pages" in report.missing_recommended
        assert "doi" in report.missing_recommended

    def test_inproceedings_booktitle(self, checker):
        """Inproceedings should require booktitle, not journal."""
        entry = {
            "ENTRYTYPE": "inproceedings",
            "ID": "conf2020",
            "author": "Doe, Jane",
            "title": "Conference Paper",
            "booktitle": "Proceedings of Test Conference",
            "year": "2020",
        }
        report = checker.check_entry(entry)
        assert report.missing_required == []
        # journal is not required for inproceedings
        assert "journal" not in report.missing_required

    def test_inproceedings_missing_booktitle(self, checker):
        """Inproceedings without booktitle should report it missing."""
        entry = {
            "ENTRYTYPE": "inproceedings",
            "ID": "conf_no_venue",
            "author": "Doe, Jane",
            "title": "No Venue",
            "year": "2020",
        }
        report = checker.check_entry(entry)
        assert "booktitle" in report.missing_required

    def test_book_author_or_editor(self, checker):
        """Book with author should not require editor."""
        entry = {
            "ENTRYTYPE": "book",
            "ID": "book2020",
            "author": "Author, Some",
            "title": "The Book",
            "publisher": "Publisher Inc",
            "year": "2020",
        }
        report = checker.check_entry(entry)
        assert "author" not in report.missing_required
        assert "editor" not in report.missing_required

    def test_empty_fields_treated_as_missing(self, checker):
        """Empty string fields should be treated as missing."""
        entry = {
            "ENTRYTYPE": "article",
            "ID": "empty",
            "author": "",
            "title": "  ",  # Whitespace only
            "journal": "Valid Journal",
            "year": "2020",
        }
        report = checker.check_entry(entry)
        assert "author" in report.missing_required
        assert "title" in report.missing_required

    def test_has_missing_fields(self, checker):
        """has_missing_fields should return correct status."""
        complete_entry = {
            "ENTRYTYPE": "misc",
            "ID": "complete",
            "title": "Complete Misc",
            "author": "Author",
            "year": "2020",
            "url": "https://example.com",
            "howpublished": "Online",
        }
        incomplete_entry = {
            "ENTRYTYPE": "misc",
            "ID": "incomplete",
            "title": "Only Title",
        }

        complete_report = checker.check_entry(complete_entry)
        incomplete_report = checker.check_entry(incomplete_entry)

        assert not checker.has_missing_fields(complete_report)
        assert checker.has_missing_fields(incomplete_report)

    def test_has_missing_required(self, checker):
        """has_missing_required should detect required field issues."""
        complete_entry = {
            "ENTRYTYPE": "article",
            "ID": "complete",
            "author": "Author",
            "title": "Title",
            "journal": "Journal",
            "year": "2020",
        }
        incomplete_entry = {
            "ENTRYTYPE": "article",
            "ID": "incomplete",
            "title": "Only Title",
        }

        complete_report = checker.check_entry(complete_entry)
        incomplete_report = checker.check_entry(incomplete_entry)

        assert not checker.has_missing_required(complete_report)
        assert checker.has_missing_required(incomplete_report)


# ------------- MissingFieldProcessor Tests -------------


class TestMissingFieldProcessor:
    """Tests for MissingFieldProcessor."""

    @pytest.fixture
    def processor_check_only(self):
        """Create a processor in check-only mode."""
        return MissingFieldProcessor(
            checker=FieldChecker(),
            filler=None,
            fill_mode="recommended",
            fill_enabled=False,
        )

    def test_complete_entry_returns_complete_action(self, processor_check_only):
        """Complete entry should return 'complete' action."""
        entry = {
            "ENTRYTYPE": "misc",
            "ID": "complete",
            "title": "Complete",
            "author": "Author",
            "year": "2020",
            "url": "https://example.com",
            "howpublished": "Online",
        }
        result = processor_check_only.process_entry(entry)
        assert result.action == "complete"
        assert not result.changed

    def test_incomplete_entry_check_only(self, processor_check_only):
        """Incomplete entry in check-only mode should not change."""
        entry = {
            "ENTRYTYPE": "article",
            "ID": "incomplete",
            "title": "Only Title",
        }
        result = processor_check_only.process_entry(entry)
        assert result.action == "unfillable"
        assert not result.changed
        assert "author" in result.report.missing_required

    def test_generate_summary(self, processor_check_only):
        """Summary generation should count correctly."""
        results = [
            FieldCheckResult(
                original={},
                updated={},
                report=MissingFieldReport("a", "article", [], [], {}, []),
                changed=False,
                action="complete",
            ),
            FieldCheckResult(
                original={},
                updated={},
                report=MissingFieldReport("b", "article", [], [], {"year": ("2020", "API")}, []),
                changed=True,
                action="filled",
            ),
            FieldCheckResult(
                original={},
                updated={},
                report=MissingFieldReport("c", "article", ["author"], [], {}, []),
                changed=False,
                action="unfillable",
            ),
        ]
        summary = processor_check_only.generate_summary(results)
        assert summary["total"] == 3
        assert summary["complete"] == 1
        assert summary["filled"] == 1
        assert summary["unfillable"] == 1

    def test_generate_json_report(self, processor_check_only):
        """JSON report should have correct structure."""
        results = [
            FieldCheckResult(
                original={"ID": "test"},
                updated={"ID": "test"},
                report=MissingFieldReport("test", "article", ["author"], ["doi"], {}, []),
                changed=False,
                action="unfillable",
            ),
        ]
        report = processor_check_only.generate_json_report(results)

        assert "summary" in report
        assert "entries" in report
        assert len(report["entries"]) == 1
        assert report["entries"][0]["key"] == "test"
        assert report["entries"][0]["action"] == "unfillable"
        assert "author" in report["entries"][0]["missing_required"]


# ------------- MissingFieldReport Tests -------------


class TestMissingFieldReport:
    """Tests for MissingFieldReport dataclass."""

    def test_report_creation(self):
        """Report should be created with correct fields."""
        report = MissingFieldReport(
            entry_key="test",
            entry_type="article",
            missing_required=["author", "year"],
            missing_recommended=["doi"],
            filled_fields={"volume": ("1", "Crossref")},
            errors=["API timeout"],
        )
        assert report.entry_key == "test"
        assert report.entry_type == "article"
        assert len(report.missing_required) == 2
        assert len(report.missing_recommended) == 1
        assert "volume" in report.filled_fields
        assert len(report.errors) == 1


# ------------- FieldCheckResult Tests -------------


class TestFieldCheckResult:
    """Tests for FieldCheckResult dataclass."""

    def test_result_creation(self):
        """Result should be created with correct fields."""
        original = {"ID": "orig", "title": "Original"}
        updated = {"ID": "orig", "title": "Original", "year": "2020"}
        report = MissingFieldReport("orig", "article", [], [], {"year": ("2020", "API")}, [])

        result = FieldCheckResult(
            original=original,
            updated=updated,
            report=report,
            changed=True,
            action="filled",
        )
        assert result.changed
        assert result.action == "filled"
        assert result.updated["year"] == "2020"


# ------------- Integration-style Tests -------------


class TestFieldCheckingIntegration:
    """Integration-style tests for field checking workflow."""

    def test_full_check_workflow(self):
        """Test complete workflow from entry to report."""
        checker = FieldChecker()
        processor = MissingFieldProcessor(
            checker=checker,
            filler=None,
            fill_enabled=False,
        )

        entries = [
            # Complete entry with all required and recommended fields
            {
                "ENTRYTYPE": "article",
                "ID": "complete",
                "author": "A",
                "title": "T",
                "journal": "J",
                "year": "2020",
                "volume": "1",
                "number": "2",
                "pages": "1-10",
                "doi": "10.1/x",
                "url": "http://x",
            },
            # Partial: has all required but missing some recommended
            {"ENTRYTYPE": "article", "ID": "partial", "author": "A", "title": "T", "journal": "J", "year": "2020"},
            # Missing: missing required fields (author, journal, year)
            {"ENTRYTYPE": "article", "ID": "missing", "title": "T"},
        ]

        results = [processor.process_entry(e) for e in entries]
        summary = processor.generate_summary(results)

        assert summary["complete"] == 1
        assert summary["partial"] == 1  # Has all required but missing recommended
        assert summary["unfillable"] == 1  # Missing required fields
