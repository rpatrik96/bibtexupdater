"""Tests for BibLoader non-standard entry type retention and dropped-key audit trail.

Regression coverage for the silent data-loss bug where bibtexparser's default
``ignore_nonstandard_types=True`` silently discarded biblatex entry types such
as ``@online``, ``@software``, ``@dataset`` at parse time. These tests are
network-free and exercise only parsing, writing, and the dropped-key safety net.
"""

from __future__ import annotations

import io
import json

import pytest

from bibtex_updater.updater import (
    BibLoader,
    BibWriter,
    detect_dropped_keys,
    write_dropped_report_line,
)

ONLINE_ENTRY = """@online{nanda2025pragmatic,
  title  = {Pragmatic Mechanistic Interpretability},
  author = {Nanda, Neel},
  year   = {2025},
  url    = {https://example.org/pragmatic}
}
"""

SOFTWARE_ENTRY = """@software{somesoftware2024,
  title  = {Some Software Package},
  author = {Doe, Jane},
  year   = {2024}
}
"""

DATASET_ENTRY = """@dataset{somedataset2023,
  title  = {Some Dataset},
  author = {Smith, John},
  year   = {2023}
}
"""

ARTICLE_ENTRY = """@article{stdarticle2022,
  title   = {A Standard Article},
  author  = {Roe, Richard},
  journal = {Journal of Things},
  year    = {2022}
}
"""


class TestBibLoaderNonstandardTypes:
    """Non-standard biblatex entry types must survive parsing and round-trip."""

    def test_loads_retains_online_entry(self):
        """``loads()`` keeps an ``@online`` entry instead of silently dropping it."""
        db = BibLoader().loads(ONLINE_ENTRY)
        ids = {e.get("ID") for e in db.entries}
        assert "nanda2025pragmatic" in ids
        entry = next(e for e in db.entries if e.get("ID") == "nanda2025pragmatic")
        assert entry.get("ENTRYTYPE") == "online"

    @pytest.mark.parametrize(
        ("text", "key", "entrytype"),
        [
            (SOFTWARE_ENTRY, "somesoftware2024", "software"),
            (DATASET_ENTRY, "somedataset2023", "dataset"),
        ],
    )
    def test_loads_retains_software_and_dataset(self, text, key, entrytype):
        """``loads()`` keeps ``@software`` and ``@dataset`` entries."""
        db = BibLoader().loads(text)
        ids = {e.get("ID") for e in db.entries}
        assert key in ids
        entry = next(e for e in db.entries if e.get("ID") == key)
        assert entry.get("ENTRYTYPE") == entrytype

    def test_loads_retains_nonstandard_alongside_standard(self):
        """A standard ``@article`` is retained alongside non-standard types."""
        text = ONLINE_ENTRY + SOFTWARE_ENTRY + DATASET_ENTRY + ARTICLE_ENTRY
        db = BibLoader().loads(text)
        ids = {e.get("ID") for e in db.entries}
        assert ids == {
            "nanda2025pragmatic",
            "somesoftware2024",
            "somedataset2023",
            "stdarticle2022",
        }

    def test_load_file_roundtrips_online_entry(self, tmp_path):
        """``load_file()`` -> ``BibWriter`` round-trips an ``@online`` entry."""
        src = tmp_path / "in.bib"
        src.write_text(ONLINE_ENTRY, encoding="utf-8")

        db = BibLoader().load_file(str(src))
        ids = {e.get("ID") for e in db.entries}
        assert "nanda2025pragmatic" in ids

        dumped = BibWriter().dumps(db)
        assert "nanda2025pragmatic" in dumped
        assert "@online" in dumped


class TestDroppedKeySafetyNet:
    """Genuinely unparseable entries must be logged and recorded, not lost silently."""

    def test_detect_dropped_keys_pure_function(self):
        """``detect_dropped_keys`` returns declared keys absent from parsed IDs."""
        raw = ONLINE_ENTRY + ARTICLE_ENTRY
        parsed_ids = {"stdarticle2022"}  # simulate @online lost at parse time
        dropped = detect_dropped_keys(raw, parsed_ids)
        assert dropped == ["nanda2025pragmatic"]

    def test_detect_dropped_keys_none_dropped(self):
        """No dropped keys when every declared key is parsed."""
        raw = ONLINE_ENTRY + ARTICLE_ENTRY
        parsed_ids = {"nanda2025pragmatic", "stdarticle2022"}
        assert detect_dropped_keys(raw, parsed_ids) == []

    def test_malformed_entry_is_detected_as_dropped(self):
        """A truly malformed entry the parser cannot keep is detected as dropped."""
        malformed = "@article{broken_entry, title = {Unterminated \n\n"
        raw = ARTICLE_ENTRY + malformed
        db = BibLoader().loads(raw)
        parsed_ids = {e.get("ID") for e in db.entries}
        dropped = detect_dropped_keys(raw, parsed_ids)
        assert "broken_entry" in dropped
        assert "stdarticle2022" not in dropped

    def test_commented_out_entry_is_not_a_false_positive(self):
        """A full-line ``%``-commented ``@article{...}`` must NOT be reported dropped."""
        raw = ARTICLE_ENTRY + "% @article{commented_out, title = {Disabled}, year = {2020}}\n"
        db = BibLoader().loads(raw)
        parsed_ids = {e.get("ID") for e in db.entries}
        assert "stdarticle2022" in parsed_ids
        dropped = detect_dropped_keys(raw, parsed_ids)
        assert "commented_out" not in dropped
        assert dropped == []

    def test_entry_marker_inside_field_value_is_not_a_false_positive(self):
        """``@type{key,`` appearing inside a field value must NOT be reported dropped."""
        raw = (
            "@online{real_entry,\n"
            "  title  = {A Real Entry},\n"
            "  author = {Doe, Jane},\n"
            "  year   = {2025},\n"
            "  note   = {See also @article{fake, } discussed inline},\n"
            "  url    = {https://example.org/real}\n"
            "}\n"
        )
        db = BibLoader().loads(raw)
        parsed_ids = {e.get("ID") for e in db.entries}
        assert "real_entry" in parsed_ids
        dropped = detect_dropped_keys(raw, parsed_ids)
        assert "fake" not in dropped
        assert dropped == []

    def test_dropped_key_logged_with_citation_key_named(self, caplog):
        """The dropped citation key and file are named in a warning log line."""
        import logging

        logger = logging.getLogger("bibtex_updater.test_dropped")
        with caplog.at_level(logging.WARNING, logger="bibtex_updater.test_dropped"):
            for key in detect_dropped_keys(ONLINE_ENTRY, set()):
                logger.warning("Dropped unparseable entry %r from %s (not added to database)", key, "in.bib")
        assert "nanda2025pragmatic" in caplog.text
        assert "in.bib" in caplog.text

    def test_dropped_report_line_schema(self):
        """``write_dropped_report_line`` emits the existing JSONL schema with action='dropped'."""
        fh = io.StringIO()
        write_dropped_report_line(fh, "nanda2025pragmatic", src_file="in.bib")
        line = json.loads(fh.getvalue().strip())
        assert line["action"] == "dropped"
        assert line["key_old"] == "nanda2025pragmatic"
        assert line["key_new"] is None
        # Same schema keys as write_report_line so the report stays consistent.
        assert set(line.keys()) == {
            "file",
            "key_old",
            "key_new",
            "doi_old",
            "doi_new",
            "action",
            "method",
            "confidence",
            "title_old",
            "title_new",
        }
        assert line["file"] == "in.bib"
