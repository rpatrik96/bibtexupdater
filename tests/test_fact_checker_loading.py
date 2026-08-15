"""Regression tests for the checker bibliography-loading boundary."""

from __future__ import annotations

import logging

from bibtex_updater import fact_checker


class _CapturingProcessor:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def process_entries(self, entries, jsonl_path=None, max_workers=1):
        self.entries = list(entries)
        return []

    def generate_summary(self, results):
        return {
            "total": len(self.entries),
            "by_category": {},
            "status_counts": {},
            "verified_count": 0,
            "abstained_count": 0,
            "problematic_count": 0,
            "coverage_incomplete_count": 0,
            "verified_rate": 0.0,
            "could_not_verify_rate": 0.0,
        }


def _run_checker(monkeypatch, bib_path):
    processor = _CapturingProcessor()
    monkeypatch.setattr(
        fact_checker,
        "build_checker_processor",
        lambda *args, **kwargs: (processor, object()),
    )
    monkeypatch.setattr(fact_checker.sys, "argv", ["bibtex-check", str(bib_path)])
    return fact_checker.main(), processor


def test_online_blog_loads_and_is_classified_as_web_reference(tmp_path, monkeypatch):
    bib_path = tmp_path / "online.bib"
    bib_path.write_text(
        """@online{nanda2025pragmatic,
  author       = {Neel Nanda and Josh Engels and Arthur Conmy},
  title        = {A Pragmatic Vision for Interpretability},
  year         = {2025},
  url          = {https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability},
  organization = {Alignment Forum}
}
""",
        encoding="utf-8",
    )

    exit_code, processor = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert len(processor.entries) == 1
    classification = fact_checker.EntryClassifier().classify(processor.entries[0])
    assert classification.category == fact_checker.EntryCategory.WEB_REFERENCE


def test_checker_loads_standard_and_nonstandard_entry_types(tmp_path, monkeypatch):
    bib_path = tmp_path / "mixed.bib"
    bib_path.write_text(
        """@online{online_key, title = {Online}, url = {https://example.com/online}}
@software{software_key, title = {Software}}
@dataset{dataset_key, title = {Dataset}}
@misc{misc_key, title = {Miscellaneous}}
""",
        encoding="utf-8",
    )

    exit_code, processor = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert len(processor.entries) == 4
    assert {entry["ID"] for entry in processor.entries} == {
        "online_key",
        "software_key",
        "dataset_key",
        "misc_key",
    }


def test_checker_errors_with_key_when_parser_drops_malformed_entry(tmp_path, monkeypatch, caplog):
    bib_path = tmp_path / "malformed.bib"
    bib_path.write_text(
        """@misc{valid_entry, title = {Valid}}
@article{broken_entry, title = {Unterminated
""",
        encoding="utf-8",
    )

    with caplog.at_level(logging.ERROR, logger="fact_checker"):
        exit_code, processor = _run_checker(monkeypatch, bib_path)

    assert exit_code == 1
    assert processor.entries == []
    assert "broken_entry" in caplog.text
