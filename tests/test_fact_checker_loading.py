"""Regression tests for the checker bibliography-loading boundary."""

from __future__ import annotations

import json
import logging

from bibtex_updater import fact_checker


class _CapturingChecker:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def check_entry(self, entry):
        self.entries.append(dict(entry))
        return fact_checker.FactCheckResult(
            entry_key=entry["ID"],
            entry_type=entry["ENTRYTYPE"],
            status=fact_checker.FactCheckStatus.SKIPPED,
            overall_confidence=0.0,
            field_comparisons={},
            best_match=None,
            api_sources_queried=[],
            api_sources_with_hits=[],
            errors=[],
        )


def _run_checker(monkeypatch, bib_path, *args):
    checker = _CapturingChecker()
    processor = fact_checker.FactCheckProcessor(checker, logging.getLogger("fact_checker"))
    monkeypatch.setattr(
        fact_checker,
        "build_checker_processor",
        lambda *args, **kwargs: (processor, object()),
    )
    monkeypatch.setattr(fact_checker.sys, "argv", ["bibtex-check", str(bib_path), *args])
    return fact_checker.main(), checker


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

    exit_code, checker = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert len(checker.entries) == 1
    classification = fact_checker.EntryClassifier().classify(checker.entries[0])
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

    exit_code, checker = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert len(checker.entries) == 4
    assert {entry["ID"] for entry in checker.entries} == {
        "online_key",
        "software_key",
        "dataset_key",
        "misc_key",
    }


def test_checker_recovers_dropped_entry_without_discarding_valid_entry(tmp_path, monkeypatch, caplog):
    bib_path = tmp_path / "malformed.bib"
    bib_path.write_text(
        """@misc{valid_entry, title = {Valid}}
@article{broken_entry, title = {Unterminated
""",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="fact_checker"):
        exit_code, checker = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert {entry["ID"] for entry in checker.entries} == {"valid_entry", "broken_entry"}
    assert "broken_entry" in caplog.text
    assert "repaired" in caplog.text.lower()


def test_brace_recovery_preserves_declared_field_values(tmp_path, monkeypatch):
    bib_path = tmp_path / "recoverable.bib"
    bib_path.write_text(
        """@article{recoverable,
  title = {Literal {Nested} Value: 50\\% and \"quotes\"},
  note = {Keep commas, @ signs, and \\LaTeX byte-identical}
""".rstrip(
            "\n"
        ),
        encoding="utf-8",
    )

    exit_code, checker = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert checker.entries == [
        {
            "title": 'Literal {Nested} Value: 50\\% and "quotes"',
            "note": "Keep commas, @ signs, and \\LaTeX byte-identical",
            "ENTRYTYPE": "article",
            "ID": "recoverable",
        }
    ]


def test_missing_key_comma_is_recovered_without_changing_fields(tmp_path, monkeypatch):
    bib_path = tmp_path / "missing-comma.bib"
    bib_path.write_text(
        "@article{missing_comma title = {Exact {Nested} Value}, year = {2024}}\n",
        encoding="utf-8",
    )

    exit_code, checker = _run_checker(monkeypatch, bib_path)

    assert exit_code == 0
    assert checker.entries == [
        {
            "title": "Exact {Nested} Value",
            "year": "2024",
            "ENTRYTYPE": "article",
            "ID": "missing_comma",
        }
    ]


def test_mixed_input_reports_every_declared_key_and_strict_gates_parse_errors(tmp_path, monkeypatch, caplog):
    bib_path = tmp_path / "mixed-malformed.bib"
    bib_path.write_text(
        """@misc{good, title = {Good}}
@article{recoverable, title = {Recovered}
@article{unrecoverable, ???}
@misc{trailing, title = {Trailing}}
""",
        encoding="utf-8",
    )
    json_report = tmp_path / "report.json"
    jsonl_report = tmp_path / "report.jsonl"

    with caplog.at_level(logging.ERROR, logger="fact_checker"):
        exit_code, checker = _run_checker(
            monkeypatch,
            bib_path,
            "--report",
            str(json_report),
            "--jsonl",
            str(jsonl_report),
        )

    report = json.loads(json_report.read_text(encoding="utf-8"))
    jsonl_rows = [json.loads(line) for line in jsonl_report.read_text(encoding="utf-8").splitlines()]
    rows_by_key = {row["key"]: row for row in report["entries"]}

    assert exit_code == 0
    assert {entry["ID"] for entry in checker.entries} == {"good", "recoverable", "trailing"}
    assert report["summary"]["total"] == 4
    assert report["summary"]["parse_error_count"] == 1
    assert report["summary"]["problematic_count"] == 0
    assert report["summary"]["abstained_count"] == 0
    assert report["summary"]["verified_count"] == 0
    assert set(rows_by_key) == {"good", "recoverable", "unrecoverable", "trailing"}
    assert rows_by_key["unrecoverable"]["status"] == "parse_error"
    assert {row["key"] for row in jsonl_rows} == set(rows_by_key)
    assert next(row for row in jsonl_rows if row["key"] == "unrecoverable")["status"] == "parse_error"
    assert "unrecoverable" in caplog.text

    caplog.clear()
    strict_report = tmp_path / "strict-report.json"
    with caplog.at_level(logging.WARNING, logger="fact_checker"):
        strict_exit, _checker = _run_checker(
            monkeypatch,
            bib_path,
            "--strict",
            "--report",
            str(strict_report),
        )

    assert strict_exit == 4
    assert json.loads(strict_report.read_text(encoding="utf-8"))["summary"]["parse_error_count"] == 1
    assert "Strict mode" in caplog.text
    assert "parse error" in caplog.text.lower()
