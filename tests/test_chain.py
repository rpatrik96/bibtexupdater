"""Tests for the resolve->check chaining (``chain.py``).

The chain runs the resolver first, then fact-checks only the entries the resolver
did NOT actively upgrade -- an upgraded entry is a clean record pulled straight
from a bibliographic database, so re-verifying it is redundant ("clean bib, fast").
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from bibtex_updater.chain import build_chain_report, select_entries_to_check
from bibtex_updater.updater import ProcessResult


def _result(key: str, action: str, **overrides) -> ProcessResult:
    entry = {"ID": key, "ENTRYTYPE": "article", "title": f"title-{key}"}
    return ProcessResult(
        original=dict(entry),
        updated=dict(entry, **overrides),
        changed=action == "upgraded",
        action=action,
    )


class TestChainPartition:
    """Only entries the resolver did not vouch for go to the checker."""

    def test_upgraded_entries_are_not_checked(self):
        results = [
            _result("a", "upgraded"),
            _result("b", "unchanged"),
            _result("c", "failed"),
            _result("d", "upgraded"),
        ]
        to_check = select_entries_to_check(results)
        assert {e["ID"] for e in to_check} == {"b", "c"}

    def test_previously_resolved_entries_are_not_rechecked(self):
        """``skipped_resolved`` (a prior run's upgrade) is also trusted, not rechecked."""
        results = [_result("a", "skipped_resolved"), _result("b", "unchanged")]
        to_check = select_entries_to_check(results)
        assert {e["ID"] for e in to_check} == {"b"}

    def test_to_check_carries_the_updated_entry(self):
        """The checker sees the cleaned/updated entry, not the original."""
        results = [_result("b", "unchanged", journal="Real Journal")]
        to_check = select_entries_to_check(results)
        assert to_check[0]["journal"] == "Real Journal"


class TestUpgradeTrust:
    """Cheap, no-network classification of whether an upgrade is safe to trust+skip."""

    @staticmethod
    def _upg(*, otitle, oauthor, oyear, utitle, uauthor, uyear) -> ProcessResult:
        orig = {"ID": "k", "title": otitle, "author": oauthor, "year": oyear}
        upd = {"ID": "k", "title": utitle, "author": uauthor, "year": uyear}
        return ProcessResult(original=orig, updated=upd, changed=True, action="upgraded")

    def test_trusted_when_title_author_year_align(self):
        from bibtex_updater.chain import classify_upgrade

        t = classify_upgrade(
            self._upg(
                otitle="Attention Is All You Need",
                oauthor="Vaswani, Ashish and Shazeer, Noam",
                oyear="2017",
                utitle="Attention Is All You Need",
                uauthor="Ashish Vaswani and Noam Shazeer",
                uyear="2017",
            )
        )
        assert t.trusted is True
        assert t.reasons == []

    def test_flags_low_title_even_with_identical_authors(self):
        """A perfect author score must NOT carry a weak title (non-compensatory)."""
        from bibtex_updater.chain import classify_upgrade

        t = classify_upgrade(
            self._upg(
                otitle="Attention Is All You Need",
                oauthor="Vaswani, Ashish",
                oyear="2017",
                utitle="Graph Neural Networks: A Comprehensive Survey",
                uauthor="Vaswani, Ashish",
                uyear="2017",
            )
        )
        assert t.trusted is False
        assert any("title" in r for r in t.reasons)

    def test_flags_low_author(self):
        from bibtex_updater.chain import classify_upgrade

        t = classify_upgrade(
            self._upg(
                otitle="Attention Is All You Need",
                oauthor="Vaswani, Ashish and Shazeer, Noam",
                oyear="2017",
                utitle="Attention Is All You Need",
                uauthor="Smith, John and Doe, Jane",
                uyear="2017",
            )
        )
        assert t.trusted is False
        assert any("author" in r for r in t.reasons)

    def test_flags_record_year_preceding_citation(self):
        from bibtex_updater.chain import classify_upgrade

        t = classify_upgrade(
            self._upg(
                otitle="Attention Is All You Need",
                oauthor="Vaswani, Ashish",
                oyear="2020",
                utitle="Attention Is All You Need",
                uauthor="Vaswani, Ashish",
                uyear="2017",
            )
        )
        assert t.trusted is False
        assert any("year" in r for r in t.reasons)

    def test_flags_large_forward_year_gap(self):
        from bibtex_updater.chain import classify_upgrade

        t = classify_upgrade(
            self._upg(
                otitle="Attention Is All You Need",
                oauthor="Vaswani, Ashish",
                oyear="2017",
                utitle="Attention Is All You Need",
                uauthor="Vaswani, Ashish",
                uyear="2027",
            )
        )
        assert t.trusted is False
        assert any("year" in r for r in t.reasons)


class TestFlaggedDecision:
    """For a flagged upgrade, independent verification decides keep vs revert."""

    def test_keep_when_record_agrees_and_status_ok(self):
        from bibtex_updater.chain import decide_flagged

        upd = {"ID": "k", "doi": "10.1/x"}
        cr = SimpleNamespace(status="VERIFIED", best_match=SimpleNamespace(doi="10.1/x"))
        decision, reasons = decide_flagged(upd, cr)
        assert decision == "keep"

    def test_revert_on_doi_disagreement(self):
        from bibtex_updater.chain import decide_flagged

        upd = {"ID": "k", "doi": "10.1/x"}
        cr = SimpleNamespace(status="VERIFIED", best_match=SimpleNamespace(doi="10.1/y"))
        decision, reasons = decide_flagged(upd, cr)
        assert decision == "revert"
        assert any("disagree" in r.lower() for r in reasons)

    def test_revert_on_problematic_status(self):
        from bibtex_updater.chain import decide_flagged

        upd = {"ID": "k", "doi": "10.1/x"}
        cr = SimpleNamespace(status="HALLUCINATED", best_match=None)
        decision, reasons = decide_flagged(upd, cr)
        assert decision == "revert"


class TestChainReport:
    """The merged report distinguishes resolved-skip from actually-checked."""

    def test_upgraded_marked_skipped_and_checked_carry_status(self):
        results = [_result("a", "upgraded"), _result("b", "unchanged")]
        check_results = [SimpleNamespace(entry_key="b", status="VERIFIED")]
        report = build_chain_report(results, check_results)

        by_key = {e["key"]: e for e in report["entries"]}
        assert by_key["a"]["check"] == "skipped_resolved"
        assert by_key["b"]["check"] == "VERIFIED"
        assert report["summary"]["resolved_skipped"] == 1
        assert report["summary"]["checked"] == 1

    def test_enum_status_is_normalized_to_value(self):
        """A FactCheckStatus enum is reported by its ``.value``, not its repr."""
        results = [_result("b", "unchanged")]
        check_results = [SimpleNamespace(entry_key="b", status=SimpleNamespace(value="hallucinated"))]
        report = build_chain_report(results, check_results)
        assert report["entries"][0]["check"] == "hallucinated"

    def test_flagged_upgrade_is_reported_with_reasons_and_decision(self):
        """A flagged, reverted upgrade is surfaced -- no silent rewrite."""
        from bibtex_updater.chain import UpgradeTrust

        results = [_result("a", "upgraded")]
        trust = {
            "a": UpgradeTrust(trusted=False, title_score=0.80, author_score=1.0, reasons=["title match 0.80 < 0.90"])
        }
        decisions = {"a": ("revert", ["independent retrieval disagrees (found 10.1/y, resolver chose 10.1/x)"])}
        check_results = [SimpleNamespace(entry_key="a", status="VERIFIED", best_match=None)]
        report = build_chain_report(results, check_results, trust_by_key=trust, decision_by_key=decisions)

        e = report["entries"][0]
        assert e["trust"] == "flagged"
        assert e["decision"] == "revert"
        assert any("title" in r for r in e["reasons"])
        assert any("disagree" in r.lower() for r in e["reasons"])
        assert report["summary"]["flagged"] == 1
        assert report["summary"]["reverted"] == 1


class TestCheckerFactory:
    """The extracted checker factory builds a working processor on shared infra."""

    def test_build_checker_processor_returns_processor_and_http(self):
        from bibtex_updater.fact_checker import build_checker_processor, build_parser

        args = build_parser().parse_args(["x.bib", "--no-cache"])
        processor, http = build_checker_processor(args, logging.getLogger("t"))
        assert processor is not None
        assert http is not None

    def test_factory_reuses_a_supplied_http(self):
        """When given an http client, the factory shares it instead of building one."""
        from bibtex_updater.fact_checker import build_checker_processor, build_parser

        args = build_parser().parse_args(["x.bib", "--no-cache"])
        processor1, http = build_checker_processor(args, logging.getLogger("t"))
        processor2, http2 = build_checker_processor(args, logging.getLogger("t"), http=http)
        assert http2 is http


class TestArgBridging:
    """Each flag lives on one CLI; the chain synthesizes the other arg namespace."""

    def test_update_then_check_flag_and_bridge(self):
        from bibtex_updater.chain import _bridge_check_args
        from bibtex_updater.updater import build_arg_parser

        update_args = build_arg_parser().parse_args(
            ["refs.bib", "--then-check", "--cache", "/tmp/c.sqlite", "--rate-limit", "30", "--max-workers", "6"]
        )
        assert update_args.then_check is True

        check_args = _bridge_check_args(update_args)
        assert check_args.cache_file == "/tmp/c.sqlite"
        assert check_args.rate_limit == 30
        assert check_args.workers == 6

    def test_check_resolve_first_flag_and_bridge(self):
        from bibtex_updater.chain import _bridge_resolve_args, _default_resolved_out
        from bibtex_updater.fact_checker import build_parser

        check_args = build_parser().parse_args(
            ["refs.bib", "--resolve-first", "--cache-file", "/tmp/c.sqlite", "--rate-limit", "25", "--workers", "5"]
        )
        assert check_args.resolve_first is True

        resolve_args = _bridge_resolve_args(check_args)
        assert resolve_args.cache == "/tmp/c.sqlite"
        assert resolve_args.rate_limit == 25
        assert resolve_args.max_workers == 5
        assert _default_resolved_out(["refs.bib"]) == "refs.resolved.bib"
