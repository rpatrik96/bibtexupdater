"""Tests for the env-gated `unpublished_at_claimed_venue` verification flag.

When OpenReview's best match for a citation is a NOT-ACCEPTED submission
(rejected / withdrawn / under-review) at the cited venue, the paper is real but
was not published there. Gated off by default (BIBTEX_CHECK_OR_UNPUBLISHED_FLAG)
pending a HALLMARK FPR check; the acceptance status is stamped on the record by
the OpenReview converter, so the verdict pipeline never needs the note.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from bibtex_updater.fact_checker import FactChecker, FactCheckerConfig, FactCheckStatus
from bibtex_updater.sources import OR_ACCEPTED, OR_NOT_ACCEPTED
from bibtex_updater.utils import PublishedRecord


def _checker():
    return FactChecker(MagicMock(), MagicMock(), MagicMock(), FactCheckerConfig(), logging.getLogger("test"))


def _rec(acceptance):
    return PublishedRecord(doi="", title="X", journal="Submitted to ICLR 2024", year=2024, acceptance=acceptance)


class TestOpenReviewUnpublishedFlag:
    def test_gate_off_by_default_returns_none(self):
        entry = {"booktitle": "ICLR 2024", "title": "X"}
        assert _checker()._check_or_unpublished(entry, _rec(OR_NOT_ACCEPTED)) is None

    def test_gate_on_not_accepted_with_claimed_venue_flags(self, monkeypatch):
        monkeypatch.setenv("BIBTEX_CHECK_OR_UNPUBLISHED_FLAG", "1")
        entry = {"booktitle": "ICLR 2024", "title": "X"}
        result = _checker()._check_or_unpublished(entry, _rec(OR_NOT_ACCEPTED))
        assert result is FactCheckStatus.UNPUBLISHED_AT_CLAIMED_VENUE

    def test_gate_on_accepted_returns_none(self, monkeypatch):
        monkeypatch.setenv("BIBTEX_CHECK_OR_UNPUBLISHED_FLAG", "1")
        entry = {"booktitle": "ICLR 2024", "title": "X"}
        assert _checker()._check_or_unpublished(entry, _rec(OR_ACCEPTED)) is None

    def test_gate_on_no_claimed_venue_returns_none(self, monkeypatch):
        monkeypatch.setenv("BIBTEX_CHECK_OR_UNPUBLISHED_FLAG", "1")
        entry = {"title": "X"}  # no booktitle/journal -> nothing claimed
        assert _checker()._check_or_unpublished(entry, _rec(OR_NOT_ACCEPTED)) is None

    def test_gate_on_preprint_claimed_venue_returns_none(self, monkeypatch):
        monkeypatch.setenv("BIBTEX_CHECK_OR_UNPUBLISHED_FLAG", "1")
        entry = {"journal": "arXiv preprint", "title": "X"}
        assert _checker()._check_or_unpublished(entry, _rec(OR_NOT_ACCEPTED)) is None

    def test_non_openreview_record_unaffected(self, monkeypatch):
        # a non-OpenReview record has acceptance=None (default) -> never flagged
        monkeypatch.setenv("BIBTEX_CHECK_OR_UNPUBLISHED_FLAG", "1")
        entry = {"booktitle": "ICLR 2024", "title": "X"}
        plain = PublishedRecord(doi="10.1/x", title="X", journal="ICLR 2024", year=2024)
        assert _checker()._check_or_unpublished(entry, plain) is None
