"""Tests for the 2026-06-02 follow-up fixes:

- shared ``is_preprint_venue`` preprint/CoRR guard,
- the OpenAlex arXiv lookup using the arXiv DOI (the bare ``arxiv:`` route 404s),
- Retry-After-aware HTTP backoff,
- the resolver credibility gate + OpenReview converter rejecting CoRR venues.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import httpx
import pytest

from bibtex_updater import PublishedRecord, Resolver
from bibtex_updater.sources import openreview_note_to_candidate_record
from bibtex_updater.utils import is_preprint_venue, retry_after_seconds


class TestIsPreprintVenue:
    @pytest.mark.parametrize(
        "venue,expected",
        [
            ("CoRR 2017", True),
            ("CoRR", True),
            ("CoRR abs/1706.03762", True),
            ("arXiv preprint", True),
            ("arXiv 2010.11929", True),
            ("bioRxiv", True),
            ("medRxiv", True),
            ("Corrosion Science", False),  # 'corr' substring but not a word boundary
            ("ICLR 2021", False),
            ("Advances in Neural Information Processing Systems", False),
            ("Journal of Machine Learning Research", False),
            ("", False),
            (None, False),
        ],
    )
    def test_classifies_venue(self, venue, expected):
        assert is_preprint_venue(venue) is expected


class TestRetryAfterSeconds:
    @staticmethod
    def _exc(retry_after=None):
        resp = MagicMock()
        resp.headers = {"Retry-After": retry_after} if retry_after is not None else {}
        exc = httpx.HTTPError("boom")
        exc.response = resp
        return exc

    def test_honors_retry_after(self):
        assert retry_after_seconds(self._exc("5"), fallback=1.0) == 5.0

    def test_caps_retry_after(self):
        assert retry_after_seconds(self._exc("9999"), fallback=1.0, cap=60.0) == 60.0

    def test_falls_back_without_header(self):
        assert retry_after_seconds(self._exc(None), fallback=2.0) == 2.0

    def test_falls_back_on_garbage_header(self):
        assert retry_after_seconds(self._exc("soon"), fallback=2.0) == 2.0

    def test_no_response_attribute(self):
        exc = httpx.HTTPError("boom")
        assert retry_after_seconds(exc, fallback=3.0) == 3.0


class TestCredibilityGateRejectsCoRR:
    @staticmethod
    def _resolver():
        return Resolver(http=MagicMock(), logger=logging.getLogger("test"))

    def test_corr_record_not_credible(self):
        rec = PublishedRecord(
            doi="",
            title="Attention Is All You Need",
            journal="CoRR 2017",
            year=2017,
            url="https://openreview.net/forum?id=x",
            type="proceedings-article",
        )
        assert self._resolver()._credible_journal_article(rec) is False

    def test_real_conference_record_credible(self):
        rec = PublishedRecord(
            doi="",
            title="Attention Is All You Need",
            journal="NeurIPS 2017",
            year=2017,
            url="https://proceedings.neurips.cc/paper/2017",
            type="proceedings-article",
        )
        assert self._resolver()._credible_journal_article(rec) is True


class TestOpenAlexArxivUsesDOIScheme:
    def test_uses_arxiv_doi_not_bare_arxiv_scheme(self):
        captured = {}

        def _req(method, url, **kwargs):
            captured["url"] = url
            resp = MagicMock()
            resp.status_code = 404
            return resp

        http = MagicMock()
        http._request.side_effect = _req
        resolver = Resolver(http=http, logger=logging.getLogger("test"))
        resolver.openalex_from_arxiv("2010.11929")
        assert "doi:10.48550/arXiv.2010.11929" in captured["url"]
        assert "arxiv:2010.11929" not in captured["url"]


class TestOpenReviewConverterDropsCoRRVenue:
    @staticmethod
    def _note(venue):
        return {
            "id": "x",
            "forum": "x",
            "content": {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani"],
                "authorids": ["~Ashish_Vaswani1"],
                "venue": venue,
            },
        }

    def test_corr_venue_zeroed(self):
        rec = openreview_note_to_candidate_record(self._note("CoRR 2017"))
        assert rec is not None
        assert rec.journal is None  # CoRR dropped, not confirmed as a venue
        assert rec.year is None  # and no published year recovered from the CoRR string

    def test_real_venue_preserved(self):
        rec = openreview_note_to_candidate_record(self._note("ICLR 2021"))
        assert rec is not None
        assert rec.journal == "ICLR 2021"
        assert rec.year == 2021
