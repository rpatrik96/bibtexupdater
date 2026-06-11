"""Tests for the OpenReview resolver stage 3c (accepted submissions only)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from bibtex_updater import AsyncResolver, Resolver
from bibtex_updater.utils import normalize_title_for_match

VIT_TITLE = "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"


def _note(title, authors, authorids, venue, venueid=None, forum="AbCd1234"):
    content = {"title": title, "authors": authors, "authorids": authorids, "venue": venue}
    if venueid is not None:
        content["venueid"] = venueid
    note = {"content": content}
    if forum is not None:
        note["id"] = forum
        note["forum"] = forum
    return note


def _vit_entry():
    return {
        "ID": "vit",
        "ENTRYTYPE": "article",
        "title": VIT_TITLE,
        "author": "Dosovitskiy, Alexey and Beyer, Lucas",
        "year": "2021",
    }


def _accepted_note():
    return _note(
        VIT_TITLE, ["Alexey Dosovitskiy", "Lucas Beyer"], ["~Alexey_Dosovitskiy1", "~Lucas_Beyer1"], "ICLR 2021"
    )


class TestStage3cAcceptedOnly:
    @pytest.fixture
    def resolver(self, fake_http, logger):
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    def test_accepted_resolves_to_proceedings(self, resolver):
        resolver.openreview = MagicMock()
        resolver.openreview.search.return_value = [_accepted_note()]
        rec = resolver._stage3c_openreview(_vit_entry(), normalize_title_for_match(VIT_TITLE))
        assert rec is not None
        assert rec.type == "proceedings-article"
        assert rec.journal == "ICLR 2021"
        assert rec.url == "https://openreview.net/forum?id=AbCd1234"
        assert rec.method == "OpenReview(search)"

    @pytest.mark.parametrize(
        "venue,venueid",
        [
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference/Withdrawn_Submission"),  # withdrawn
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference/Rejected_Submission"),  # rejected
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference"),  # under review
            ("CoRR 2020", "dblp.org/journals/CORR/2020"),  # preprint mirror
        ],
    )
    def test_not_accepted_does_not_resolve(self, resolver, venue, venueid):
        note = _note(
            VIT_TITLE, ["Alexey Dosovitskiy", "Lucas Beyer"], ["~Alexey_Dosovitskiy1", "~Lucas_Beyer1"], venue, venueid
        )
        resolver.openreview = MagicMock()
        resolver.openreview.search.return_value = [note]
        rec = resolver._stage3c_openreview(_vit_entry(), normalize_title_for_match(VIT_TITLE))
        assert rec is None

    def test_title_mismatch_does_not_resolve(self, resolver):
        note = _note(
            "A Totally Different Paper About Bananas", ["Alexey Dosovitskiy"], ["~Alexey_Dosovitskiy1"], "ICLR 2021"
        )
        resolver.openreview = MagicMock()
        resolver.openreview.search.return_value = [note]
        rec = resolver._stage3c_openreview(_vit_entry(), normalize_title_for_match(VIT_TITLE))
        assert rec is None

    def test_no_forum_id_does_not_resolve(self, resolver):
        note = _note(
            VIT_TITLE,
            ["Alexey Dosovitskiy", "Lucas Beyer"],
            ["~Alexey_Dosovitskiy1", "~Lucas_Beyer1"],
            "ICLR 2021",
            forum=None,
        )
        resolver.openreview = MagicMock()
        resolver.openreview.search.return_value = [note]
        rec = resolver._stage3c_openreview(_vit_entry(), normalize_title_for_match(VIT_TITLE))
        assert rec is None

    def test_resolver_wires_openreview_client(self, resolver):
        from bibtex_updater.sources import OpenReviewClient

        assert isinstance(resolver.openreview, OpenReviewClient)


class TestAsyncStage3cAcceptedOnly:
    class _Resp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return self._payload

    class _Http:
        def __init__(self, payload):
            self._payload = payload

        async def get(self, url, service=None, params=None, accept=None):
            return TestAsyncStage3cAcceptedOnly._Resp(self._payload)

    def test_async_accepted_resolves(self, logger):
        resolver = AsyncResolver(http=self._Http({"notes": [_accepted_note()]}), logger=logger)
        rec = asyncio.run(resolver._openreview_search(_vit_entry(), normalize_title_for_match(VIT_TITLE)))
        assert rec is not None
        assert rec.type == "proceedings-article"
        assert rec.journal == "ICLR 2021"
        assert rec.method == "OpenReview(search,parallel)"

    def test_async_rejected_does_not_resolve(self, logger):
        note = _note(
            VIT_TITLE,
            ["Alexey Dosovitskiy", "Lucas Beyer"],
            ["~Alexey_Dosovitskiy1", "~Lucas_Beyer1"],
            "Submitted to ICLR 2024",
            "ICLR.cc/2024/Conference/Rejected_Submission",
        )
        resolver = AsyncResolver(http=self._Http({"notes": [note]}), logger=logger)
        rec = asyncio.run(resolver._openreview_search(_vit_entry(), normalize_title_for_match(VIT_TITLE)))
        assert rec is None
