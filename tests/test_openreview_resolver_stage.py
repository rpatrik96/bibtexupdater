"""Tests for the OpenReview resolver stage 3c (accepted submissions only)."""

from __future__ import annotations

import asyncio
import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from bibtex_updater import AsyncResolver, Resolver
from bibtex_updater.sources import build_openreview_paperhashes
from bibtex_updater.utils import (
    OPENREVIEW_API,
    OPENREVIEW_API_V2,
    AsyncHttpClient,
    AsyncRateLimiterRegistry,
    OpenReviewAuth,
    latex_to_plain,
    normalize_title_for_match,
)

VIT_TITLE = "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

NOTES_V1 = f"{OPENREVIEW_API}/notes"
NOTES_V2 = f"{OPENREVIEW_API_V2}/notes"
SEARCH_V1 = f"{OPENREVIEW_API}/notes/search"
SEARCH_V2 = f"{OPENREVIEW_API_V2}/notes/search"


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

        async def get(self, url, service=None, params=None, accept=None, **kwargs):
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


# ===========================================================================
# #67: the async stage asks both hosts, under the run's credentials
# ===========================================================================


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {"notes": []}

    def json(self):
        return self._payload


def _notes(*notes) -> _Resp:
    return _Resp(200, {"notes": list(notes)})


def _refused(status=403) -> _Resp:
    return _Resp(status, {"name": "ChallengeRequiredError", "message": "Challenge verification required"})


class _ScriptedHttp:
    """Stands in for :class:`AsyncHttpClient`: answers by URL, records every call.

    Each URL maps to a list of responses consumed in order (the last one
    repeats); an exception in the list is raised. URLs without a script answer
    an empty note list. ``openreview_auth`` is what the resolver reads to tell
    an authenticated refusal from an anonymous one.
    """

    def __init__(self, script=None, openreview_auth=None):
        self._script = {url: list(responses) for url, responses in (script or {}).items()}
        self.openreview_auth = openreview_auth
        self.calls: list[dict] = []

    async def get(self, url, service=None, params=None, accept=None, rate_limit_service=None):
        self.calls.append(
            {"url": url, "params": dict(params or {}), "service": service, "rate_limit_service": rate_limit_service}
        )
        queue = self._script.get(url)
        if not queue:
            return _notes()
        nxt = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def urls(self) -> list[str]:
        return [c["url"] for c in self.calls]


class _Authenticated:
    """An authenticated run, as the resolver sees it: a token for every OpenReview URL."""

    def token_for_url(self, url):
        return "run-token"


def _search(resolver, entry=None):
    entry = entry or _vit_entry()
    return asyncio.run(resolver._openreview_search(entry, normalize_title_for_match(entry["title"])))


class TestAsyncStage3cAsksBothHosts:
    """The two hosts hold disjoint sets of notes (#63): v2 has ICLR 2024+,
    NeurIPS 2023+, TMLR and COLM, v1 the pre-2023 venues. The async stage asked
    v1 alone, so it could never confirm the modern half of a bibliography."""

    def test_the_paperhash_lookup_goes_to_v2_then_v1(self, logger):
        http = _ScriptedHttp({NOTES_V2: [_notes()], NOTES_V1: [_notes(_accepted_note())]})
        rec = _search(AsyncResolver(http=http, logger=logger))
        assert rec is not None and rec.journal == "ICLR 2021"
        assert http.urls() == [NOTES_V2, NOTES_V1]
        assert all("paperhash" in c["params"] for c in http.calls)

    def test_a_hit_on_v2_does_not_ask_v1(self, logger):
        http = _ScriptedHttp({NOTES_V2: [_notes(_accepted_note())]})
        assert _search(AsyncResolver(http=http, logger=logger)) is not None
        assert http.urls() == [NOTES_V2]

    def test_every_request_is_tagged_as_openreview_so_the_client_attaches_the_token(self, logger):
        """``AsyncHttpClient`` keys the bearer token on ``service="openreview"``."""
        http = _ScriptedHttp({SEARCH_V1: [_notes(_accepted_note())]})
        _search(AsyncResolver(http=http, logger=logger))
        assert http.urls() == [NOTES_V2, NOTES_V1, SEARCH_V2, SEARCH_V1]
        assert [c["service"] for c in http.calls] == ["openreview"] * 4
        # One source, two declared limits: 180/min on /notes, 5/min on /notes/search.
        assert [c["rate_limit_service"] for c in http.calls] == [
            "openreview",
            "openreview",
            "openreview_search",
            "openreview_search",
        ]

    def test_the_term_fallback_runs_on_both_hosts_after_a_miss(self, logger):
        http = _ScriptedHttp({SEARCH_V1: [_notes(_accepted_note())]})
        rec = _search(AsyncResolver(http=http, logger=logger))
        assert rec is not None
        term_calls = [c for c in http.calls if c["url"] in (SEARCH_V1, SEARCH_V2)]
        assert [c["url"] for c in term_calls] == [SEARCH_V2, SEARCH_V1]
        assert all(c["params"]["term"] == latex_to_plain(VIT_TITLE).strip() for c in term_calls)

    def test_both_index_forms_of_the_hash_are_issued_for_an_accented_author(self, logger):
        """OpenReview keeps Latin-1 diacritics in the index, so the folded
        surname alone misses every accented author; both forms go out, built
        from the raw BibTeX name as the checker does."""
        entry = dict(_vit_entry(), author="Akyürek, Ekin and Beyer, Lucas")
        http = _ScriptedHttp()
        _search(AsyncResolver(http=http, logger=logger), entry)
        hashes = [c["params"]["paperhash"] for c in http.calls if c["url"] == NOTES_V2]
        assert hashes == build_openreview_paperhashes(VIT_TITLE, "Akyürek, Ekin")
        assert len(hashes) == 2 and hashes[0].startswith("akyürek|") and hashes[1].startswith("akyurek|")


class TestAsyncStage3cRefusals:
    """A 401/403 is handled the way #65 settled it for the sync client: an
    anonymous refusal is a configuration state and latches the endpoint for the
    run; an authenticated one is a blip, reported once and forgotten."""

    def test_an_anonymous_refusal_latches_each_endpoint_and_is_reported_once(self, logger, caplog):
        http = _ScriptedHttp({NOTES_V2: [_refused()], NOTES_V1: [_refused()]})
        resolver = AsyncResolver(http=http, logger=logger)
        with caplog.at_level(logging.WARNING, logger=logger.name):
            for _ in range(3):
                assert _search(resolver) is None
        # One request per host on the first entry only; the fallback still runs for every entry.
        assert sorted(u for u in http.urls() if u in (NOTES_V1, NOTES_V2)) == sorted([NOTES_V1, NOTES_V2])
        assert http.urls().count(SEARCH_V2) == 3 and http.urls().count(SEARCH_V1) == 3
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert all("OPENREVIEW_USERNAME" in w and "anonymous" in w for w in warnings)

    def test_the_anonymous_fallback_still_contributes(self, logger):
        """No credentials, both ``/notes`` gated, and the full-text search still resolves the entry."""
        http = _ScriptedHttp({NOTES_V2: [_refused()], NOTES_V1: [_refused()], SEARCH_V2: [_notes(_accepted_note())]})
        rec = _search(AsyncResolver(http=http, logger=logger))
        assert rec is not None and rec.method == "OpenReview(search,parallel)"

    def test_an_authenticated_refusal_does_not_latch(self, logger, caplog):
        """One refused v2 lookup, then the next entry asks v2 again and gets its note."""
        http = _ScriptedHttp(
            {NOTES_V2: [_refused(), _refused(), _notes(_accepted_note())]}, openreview_auth=_Authenticated()
        )
        resolver = AsyncResolver(http=http, logger=logger)
        with caplog.at_level(logging.WARNING, logger=logger.name):
            assert _search(resolver) is None
            assert _search(resolver) is None
            rec = _search(resolver)
        assert rec is not None and rec.journal == "ICLR 2021"
        assert http.urls().count(NOTES_V2) == 3
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "authenticated" in warnings[0]

    def test_a_refusal_on_one_host_leaves_the_other_alone(self, logger):
        """v1 and v2 index disjoint venues, so one gate must not close both."""
        http = _ScriptedHttp({NOTES_V2: [_refused()], NOTES_V1: [_notes(_accepted_note())]})
        rec = _search(AsyncResolver(http=http, logger=logger))
        assert rec is not None
        assert http.urls() == [NOTES_V2, NOTES_V1]

    def test_a_transient_failure_does_not_latch(self, logger):
        """A timeout or an exhausted retry budget is not a refusal; the next entry asks again."""
        http = _ScriptedHttp({NOTES_V2: [RuntimeError("Network failure after retries"), _notes(_accepted_note())]})
        resolver = AsyncResolver(http=http, logger=logger)
        assert _search(resolver) is None
        assert _search(resolver) is not None
        assert http.urls().count(NOTES_V2) == 2

    def test_a_401_is_a_refusal_too(self, logger):
        http = _ScriptedHttp({NOTES_V2: [_refused(401)], NOTES_V1: [_refused(401)]})
        resolver = AsyncResolver(http=http, logger=logger)
        for _ in range(2):
            assert _search(resolver) is None
        assert sorted(u for u in http.urls() if u in (NOTES_V1, NOTES_V2)) == sorted([NOTES_V1, NOTES_V2])

    def test_no_credentials_and_nothing_found_returns_none_cleanly(self, logger, caplog):
        http = _ScriptedHttp({NOTES_V2: [_refused()], NOTES_V1: [_refused()]})
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            assert _search(AsyncResolver(http=http, logger=logger)) is None
        assert "OPENREVIEW_PASSWORD" in caplog.text


class _FakeLoginClient:
    """Stands in for the bare ``httpx.Client`` the login POST goes out on; records the thread."""

    def __init__(self, token):
        self._token = token
        self.calls: list[tuple[str, dict]] = []
        self.threads: list[int] = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        self.calls.append((url, dict(json or {})))
        self.threads.append(threading.get_ident())
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"token": self._token, "user": {"id": "~Test_User1"}}
        return resp


class _GatedTransport:
    """Fake ``httpx.AsyncClient``: ``/notes`` answers 200 to a bearer token and 403 without one."""

    def __init__(self, token, note):
        self._token = token
        self._note = note
        self.calls: list[tuple[str, dict]] = []

    async def request(self, method, url, params=None, json=None, headers=None):
        headers = dict(headers or {})
        self.calls.append((url, headers))
        if url.endswith("/notes/search"):
            return _notes()
        if headers.get("Authorization") == f"Bearer {self._token}":
            return _notes(self._note)
        return _refused()

    async def aclose(self):
        return None


class TestAsyncStage3cCarriesCredentials:
    """End to end: the same credentials ``bibtex-check`` uses, on the async path.

    Before #67 ``AsyncHttpClient`` took no ``openreview_auth``, so this request
    went out anonymous, met the challenge gate, and the stage returned nothing.
    """

    @pytest.fixture(autouse=True)
    def _no_pacing(self, monkeypatch):
        async def _wait(self_, service):
            return None

        monkeypatch.setattr("bibtex_updater.utils.AsyncRateLimiterRegistry.wait", _wait)

    def test_the_resolver_authenticates_through_the_async_client(self, logger, tmp_path):
        # The same config surface the checker reads; the token store is pointed
        # at a scratch file so the run never touches the developer's real cache.
        auth = OpenReviewAuth.from_env(
            env={
                "OPENREVIEW_USERNAME": "a@b.c",
                "OPENREVIEW_PASSWORD": "pw",
                "BIBTEX_CHECK_OPENREVIEW_TOKEN_CACHE": str(tmp_path / "openreview-tokens.json"),
            }
        )
        assert auth is not None
        http = AsyncHttpClient(rate_limiters=AsyncRateLimiterRegistry(), cache=None, openreview_auth=auth)
        transport = _GatedTransport("run-token", _accepted_note())
        http._client = transport  # type: ignore[assignment]
        login = _FakeLoginClient("run-token")

        with patch("bibtex_updater.utils.httpx.Client", login):
            rec = _search(AsyncResolver(http=http, logger=logger))

        assert rec is not None and rec.journal == "ICLR 2021"
        assert transport.calls[0][0] == NOTES_V2
        assert transport.calls[0][1]["Authorization"] == "Bearer run-token"
        assert login.calls == [(f"{OPENREVIEW_API_V2}/login", {"id": "a@b.c", "password": "pw"})]
        # The blocking login POST ran on a worker thread, not on the event loop.
        assert login.threads == [login.threads[0]] and login.threads[0] != threading.get_ident()

    def test_without_credentials_the_same_run_is_refused_and_says_so(self, logger, caplog):
        http = AsyncHttpClient(rate_limiters=AsyncRateLimiterRegistry(), cache=None)
        transport = _GatedTransport("run-token", _accepted_note())
        http._client = transport  # type: ignore[assignment]
        with caplog.at_level(logging.WARNING, logger=logger.name):
            rec = _search(AsyncResolver(http=http, logger=logger))
        assert rec is None
        assert all("Authorization" not in h for _, h in transport.calls)
        assert "OPENREVIEW_USERNAME" in caplog.text
