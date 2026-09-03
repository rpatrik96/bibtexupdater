"""OpenReview's challenge gate, and the optional credentials that get past it.

OpenReview stopped serving ``/notes`` to anonymous callers. Both
``api.openreview.net/notes?paperhash=`` -- the exact title + first-author lookup
the cascade relies on -- and ``api2.openreview.net/notes`` answer
``403 ChallengeRequiredError``. Measured during a 5,043-reference screening run:
68 of 68 sampled OpenReview lookups failed with 403, so the source contributed
nothing while still costing a round trip and a circuit-breaker tick per entry.

Two things follow, and both are tested here. A 403 is a refusal, never an
answer, so it can never support the exhaustive ``not_found`` claim: the lookup
keeps failing for every entry. And a refusal is a configuration state rather
than a blip, so the refused endpoint is latched for the run and later entries
fail without issuing a request at all.

Credentials are optional throughout: every test that does not supply them
asserts the anonymous path still behaves, and the full-text fallbacks stay
where they were, reached after the exact lookup answered and found nothing.

All network access is faked; no test here touches a live host.
"""

from __future__ import annotations

import base64
import json
import logging
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bibtex_updater.sources import OpenReviewClient
from bibtex_updater.utils import (
    OPENREVIEW_API,
    OPENREVIEW_API_V2,
    HttpClient,
    OpenReviewAuth,
    OpenReviewTokenStore,
    RateLimiterRegistry,
    SourceUnavailableError,
    _default_openreview_token_cache_path,
    _jwt_expiry,
    raise_for_failed_lookup,
)

PASSWORD = "correct-horse-battery-staple"
TOKEN_V1 = "v1-token-aaaaaaaaaaaa"
TOKEN_V2 = "v2-token-bbbbbbbbbbbb"

NOTES_V1 = f"{OPENREVIEW_API}/notes"
NOTES_V2 = f"{OPENREVIEW_API_V2}/notes"
SEARCH_V1 = f"{OPENREVIEW_API}/notes/search"
SEARCH_V2 = f"{OPENREVIEW_API_V2}/notes/search"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """No retry backoff and no rate-limiter pacing."""
    monkeypatch.setattr("bibtex_updater.utils.time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("bibtex_updater.utils.RateLimiter.wait", lambda self: None)


def _notes(notes) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = {"notes": list(notes)}
    return resp


def _status(code: int, name: str = "ChallengeRequiredError") -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = {
        "name": name,
        "message": "Challenge verification required",
        "status": code,
    }
    return resp


def _note(title="Adam: A Method", ident="note1"):
    return {
        "id": ident,
        "forum": ident,
        "content": {
            "title": title,
            "authors": ["D Kingma"],
            "authorids": ["~Diederik_Kingma1"],
            "venue": "ICLR 2015",
            "venueid": "ICLR.cc/2015/Conference",
        },
    }


def _urls(http_mock) -> list[str]:
    """The URLs a mocked ``HttpClient._request`` was asked for, in order."""
    return [call.args[1] if len(call.args) > 1 else call.kwargs["url"] for call in http_mock._request.call_args_list]


# ===========================================================================
# A refusal carries its status, so callers can tell it from a throttle
# ===========================================================================


class TestFailedLookupCarriesStatus:
    def test_403_is_a_failed_lookup_with_its_status(self):
        with pytest.raises(SourceUnavailableError) as exc_info:
            raise_for_failed_lookup("openreview", NOTES_V1, 403)
        exc = exc_info.value
        assert exc.status_code == 403
        # The host answered, so the network is demonstrably up. The lookup still
        # failed: a refusal says nothing about whether the paper exists.
        assert exc.transport_failure is False

    def test_5xx_keeps_transport_semantics(self):
        with pytest.raises(SourceUnavailableError) as exc_info:
            raise_for_failed_lookup("openreview", NOTES_V1, 503)
        assert exc_info.value.status_code == 503
        assert exc_info.value.transport_failure is True

    def test_absence_status_still_answers(self):
        # 404 asserts absence and must keep raising nothing.
        assert raise_for_failed_lookup("openreview", NOTES_V1, 404) is None


# ===========================================================================
# The challenge gate: a 403 is never absence, and is asked once per run
# ===========================================================================


class TestChallengeGatedNotesEndpoint:
    def test_403_is_a_failed_lookup_not_an_empty_result(self):
        """The claim ``not_found`` rests on is that every source answered.

        A challenge gate is a refusal, so OpenReview has said nothing about the
        entry and the cascade must not read the missing answer as absence.
        """
        http = MagicMock()
        http._request.return_value = _status(403)
        client = OpenReviewClient(http=http)

        with pytest.raises(SourceUnavailableError) as exc_info:
            client.search("blob", title="Adam: A Method", first_author="kingma")
        assert exc_info.value.status_code == 403
        assert exc_info.value.service == "openreview"

    def test_a_refused_endpoint_is_asked_once_per_run(self):
        """68 of 68 sampled lookups in one screening run were refused alike.

        Re-issuing the same refused request for every entry buys a round trip
        and a circuit-breaker tick for an answer that is not coming, so the
        refusal is remembered and later entries fail without a request.
        """
        http = MagicMock()
        http.openreview_auth = None  # anonymous: what the latch is for
        http._request.return_value = _status(403)
        client = OpenReviewClient(http=http)

        for _ in range(5):
            with pytest.raises(SourceUnavailableError):
                client.search("blob", title="Adam: A Method", first_author="kingma")

        # Both hosts are asked once, then both are latched.
        assert _urls(http) == [NOTES_V2, NOTES_V1]

    def test_the_latch_still_fails_every_entry_it_skips(self):
        """Skipping the request must not quietly become an answer.

        A skipped lookup is still a failed lookup: it keeps blocking the
        exhaustive ``not_found`` claim exactly as the first refusal did, and it
        names the missing configuration.
        """
        http = MagicMock()
        http.openreview_auth = None  # anonymous: what the latch is for
        http._request.return_value = _status(403)
        client = OpenReviewClient(http=http)
        with pytest.raises(SourceUnavailableError):
            client.search("blob", title="Adam: A Method", first_author="kingma")

        with pytest.raises(SourceUnavailableError) as exc_info:
            client.search("blob", title="Another Paper Entirely", first_author="hopper")
        exc = exc_info.value
        assert exc.status_code == 403
        assert exc.transport_failure is False
        assert "OPENREVIEW_USERNAME" in str(exc)

    def test_a_transient_failure_does_not_latch(self):
        """Only a refusal is a configuration state.

        A timeout or a 5xx is transient and belongs to the circuit breaker,
        which already paces it, so the endpoint is asked again next entry.
        """
        http = MagicMock()
        # Both hosts 5xx on the first entry, then v2 answers on the second.
        http._request.side_effect = [_status(503), _status(503), _notes([_note()])]
        client = OpenReviewClient(http=http)

        with pytest.raises(SourceUnavailableError):
            client.search("blob", title="Adam: A Method", first_author="kingma")
        assert client.search("blob", title="Adam: A Method", first_author="kingma") == [_note()]
        assert _urls(http) == [NOTES_V2, NOTES_V1, NOTES_V2]

    def test_a_refusal_does_not_spend_the_search_budget(self):
        """The full-text fallbacks are not a substitute for the exact lookup.

        ``/notes/search`` declares 5 requests per minute on the v1 host and 20
        on v2, against 180 on ``/notes``. Routing every entry of a
        5,000-reference run through them would make OpenReview the bottleneck
        for the whole cascade, and they answer with reviews and unrelated notes
        besides. They stay where they were: reached after the exact lookup
        answered and found nothing.
        """
        http = MagicMock()
        http._request.return_value = _status(403)
        client = OpenReviewClient(http=http)
        with pytest.raises(SourceUnavailableError):
            client.search("blob", title="Adam: A Method", first_author="kingma")
        assert SEARCH_V1 not in _urls(http)
        assert SEARCH_V2 not in _urls(http)

    def test_the_two_endpoint_families_are_paced_separately(self):
        """One source, three declared limits.

        ``/notes`` allows 180 requests per minute, v1 ``/notes/search`` allows 5
        and v2 ``/notes/search`` allows 20, so a single limiter either throttles
        the exact lookup to a crawl or walks the search endpoints into a 429.
        The circuit breaker and ``sources_failed`` still see one source.
        """
        http = MagicMock()
        http._request.side_effect = [_notes([]), _notes([]), _notes([]), _notes([])]
        OpenReviewClient(http=http).search("blob", title="Adam: A Method", first_author="kingma")

        tagged = [(c.kwargs["service"], c.kwargs["rate_limit_service"]) for c in http._request.call_args_list]
        assert tagged == [
            ("openreview", "openreview"),
            ("openreview", "openreview"),
            ("openreview", "openreview_search"),
            ("openreview", "openreview_search"),
        ]

    def test_a_clean_empty_search_is_still_absence(self):
        """Nothing above may weaken the ordinary miss.

        Every endpoint answered and none holds the paper, so the search reports
        an empty result and ``not_found`` remains available to the cascade.
        """
        http = MagicMock()
        http._request.side_effect = [_notes([]), _notes([]), _notes([]), _notes([])]
        client = OpenReviewClient(http=http)
        assert client.search("blob", title="Adam: A Method", first_author="kingma") == []

    def test_the_term_fallbacks_run_after_a_genuine_miss(self):
        """An authenticated run that misses on paperhash still gets both hosts.

        LaTeX escapes and author-name drift make the exact hash miss on papers
        OpenReview does hold, and a note on one host is invisible on the other.
        """
        http = MagicMock()
        http._request.side_effect = [_notes([]), _notes([]), _notes([]), _notes([_note()])]
        client = OpenReviewClient(http=http)

        assert client.search("blob", title="Adam: A Method", first_author="kingma") == [_note()]
        assert _urls(http) == [NOTES_V2, NOTES_V1, SEARCH_V2, SEARCH_V1]


# ===========================================================================
# The credentials themselves
# ===========================================================================


class _FakeLoginClient:
    """Stands in for the bare ``httpx.Client`` the login POST goes out on."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None):
        self.calls.append((url, dict(json or {})))
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _login_ok(token: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"token": token, "user": {"id": "~Test_User1"}}
    return resp


class TestOpenReviewAuthFromEnv:
    def test_absent_credentials_leave_the_caller_anonymous(self):
        assert OpenReviewAuth.from_env(env={}) is None

    def test_either_half_alone_is_not_credentials(self):
        assert OpenReviewAuth.from_env(env={"OPENREVIEW_USERNAME": "a@b.c"}) is None
        assert OpenReviewAuth.from_env(env={"OPENREVIEW_PASSWORD": PASSWORD}) is None

    def test_both_halves_build_credentials(self):
        auth = OpenReviewAuth.from_env(env={"OPENREVIEW_USERNAME": "a@b.c", "OPENREVIEW_PASSWORD": PASSWORD})
        assert auth is not None
        assert auth.username == "a@b.c"

    def test_an_explicit_username_wins_over_the_environment(self):
        auth = OpenReviewAuth.from_env(
            "flag@example.org",
            env={"OPENREVIEW_USERNAME": "env@example.org", "OPENREVIEW_PASSWORD": PASSWORD},
        )
        assert auth is not None
        assert auth.username == "flag@example.org"


class TestOpenReviewAuthLogin:
    def test_one_login_per_host_and_the_token_is_reused(self):
        fake = _FakeLoginClient([_login_ok(TOKEN_V1)])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(f"{NOTES_V1}?paperhash=x") == TOKEN_V1
            assert auth.token_for_url(f"{SEARCH_V1}?term=y") == TOKEN_V1
        assert len(fake.calls) == 1
        assert fake.calls[0][0] == f"{OPENREVIEW_API}/login"
        assert fake.calls[0][1] == {"id": "a@b.c", "password": PASSWORD}

    def test_an_unrelated_origin_gets_its_own_token(self):
        """Token sharing is scoped to the hosts that honour one another's tokens.

        The two OpenReview hosts do (see :class:`TestOneLoginServesBothHosts`);
        anything else keys on its own origin and logs in for itself.
        """
        fake = _FakeLoginClient([_login_ok(TOKEN_V1), _login_ok(TOKEN_V2)])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V1) == TOKEN_V1
            assert auth.token_for_url("https://mirror.example.org/notes") == TOKEN_V2
        assert [url for url, _ in fake.calls] == [
            f"{OPENREVIEW_API}/login",
            "https://mirror.example.org/login",
        ]

    def test_a_rejected_login_degrades_to_anonymous(self, caplog):
        """Bad credentials must not end a 5,043-entry run."""
        fake = _FakeLoginClient([_status(400)])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with caplog.at_level(logging.DEBUG), patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V1) is None
            # And it is not re-attempted for every later entry.
            assert auth.token_for_url(NOTES_V1) is None
        assert len(fake.calls) == 1
        assert "continuing anonymously" in caplog.text

    def test_an_unreachable_login_host_degrades_to_anonymous(self, caplog):
        fake = _FakeLoginClient([httpx.ConnectError("no route to host")])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with caplog.at_level(logging.DEBUG), patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V1) is None
        assert "continuing anonymously" in caplog.text

    def test_a_tokenless_body_is_a_failed_login(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"user": {"id": "~Test_User1"}}
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([resp])):
            assert auth.token_for_url(NOTES_V1) is None

    def test_invalidate_forces_one_fresh_login(self):
        fake = _FakeLoginClient([_login_ok(TOKEN_V1), _login_ok("second-token")])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V1) == TOKEN_V1
            auth.invalidate(NOTES_V1)
            assert auth.token_for_url(NOTES_V1) == "second-token"
        assert len(fake.calls) == 2


class TestCredentialsNeverReachTheLog:
    def test_neither_password_nor_token_is_logged(self, caplog):
        fake = _FakeLoginClient([_status(400), _login_ok(TOKEN_V1)])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with caplog.at_level(logging.DEBUG), patch("bibtex_updater.utils.httpx.Client", fake):
            auth.token_for_url(NOTES_V1)
            auth.token_for_url(SEARCH_V2)
        assert caplog.text  # the failure and the success were both reported
        assert PASSWORD not in caplog.text
        assert TOKEN_V1 not in caplog.text

    def test_the_repr_hides_both(self):
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V1)])):
            auth.token_for_url(NOTES_V1)
        for text in (repr(auth), str(auth), f"{auth}"):
            assert PASSWORD not in text
            assert TOKEN_V1 not in text


# ===========================================================================
# The transport: the token rides on OpenReview requests and nothing else
# ===========================================================================


def _http(auth=None, response=None, side_effect=None) -> HttpClient:
    http = HttpClient(
        timeout=1.0,
        user_agent="test",
        rate_limiter=RateLimiterRegistry(),
        cache=None,
        openreview_auth=auth,
    )
    http.client = MagicMock()
    if side_effect is not None:
        http.client.request.side_effect = side_effect
    else:
        http.client.request.return_value = response
    return http


def _sent_headers(http) -> list[dict]:
    return [call.kwargs["headers"] for call in http.client.request.call_args_list]


def _urls_sent(http) -> list[str]:
    """The URLs a mocked shared client actually put on the wire, in order."""
    return [call.args[1] for call in http.client.request.call_args_list]


def _sent(http) -> list[tuple[str, dict]]:
    """Each request as ``(url, headers)``, so a header can be tied to its host."""
    return list(zip(_urls_sent(http), _sent_headers(http), strict=True))


class TestHttpClientCarriesTheToken:
    def test_an_anonymous_client_sends_no_authorization_header(self):
        """The default is unchanged: no credentials, no header, same request."""
        http = _http(response=_notes([]))
        http._request("GET", NOTES_V1, params={"paperhash": "x"}, service="openreview")
        assert "Authorization" not in _sent_headers(http)[0]

    def test_an_authenticated_client_sends_the_bearer_token(self):
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V1)])):
            http = _http(auth=auth, response=_notes([_note()]))
            resp = http._request("GET", NOTES_V1, params={"paperhash": "x"}, service="openreview")
        assert resp.status_code == 200
        assert _sent_headers(http)[0]["Authorization"] == f"Bearer {TOKEN_V1}"

    def test_the_token_rides_on_openreview_requests_only(self):
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V1)])):
            http = _http(auth=auth, response=_notes([]))
            http._request("GET", "https://api.crossref.org/works", service="crossref")
        assert "Authorization" not in _sent_headers(http)[0]

    def test_where_an_anonymous_client_is_refused_an_authenticated_one_succeeds(self):
        """The whole point, in one test.

        The same request, with and without credentials: 403 and nothing, against
        200 and the note.
        """
        anonymous = _http(response=_status(403))
        with pytest.raises(SourceUnavailableError):
            OpenReviewClient(http=anonymous).search("b", title="Adam: A Method", first_author="kingma")

        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V1)])):
            authenticated = _http(auth=auth, response=_notes([_note()]))
            out = OpenReviewClient(http=authenticated).search("b", title="Adam: A Method", first_author="kingma")
        assert out == [_note()]
        assert _sent_headers(authenticated)[0]["Authorization"] == f"Bearer {TOKEN_V1}"

    def test_a_401_refreshes_the_token_exactly_once(self):
        """``401 TokenExpiredError`` is the one status a re-login answers.

        The retry happens once per request rather than on every attempt.
        """
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        fake = _FakeLoginClient([_login_ok(TOKEN_V1), _login_ok("fresh-token")])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            http = _http(
                auth=auth,
                side_effect=[_status(401, "TokenExpiredError"), _notes([_note()])],
            )
            resp = http._request("GET", NOTES_V1, params={"paperhash": "x"}, service="openreview")
        assert resp.status_code == 200
        assert len(fake.calls) == 2
        headers = _sent_headers(http)
        assert headers[0]["Authorization"] == f"Bearer {TOKEN_V1}"
        assert headers[1]["Authorization"] == "Bearer fresh-token"

    def test_a_403_does_not_spend_a_login(self):
        """``403 ChallengeRequiredError`` says the request was never recognized.

        Logging in again answers a question that was not asked, and logins are
        the scarce resource: four in roughly two minutes and OpenReview starts
        refusing them. The refusal is handed back for
        ``raise_for_failed_lookup`` to classify instead.
        """
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        fake = _FakeLoginClient([_login_ok(TOKEN_V1)])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            http = _http(auth=auth, response=_status(403))
            resp = http._request("GET", NOTES_V1, params={"paperhash": "x"}, service="openreview")
        assert resp.status_code == 403
        assert http.client.request.call_count == 1
        assert len(fake.calls) == 1

    def test_a_persistent_401_is_reported_not_retried_forever(self):
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        fake = _FakeLoginClient([_login_ok(TOKEN_V1), _login_ok("fresh-token")])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            http = _http(auth=auth, response=_status(401, "TokenExpiredError"))
            resp = http._request("GET", NOTES_V1, params={"paperhash": "x"}, service="openreview")
        # Two attempts: the original token, then one refreshed one.
        assert resp.status_code == 401
        assert http.client.request.call_count == 2

    def test_the_login_never_goes_through_the_shared_client(self):
        """The password must not reach the rate limiter, the retry loop or the cache.

        The shared client writes every 200 JSON response to disk keyed on the
        request, so a login routed through it would persist both the credentials
        and the token.
        """
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V1)])):
            http = _http(auth=auth, response=_notes([]))
            http._request("GET", NOTES_V1, params={"paperhash": "x"}, service="openreview")
        urls = [call.args[1] for call in http.client.request.call_args_list]
        assert urls == [NOTES_V1]
        for call in http.client.request.call_args_list:
            assert PASSWORD not in str(call)


# ===========================================================================
# One login for the whole fleet
# ===========================================================================


def _jwt(exp_epoch: float) -> str:
    """A syntactically real JWT whose payload carries ``exp``. Never verified."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"exp": int(exp_epoch)}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.signature"


class TestTokenReuseAcrossProcesses:
    """``bibtex-check`` used to log in once per process, which is right for one
    long run and wrong for a sharded one: three shards produced about 45 logins
    and OpenReview answered ``429`` to 60 of them, degrading those runs to
    anonymous. ``/login`` refuses after four logins in roughly two minutes,
    while the JWT it issues is valid for 24 hours and authenticates both hosts.
    """

    def _auth(self, tmp_path, username="a@b.c"):
        store = OpenReviewTokenStore(tmp_path / "openreview-tokens.json")
        return OpenReviewAuth(username, PASSWORD, token_store=store)

    def test_a_second_client_reuses_the_cached_token(self, tmp_path):
        token = _jwt(time.time() + 24 * 3600)
        first = self._auth(tmp_path)
        fake = _FakeLoginClient([_login_ok(token)])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert first.token_for_url(NOTES_V1) == token

        # A separate process is a separate object with an empty memory cache.
        second = self._auth(tmp_path)
        no_login = _FakeLoginClient([])
        with patch("bibtex_updater.utils.httpx.Client", no_login):
            assert second.token_for_url(NOTES_V1) == token
        assert no_login.calls == []
        assert len(fake.calls) == 1

    def test_the_cache_file_is_not_world_readable(self, tmp_path):
        path = tmp_path / "openreview-tokens.json"
        auth = OpenReviewAuth("a@b.c", PASSWORD, token_store=OpenReviewTokenStore(path))
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(_jwt(time.time() + 3600))])):
            auth.token_for_url(NOTES_V1)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_the_cache_holds_no_password_and_no_username(self, tmp_path):
        path = tmp_path / "openreview-tokens.json"
        auth = OpenReviewAuth("a@b.c", PASSWORD, token_store=OpenReviewTokenStore(path))
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(_jwt(time.time() + 3600))])):
            auth.token_for_url(NOTES_V1)
        body = path.read_text()
        assert PASSWORD not in body
        assert "a@b.c" not in body

    def test_another_account_does_not_pick_up_the_token(self, tmp_path):
        token = _jwt(time.time() + 3600)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(token)])):
            self._auth(tmp_path, "a@b.c").token_for_url(NOTES_V1)

        other = self._auth(tmp_path, "other@b.c")
        fake = _FakeLoginClient([_login_ok("other-token")])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert other.token_for_url(NOTES_V1) == "other-token"
        assert len(fake.calls) == 1

    def test_an_expired_cached_token_is_not_served(self, tmp_path):
        store = OpenReviewTokenStore(tmp_path / "openreview-tokens.json")
        store.put("a@b.c", OPENREVIEW_API, _jwt(time.time() - 60))
        assert store.get("a@b.c", OPENREVIEW_API) is None

    def test_a_refusal_drops_the_shared_copy(self, tmp_path):
        # Another process presenting the same expired token would be refused
        # identically, so invalidation has to reach the file.
        store = OpenReviewTokenStore(tmp_path / "openreview-tokens.json")
        auth = OpenReviewAuth("a@b.c", PASSWORD, token_store=store)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(_jwt(time.time() + 3600))])):
            auth.token_for_url(NOTES_V1)
        assert store.get("a@b.c", OPENREVIEW_API) is not None
        auth.invalidate(NOTES_V1)
        assert store.get("a@b.c", OPENREVIEW_API) is None

    def test_persistence_can_be_turned_off(self, tmp_path):
        env = {
            "OPENREVIEW_USERNAME": "a@b.c",
            "OPENREVIEW_PASSWORD": PASSWORD,
            "BIBTEX_CHECK_OPENREVIEW_TOKEN_CACHE": "0",
        }
        auth = OpenReviewAuth.from_env(env=env)
        assert auth is not None
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(_jwt(time.time() + 3600))])):
            auth.token_for_url(NOTES_V1)

        fresh = OpenReviewAuth.from_env(env=env)
        fake = _FakeLoginClient([_login_ok("second-token")])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert fresh.token_for_url(NOTES_V1) == "second-token"
        assert len(fake.calls) == 1

    def test_the_cache_path_can_be_moved(self, tmp_path):
        path = tmp_path / "elsewhere" / "tokens.json"
        env = {
            "OPENREVIEW_USERNAME": "a@b.c",
            "OPENREVIEW_PASSWORD": PASSWORD,
            "BIBTEX_CHECK_OPENREVIEW_TOKEN_CACHE": str(path),
        }
        auth = OpenReviewAuth.from_env(env=env)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(_jwt(time.time() + 3600))])):
            auth.token_for_url(NOTES_V1)
        assert path.exists()

    def test_the_default_path_is_the_user_cache_never_the_repo(self):
        path = _default_openreview_token_cache_path()
        assert path.name == "openreview-tokens.json"
        assert path.parent.name == "bibtex-updater"
        assert Path.cwd() not in path.parents

    def test_an_unreadable_cache_degrades_to_logging_in(self, tmp_path):
        path = tmp_path / "openreview-tokens.json"
        path.write_text("not json at all")
        auth = OpenReviewAuth("a@b.c", PASSWORD, token_store=OpenReviewTokenStore(path))
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok("fresh")])):
            assert auth.token_for_url(NOTES_V1) == "fresh"

    def test_expiry_comes_from_the_token_itself(self):
        assert _jwt_expiry(_jwt(1800000000)) == 1800000000.0
        assert _jwt_expiry("not-a-jwt") is None


# ===========================================================================
# A refusal to an authenticated request is a blip, not a configuration state
# ===========================================================================


class TestAnAuthenticatedRefusalIsNotLatched:
    """The defect a 5,043-reference screening run surfaced.

    The latch was built for the anonymous case, where a 403 states a standing
    configuration: no credentials, no ``/notes``, and re-asking per entry buys
    nothing. A credentialled run is the opposite case. Measured over three
    concurrent shards of that run under v1.10.0: 844 authenticated
    ``api2.openreview.net/notes`` responses came back 200 and were cached, while
    each 250-entry process took exactly one 403 -- and that single refusal
    latched the host, so 160 to 173 of the remaining entries in the process
    reported ``challenge-gated for anonymous callers; set OPENREVIEW_USERNAME /
    OPENREVIEW_PASSWORD`` on a run where both were set. OpenReview holds ICLR
    2024+, NeurIPS 2023+, TMLR and COLM on v2 alone, so the source contributed
    nothing to the modern half of the bibliography.
    """

    def test_a_transient_403_does_not_gag_the_rest_of_the_run(self):
        """One refused v2 lookup, then the next entry asks v2 again -- with the token."""
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V2)])):
            http = _http(
                auth=auth,
                side_effect=[_status(403), _notes([]), _notes([_note()])],
            )
            client = OpenReviewClient(http=http)
            with pytest.raises(SourceUnavailableError):
                client.search("q", title="Adam: A Method", first_author="kingma")
            out = client.search("q", title="Adam: A Method", first_author="kingma")

        # The second entry reached v2, which is where every 2023+ venue lives.
        second_lookup = _urls_sent(http)[2:]
        assert NOTES_V2 in second_lookup
        v2_headers = [h for u, h in _sent(http)[2:] if u == NOTES_V2]
        assert v2_headers and all(h["Authorization"] == f"Bearer {TOKEN_V2}" for h in v2_headers)
        assert out == [_note()]

    def test_an_anonymous_403_still_latches(self):
        """The behaviour the latch was built for is unchanged.

        No credentials means no ``/notes`` for the whole run, and re-issuing the
        refused request per entry costs a round trip for an answer that is not
        coming.
        """
        http = _http(response=_status(403))
        client = OpenReviewClient(http=http)
        for _ in range(3):
            with pytest.raises(SourceUnavailableError):
                client.search("q", title="Adam: A Method", first_author="kingma")
        # Two requests in total: one per host, on the first entry only.
        assert sorted(set(_urls_sent(http))) == sorted({NOTES_V1, NOTES_V2})
        assert http.client.request.call_count == 2

    def test_a_refusal_on_one_host_leaves_the_other_alone(self):
        """v1 and v2 index disjoint venues, so one gate must not close both."""
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(TOKEN_V2)])):
            http = _http(auth=auth, side_effect=[_status(403), _notes([_note()])])
            out = OpenReviewClient(http=http).search("q", title="Adam: A Method", first_author="kingma")
        assert out == [_note()]
        assert _urls_sent(http) == [NOTES_V2, NOTES_V1]


class TestOneLoginServesBothHosts:
    """OpenReview mints one JWT and both hosts accept it.

    Verified live: a token from ``POST api2.openreview.net/login`` answers 200
    on ``api.openreview.net/notes?paperhash=`` and on ``api2/notes``, where an
    anonymous caller is refused on both. Logins are the scarce resource --
    roughly four in two minutes and OpenReview starts answering 429 -- so
    querying both hosts must cost one login, not one per host.
    """

    def test_a_token_minted_at_one_host_is_presented_at_the_other(self):
        fake = _FakeLoginClient([_login_ok(TOKEN_V2)])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V2) == TOKEN_V2
            assert auth.token_for_url(NOTES_V1) == TOKEN_V2
            assert auth.token_for_url(SEARCH_V1) == TOKEN_V2
        assert [url for url, _ in fake.calls] == [f"{OPENREVIEW_API_V2}/login"]

    def test_a_shared_token_is_read_from_the_cache_for_either_host(self, tmp_path):
        """A sharded fleet logs in once between all its shards, for both hosts."""
        store = OpenReviewTokenStore(tmp_path / "openreview-tokens.json")
        token = _jwt(time.time() + 3600)
        store.put("a@b.c", OPENREVIEW_API_V2, token)
        auth = OpenReviewAuth("a@b.c", PASSWORD, token_store=store)
        fake = _FakeLoginClient([])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V1) == token
        assert fake.calls == []

    def test_invalidating_a_shared_token_drops_it_everywhere(self, tmp_path):
        """An expired token is refused at both hosts, so both copies must go."""
        store = OpenReviewTokenStore(tmp_path / "openreview-tokens.json")
        auth = OpenReviewAuth("a@b.c", PASSWORD, token_store=store)
        token = _jwt(time.time() + 3600)
        with patch("bibtex_updater.utils.httpx.Client", _FakeLoginClient([_login_ok(token)])):
            assert auth.token_for_url(NOTES_V2) == token
            assert auth.token_for_url(NOTES_V1) == token
        auth.invalidate(NOTES_V1)
        assert store.get("a@b.c", OPENREVIEW_API_V2) is None
        assert store.get("a@b.c", OPENREVIEW_API) is None
        fake = _FakeLoginClient([_login_ok("second-token")])
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V2) == "second-token"
        assert len(fake.calls) == 1

    def test_a_disabled_host_still_uses_a_sibling_token(self):
        """A 429 on one ``/login`` must not cost the run its authenticated lookups."""
        fake = _FakeLoginClient([_status(429), _login_ok(TOKEN_V1)])
        auth = OpenReviewAuth("a@b.c", PASSWORD)
        with patch("bibtex_updater.utils.httpx.Client", fake):
            assert auth.token_for_url(NOTES_V2) is None
            assert auth.token_for_url(NOTES_V1) == TOKEN_V1
            assert auth.token_for_url(NOTES_V2) == TOKEN_V1
        assert len(fake.calls) == 2
