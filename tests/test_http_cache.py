"""HttpClient response caching, including non-JSON (arXiv Atom XML) bodies.

``HttpClient._request`` used to cache ONLY responses whose Content-Type starts
with ``application/json``, so arXiv Atom XML responses were re-fetched on every
run despite the SqliteCache. Non-JSON text responses are now stored in a
versioned envelope (``{"__btu_cache_v2__": true, "ct": ..., "text": ...}``)
while JSON responses keep the legacy raw-body format, so existing cache files
keep working in both directions.

All tests are hermetic: the httpx client is replaced by a counting stub.
"""

from __future__ import annotations

import json
import threading

import httpx

from bibtex_updater.fact_checker import ArxivClient
from bibtex_updater.utils import HttpClient, RateLimiterRegistry, SqliteCache

ATOM_BODY = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<feed xmlns="http://www.w3.org/2005/Atom">\n'
    "  <entry><title>Café Embeddings: Müller et al.</title></entry>\n"
    "</feed>\n"
)


class _CountingTransport:
    """``httpx.Client`` stand-in: serves canned responses, counts round-trips."""

    def __init__(self, status_code: int, content: str, content_type: str):
        self.status_code = status_code
        self.content = content
        self.content_type = content_type
        self.count = 0
        self._lock = threading.Lock()

    def request(self, method, url, **kwargs):
        with self._lock:
            self.count += 1
        return httpx.Response(
            self.status_code,
            content=self.content.encode("utf-8"),
            headers={"Content-Type": self.content_type},
            request=httpx.Request(method, str(url)),
        )


def _make_http(tmp_path, status_code=200, content=ATOM_BODY, content_type="application/atom+xml; charset=utf-8"):
    cache = SqliteCache(str(tmp_path / "cache.db"))
    registry = RateLimiterRegistry(dict.fromkeys(RateLimiterRegistry.DEFAULT_LIMITS, 100_000))
    http = HttpClient(timeout=5.0, user_agent="test", rate_limiter=registry, cache=cache)
    http.client.close()  # replace the real client with the stub
    transport = _CountingTransport(status_code, content, content_type)
    http.client = transport  # type: ignore[assignment]
    return http, transport


class TestNonJsonCaching:
    """Atom XML (and other text) 200s are cached and replayed faithfully."""

    def test_second_identical_request_served_from_cache(self, tmp_path):
        http, transport = _make_http(tmp_path)
        params = {"id_list": "2003.08934"}

        first = http._request("GET", "https://export.arxiv.org/api/query", params=params, accept="application/atom+xml")
        assert transport.count == 1
        assert first.status_code == 200

        second = http._request(
            "GET", "https://export.arxiv.org/api/query", params=params, accept="application/atom+xml"
        )

        # Zero additional transport calls; the body and Content-Type survive
        # the cache round-trip exactly (including non-ASCII characters).
        assert transport.count == 1
        assert second.headers.get("X-From-Cache") == "1"
        assert second.text == first.text == ATOM_BODY
        assert second.headers.get("Content-Type") == "application/atom+xml; charset=utf-8"

    def test_arxiv_fetch_atom_cached_end_to_end(self, tmp_path):
        """The motivating case: ArxivClient.fetch_atom stops re-fetching."""
        http, transport = _make_http(tmp_path)
        arxiv = ArxivClient(http)

        first = arxiv.fetch_atom("2003.08934")
        second = arxiv.fetch_atom("2003.08934")

        assert transport.count == 1
        assert first == second == ATOM_BODY

    def test_distinct_requests_not_conflated(self, tmp_path):
        http, transport = _make_http(tmp_path)

        http._request("GET", "https://export.arxiv.org/api/query", params={"id_list": "a"})
        http._request("GET", "https://export.arxiv.org/api/query", params={"id_list": "b"})

        assert transport.count == 2

    def test_non_200_text_not_cached(self, tmp_path):
        """Only 200s are cached: a 404 must be re-fetched next time."""
        http, transport = _make_http(tmp_path, status_code=404)

        http._request("GET", "https://export.arxiv.org/api/query", params={"id_list": "x"})
        http._request("GET", "https://export.arxiv.org/api/query", params={"id_list": "x"})

        assert transport.count == 2

    def test_binary_content_type_not_cached(self, tmp_path):
        """Non-text bodies (PDF, ...) stay uncached (cache stores JSON values)."""
        http, transport = _make_http(tmp_path, content="%PDF-1.4", content_type="application/pdf")

        http._request("GET", "https://example.org/paper.pdf")
        http._request("GET", "https://example.org/paper.pdf")

        assert transport.count == 2


class TestJsonCachingCompatibility:
    """JSON responses keep the legacy raw-body cache format, and legacy values
    written by older versions are still readable."""

    def test_json_cached_in_legacy_format(self, tmp_path):
        body = {"message": {"items": [{"DOI": "10.1234/x"}]}}
        http, transport = _make_http(tmp_path, content=json.dumps(body), content_type="application/json")
        params = {"rows": 1}

        http._request("GET", "https://api.crossref.org/works", params=params, accept="application/json")

        # The stored value is the RAW decoded body -- not the v2 envelope -- so
        # a downgraded tool version could still read this cache file.
        key = json.dumps(
            {"m": "GET", "u": "https://api.crossref.org/works", "p": params, "a": "application/json", "j": None},
            sort_keys=True,
        )
        assert http.cache.get(key) == body

        replay = http._request("GET", "https://api.crossref.org/works", params=params, accept="application/json")
        assert transport.count == 1
        assert replay.json() == body
        assert replay.headers.get("X-From-Cache") == "1"

    def test_legacy_plain_json_cache_value_still_readable(self, tmp_path):
        """A plain JSON value written by an OLD version replays as before."""
        http, transport = _make_http(tmp_path)
        legacy_body = {"message": {"items": [{"DOI": "10.1234/legacy"}]}}
        params = {"rows": 5}
        key = json.dumps(
            {"m": "GET", "u": "https://api.crossref.org/works", "p": params, "a": "application/json", "j": None},
            sort_keys=True,
        )
        http.cache.set(key, legacy_body)

        resp = http._request("GET", "https://api.crossref.org/works", params=params, accept="application/json")

        assert transport.count == 0  # served from the legacy cache value
        assert resp.status_code == 200
        assert resp.json() == legacy_body

    def test_non_200_json_not_cached(self, tmp_path):
        """A 404 JSON body must not be cached (it would replay as a 200)."""
        http, transport = _make_http(tmp_path, status_code=404, content="{}", content_type="application/json")

        http._request("GET", "https://api.crossref.org/works/10.1/missing")
        http._request("GET", "https://api.crossref.org/works/10.1/missing")

        assert transport.count == 2
