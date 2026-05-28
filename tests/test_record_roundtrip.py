"""Self-consistency oracle for record <-> entry round-tripping.

Core invariant
--------------
A ``PublishedRecord`` turned into a BibTeX entry (the exact same data, just in
the citation representation) MUST verify against itself with zero field
mismatches. Any failure here is a *comparison asymmetry* bug: the entry side and
the record side normalize the same surname / venue / title differently, which in
production surfaces as a false AUTHOR_MISMATCH / HALLUCINATED / VENUE_MISMATCH for
a perfectly-cited paper.

The entry is built from the record using the production code path
(``Updater.update_entry``), so this also exercises the converter that real runs
use. We then feed (entry, record) into ``FactChecker._compare_all_fields`` and
assert every comparison ``.matches``.

A second oracle exercises the resolver match-score path: the match score of a
record against an entry built from itself must clear the resolver
``MATCH_THRESHOLD`` (~0.85). Both ``_compare_all_fields`` and
``_compute_match_score`` are lightweight (no network), so we instantiate the real
``FactChecker`` / ``Resolver`` with mocked HTTP clients.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    CrossrefClient,
    DBLPClient,
    FactChecker,
    FactCheckerConfig,
    SemanticScholarClient,
)
from bibtex_updater.updater import PreprintDetection, Resolver, Updater
from bibtex_updater.utils import (
    HttpClient,
    PublishedRecord,
    authors_last_names,
    normalize_title_for_match,
)

# ------------- Fixtures (replicated, not imported, per task) -------------


@pytest.fixture
def logger():
    return logging.getLogger("test_record_roundtrip")


@pytest.fixture
def fake_http():
    """Mock HTTP client that never makes real requests."""
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def checker(fake_http, logger):
    """Real FactChecker with mocked API clients (no network)."""
    return FactChecker(
        CrossrefClient(fake_http),
        DBLPClient(fake_http),
        SemanticScholarClient(fake_http),
        FactCheckerConfig(),
        logger,
    )


class _FakeHttpClient(HttpClient):
    """HttpClient subclass that refuses network access."""

    def __init__(self):  # noqa: D401 - intentionally skip parent network setup
        pass

    def _request(self, *args, **kwargs):
        raise NotImplementedError("network disabled in tests")


@pytest.fixture
def resolver(logger):
    """Real Resolver instance (match-score path is pure / no network)."""
    return Resolver(_FakeHttpClient(), logger)


@pytest.fixture
def updater():
    return Updater(keep_preprint_note=False, rekey=False)


# ------------- Representative records (each stresses a past failure) ----------


def _records() -> list[tuple[str, PublishedRecord, tuple]]:
    """(id, record, marks) triples. ``marks`` carries any xfail for the case."""
    return [
        (
            "plain_ascii",
            PublishedRecord(
                doi="10.1234/jml.2020.001",
                title="Deep Learning for Natural Language Processing",
                authors=[
                    {"given": "John", "family": "Smith"},
                    {"given": "Jane", "family": "Doe"},
                ],
                journal="Journal of Machine Learning Research",
                year=2020,
                type="journal-article",
            ),
            (),
        ),
        (
            "nobiliary_particle",
            PublishedRecord(
                doi="10.1234/wavenet.2016",
                title="WaveNet: A Generative Model for Raw Audio",
                authors=[
                    {"given": "Aaron", "family": "van den Oord"},
                    {"given": "Sander", "family": "Dieleman"},
                ],
                journal="Speech Synthesis Workshop",
                year=2016,
                type="journal-article",
            ),
            (),
        ),
        (
            "multiword_particle",
            PublishedRecord(
                doi="10.1234/vision.2019",
                title="A Model of Cortical Self-Organization",
                authors=[
                    {"given": "Peter", "family": "von der Malsburg"},
                    {"given": "Maria", "family": "de la Cruz"},
                ],
                journal="Biological Cybernetics",
                year=2019,
                type="journal-article",
            ),
            (),
        ),
        (
            "diacritics",
            PublishedRecord(
                doi="10.1234/kernel.2002",
                title="Learning with Kernels",
                authors=[
                    {"given": "Bernhard", "family": "Schölkopf"},
                    {"given": "Klaus-Robert", "family": "Müller"},
                ],
                journal="Neural Computation",
                year=2002,
                type="journal-article",
            ),
            (),
        ),
        (
            "conference",
            PublishedRecord(
                doi="10.1234/neurips.2017.attention",
                title="Attention Is All You Need",
                authors=[
                    {"given": "Ashish", "family": "Vaswani"},
                    {"given": "Noam", "family": "Shazeer"},
                ],
                journal="Advances in Neural Information Processing Systems",
                year=2017,
                type="proceedings-article",
            ),
            (),
        ),
        (
            "many_authors",
            PublishedRecord(
                doi="10.1234/bigteam.2021",
                title="Scaling Laws for Neural Language Models",
                authors=[
                    {"given": "Jared", "family": "Kaplan"},
                    {"given": "Sam", "family": "McCandlish"},
                    {"given": "Tom", "family": "Henighan"},
                    {"given": "Tom", "family": "Brown"},
                    {"given": "Benjamin", "family": "Chess"},
                    {"given": "Rewon", "family": "Child"},
                ],
                journal="Journal of Machine Learning Research",
                year=2021,
                type="journal-article",
            ),
            (),
        ),
    ]


_RECORD_PARAMS = [pytest.param(rec, marks=marks, id=name) for name, rec, marks in _records()]


# ------------- Oracle 1: _compare_all_fields self-consistency ----------------


class TestCompareAllFieldsRoundtrip:
    """A record -> entry must verify against itself with no field mismatch."""

    @pytest.mark.parametrize("record", _RECORD_PARAMS)
    def test_all_fields_match(self, checker, updater, record):
        detection = PreprintDetection(is_preprint=False)
        entry = updater.update_entry({"ID": "x", "ENTRYTYPE": "article"}, record, detection)

        comparisons = checker._compare_all_fields(entry, record)

        for field in ("title", "author", "year", "venue"):
            assert field in comparisons, f"missing comparison: {field}"
            comp = comparisons[field]
            assert comp.matches, (
                f"{field} falsely mismatched on self-roundtrip: "
                f"entry={comp.entry_value!r} api={comp.api_value!r} "
                f"score={comp.similarity_score:.3f}"
            )


# ------------- Oracle 2: resolver match-score self-consistency ----------------


class TestMatchScoreRoundtrip:
    """A record scored against an entry built from itself clears the threshold."""

    @pytest.mark.parametrize("record", _RECORD_PARAMS)
    def test_match_score_above_threshold(self, resolver, updater, record):
        detection = PreprintDetection(is_preprint=False)
        entry = updater.update_entry({"ID": "x", "ENTRYTYPE": "article"}, record, detection)

        title_norm = normalize_title_for_match(entry.get("title", ""))
        authors_ref = authors_last_names(entry.get("author", ""))

        score = resolver._compute_match_score(title_norm, record, authors_ref)

        assert score >= resolver.MATCH_THRESHOLD, (
            f"self-roundtrip match score {score:.3f} below "
            f"MATCH_THRESHOLD {resolver.MATCH_THRESHOLD} "
            f"(title={title_norm!r}, authors={authors_ref!r})"
        )
