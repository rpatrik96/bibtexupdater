"""Tests for the preprint-citation convention and non-canonical venue names.

Measured on a 5,043-reference corpus (2026-09): 1,024 references came back with
``venue`` as the sole disagreeing field, every one of them found by the cascade
and confirmed on title and authors. Three defects produced them.

1. **The convention itself.** ``journal = {arXiv preprint arXiv:2408.05147}`` is
   what Google Scholar's BibTeX export emits, so it is everywhere -- 764 of the
   1,024. The checker read it as a PUBLISHED-venue claim that an arXiv record
   could not confirm, and abstained (``NON_COMPARABLE`` -> ``UNCONFIRMED``). It
   is not a published-venue claim at all; it says the work is a preprint, which
   is what the record confirms. :func:`is_preprint_server_venue` now recognizes
   every common spelling and the venue field confirms.

2. **Brace-protected venue names.** Zotero/Better BibTeX writes ``The {{Twelfth
   International Conference}} on {{Learning Representations}}``. The braces
   survived venue normalization, no alias matched, and the fuzzy score against
   ``ICLR`` fell to 0.09 -- a hard ``VENUE_MISMATCH`` on a correctly cited ICLR
   paper. ``_normalize_venue_for_matching`` now runs ``latex_to_plain`` first.

3. **COLM missing from the alias map**, in every form it is cited in
   (``Conference on Language Modeling``, with or without the ``(COLM)`` gloss,
   with or without an ordinal edition prefix).

Plus the lookup-aid case: an entry that carries an arXiv identifier ALONGSIDE a
real published venue must be scored against the published venue, not against
the identifier.

The negative tests at the bottom pin the property that matters: none of this
may let two different conferences confirm each other.
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
    FactCheckStatus,
    SemanticScholarClient,
    venues_match,
)
from bibtex_updater.matching import MatchOutcome, get_canonical_venue, is_preprint_server_venue
from bibtex_updater.utils import PublishedRecord, entry_venue


@pytest.fixture
def logger():
    return logging.getLogger("test_preprint_venue_convention")


@pytest.fixture
def empty_http():
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    return mock


@pytest.fixture
def checker(empty_http, logger):
    return FactChecker(
        CrossrefClient(empty_http),
        DBLPClient(empty_http),
        SemanticScholarClient(empty_http),
        FactCheckerConfig(),
        logger,
    )


# --------------------------------------------------------------------------
# 1. The preprint-citation convention, in every spelling seen in the corpus
# --------------------------------------------------------------------------

#: Counts are the 2026-09 corpus occurrences among the 1,024 venue-only findings.
PREPRINT_VENUE_FORMS = [
    "arXiv preprint arXiv:2408.05147",  # 667 -- Google Scholar's export
    "arXiv preprint",  # 60
    "CoRR",  # 19 -- DBLP's label for arXiv
    "arXiv:2408.05147",  # 12
    "\\href{https://arxiv.org/abs/2408.05147}{arXiv:2408.05147}",  # 8
    "arXiv e-prints",
    "arXiv",
    "CoRR abs/2408.05147",
    "arXiv preprint arXiv:math.GT/0309136",  # old-style identifier
    "arXiv:math.GT/0309136",
    "bioRxiv",
    "SSRN Electronic Journal",
    "Preprint",
]


@pytest.mark.parametrize("venue", PREPRINT_VENUE_FORMS)
def test_preprint_server_venue_recognized(venue):
    assert is_preprint_server_venue(venue) is True


@pytest.mark.parametrize(
    "venue",
    [
        "International Conference on Learning Representations",
        "Advances in Neural Information Processing Systems",
        "Journal of Machine Learning Research",
        "Nature",
        # ``corr`` is anchored at a word boundary precisely so these do not fire.
        "Corrosion Science",
        "Recording Industry Journal",
        # A publisher series is NOT a preprint server: it names a real channel
        # that merely cannot pin one venue, so it keeps abstaining.
        "Proceedings of Machine Learning Research",
        "Lecture Notes in Computer Science",
        "OpenReview",
    ],
)
def test_real_venue_is_not_a_preprint_server(venue):
    assert is_preprint_server_venue(venue) is False


def _gemma_scope_entry(journal: str) -> dict[str, str]:
    """The regression case: a real paper, found by arXiv, venue the sole finding."""
    return {
        "ID": "lieberum2024gemmascope",
        "ENTRYTYPE": "article",
        "title": "{Gemma Scope}: Open Sparse Autoencoders Everywhere All At Once on {Gemma 2}",
        "author": "Lieberum, Tom and Rajamanoharan, Senthooran and Conmy, Arthur",
        "journal": journal,
        "year": "2024",
    }


def _gemma_scope_record() -> PublishedRecord:
    return PublishedRecord(
        doi="10.48550/arXiv.2408.05147",
        title="Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2",
        authors=[
            {"given": "Tom", "family": "Lieberum"},
            {"given": "Senthooran", "family": "Rajamanoharan"},
            {"given": "Arthur", "family": "Conmy"},
        ],
        journal="arXiv",
        year=2024,
        arxiv_id="2408.05147",
    )


@pytest.mark.parametrize("venue", PREPRINT_VENUE_FORMS)
def test_preprint_citation_confirms_the_venue_field(checker, venue):
    """Every spelling of the convention leaves the venue CONFIRMED, not abstained."""
    comparisons = checker._compare_all_fields(_gemma_scope_entry(venue), _gemma_scope_record())
    assert comparisons["venue"].resolved_outcome is MatchOutcome.MATCH
    assert comparisons["venue"].is_confirmed is True
    assert comparisons["venue"].note == "Preprint-server citation; no published venue claimed"


def test_gemma_scope_comes_back_clean(checker):
    """The named regression case: verified, nothing mismatched, nothing unconfirmed."""
    entry = _gemma_scope_entry("arXiv preprint arXiv:2408.05147")
    comparisons = checker._compare_all_fields(entry, _gemma_scope_record())
    assert [n for n, c in comparisons.items() if c.is_mismatch] == []
    assert [n for n, c in comparisons.items() if c.is_non_confirming] == []
    status = checker._determine_status(0.95, comparisons, ["arxiv"], "article")
    assert status is FactCheckStatus.VERIFIED


def test_preprint_citation_matched_against_a_published_record_still_confirms(checker):
    """A preprint citation whose paper was LATER published is still a valid citation.

    The entry claims no published venue, so the published record has nothing to
    contradict -- and the venue must not read as unconfirmed either.
    """
    entry = _gemma_scope_entry("arXiv preprint arXiv:2408.05147")
    published = PublishedRecord(
        doi="10.1000/real",
        title="Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2",
        authors=[
            {"given": "Tom", "family": "Lieberum"},
            {"given": "Senthooran", "family": "Rajamanoharan"},
            {"given": "Arthur", "family": "Conmy"},
        ],
        journal="BlackboxNLP",
        year=2024,
    )
    assert published.journal  # the record does name a real venue
    comparisons = checker._compare_all_fields(entry, published)
    assert comparisons["venue"].resolved_outcome is MatchOutcome.MATCH


# --------------------------------------------------------------------------
# 2. An arXiv identifier is a lookup aid, never a competing venue claim
# --------------------------------------------------------------------------


def test_published_venue_wins_over_a_preprint_journal_string():
    """``journal = {arXiv preprint ...}`` beside a real ``booktitle``.

    Reading ``journal`` first discarded the published claim entirely.
    """
    entry = {
        "journal": "arXiv preprint arXiv:1706.03762",
        "booktitle": "Advances in Neural Information Processing Systems",
    }
    assert entry_venue(entry) == "Advances in Neural Information Processing Systems"


def test_preprint_journal_is_still_returned_when_it_is_the_only_venue_field():
    assert entry_venue({"journal": "arXiv preprint arXiv:1706.03762"}) == "arXiv preprint arXiv:1706.03762"


def test_venue_field_precedence_is_otherwise_unchanged():
    assert entry_venue({"journal": "Nature", "booktitle": "ICML"}) == "Nature"
    assert entry_venue({"booktitle": "ICML", "series": "LNCS"}) == "ICML"
    assert entry_venue({"howpublished": "Communications of the ACM"}) == "Communications of the ACM"
    assert entry_venue({}) == ""


@pytest.mark.parametrize(
    "carrier",
    [
        {"eprint": "1706.03762", "archiveprefix": "arXiv"},
        {"note": "arXiv:1706.03762"},
        {"url": "https://arxiv.org/abs/1706.03762"},
        {"doi": "10.48550/arXiv.1706.03762"},
        {"journal": "arXiv preprint arXiv:1706.03762"},
    ],
)
def test_arxiv_identifier_beside_a_published_venue_confirms_that_venue(checker, carrier):
    """The identifier makes the entry findable; the published venue is the claim."""
    entry = {
        "ID": "vaswani2017attention",
        "ENTRYTYPE": "inproceedings",
        "title": "Attention Is All You Need",
        "author": "Vaswani, Ashish and Shazeer, Noam",
        "booktitle": "Advances in Neural Information Processing Systems",
        "year": "2017",
        **carrier,
    }
    record = PublishedRecord(
        doi="10.1000/neurips",
        title="Attention Is All You Need",
        authors=[{"given": "Ashish", "family": "Vaswani"}, {"given": "Noam", "family": "Shazeer"}],
        journal="NeurIPS",
        year=2017,
    )
    comparisons = checker._compare_all_fields(entry, record)
    assert comparisons["venue"].resolved_outcome is MatchOutcome.MATCH
    assert comparisons["venue"].entry_value == "Advances in Neural Information Processing Systems"


# --------------------------------------------------------------------------
# 3. Non-canonical venue names: braces, glosses, ordinal editions, COLM
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("venue", "canonical"),
    [
        # Brace-protected (Zotero / Better BibTeX). These were hard MISMATCHes.
        ("The {{Twelfth International Conference}} on {{Learning Representations}}", "iclr"),
        ("International {{Conference}} on {{Learning Representations}}", "iclr"),
        ("Advances in {{Neural Information Processing Systems}}", "neurips"),
        # Full expansion, with and without its own acronym gloss.
        ("International Conference on Learning Representations", "iclr"),
        ("International Conference on Learning Representations (ICLR)", "iclr"),
        ("Advances in Neural Information Processing Systems (NeurIPS)", "neurips"),
        # COLM, in every form the corpus cites it in.
        ("Conference on Language Modeling", "colm"),
        ("Conference on Language Modeling (COLM)", "colm"),
        ("Conference on Language Modeling ({COLM})", "colm"),
        ("First Conference on Language Modeling", "colm"),
        ("Second Conference on Language Modeling", "colm"),
        ("Third Conference on Language Modeling", "colm"),
        ("Proceedings of the Conference on Language Modeling (COLM)", "colm"),
        ("COLM", "colm"),
    ],
)
def test_non_canonical_venue_names_canonicalize(venue, canonical):
    assert get_canonical_venue(venue) == canonical


@pytest.mark.parametrize(
    ("claimed", "recorded"),
    [
        ("The {{Twelfth International Conference}} on {{Learning Representations}}", "ICLR"),
        ("International {{Conference}} on {{Learning Representations}}", "ICLR 2024"),
        ("Conference on Language Modeling ({COLM})", "Second Conference on Language Modeling"),
        ("Third Conference on Language Modeling", "COLM"),
        ("International Conference on Learning Representations", "ICLR"),
    ],
)
def test_correct_venue_in_a_non_canonical_form_matches(claimed, recorded):
    assert venues_match(claimed, recorded).outcome is MatchOutcome.MATCH


# --------------------------------------------------------------------------
# NEGATIVE TESTS -- genuine cross-venue mismatch detection must survive
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claimed", "recorded"),
    [
        # The named requirement: NeurIPS claimed for an ICML paper still flags.
        ("Advances in Neural Information Processing Systems", "International Conference on Machine Learning"),
        ("NeurIPS", "ICML"),
        # Expansion-vs-expansion across two different conferences.
        ("International Conference on Learning Representations", "International Conference on Machine Learning"),
        # The newly-aliased venue must not swallow its neighbours.
        ("Conference on Language Modeling", "International Conference on Learning Representations"),
        ("COLM", "ICLR"),
        ("Second Conference on Language Modeling", "NeurIPS"),
        # Brace-stripping must not merge two different braced venues.
        ("The {{Twelfth International Conference}} on {{Learning Representations}}", "ICML"),
        ("Advances in {{Neural Information Processing Systems}}", "ICLR"),
    ],
)
def test_different_conferences_still_mismatch(claimed, recorded):
    assert venues_match(claimed, recorded).outcome is MatchOutcome.MISMATCH


def test_neurips_claimed_for_an_icml_paper_still_reaches_venue_mismatch(checker):
    entry = {
        "ID": "wrongvenue",
        "ENTRYTYPE": "inproceedings",
        "title": "A Real Paper About Representations",
        "author": "Doe, Jane and Roe, Richard",
        "booktitle": "Advances in Neural Information Processing Systems",
        "year": "2023",
        # An arXiv ID on the entry must not buy the wrong venue an abstention.
        "note": "arXiv:2301.00001",
    }
    record = PublishedRecord(
        doi="10.1000/icml",
        title="A Real Paper About Representations",
        authors=[{"given": "Jane", "family": "Doe"}, {"given": "Richard", "family": "Roe"}],
        journal="International Conference on Machine Learning",
        year=2023,
    )
    comparisons = checker._compare_all_fields(entry, record)
    assert comparisons["venue"].resolved_outcome is MatchOutcome.MISMATCH
    status = checker._determine_status(0.95, comparisons, ["crossref"], "inproceedings")
    assert status is FactCheckStatus.VENUE_MISMATCH


def test_workshop_is_still_distinct_from_its_host_conference():
    """``workshop`` is a satellite event with its own proceedings, not a track."""
    result = venues_match(
        "International Conference on Machine Learning",
        "ICML 2025 Workshop on Reliable and Responsible Foundation Models",
    )
    assert result.outcome is MatchOutcome.MISMATCH


def test_a_published_venue_claim_against_a_preprint_record_still_abstains(checker):
    """The fix is scoped to the CLAIM side.

    An entry that really does claim ICLR, matched to an arXiv record, still
    cannot have that claim confirmed -- it abstains, exactly as before, and the
    abstention is reported under ``unconfirmed``, never as a mismatch.
    """
    entry = {
        "ID": "kingma2015adam",
        "ENTRYTYPE": "inproceedings",
        "title": "Adam: A Method for Stochastic Optimization",
        "author": "Kingma, Diederik P. and Ba, Jimmy",
        "booktitle": "International Conference on Learning Representations",
        "year": "2015",
    }
    record = PublishedRecord(
        doi="10.48550/arXiv.1412.6980",
        title="Adam: A Method for Stochastic Optimization",
        authors=[{"given": "Diederik P.", "family": "Kingma"}, {"given": "Jimmy", "family": "Ba"}],
        journal="arXiv",
        year=2015,
        arxiv_id="1412.6980",
    )
    comparisons = checker._compare_all_fields(entry, record)
    assert comparisons["venue"].resolved_outcome is MatchOutcome.NON_COMPARABLE
    assert comparisons["venue"].is_mismatch is False
    assert comparisons["venue"].is_non_confirming is True
