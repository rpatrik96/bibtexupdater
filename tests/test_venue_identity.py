"""Tests for identifier-based venue identity (cross-source consensus, Task 1).

``_detect_cross_source_venue_mismatch`` used to require BOTH the entry venue
and >= 2 source venues to canonicalize through the hand-curated ~45-entry
ML/CS alias map, so any wrong-venue claim outside that map abstained (HALLMARK
``wrong_venue`` / ``arxiv_version_mismatch`` false negatives).

Venue identity is now also established by identifiers carried on
``PublishedRecord``:

* ``issn``        -- Crossref ``ISSN`` list / OpenAlex source ``issn``/``issn_l``
* ``venue_source_id`` -- OpenAlex source id URL
* ``venue_key``   -- DBLP venue stream key (``conf/icml``, ``journals/corr``)

Two records that share any of these belong to the same venue group even when
their venue strings differ and neither canonicalizes. FPR guards preserved:
any order-reliable source agreeing with the entry suppresses the flag;
non-canonicalizable entry venues flag ONLY against identifier-corroborated
groups; preprint/series records (including ``journals/corr`` keys) never
anchor; ambiguous multi-group consensus abstains.
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
from bibtex_updater.matching import MatchOutcome
from bibtex_updater.sources import openalex_work_to_candidate_record
from bibtex_updater.utils import (
    PublishedRecord,
    crossref_message_to_record,
    dblp_hit_to_candidate_record,
    dblp_hit_to_record,
    dblp_stream_key,
    normalize_issn,
)


@pytest.fixture
def logger():
    return logging.getLogger("test_venue_identity")


@pytest.fixture
def checker(logger):
    mock = MagicMock()
    mock._request.return_value = MagicMock(status_code=404, json=lambda: {})
    crossref = CrossrefClient(mock)
    dblp = DBLPClient(mock)
    s2 = SemanticScholarClient(mock)
    return FactChecker(crossref, dblp, s2, FactCheckerConfig(), logger)


def _record(
    venue: str,
    *,
    order_reliable: bool = True,
    issn: tuple[str, ...] = (),
    venue_source_id: str | None = None,
    venue_key: str | None = None,
) -> PublishedRecord:
    return PublishedRecord(
        doi="10.0/x",
        title="A Paper",
        authors=[{"given": "Ada", "family": "Lovelace"}],
        journal=venue,
        year=2023,
        order_reliable=order_reliable,
        issn=issn,
        venue_source_id=venue_source_id,
        venue_key=venue_key,
    )


# ===========================================================================
# Helpers: normalize_issn / dblp_stream_key
# ===========================================================================


class TestNormalizeIssn:
    def test_hyphenated_uppercase_passthrough(self):
        assert normalize_issn("1532-4435") == "1532-4435"

    def test_unhyphenated_gets_hyphen(self):
        assert normalize_issn("15324435") == "1532-4435"

    def test_lowercase_x_check_char_uppercased(self):
        assert normalize_issn("2167-647x") == "2167-647X"

    def test_junk_and_empty_rejected(self):
        assert normalize_issn("") is None
        assert normalize_issn(None) is None
        assert normalize_issn("not-an-issn") is None
        assert normalize_issn("12345") is None
        assert normalize_issn("1234-56789") is None


class TestDblpStreamKey:
    def test_conf_key(self):
        assert dblp_stream_key({"key": "conf/icml/Smith23"}) == "conf/icml"

    def test_journals_corr_key(self):
        assert dblp_stream_key({"key": "journals/corr/abs-2301-00001"}) == "journals/corr"

    def test_url_fallback(self):
        info = {"url": "https://dblp.org/rec/conf/icml/Smith23"}
        assert dblp_stream_key(info) == "conf/icml"

    def test_unknown_shapes_yield_none(self):
        assert dblp_stream_key({}) is None
        assert dblp_stream_key({"key": "homepages/x/Smith"}) is None
        assert dblp_stream_key({"key": "conf/icml"}) is None  # no record id segment
        assert dblp_stream_key({"key": 42}) is None
        assert dblp_stream_key({"url": "https://dblp.org/pid/x"}) is None


# ===========================================================================
# Converters populate the venue-identity fields
# ===========================================================================


class TestConverterVenueIdentity:
    def test_defaults_are_empty(self):
        rec = PublishedRecord(doi="10.0/x")
        assert rec.issn == ()
        assert rec.venue_source_id is None
        assert rec.venue_key is None

    def test_crossref_issn_populated_and_deduped(self):
        msg = {
            "DOI": "10.1234/abc",
            "title": ["A Paper"],
            "container-title": ["Journal of Machine Learning Research"],
            "ISSN": ["1532-4435", "15324435", "1533-7928", "junk"],
        }
        rec = crossref_message_to_record(msg)
        assert rec is not None
        assert rec.issn == ("1532-4435", "1533-7928")

    def test_crossref_missing_issn_is_empty(self):
        rec = crossref_message_to_record({"DOI": "10.1234/abc", "title": ["A Paper"]})
        assert rec is not None
        assert rec.issn == ()

    def test_openalex_source_id_and_issn_populated(self):
        work = {
            "title": "A Paper",
            "publication_year": 2023,
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {
                "source": {
                    "id": "https://openalex.org/S1983995261",
                    "display_name": "Journal of Machine Learning Research",
                    "issn_l": "1532-4435",
                    "issn": ["1532-4435", "1533-7928"],
                }
            },
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec is not None
        assert rec.venue_source_id == "https://openalex.org/S1983995261"
        assert rec.issn == ("1532-4435", "1533-7928")

    def test_openalex_missing_source_yields_empty_identity(self):
        work = {
            "title": "A Paper",
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {"source": None},
        }
        rec = openalex_work_to_candidate_record(work)
        assert rec is not None
        assert rec.venue_source_id is None
        assert rec.issn == ()

    def test_dblp_candidate_converter_sets_venue_key(self):
        hit = {
            "info": {
                "key": "conf/icml/Smith23",
                "title": "A Paper",
                "venue": "ICML",
                "year": "2023",
                "ee": "https://example.org/p",
                "authors": {"author": [{"text": "Ada Lovelace"}]},
            }
        }
        rec = dblp_hit_to_candidate_record(hit)
        assert rec is not None
        assert rec.venue_key == "conf/icml"

    def test_dblp_strict_converter_sets_venue_key_from_url(self):
        hit = {
            "info": {
                "url": "https://dblp.org/rec/journals/jmlr/Smith23",
                "title": "A Paper",
                "venue": "Journal of Machine Learning Research",
                "year": "2023",
                "ee": "https://example.org/p",
                "doi": "10.5555/12345",
                "authors": {"author": [{"text": "Ada Lovelace"}]},
            }
        }
        rec = dblp_hit_to_record(hit)
        assert rec is not None
        assert rec.venue_key == "journals/jmlr"


# ===========================================================================
# Identifier-based cross-source venue consensus
# ===========================================================================


class TestIdentifierVenueConsensus:
    def test_issn_group_agreeing_with_entry_does_not_flag(self, checker):
        """Spec test 1 (no regression): Crossref 'J. Mach. Learn. Res.' +
        OpenAlex 'Journal of Machine Learning Research' share an ISSN; the
        entry says 'JMLR'. Both canonicalize with the entry -> agreement, no
        flag."""
        per_source = {
            "crossref": _record("J. Mach. Learn. Res.", issn=("1532-4435",)),
            "openalex": _record("Journal of Machine Learning Research", issn=("1532-4435",)),
        }
        assert checker._detect_cross_source_venue_mismatch("JMLR", per_source) is None

    def test_canonical_entry_vs_issn_matched_unknown_journal_flags(self, checker):
        """Spec test 2: entry 'AISTATS' (canonicalizable), two sources
        ISSN-matched to 'Foo Journal of Bar' (NOT in the alias map) ->
        MISMATCH fires where the canonical-only logic abstained."""
        per_source = {
            "crossref": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "openalex": _record("Foo Journal of Bar", issn=("1234-5678", "8765-4321")),
        }
        consensus = checker._detect_cross_source_venue_mismatch("AISTATS", per_source)
        assert consensus == "Foo Journal of Bar"

    def test_noncanonical_entry_vs_source_id_matched_conference_flags(self, checker):
        """Spec test 3: entry 'Journal of Advanced Neural Computing' (not
        canonicalizable), two order-reliable records share a venue_source_id
        and are named 'International Conference on Machine Learning' ->
        MISMATCH fires."""
        sid = "https://openalex.org/S1234567890"
        per_source = {
            "crossref": _record("International Conference on Machine Learning", venue_source_id=sid),
            "openalex": _record("International Conference on Machine Learning", venue_source_id=sid),
        }
        consensus = checker._detect_cross_source_venue_mismatch("Journal of Advanced Neural Computing", per_source)
        assert consensus == "icml"

    def test_noncanonical_entry_without_identifier_corroboration_abstains(self, checker):
        """Spec test 4: same shape as test 3 but the two records only
        fuzzy-resemble each other (no shared identifier, no canonical) ->
        abstain. Refuting a venue string the alias map has never seen needs
        identifier-grade corroboration, not string similarity."""
        per_source = {
            "crossref": _record("International Journal of Quantum Robotics"),
            "openalex": _record("Intl. Journal of Quantum Robotics"),
        }
        assert checker._detect_cross_source_venue_mismatch("Journal of Advanced Neural Computing", per_source) is None

    def test_one_source_agreeing_with_entry_suppresses_flag(self, checker):
        """Spec test 5 (FPR guard): one order-reliable source agrees with the
        entry -> no flag, even though two other sources share an ISSN-matched
        different venue."""
        per_source = {
            "crossref": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "openalex": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "dblp": _record("Artificial Intelligence and Statistics"),  # agrees with AISTATS
        }
        assert checker._detect_cross_source_venue_mismatch("AISTATS", per_source) is None

    def test_preprint_records_never_anchor_string_marker(self, checker):
        """Spec test 6a: arXiv-venue records cannot anchor a consensus even
        when they share identifiers."""
        per_source = {
            "crossref": _record("arXiv preprint arXiv:2301.00001", issn=("1234-5678",)),
            "openalex": _record("arXiv preprint arXiv:2301.00001", issn=("1234-5678",)),
        }
        assert checker._detect_cross_source_venue_mismatch("AISTATS", per_source) is None

    def test_corr_venue_key_never_anchors(self, checker):
        """Spec test 6b: a DBLP record whose venue STRING does not look like a
        preprint but whose venue_key is the CoRR stream is excluded by the
        identifier-side preprint marker."""
        per_source = {
            "crossref": _record("Computing Research Repository", venue_key="journals/corr"),
            "openalex": _record("Computing Research Repository", venue_key="journals/corr"),
        }
        assert checker._detect_cross_source_venue_mismatch("AISTATS", per_source) is None

    def test_preprint_entry_venue_abstains(self, checker):
        """A preprint claim on the ENTRY side is never refuted by published
        consensus (the published twin legitimately coexists)."""
        per_source = {
            "crossref": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "openalex": _record("Foo Journal of Bar", issn=("1234-5678",)),
        }
        assert checker._detect_cross_source_venue_mismatch("arXiv preprint arXiv:2301.00001", per_source) is None

    def test_two_conflicting_consensus_groups_abstain(self, checker):
        """Sources disagreeing among themselves (two identifier-corroborated
        groups) is ambiguity, not consensus -> abstain."""
        per_source = {
            "crossref": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "openalex": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "dblp": _record("Baz Quarterly Review", issn=("1111-2222",)),
            "openreview": _record("Baz Quarterly Review", issn=("1111-2222",)),
        }
        assert checker._detect_cross_source_venue_mismatch("AISTATS", per_source) is None

    def test_order_unreliable_sources_never_anchor(self, checker):
        """Identifier links between order-UNRELIABLE records do not build a
        consensus group (mirrors the existing order-reliable gate)."""
        per_source = {
            "semanticscholar": _record("Foo Journal of Bar", issn=("1234-5678",), order_reliable=False),
            "openalex": _record("Foo Journal of Bar", issn=("1234-5678",)),
        }
        assert checker._detect_cross_source_venue_mismatch("AISTATS", per_source) is None

    def test_dblp_venue_key_link_groups_records(self, checker):
        """venue_key equality is a grouping identifier: two sources reporting
        the same DBLP conf stream under different display strings form an
        identifier-corroborated consensus."""
        per_source = {
            "dblp": _record("Foo Symposium (FOOS)", venue_key="conf/foos"),
            "crossref": _record("Foo Symp.", venue_key="conf/foos"),
        }
        consensus = checker._detect_cross_source_venue_mismatch("Journal of Advanced Neural Computing", per_source)
        assert consensus == "Foo Symposium (FOOS)"

    def test_compare_all_fields_downgrades_venue_via_identifier_consensus(self, checker):
        """Wiring: with a preprint-twin best match (venue NON_COMPARABLE), the
        identifier-based consensus downgrades the venue comparison to MISMATCH
        and the note names the consensus venue."""
        entry = {
            "ID": "x",
            "ENTRYTYPE": "article",
            "title": "A Paper",
            "author": "Lovelace, Ada",
            "journal": "AISTATS",
            "year": "2023",
        }
        best = _record("arXiv preprint arXiv:2301.00001")
        per_source = {
            "crossref": _record("Foo Journal of Bar", issn=("1234-5678",)),
            "openalex": _record("Foo Journal of Bar", issn=("1234-5678",)),
        }
        comparisons = checker._compare_all_fields(entry, best, per_source_records=per_source)
        assert comparisons["venue"].outcome is MatchOutcome.MISMATCH
        assert "Foo Journal of Bar" in (comparisons["venue"].note or "")
