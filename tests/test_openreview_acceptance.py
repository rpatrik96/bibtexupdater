"""Tests for the openreview_acceptance() status classifier."""

from __future__ import annotations

import pytest

from bibtex_updater.sources import (
    OR_ACCEPTED,
    OR_NOT_ACCEPTED,
    OR_PREPRINT,
    OR_UNKNOWN,
    openreview_acceptance,
)


def _note(venue=None, venueid=None):
    content = {}
    if venue is not None:
        content["venue"] = venue
    if venueid is not None:
        content["venueid"] = venueid
    return {"content": content}


class TestOpenReviewAcceptance:
    @pytest.mark.parametrize(
        "venue,venueid,expected",
        [
            # accepted: DBLP conference import (the common case for indexed papers)
            ("ICLR 2021", "dblp.org/conf/ICLR/2021", OR_ACCEPTED),
            # accepted: native conference venueid
            ("ICLR 2024 poster", "ICLR.cc/2024/Conference", OR_ACCEPTED),
            ("NeurIPS 2023 oral", "NeurIPS.cc/2023/Conference", OR_ACCEPTED),
            # accepted: clean venue string with a year, no negative markers, no venueid
            ("ICLR 2021", None, OR_ACCEPTED),
            # preprint: CoRR / arXiv mirror
            ("CoRR 2020", "dblp.org/journals/CORR/2020", OR_PREPRINT),
            ("arXiv 2010.11929", None, OR_PREPRINT),
            # not accepted: withdrawn / rejected venueid
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference/Withdrawn_Submission", OR_NOT_ACCEPTED),
            ("Submitted to NeurIPS 2023", "NeurIPS.cc/2023/Conference/Rejected_Submission", OR_NOT_ACCEPTED),
            # not accepted: 'Submitted to' venue (under review), even with a Conference venueid
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference", OR_NOT_ACCEPTED),
            # unknown: nothing usable
            (None, None, OR_UNKNOWN),
            ("", "", OR_UNKNOWN),
            # unknown: yearless venue + unrecognized venueid (e.g. TMLR continuous) -> conservative
            ("Transactions on Machine Learning Research", "TMLR", OR_UNKNOWN),
        ],
    )
    def test_classification(self, venue, venueid, expected):
        assert openreview_acceptance(_note(venue, venueid)) == expected

    def test_preprint_takes_precedence_over_year(self):
        # a CoRR note carries a year but must classify PREPRINT, never ACCEPTED
        assert openreview_acceptance(_note("CoRR 2017", "dblp.org/journals/CORR/2017")) == OR_PREPRINT

    def test_empty_or_malformed_note(self):
        assert openreview_acceptance({}) == OR_UNKNOWN
        assert openreview_acceptance({"content": None}) == OR_UNKNOWN
        assert openreview_acceptance({"content": "garbage"}) == OR_UNKNOWN


def _v2_note(venue=None, venueid=None):
    """API v2 note: every content field wrapped as ``{"value": ...}``."""
    content = {}
    if venue is not None:
        content["venue"] = {"value": venue}
    if venueid is not None:
        content["venueid"] = {"value": venueid}
    return {"content": content}


class TestOpenReviewAcceptanceV2Shapes:
    """The classifier must unwrap API v2 ``{"value": ...}`` content fields and
    mirror the v1 vocabulary (rejected/withdrawn/Submitted-to/native venueids).
    """

    @pytest.mark.parametrize(
        "venue,venueid,expected",
        [
            # accepted: native conference venueids (the v2-era shape)
            ("ICLR 2024 poster", "ICLR.cc/2024/Conference", OR_ACCEPTED),
            ("NeurIPS 2023 oral", "NeurIPS.cc/2023/Conference", OR_ACCEPTED),
            # not accepted: rejected / withdrawn / desk-rejected venueids
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference/Rejected_Submission", OR_NOT_ACCEPTED),
            (None, "NeurIPS.cc/2023/Conference/Withdrawn_Submission", OR_NOT_ACCEPTED),
            (None, "ICLR.cc/2024/Conference/Desk_Rejected_Submission", OR_NOT_ACCEPTED),
            # not accepted: under review ("Submitted to ..."), clean venueid
            ("Submitted to NeurIPS 2023", "NeurIPS.cc/2023/Conference", OR_NOT_ACCEPTED),
            # preprint mirror
            ("CoRR 2023", "dblp.org/journals/CORR/2023", OR_PREPRINT),
            # unknown: nothing usable
            (None, None, OR_UNKNOWN),
        ],
    )
    def test_v2_wrapped_classification(self, venue, venueid, expected):
        assert openreview_acceptance(_v2_note(venue, venueid)) == expected

    def test_v1_and_v2_shapes_agree(self):
        cases = [
            ("ICLR 2024 poster", "ICLR.cc/2024/Conference"),
            ("Submitted to ICLR 2024", "ICLR.cc/2024/Conference/Rejected_Submission"),
            ("CoRR 2023", "dblp.org/journals/CORR/2023"),
            (None, None),
        ]
        for venue, venueid in cases:
            assert openreview_acceptance(_note(venue, venueid)) == openreview_acceptance(_v2_note(venue, venueid))
