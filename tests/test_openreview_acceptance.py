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
