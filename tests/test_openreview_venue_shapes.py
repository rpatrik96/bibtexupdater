"""OpenReview note ``venue``/``venueid`` may not be a string.

A live run crashed with ``'list' object has no attribute 'lower'`` while checking
``@proceedings`` entries. The OpenReview converter reads::

    venue = _content_value(content, "venue") or _content_value(content, "venueid")
    if is_preprint_venue(venue):   # -> venue.lower()

Every neighbouring field is coerced defensively (``str(title)``, ``str(raw_year)``,
isinstance guards on ``authors``/``authorids``) -- ``venue`` was the one that was
not, and OpenReview returns ``venueid`` as a LIST for some notes. The exception
was swallowed per-entry, so the entry was silently dropped from the report: worse
than a wrong verdict, because nothing signals the gap.

The converter must survive any shape and still produce a usable record.
"""

from __future__ import annotations

from bibtex_updater.sources import openreview_note_to_candidate_record


def _note(venue_field: object, key: str = "venue") -> dict:
    return {
        "content": {
            "title": {"value": "A Real Paper Title"},
            "authors": {"value": ["Ada Lovelace"]},
            "authorids": {"value": ["~Ada_Lovelace1"]},
            key: {"value": venue_field},
        }
    }


class TestVenueShapeRobustness:
    def test_list_valued_venue_does_not_crash(self):
        rec = openreview_note_to_candidate_record(_note(["ICLR.cc/2024/Conference"]))
        assert rec is not None
        assert isinstance(rec.journal, str) or rec.journal is None

    def test_list_valued_venueid_does_not_crash(self):
        rec = openreview_note_to_candidate_record(_note(["NeurIPS.cc/2023/Conference"], key="venueid"))
        assert rec is not None
        assert isinstance(rec.journal, str) or rec.journal is None

    def test_list_venue_keeps_its_first_usable_string(self):
        rec = openreview_note_to_candidate_record(_note(["ICLR 2024", "ignored"]))
        assert rec is not None and rec.journal == "ICLR 2024"

    def test_empty_list_venue_yields_no_venue(self):
        rec = openreview_note_to_candidate_record(_note([]))
        assert rec is not None and not rec.journal

    def test_list_of_non_strings_yields_no_venue(self):
        rec = openreview_note_to_candidate_record(_note([{"nested": 1}, 7]))
        assert rec is not None and not rec.journal

    def test_non_string_scalar_venue_does_not_crash(self):
        rec = openreview_note_to_candidate_record(_note(2024))
        assert rec is not None

    def test_preprint_venue_in_a_list_is_still_dropped(self):
        """The CoRR gate must survive the coercion, not be bypassed by it."""
        rec = openreview_note_to_candidate_record(_note(["CoRR 2017"]))
        assert rec is not None and not rec.journal

    def test_plain_string_venue_is_unchanged(self):
        rec = openreview_note_to_candidate_record(_note("ICLR 2024"))
        assert rec is not None and rec.journal == "ICLR 2024"

    def test_year_still_recovered_from_a_list_venue(self):
        rec = openreview_note_to_candidate_record(_note(["ICLR 2024"]))
        assert rec is not None and rec.year == 2024
