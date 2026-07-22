"""Entry-field fallbacks: ``howpublished`` as a venue, ``editor`` as an author.

Two BibTeX fields the checker never read, each costing a systematic block of
false positives on the Varga bibliography:

* ``@misc`` front matter carries its venue in ``howpublished``
  (``howpublished = {Infocommunications Journal}``). Venue extraction was
  ``journal or booktitle``, so 24 entries reported "No venue claimed" -- the
  venue then matched trivially, and the strongest available disambiguator never
  constrained retrieval.
* ``@proceedings``/``@book`` name their people in ``editor``, not ``author``.
  ``entry.get("author", "")`` yielded an EMPTY author list for all five
  ``@proceedings`` entries, every one of which was flagged.

A URL-valued ``howpublished`` is a web reference, not a venue, and must not be
promoted -- that is what the existing web-reference path handles.
"""

from bibtex_updater.utils import entry_authors, entry_venue


class TestEntryVenue:
    """``howpublished``/``series`` back the venue when journal/booktitle are absent."""

    def test_journal_wins(self):
        assert entry_venue({"journal": "Nature", "howpublished": "Elsewhere"}) == "Nature"

    def test_booktitle_wins_over_howpublished(self):
        assert entry_venue({"booktitle": "NeurIPS", "howpublished": "Elsewhere"}) == "NeurIPS"

    def test_howpublished_used_when_no_journal_or_booktitle(self):
        entry = {"title": "Impactful Surveys", "howpublished": "Infocommunications Journal"}
        assert entry_venue(entry) == "Infocommunications Journal"

    def test_series_used_as_last_resort(self):
        assert entry_venue({"series": "Lecture Notes in Computer Science"}) == "Lecture Notes in Computer Science"

    def test_no_venue_claim_returns_empty(self):
        assert entry_venue({"title": "Something"}) == ""

    def test_whitespace_only_is_no_claim(self):
        assert entry_venue({"howpublished": "   "}) == ""

    def test_missing_fields_return_empty(self):
        assert entry_venue({}) == ""


class TestEntryVenueRejectsUrls:
    """A URL-valued ``howpublished`` is a web reference, never a venue."""

    def test_bare_url_rejected(self):
        assert entry_venue({"howpublished": "https://example.org/post"}) == ""

    def test_http_url_rejected(self):
        assert entry_venue({"howpublished": "http://example.org"}) == ""

    def test_latex_url_macro_rejected(self):
        assert entry_venue({"howpublished": "\\url{https://example.org}"}) == ""

    def test_www_prefix_rejected(self):
        assert entry_venue({"howpublished": "www.example.org"}) == ""

    def test_online_boilerplate_rejected(self):
        # "[Online]. Available: ..." is a citation-style marker, not a venue.
        assert entry_venue({"howpublished": "[Online]"}) == ""

    def test_real_venue_containing_a_dot_survives(self):
        assert entry_venue({"howpublished": "Proc. IEEE INFOCOM"}) == "Proc. IEEE INFOCOM"


class TestEntryAuthors:
    """``editor`` backs ``author`` for volume-level entry types."""

    def test_author_wins(self):
        assert entry_authors({"author": "Varga, Pal", "editor": "Someone Else"}) == "Varga, Pal"

    def test_editor_used_when_no_author(self):
        entry = {"editor": "Varga, P{\\'a}l and Wauters, Tim", "title": "CNSM 2024"}
        assert entry_authors(entry) == "Varga, P{\\'a}l and Wauters, Tim"

    def test_no_people_returns_empty(self):
        assert entry_authors({"title": "Anonymous Proceedings"}) == ""

    def test_whitespace_only_author_falls_back_to_editor(self):
        assert entry_authors({"author": "  ", "editor": "Varga, Pal"}) == "Varga, Pal"

    def test_missing_fields_return_empty(self):
        assert entry_authors({}) == ""
