"""Volume-level entry types: ``@proceedings`` titles are venue names.

A ``@proceedings`` entry describes a whole conference volume, so its ``title``
IS the conference name -- carrying exactly the boilerplate a venue name carries
(a leading year, an ordinal, a ``Proceedings of the`` prefix, a trailing
``(ACRONYM YEAR)``). Comparing it verbatim against the indexed container name
flagged every such entry in the Varga run, including one at title similarity
**0.97**: ``... (CNSM 2024)`` vs ``... (CNSM)`` differs by five characters, which
``is_near_miss_title`` (min edit distance 3) reads as deliberate tampering.

For a volume title the fix is to normalize it the way venue names are already
normalized, and to switch off the near-miss rule -- a year or ordinal delta in a
conference name is boilerplate, not fabrication.
"""

from bibtex_updater.matching import is_volume_entry_type, normalize_volume_title


class TestVolumeEntryTypes:
    def test_proceedings_is_volume(self):
        assert is_volume_entry_type("proceedings")

    def test_case_insensitive(self):
        assert is_volume_entry_type("Proceedings")

    def test_inproceedings_is_not_volume(self):
        # A paper IN proceedings has a real paper title; only the volume itself
        # is named after the conference.
        assert not is_volume_entry_type("inproceedings")

    def test_article_is_not_volume(self):
        assert not is_volume_entry_type("article")

    def test_empty_is_not_volume(self):
        assert not is_volume_entry_type("")


class TestNormalizeVolumeTitle:
    """Conference-name boilerplate is stripped before comparison."""

    def test_proceedings_of_the_prefix_dropped(self):
        assert "proceedings of" not in normalize_volume_title(
            "Proceedings of the 18th IEEE/IFIP Network Operations and Management Symposium"
        )

    def test_leading_year_dropped(self):
        assert "2024" not in normalize_volume_title("2024 20th International Conference on Network Management")

    def test_trailing_acronym_year_matches_bare_acronym(self):
        entry = "2024 20th International Conference on Network and Service Management (CNSM 2024)"
        api = "2024 20th International Conference on Network and Service Management (CNSM)"
        assert normalize_volume_title(entry) == normalize_volume_title(api)

    def test_noms_proceedings_prefix_and_year_suffix(self):
        entry = "Proceedings of the 18th IEEE/IFIP Network Operations and Management Symposium (NOMS 2022)"
        api = "NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium"
        # Not identical, but the shared conference name must dominate.
        assert "noms" in normalize_volume_title(entry)
        assert "proceedings of" not in normalize_volume_title(entry)
        assert "2022" not in normalize_volume_title(api)

    def test_distinct_conferences_stay_distinct(self):
        a = normalize_volume_title("International Conference on Network and Service Management")
        b = normalize_volume_title("International Conference on Machine Learning")
        assert a != b

    def test_empty_title(self):
        assert normalize_volume_title("") == ""
