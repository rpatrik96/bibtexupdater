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

from bibtex_updater.matching import is_volume_entry_type, normalize_volume_title, volume_title_subsumed


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

    def test_record_subtitle_does_not_make_it_a_different_volume(self):
        """Publishers store the volume's full descriptive title.

        The entry cites "Computer Vision -- ECCV 2016"; Crossref's record for the
        same DOI is "Computer Vision - ECCV 2016: 14th European Conference,
        Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part I".
        Same volume, and the DOI is the entry's own identifier.
        """
        entry = normalize_volume_title("Computer Vision -- {ECCV} 2016")
        api = normalize_volume_title(
            "Computer Vision - ECCV 2016: 14th European Conference, Amsterdam, "
            "The Netherlands, October 11-14, 2016, Proceedings, Part I"
        )
        assert entry and api
        assert entry in api


class TestVolumeTitleSubsumption:
    """A volume's cited title is the record's title minus a descriptive suffix.

    Indexes append the meeting's place and dates to a proceedings title
    ("... Computational Linguistics, ACL 2020, Online, July 5-10, 2020"). The
    cited short form is the same volume, and the length gap alone is what sinks
    the fuzzy score below the title threshold.
    """

    def test_acl_proceedings_with_place_and_date_suffix(self):
        assert volume_title_subsumed(
            "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
            "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, "
            "ACL 2020, Online, July 5-10, 2020",
        )

    def test_eccv_proceedings_with_part_suffix(self):
        assert volume_title_subsumed(
            "Computer Vision -- {ECCV} 2016",
            "Computer Vision - ECCV 2016: 14th European Conference, Amsterdam, "
            "The Netherlands, October 11-14, 2016, Proceedings, Part I",
        )

    def test_a_different_conference_is_not_subsumed(self):
        assert not volume_title_subsumed(
            "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
            "Advances in Cryptology - CRYPTO 2019, 39th Annual International Cryptology Conference",
        )

    def test_empty_sides_are_not_subsumed(self):
        assert not volume_title_subsumed("", "Anything At All Here")
        assert not volume_title_subsumed("Anything At All Here", "")


class TestVolumeComparisonEndToEnd:
    """``_compare_all_fields`` must apply the subsumption rule, not just the DOI path."""

    def test_acl_volume_title_confirms_against_suffixed_record(self):
        import logging
        from unittest.mock import MagicMock

        from bibtex_updater.fact_checker import (
            CrossrefClient,
            DBLPClient,
            FactChecker,
            FactCheckerConfig,
            PublishedRecord,
            SemanticScholarClient,
        )

        http = MagicMock()
        http._request.return_value = MagicMock(status_code=404, json=lambda: {})
        checker = FactChecker(
            CrossrefClient(http),
            DBLPClient(http),
            SemanticScholarClient(http),
            FactCheckerConfig(),
            logging.getLogger("volume-cmp-test"),
        )
        entry = {
            "ID": "g3_acl2020proceedings",
            "ENTRYTYPE": "proceedings",
            "editor": "Dan Jurafsky and Joyce Chai and Natalie Schluter and Joel Tetreault",
            "title": "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
            "year": "2020",
        }
        record = PublishedRecord(
            doi="10.18653/v1/2020.acl-main",
            url=None,
            title=(
                "Proceedings of the 58th Annual Meeting of the Association for Computational "
                "Linguistics, ACL 2020, Online, July 5-10, 2020"
            ),
            authors=[
                {"given": "Dan", "family": "Jurafsky"},
                {"given": "Joyce", "family": "Chai"},
                {"given": "Natalie", "family": "Schluter"},
                {"given": "Joel R.", "family": "Tetreault"},
            ],
            journal="ACL",
            year=2020,
        )
        comparisons = checker._compare_all_fields(entry, record)
        assert comparisons["title"].is_confirmed, comparisons["title"].similarity_score


class TestDoiConsistencyForVolumes:
    """The DOI-consistency check must use volume normalization too.

    It has its OWN title comparison, so a volume-title fix applied only in
    ``_compare_all_fields`` leaves this path flagging DOI_MISMATCH on a
    correctly-cited proceedings volume.
    """

    def _checker(self):
        import logging
        from unittest.mock import MagicMock

        from bibtex_updater.fact_checker import (
            CrossrefClient,
            DBLPClient,
            FactChecker,
            FactCheckerConfig,
            SemanticScholarClient,
        )

        http = MagicMock()
        http._request.return_value = MagicMock(status_code=404, json=lambda: {})
        return FactChecker(
            CrossrefClient(http),
            DBLPClient(http),
            SemanticScholarClient(http),
            FactCheckerConfig(),
            logging.getLogger("volume-doi-test"),
        )

    def test_proceedings_doi_with_fuller_record_title_is_not_a_mismatch(self):
        from bibtex_updater.fact_checker import PublishedRecord

        checker = self._checker()
        record = PublishedRecord(
            doi="10.1007/978-3-319-46448-0",
            url=None,
            title=(
                "Computer Vision - ECCV 2016: 14th European Conference, Amsterdam, "
                "The Netherlands, October 11-14, 2016, Proceedings, Part I"
            ),
            authors=[],
            journal="Lecture Notes in Computer Science",
            year=2016,
        )
        checker._structured_record_by_doi = lambda _doi: record
        entry = {
            "ID": "g3_eccv2016proceedings",
            "ENTRYTYPE": "proceedings",
            "editor": "Bastian Leibe and Jiri Matas and Nicu Sebe and Max Welling",
            "title": "Computer Vision -- {ECCV} 2016",
            "year": "2016",
            "doi": "10.1007/978-3-319-46448-0",
        }
        assert checker._check_doi_consistency(entry) is None

    def test_a_genuinely_wrong_doi_on_a_volume_still_flags(self):
        from bibtex_updater.fact_checker import FactCheckStatus, PublishedRecord

        checker = self._checker()
        record = PublishedRecord(
            doi="10.1007/978-3-319-46448-0",
            url=None,
            title="Advances in Cryptology - CRYPTO 2019",
            authors=[],
            journal="Lecture Notes in Computer Science",
            year=2019,
        )
        checker._structured_record_by_doi = lambda _doi: record
        entry = {
            "ID": "x",
            "ENTRYTYPE": "proceedings",
            "title": "Computer Vision -- {ECCV} 2016",
            "year": "2016",
            "doi": "10.1007/978-3-319-46448-0",
        }
        result = checker._check_doi_consistency(entry)
        assert result is not None and result.status is FactCheckStatus.DOI_MISMATCH
