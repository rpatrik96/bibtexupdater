"""Publisher-series venues and venue-name subsumption.

Two systematic venue false positives from the Varga run, both firing on entries
whose title matched the record EXACTLY (so the paper was certainly right):

* **Series.** Crossref/OpenAlex return the publisher SERIES as container-title
  for Springer/IFIP proceedings, so ``booktitle = {International Conference on
  Entertainment Computing}`` was compared against ``Lecture Notes in Computer
  Science`` and reported as a venue mismatch. ``_SERIES_MARKERS`` listed only
  PMLR/JMLR W&CP -- an ML-only roster that leaves the entire Springer ecosystem
  (LNCS, CCIS, LNICST, Studies in Computational Intelligence) unrecognized.
  A series spans many distinct conferences, so it cannot pin one venue: the
  comparison is NON_COMPARABLE, never a mismatch.

* **Subsumption.** Databases routinely store a shortened container name --
  ``2021 IFIP/IEEE International Symposium on Integrated Network Management
  (IM)`` indexed as ``Integrated Network Management``, or a subtitle dropped
  (``...: New Methods and Practice for the Networked Society``). One name
  containing the other is the same venue, not two different ones.

Subsumption must NOT swallow a workshop into its parent conference, which is a
genuinely different venue -- hence the blocking-token guard.
"""

from bibtex_updater.fact_checker import venues_match
from bibtex_updater.matching import MatchOutcome, is_preprint_or_series_venue


class TestSpringerIfipSeriesRecognized:
    """Publisher series that span many conferences cannot confirm one venue."""

    def test_lecture_notes_in_computer_science(self):
        assert is_preprint_or_series_venue("Lecture Notes in Computer Science")

    def test_lecture_notes_in_networks_and_systems(self):
        assert is_preprint_or_series_venue("Lecture Notes in Networks and Systems")

    def test_lncs_with_volume_suffix(self):
        assert is_preprint_or_series_venue("Lecture Notes in Computer Science, vol 13495")

    def test_studies_in_computational_intelligence(self):
        assert is_preprint_or_series_venue("Studies in Computational Intelligence")

    def test_communications_in_computer_and_information_science(self):
        assert is_preprint_or_series_venue("Communications in Computer and Information Science")

    def test_ifip_federation_umbrella(self):
        assert is_preprint_or_series_venue("IFIP International Federation for Information Processing")

    def test_ifip_advances(self):
        assert is_preprint_or_series_venue("IFIP Advances in Information and Communication Technology")

    def test_advances_in_intelligent_systems_and_computing(self):
        assert is_preprint_or_series_venue("Advances in Intelligent Systems and Computing")

    def test_lnicst(self):
        assert is_preprint_or_series_venue(
            "Lecture Notes of the Institute for Computer Sciences, "
            "Social Informatics and Telecommunications Engineering"
        )

    def test_springerbriefs(self):
        assert is_preprint_or_series_venue("SpringerBriefs in Applied Sciences and Technology")


class TestRealVenuesNotMistakenForSeries:
    """Guard: the series roster must not swallow real, specific venues."""

    def test_jmlr_journal_still_specific(self):
        assert not is_preprint_or_series_venue("Journal of Machine Learning Research")

    def test_named_conference_still_specific(self):
        assert not is_preprint_or_series_venue("International Conference on Entertainment Computing")

    def test_ieee_transactions_still_specific(self):
        assert not is_preprint_or_series_venue("IEEE Transactions on Network and Service Management")

    def test_nature_still_specific(self):
        assert not is_preprint_or_series_venue("Nature")

    def test_ifip_named_conference_still_specific(self):
        # A specific IFIP *conference* is a venue; only the umbrella series is not.
        assert not is_preprint_or_series_venue("IFIP/IEEE International Symposium on Integrated Network Management")


class TestSeriesVenueComparisonIsNonComparable:
    """A series on the record side abstains rather than contradicting the entry."""

    def test_lncs_record_does_not_contradict_conference_claim(self):
        result = venues_match(
            "International Conference on Entertainment Computing", "Lecture Notes in Computer Science"
        )
        assert result.outcome is MatchOutcome.NON_COMPARABLE

    def test_studies_in_computational_intelligence_record(self):
        result = venues_match("Complex Networks and Their Applications XII", "Studies in Computational Intelligence")
        assert result.outcome is MatchOutcome.NON_COMPARABLE

    def test_ifip_umbrella_record(self):
        result = venues_match(
            "EUNICE 2005: Networks and Applications Towards a Ubiquitously Connected World",
            "IFIP International Federation for Information Processing",
        )
        assert result.outcome is MatchOutcome.NON_COMPARABLE


class TestVenueSubsumption:
    """One venue name containing the other is the same venue."""

    def test_organizer_prefix_and_acronym_dropped_by_index(self):
        result = venues_match(
            "2021 IFIP/IEEE International Symposium on Integrated Network Management (IM)",
            "Integrated Network Management",
        )
        assert result.outcome is MatchOutcome.MATCH

    def test_subtitle_dropped_by_index(self):
        result = venues_match(
            "Advances in Information Systems Development: New Methods and Practice for the Networked Society",
            "Advances in Information Systems Development",
        )
        assert result.outcome is MatchOutcome.MATCH

    def test_subsumption_is_symmetric(self):
        result = venues_match(
            "Integrated Network Management",
            "2021 IFIP/IEEE International Symposium on Integrated Network Management (IM)",
        )
        assert result.outcome is MatchOutcome.MATCH


class TestSubsumptionGuards:
    """Subsumption must not merge genuinely different venues."""

    def test_workshop_is_not_its_parent_conference(self):
        result = venues_match(
            "International Conference on Machine Learning Workshop on Foundation Models",
            "International Conference on Machine Learning",
        )
        assert result.outcome is not MatchOutcome.MATCH

    def test_companion_volume_is_not_the_main_conference(self):
        result = venues_match(
            "Companion Proceedings of the ACM Web Conference",
            "ACM Web Conference",
        )
        assert result.outcome is not MatchOutcome.MATCH

    def test_short_fragment_does_not_subsume(self):
        # Fewer than three tokens is too generic to establish identity.
        result = venues_match("International Conference on Robotics and Automation", "Automation")
        assert result.outcome is not MatchOutcome.MATCH

    def test_workshop_whose_own_name_was_shortened_still_matches(self):
        """A workshop is a venue too: dropping "International Workshop on" from
        ITS OWN name is index truncation, not a satellite/parent confusion.

        The discriminator is whether the longer side adds substantive TOPIC
        words beyond the marker and boilerplate. "ICML Workshop on Foundation
        Models" adds "foundation models" -- a different event. "International
        Workshop on IP Operations and Management" adds nothing but boilerplate.
        """
        result = venues_match(
            "International Workshop on IP Operations and Management",
            "IP Operations and Management",
        )
        assert result.outcome is MatchOutcome.MATCH

    def test_symposium_prefix_truncation_still_matches(self):
        result = venues_match(
            "International Workshop on Quality of Service",
            "Quality of Service",
        )
        assert result.outcome is MatchOutcome.MATCH

    def test_unrelated_venues_still_mismatch(self):
        result = venues_match("Nature", "Science")
        assert result.outcome is MatchOutcome.MISMATCH

    def test_different_ieee_transactions_still_mismatch(self):
        result = venues_match(
            "IEEE Transactions on Network and Service Management",
            "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        )
        assert result.outcome is MatchOutcome.MISMATCH
