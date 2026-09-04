"""Venue comparison: expand ISO-4 abbreviations, and abstain on unknown names.

Two defects, both producing venue disagreements on correct citations.

ISO-4 is the standard abbreviation for journal titles and the form a large share
of real ``.bib`` files carry. ``EXPANDED_VENUE_ALIASES`` covers ML and CS
conferences by acronym and full name, so an abbreviated journal canonicalises to
nothing and falls to a token sort below threshold. Measured before the fix:
``ACM Trans. Graph.`` 0.70, ``Proc. Natl. Acad. Sci. U.S.A.`` 0.60,
``Annu. Rev. Stat. Appl.`` 0.55 -- three real papers reported as mis-venued.

Separately, the terminal branch returned MISMATCH for any pair that reached it,
including pairs where the comparator recognised NEITHER name. Not recognising a
name is not evidence that two names differ. On a workshop corpus, 920 of 5,043
references came back with venue as the sole disagreement, 127 of them with no
journal field at all and a correct venue name in non-canonical form.

The abstention is deliberately narrow. MISMATCH is kept wherever there are
positive grounds: both sides canonicalising to different known venues, a
satellite-event asymmetry, or both sides having stated an acronym.
"""

from __future__ import annotations

import pytest

from bibtex_updater.fact_checker import venues_match
from bibtex_updater.matching import (
    MatchOutcome,
    expand_ltwa_abbreviations,
    venue_abbreviation_matches,
)


class TestIso4Expansion:
    @pytest.mark.parametrize(
        ("abbreviated", "full"),
        [
            ("ACM Trans. Graph.", "ACM Transactions on Graphics"),
            ("Proc. Natl. Acad. Sci. U.S.A.", "Proceedings of the National Academy of Sciences"),
            ("Annu. Rev. Stat. Appl.", "Annual Review of Statistics and Its Application"),
            (
                "IEEE Trans. Pattern Anal. Mach. Intell.",
                "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            ),
            ("J. Mach. Learn. Res.", "Journal of Machine Learning Research"),
            ("Int. J. Comput. Vis.", "International Journal of Computer Vision"),
            ("Nat. Mach. Intell.", "Nature Machine Intelligence"),
        ],
    )
    def test_abbreviated_journal_matches_its_full_name(self, abbreviated, full):
        assert venues_match(abbreviated, full).outcome is MatchOutcome.MATCH

    def test_expansion_is_idempotent(self):
        """Full forms are not themselves keys, so re-expanding changes nothing."""
        once = expand_ltwa_abbreviations("Proc. Natl. Acad. Sci.")
        assert expand_ltwa_abbreviations(once) == once

    def test_expansion_only_ever_reports_a_positive(self):
        """It can clear a false disagreement; it must not manufacture one."""
        assert venue_abbreviation_matches("NeurIPS", "ICML") is False
        assert venue_abbreviation_matches("", "ICML") is False
        assert venue_abbreviation_matches("ICML", "") is False

    def test_unmapped_words_pass_through(self):
        """An incomplete map is safe: unknown words are left alone."""
        assert "quixotic" in expand_ltwa_abbreviations("Quixotic Trans. Foo")


class TestDifferentVenuesStillMismatch:
    """The negatives. Both reviewers of this change asked for these explicitly.

    A matcher loosened to abstain on unknown names must not start abstaining on
    known-different ones -- wrong venue is a real hallucination class, 8 of 119
    field-level defects on the workshop corpus that motivated the change.
    """

    @pytest.mark.parametrize(
        ("venue_a", "venue_b"),
        [
            ("NeurIPS", "ICML"),
            ("ACL", "EMNLP"),
            ("CVPR", "ICCV"),
            ("Neural Information Processing Systems", "International Conference on Machine Learning"),
        ],
    )
    def test_two_different_known_venues_mismatch(self, venue_a, venue_b):
        assert venues_match(venue_a, venue_b).outcome is MatchOutcome.MISMATCH

    def test_workshop_does_not_match_its_host_conference(self):
        assert venues_match("ICML Workshop on Foo", "ICML").outcome is MatchOutcome.MISMATCH

    def test_declared_acronyms_that_differ_are_a_real_mismatch(self):
        """Both sides stated a shorthand, so the pair is comparable."""
        result = venues_match(
            "2023 19th International Conference on Network and Service Management (CNSM)",
            "NOMS",
        )
        assert result.outcome is MatchOutcome.MISMATCH

    def test_two_expanding_journals_that_differ_still_mismatch_or_abstain(self):
        """Expansion must not collapse distinct journals into a match."""
        result = venues_match("J. Mach. Learn. Res.", "J. Artif. Intell. Res.")
        assert result.outcome is not MatchOutcome.MATCH


class TestAbstainOnUnrecognisedNames:
    def test_two_unknown_dissimilar_venues_are_non_comparable(self):
        """Not recognising either name is not evidence that they differ."""
        result = venues_match("Journal of Irreproducible Results", "Baltic Journal of Herpetology")
        assert result.outcome is MatchOutcome.NON_COMPARABLE

    def test_a_thinly_indexed_real_venue_is_not_called_a_disagreement(self):
        """The COLM case: real, but absent from the alias map for a while.

        A venue the comparator has never heard of, cited correctly, must not be
        reported as mis-venued just because the map is incomplete. Note neither
        side may carry a satellite marker ("workshop", "symposium"): that
        asymmetry is checked earlier and is a genuine mismatch on its own, so a
        fixture containing one would test the wrong branch.
        """
        result = venues_match(
            "Third Conference on Obscure But Real Methods",
            "Meeting on Entirely Different Things",
        )
        assert result.outcome is MatchOutcome.NON_COMPARABLE

    def test_one_side_known_still_mismatches(self):
        """Abstention needs BOTH sides unrecognised; one known name is a signal."""
        result = venues_match("NeurIPS", "Baltic Journal of Herpetology")
        assert result.outcome is MatchOutcome.MISMATCH
