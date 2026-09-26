"""Metamorphic / symmetry properties for author comparison.

Each subtle false-mismatch bug can be stated as an *invariance*: a TRUE match's
verdict must not change when the SAME underlying name is written in a different
but equivalent citation style. We hold one side fixed and apply a
representation-preserving transform to the other, then assert the author
comparison still matches.

We reproduce the production author-comparison contract from
``FactChecker._compare_all_fields``:

* the ENTRY side is keyed with ``authors_last_names`` (which routes each name
  through ``last_name_from_person``);
* the RECORD side is keyed with ``PublishedRecord.surname_keys`` -- the single
  source of truth that also routes each ``family`` through
  ``last_name_from_person``, so the two sides are derived symmetrically;
* the two key lists are scored with ``combined_author_score`` and compared to
  ``FactCheckerConfig.author_threshold``.

The point of routing BOTH sides through the same reduction is that these
invariances hold by construction. Each property below would have failed under
the historical asymmetric contract (entry reduced to a token, record kept raw),
which is exactly the class of bug this suite guards against.

``hypothesis`` is NOT installed in this environment, so the property tests below
are written as thorough ``@pytest.mark.parametrize`` example tables rather than
generators. The file therefore runs under plain pytest.
"""

from __future__ import annotations

import pytest

from bibtex_updater.fact_checker import FactCheckerConfig
from bibtex_updater.matching import combined_author_score
from bibtex_updater.utils import (
    PublishedRecord,
    authors_last_names,
    last_name_from_person,
)

AUTHOR_THRESHOLD = FactCheckerConfig().author_threshold


# ------------- Production-contract helpers (mirror _compare_all_fields) --------


def _entry_keys(author_field: str) -> list[str]:
    """Entry-side surname keys, exactly as _compare_all_fields derives them."""
    return authors_last_names(author_field, limit=10)


def _record_keys(families: list[str]) -> list[str]:
    """Record-side surname keys, exactly as _compare_all_fields derives them.

    Mirrors the production contract: the record side goes through
    ``PublishedRecord.surname_keys`` (the single source of truth), which routes
    each ``family`` through the same ``last_name_from_person`` the entry side
    uses.
    """
    return PublishedRecord(doi="x", authors=[{"family": f} for f in families]).surname_keys(limit=10)


def _author_matches(author_field: str, families: list[str]) -> bool:
    """Replicate the author FieldComparison.matches decision."""
    score = combined_author_score(_entry_keys(author_field), _record_keys(families), 0.5, 0.5)
    return score >= AUTHOR_THRESHOLD


# ------------- Property 1: surname-key invariance under name-order reorder -----


class TestGivenFamilyVsFamilyGivenKey:
    """``last_name_from_person`` must yield the same surname key for
    'Given Family' and 'Family, Given' representations of one person.

    This is the hypothesis-style property, expressed as a generated-by-hand
    table of (given, family) pairs since hypothesis is unavailable.
    """

    # Plain ASCII single-token surnames: the invariant holds today.
    _ASCII = [
        ("John", "Smith"),
        ("Jane", "Doe"),
        ("Ashish", "Vaswani"),
        ("Yoshua", "Bengio"),
        ("Fei-Fei", "Li"),
        ("Kyunghyun", "Cho"),
        ("Quoc", "Le"),
        ("Geoffrey", "Hinton"),
    ]

    @pytest.mark.parametrize(("given", "family"), _ASCII)
    def test_reorder_surname_key_invariant_ascii(self, given, family):
        gf = last_name_from_person(f"{given} {family}")
        fg = last_name_from_person(f"{family}, {given}")
        assert gf == fg, f"surname key differs by name order: {gf!r} != {fg!r}"

    # Diacritic surnames: invariant also holds (diacritics are stripped both ways).
    _DIACRITIC = [
        ("Bernhard", "Schölkopf"),
        ("Klaus-Robert", "Müller"),
        ("Yann", "LeCun"),
        ("François", "Chollet"),
    ]

    @pytest.mark.parametrize(("given", "family"), _DIACRITIC)
    def test_reorder_surname_key_invariant_diacritic(self, given, family):
        gf = last_name_from_person(f"{given} {family}")
        fg = last_name_from_person(f"{family}, {given}")
        assert gf == fg, f"surname key differs by name order: {gf!r} != {fg!r}"

    # Particle surnames: both forms reduce to the final, most distinctive token.
    _PARTICLE = [
        ("Aaron", "van den Oord"),
        ("Peter", "von der Malsburg"),
        ("Maria", "de la Cruz"),
        ("Richard", "von Mises"),
    ]

    @pytest.mark.parametrize(("given", "family"), _PARTICLE)
    def test_reorder_surname_key_invariant_particle(self, given, family):
        gf = last_name_from_person(f"{given} {family}")
        fg = last_name_from_person(f"{family}, {given}")
        assert gf == fg, f"surname key differs by name order: {gf!r} != {fg!r}"


# ------------- Property 2: comparison symmetry --------------------------------


class TestAuthorComparisonSymmetry:
    """``combined_author_score`` must be symmetric in its two arguments: swapping
    entry-keys and record-keys cannot change the score.
    """

    _PAIRS = [
        (["smith", "doe"], ["smith", "doe"]),
        (["smith", "doe"], ["doe", "smith"]),
        (["vaswani", "shazeer", "parmar"], ["vaswani", "shazeer"]),
        (["scholkopf"], ["scholkopf"]),
        (["oord"], ["van den oord"]),
        (["sun"], ["sun 0020"]),
        ([], ["smith"]),
    ]

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_score_symmetric(self, a, b):
        assert combined_author_score(a, b, 0.5, 0.5) == pytest.approx(combined_author_score(b, a, 0.5, 0.5))


# ------------- Property 3: a true match survives entry-side transforms --------

# Base matching pair: entry author field <-> record family list, same people.
_BASE_ENTRY = "Ashish Vaswani and Noam Shazeer"
_BASE_FAMILIES = ["Vaswani", "Shazeer"]


def test_base_pair_matches():
    """Sanity: the untransformed base pair is a TRUE match."""
    assert _author_matches(_BASE_ENTRY, _BASE_FAMILIES)


class TestEntrySideTransformsInvariant:
    """Apply a citation-style transform to the ENTRY author field (record held
    fixed); a true match must stay a match.
    """

    # (id, transformed_entry_author, marks)
    _CASES = [
        # "Given Family" -> "Family, Given" reordering.
        ("reorder_comma", "Vaswani, Ashish and Shazeer, Noam", ()),
        # Add diacritics on the entry side only.
        ("add_diacritics", "Áshish Vaswani and Noam Shâzeer", ()),
        # Remove a middle name / initial (still same surnames).
        ("extra_initials", "Ashish K. Vaswani and Noam M. Shazeer", ()),
    ]

    @pytest.mark.parametrize(
        ("entry_author",),
        [pytest.param(c[1], marks=c[2], id=c[0]) for c in _CASES],
    )
    def test_entry_transform_preserves_match(self, entry_author):
        assert _author_matches(entry_author, _BASE_FAMILIES)


class TestRecordSideTransformsInvariant:
    """Apply a transform to the RECORD family list (entry held fixed); a true
    match must stay a match.
    """

    _CASES = [
        # Add diacritics on the record side only.
        ("add_diacritics", ["Váswani", "Shazéer"], ()),
        # DBLP-style homonym suffix on a record family name: stripped by the key.
        ("dblp_suffix_0020", ["Vaswani 0020", "Shazeer"], ()),
        ("dblp_suffix_0001", ["Vaswani", "Shazeer 0001"], ()),
    ]

    @pytest.mark.parametrize(
        ("families",),
        [pytest.param(c[1], marks=c[2], id=c[0]) for c in _CASES],
    )
    def test_record_transform_preserves_match(self, families):
        assert _author_matches(_BASE_ENTRY, families)


# ------------- Property 4: particle placement, both sides ---------------------


class TestParticlePlacementInvariant:
    """A nobiliary-particle author must match itself regardless of which side
    spells the particle, because both sides reduce the family to its final,
    most distinctive token through ``last_name_from_person``.
    """

    # Entry uses 'Given Family'; record keeps the full family. Both reduce to
    # the same final token, so they match.
    @pytest.mark.parametrize(
        ("entry_author", "families"),
        [
            ("Aaron van den Oord", ["van den Oord"]),
            ("Peter von der Malsburg", ["von der Malsburg"]),
            ("Maria de la Cruz", ["de la Cruz"]),
        ],
        ids=["van_den_oord", "von_der_malsburg", "de_la_cruz"],
    )
    def test_given_family_particle_matches_record(self, entry_author, families):
        assert _author_matches(entry_author, families)

    # Entry uses 'Family, Given' (full particle preserved); this DOES match
    # today and must keep matching.
    @pytest.mark.parametrize(
        ("entry_author", "families"),
        [
            ("van den Oord, Aaron", ["van den Oord"]),
            ("von der Malsburg, Peter", ["von der Malsburg"]),
            ("de la Cruz, Maria", ["de la Cruz"]),
        ],
        ids=["van_den_oord", "von_der_malsburg", "de_la_cruz"],
    )
    def test_family_given_particle_matches_record(self, entry_author, families):
        assert _author_matches(entry_author, families)


# ------------- Property 5: record-level canonical venue accessor --------------


class TestRecordCanonicalVenue:
    """``PublishedRecord.canonical_venue`` is the single record-side venue
    accessor; it must agree with ``get_canonical_venue`` and must NOT collapse
    distinct sibling journals onto a generic parent venue.
    """

    @pytest.mark.parametrize(
        ("journal", "expected"),
        [
            ("NeurIPS", "neurips"),
            ("Advances in Neural Information Processing Systems", "neurips"),
            ("Nature", "nature"),
        ],
    )
    def test_canonical_venue_matches_helper(self, journal, expected):
        from bibtex_updater.matching import get_canonical_venue

        rec = PublishedRecord(doi="x", journal=journal)
        assert rec.canonical_venue == get_canonical_venue(journal) == expected

    @pytest.mark.parametrize("journal", ["Nature Physics", "Science Robotics", "PNAS Nexus"])
    def test_sibling_journal_not_collapsed(self, journal):
        """A sibling journal must not canonicalize to its generic parent."""
        rec = PublishedRecord(doi="x", journal=journal)
        assert rec.canonical_venue != "nature"
        assert rec.canonical_venue != "science"
        assert rec.canonical_venue != "pnas"
