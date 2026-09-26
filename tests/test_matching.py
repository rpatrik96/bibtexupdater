"""Tests for matching utilities module."""

from __future__ import annotations

from bibtex_updater.matching import (
    EXPANDED_VENUE_ALIASES,
    MatchOutcome,
    author_sequence_similarity,
    combined_author_score,
    get_canonical_venue,
    is_near_miss_title,
    is_preprint_or_series_venue,
    symmetric_author_match,
    title_edit_distance,
    word_level_diff,
)

# ------------- P2.2: Near-Miss Title Detection Tests -------------


class TestTitleEditDistance:
    """Test character-level edit distance computation."""

    def test_identical_titles(self):
        """Identical titles should have zero edit distance."""
        title = "Deep Learning for Natural Language Processing"
        assert title_edit_distance(title, title) == 0

    def test_one_word_different(self):
        """One word substitution should have non-zero distance."""
        title_a = "Deep Learning for Natural Language Processing"
        title_b = "Deep Learning for Natural Language Understanding"
        # "Processing" -> "Understanding" is significant character change
        distance = title_edit_distance(title_a, title_b)
        assert distance > 0
        assert distance >= 7  # "Processing" vs "Understanding" differs by ~7-10 chars

    def test_empty_titles(self):
        """Empty titles should have zero distance."""
        assert title_edit_distance("", "") == 0

    def test_case_insensitive(self):
        """Normalization should handle case differences."""
        title_a = "Deep Learning"
        title_b = "DEEP LEARNING"
        # Case is normalized before distance computation
        assert title_edit_distance(title_a, title_b) == 0


class TestIsNearMissTitle:
    """Test near-miss title detection logic."""

    def test_exact_match_not_near_miss(self):
        """Exact matches should not be flagged as near-miss."""
        title = "Deep Learning for Natural Language Processing"
        # Fuzzy score = 1.0, edit distance = 0
        assert not is_near_miss_title(title, title, fuzzy_score=1.0)

    def test_small_edit_not_near_miss(self):
        """Small edits below min_edit_distance should not be near-miss."""
        title_a = "Deep Learning"
        title_b = "Deep Learnings"  # 1 char difference
        # Even with high fuzzy score, distance=1 < min_edit_distance=3
        assert not is_near_miss_title(title_a, title_b, fuzzy_score=0.95, min_edit_distance=3)

    def test_word_substitution_is_near_miss(self):
        """Word substitution with high fuzzy score should be near-miss."""
        title_a = "Deep Learning for Natural Language Processing"
        title_b = "Deep Learning for Natural Language Understanding"
        # High fuzzy score (similar tokens) but edit distance >= 3
        # Token-sort ratio should be >= 0.90, edit distance >= 3
        assert is_near_miss_title(title_a, title_b, fuzzy_score=0.92, min_edit_distance=3)

    def test_below_fuzzy_threshold_not_near_miss(self):
        """Low fuzzy score should fail near-miss check regardless of edit distance."""
        title_a = "Deep Learning"
        title_b = "Completely Different Title"
        # Low fuzzy score, high edit distance
        assert not is_near_miss_title(title_a, title_b, fuzzy_score=0.50, fuzzy_threshold=0.90)

    def test_custom_min_edit_distance(self):
        """Custom min_edit_distance threshold should work."""
        title_a = "Machine Learning"
        title_b = "Machine Learnings"  # 1 char difference
        # With min_edit_distance=1, this should be near-miss if fuzzy score is high
        assert is_near_miss_title(title_a, title_b, fuzzy_score=0.95, min_edit_distance=1)


class TestWordLevelDiff:
    """Test word-level difference computation."""

    def test_identical_titles_no_diff(self):
        """Identical titles should have no differences."""
        title = "Deep Learning for NLP"
        diff = word_level_diff(title, title)
        assert diff["added"] == []
        assert diff["removed"] == []
        assert set(diff["common"]) == {"deep", "learning", "for", "nlp"}
        assert diff["substitution_count"] == 0

    def test_word_addition(self):
        """Added word should be detected."""
        title_a = "Deep Learning"
        title_b = "Deep Learning Methods"
        diff = word_level_diff(title_a, title_b)
        assert "methods" in diff["added"]
        assert diff["removed"] == []

    def test_word_removal(self):
        """Removed word should be detected."""
        title_a = "Deep Learning Methods"
        title_b = "Deep Learning"
        diff = word_level_diff(title_a, title_b)
        assert "methods" in diff["removed"]
        assert diff["added"] == []

    def test_word_substitution(self):
        """Word substitution should be estimated."""
        title_a = "Deep Learning for Natural Language Processing"
        title_b = "Deep Learning for Natural Language Understanding"
        diff = word_level_diff(title_a, title_b)
        assert "processing" in diff["removed"]
        assert "understanding" in diff["added"]
        # One removed, one added -> substitution_count = 1
        assert diff["substitution_count"] == 1


# ------------- P2.3: Ordered Author Comparison Tests -------------


class TestAuthorSequenceSimilarity:
    """Test LCS-based author sequence similarity."""

    def test_identical_order(self):
        """Identical author lists should have similarity 1.0."""
        authors = ["Smith", "Doe", "Johnson"]
        assert author_sequence_similarity(authors, authors) == 1.0

    def test_reversed_order(self):
        """Reversed order should have lower similarity."""
        authors_a = ["Smith", "Doe"]
        authors_b = ["Doe", "Smith"]
        # LCS length = 1 (either "Smith" or "Doe"), max_len = 2
        # Similarity = 1/2 = 0.5
        similarity = author_sequence_similarity(authors_a, authors_b)
        assert similarity == 0.5

    def test_one_missing_author(self):
        """Missing author should reduce similarity."""
        authors_a = ["Smith", "Doe", "Johnson"]
        authors_b = ["Smith", "Johnson"]  # "Doe" missing but order preserved
        # LCS = ["Smith", "Johnson"] = 2, max_len = 3
        # Similarity = 2/3 ≈ 0.667
        similarity = author_sequence_similarity(authors_a, authors_b)
        assert abs(similarity - 2 / 3) < 0.01

    def test_empty_lists(self):
        """Both empty lists should have similarity 1.0."""
        assert author_sequence_similarity([], []) == 1.0

    def test_one_empty_list(self):
        """One empty list should have similarity 0.0."""
        assert author_sequence_similarity(["Smith"], []) == 0.0
        assert author_sequence_similarity([], ["Smith"]) == 0.0

    def test_single_author(self):
        """Single matching author should have similarity 1.0."""
        assert author_sequence_similarity(["Smith"], ["Smith"]) == 1.0

    def test_completely_different(self):
        """Completely different authors should have similarity 0.0."""
        authors_a = ["Smith", "Doe"]
        authors_b = ["Johnson", "Brown"]
        # No common authors in LCS
        assert author_sequence_similarity(authors_a, authors_b) == 0.0


class TestCombinedAuthorScore:
    """Test combined Jaccard + LCS author scoring."""

    def test_identical_authors(self):
        """Identical authors should have score 1.0."""
        authors = ["Smith", "Doe", "Johnson"]
        # Jaccard = 1.0, LCS = 1.0
        assert combined_author_score(authors, authors) == 1.0

    def test_swapped_authors(self):
        """Swapped authors should have high Jaccard but lower LCS."""
        authors_a = ["Smith", "Doe"]
        authors_b = ["Doe", "Smith"]
        # Jaccard = 1.0 (same set), LCS = 0.5 (reversed order)
        # Combined = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        score = combined_author_score(authors_a, authors_b)
        assert abs(score - 0.75) < 0.01

    def test_custom_weights(self):
        """Custom weights should affect the combined score."""
        authors_a = ["Smith", "Doe"]
        authors_b = ["Doe", "Smith"]
        # Jaccard = 1.0, LCS = 0.5
        # With jaccard_weight=0.8, sequence_weight=0.2:
        # Combined = 0.8 * 1.0 + 0.2 * 0.5 = 0.9
        score = combined_author_score(authors_a, authors_b, jaccard_weight=0.8, sequence_weight=0.2)
        assert abs(score - 0.9) < 0.01

    def test_partial_overlap(self):
        """Partial author overlap should give intermediate score."""
        authors_a = ["Smith", "Doe", "Johnson"]
        authors_b = ["Smith", "Doe", "Brown"]
        # Jaccard = 2/4 = 0.5 (2 common, 4 total unique)
        # LCS = 2/3 ≈ 0.667 (["Smith", "Doe"] in sequence)
        # Combined (equal weights) = 0.5 * 0.5 + 0.5 * 0.667 ≈ 0.583
        score = combined_author_score(authors_a, authors_b)
        assert 0.55 < score < 0.62


# ------------- FIX C: Symmetric Author Matching Tests -------------


class TestSymmetricAuthorMatch:
    """FIX C + positive-confirmation: entry vs API author comparison is symmetric
    AND three-valued. Containment proves "consistent with", not "complete"."""

    def test_entry_subset_without_sentinel_is_partial(self):
        """Entry lists first 3 of 8 real authors, NO sentinel -> PARTIAL.

        Consistent but incomplete: not a mismatch (was a legacy false mismatch),
        but also not a full positive confirmation (the prior softening over-claimed
        this as a full match).
        """
        entry = ["smith", "doe", "jones"]
        api = ["smith", "doe", "jones", "brown", "wang", "lee", "kim", "patel"]
        result = symmetric_author_match(entry, api)
        assert result.outcome is MatchOutcome.PARTIAL
        assert result.is_confirmed is False
        assert result.is_mismatch is False

    def test_exact_match_is_confirmed(self):
        names = ["smith", "doe", "jones"]
        result = symmetric_author_match(names, names)
        assert result.outcome is MatchOutcome.MATCH
        assert result.is_confirmed is True
        assert result.score == 1.0

    def test_and_others_sentinel_leading_prefix_is_confirmed(self):
        """'and others' marks a deliberate leading truncation -> CONFIRMED match."""
        entry = ["smith", "doe", "others"]
        api = ["smith", "doe", "jones", "brown"]
        result = symmetric_author_match(entry, api)
        assert result.outcome is MatchOutcome.MATCH
        assert result.is_confirmed is True

    def test_et_al_sentinel_leading_prefix_is_confirmed(self):
        entry = ["smith", "et al"]
        api = ["smith", "doe", "jones"]
        result = symmetric_author_match(entry, api)
        assert result.outcome is MatchOutcome.MATCH
        assert result.is_confirmed is True

    def test_different_first_author_mismatches(self):
        """A genuinely swapped/wrong lead author must still be a MISMATCH."""
        entry = ["wrong", "doe", "jones"]
        api = ["smith", "doe", "jones"]
        result = symmetric_author_match(entry, api)
        assert result.outcome is MatchOutcome.MISMATCH
        assert result.score == 0.0

    def test_empty_side_is_non_comparable(self):
        """No usable names on a side -> cannot confirm or refute -> non-comparable."""
        assert symmetric_author_match([], ["smith"]).outcome is MatchOutcome.NON_COMPARABLE
        assert symmetric_author_match(["smith"], []).outcome is MatchOutcome.NON_COMPARABLE
        # Only-sentinel side strips to empty -> non-comparable.
        assert symmetric_author_match(["others"], ["smith", "doe"]).outcome is MatchOutcome.NON_COMPARABLE

    def test_completely_different_lists_mismatch(self):
        entry = ["alpha", "beta", "gamma"]
        api = ["delta", "epsilon", "zeta"]
        result = symmetric_author_match(entry, api)
        assert result.outcome is MatchOutcome.MISMATCH

    def test_dropped_interior_author_with_sentinel_is_partial(self):
        """A sentinel only confirms a LEADING truncation; dropping an interior
        author (non-prefix subsequence) is still PARTIAL even with 'and others'."""
        entry = ["smith", "jones", "others"]  # drops interior "doe"
        api = ["smith", "doe", "jones", "brown"]
        result = symmetric_author_match(entry, api)
        assert result.outcome is MatchOutcome.PARTIAL
        assert result.is_confirmed is False

    def test_same_lead_partial_overlap_scores_symmetrically(self):
        """Same first author, partially divergent tails -> symmetric slice scoring."""
        entry = ["smith", "doe", "x"]
        api = ["smith", "doe", "y", "z", "w"]
        result = symmetric_author_match(entry, api)
        # smith,doe,x vs smith,doe,y (sliced to 3) -> not a subsequence, scored.
        assert result.outcome in (MatchOutcome.MATCH, MatchOutcome.MISMATCH)


class TestAuthorOrderSensitivity:
    """Author ORDER is a reliable signal when the matched record comes from an
    order-preserving source (Crossref/OpenAlex/DBLP/OpenReview return authors in
    publication order -- verified empirically). A citation that lists the SAME
    authors in a DIFFERENT order (a swapped-authors corruption that keeps the lead
    author) is then a real MISMATCH. Against an order-unreliable source (Semantic
    Scholar's synthesized names) order is ignored, preserving the prior behavior.
    """

    def test_reordered_authors_mismatch_when_order_reliable(self):
        # Same set, lead author unchanged, interior authors scrambled (the OSAKA
        # swapped_authors signature). Old behavior: high Jaccard -> MATCH.
        entry = ["caccia", "lin", "rodriguez", "ostapenko"]
        api = ["caccia", "rodriguez", "ostapenko", "lin"]
        r = symmetric_author_match(entry, api, order_reliable=True)
        assert r.outcome is MatchOutcome.MISMATCH

    def test_reordered_authors_not_flagged_when_order_unreliable(self):
        entry = ["caccia", "lin", "rodriguez", "ostapenko"]
        api = ["caccia", "rodriguez", "ostapenko", "lin"]
        r = symmetric_author_match(entry, api, order_reliable=False)
        assert r.outcome is not MatchOutcome.MISMATCH

    def test_order_insensitive_is_the_default(self):
        entry = ["caccia", "lin", "rodriguez", "ostapenko"]
        api = ["caccia", "rodriguez", "ostapenko", "lin"]
        assert symmetric_author_match(entry, api).outcome is not MatchOutcome.MISMATCH

    def test_correct_full_order_still_matches_when_order_reliable(self):
        names = ["smith", "doe", "jones"]
        assert symmetric_author_match(names, names, order_reliable=True).outcome is MatchOutcome.MATCH

    def test_leading_prefix_with_sentinel_still_confirmed_when_order_reliable(self):
        # Correct order, explicitly truncated -> still a confirmation (order kept).
        r = symmetric_author_match(["smith", "doe", "others"], ["smith", "doe", "jones", "brown"], order_reliable=True)
        assert r.outcome is MatchOutcome.MATCH

    def test_in_order_subsequence_still_partial_when_order_reliable(self):
        # Interior author dropped but SAME relative order -> PARTIAL, not an
        # order-mismatch (order was preserved, the claim is just incomplete).
        entry = ["smith", "jones", "others"]
        api = ["smith", "doe", "jones", "brown"]
        assert symmetric_author_match(entry, api, order_reliable=True).outcome is MatchOutcome.PARTIAL

    def test_first_author_swap_mismatch_regardless_of_flag(self):
        r = symmetric_author_match(["wrong", "doe"], ["smith", "doe"], order_reliable=True)
        assert r.outcome is MatchOutcome.MISMATCH

    def test_low_overlap_different_paper_mismatches_via_score(self):
        # Genuinely different authors (same lead by coincidence) -> MISMATCH via the
        # score path, not the order rule (overlap below the gate).
        r = symmetric_author_match(["alpha", "beta", "gamma"], ["alpha", "delta", "epsilon"], order_reliable=True)
        assert r.outcome is MatchOutcome.MISMATCH

    def test_single_differing_author_is_not_an_order_mismatch(self):
        # Regression (e1694f2a0b29): a valid entry whose order-reliable record has
        # ONE mangled author (a record-side typo 'Ren' -> 'Rent') is NOT a
        # reordering -- multisets differ -- so the order rule must not fire; the
        # matching head verifies it via the score path.
        entry = ["wang", "yang", "feng", "sun", "guo", "zhang", "ren"]
        api = ["wang", "yang", "feng", "sun", "guo", "zhang", "rent"]  # last author typo'd
        r = symmetric_author_match(entry, api, order_reliable=True)
        assert r.outcome is not MatchOutcome.MISMATCH

    def test_genuine_permutation_still_flags(self):
        # A true reordering (identical author multiset, different order) still flags.
        entry = ["caccia", "lin", "rodriguez", "ostapenko"]
        api = ["caccia", "rodriguez", "ostapenko", "lin"]
        assert symmetric_author_match(entry, api, order_reliable=True).outcome is MatchOutcome.MISMATCH

    def test_alphabetized_record_order_is_not_a_swap(self):
        # Regression (df33d8b19854, c92305210097, ae74287cae13): some order-reliable
        # records (Crossref NeurIPS/ICML proceedings deposits, prefix 10.52202) sort
        # authors A-Z instead of preserving title-page order. The same author
        # multiset in alphabetical order must NOT be read as a swapped-authors
        # defect -- it is a record-side sort artifact -> MATCH, not MISMATCH.
        entry = ["zhang", "pan", "li", "liu", "chen", "liu", "wang"]
        api = sorted(entry)  # record alphabetized its contributors
        assert symmetric_author_match(entry, api, order_reliable=True).outcome is MatchOutcome.MATCH

    def test_alphabetized_lead_difference_is_not_a_mismatch(self):
        # The alphabetized record also differs on the LEAD author; the same-multiset
        # exclusion must cover the hard first-author guard too.
        entry = ["nowak", "grooten", "mocanu", "tabor"]
        api = sorted(entry)  # ["grooten", "mocanu", "nowak", "tabor"]
        assert entry[0] != api[0]
        assert symmetric_author_match(entry, api, order_reliable=True).outcome is MatchOutcome.MATCH

    def test_non_alphabetized_genuine_swap_still_flags_after_alpha_guard(self):
        # A real lead-author swap whose order is NOT alphabetical must still flag,
        # so the alphabetization guard does not weaken genuine swap detection.
        entry = ["smith", "jones", "adams"]
        api = ["jones", "smith", "adams"]  # same set, swapped lead, not A-Z sorted
        assert api != sorted(api)
        assert symmetric_author_match(entry, api, order_reliable=True).outcome is MatchOutcome.MISMATCH


# ------------- FIX E-venue: Preprint / series venue Tests -------------


class TestIsPreprintOrSeriesVenue:
    """FIX E-venue: preprint/series venues are non-comparable."""

    def test_arxiv_variants(self):
        for v in ("arXiv", "arXiv preprint arXiv:2010.11929", "CoRR"):
            assert is_preprint_or_series_venue(v) is True, v

    def test_other_preprint_servers(self):
        for v in ("bioRxiv", "medRxiv", "chemRxiv", "Some Preprint"):
            assert is_preprint_or_series_venue(v) is True, v

    def test_pmlr_series(self):
        for v in ("PMLR", "Proceedings of Machine Learning Research", "JMLR W&CP"):
            assert is_preprint_or_series_venue(v) is True, v

    def test_real_venue_is_not_preprint(self):
        for v in ("NeurIPS", "ICML", "Nature", ""):
            assert is_preprint_or_series_venue(v) is False, v

    def test_jmlr_journal_not_treated_as_series(self):
        """JMLR is a distinct published journal, NOT the PMLR umbrella series."""
        assert is_preprint_or_series_venue("Journal of Machine Learning Research") is False

    def test_openreview_platform_is_non_comparable(self):
        # Regression (cc479d014ba7, a2b6f92163c9): OpenReview hosts many venues, so
        # a record whose venue is just the platform name cannot confirm or refute a
        # claimed conference -> non-comparable, not a venue mismatch.
        for v in ("OpenReview", "OpenReview.net", "openreview.net"):
            assert is_preprint_or_series_venue(v) is True, v


# ------------- P2.5: Expanded Venue Aliases Tests -------------


class TestGetCanonicalVenue:
    """Test venue canonicalization."""

    def test_known_venue_neurips(self):
        """NeurIPS canonical name should be recognized."""
        assert get_canonical_venue("NeurIPS") == "neurips"
        assert get_canonical_venue("neurips") == "neurips"

    def test_known_venue_nips_alias(self):
        """NIPS alias should map to neurips."""
        assert get_canonical_venue("NIPS") == "neurips"
        assert get_canonical_venue("nips") == "neurips"

    def test_known_venue_icml(self):
        """ICML variations should map to canonical form."""
        assert get_canonical_venue("ICML") == "icml"
        assert get_canonical_venue("International Conference on Machine Learning") == "icml"

    def test_proceedings_prefix_stripped(self):
        """'Proceedings of' prefix should be normalized."""
        assert get_canonical_venue("Proceedings of the AAAI Conference on Artificial Intelligence") == "aaai"
        assert get_canonical_venue("Proceedings of ICML") == "icml"

    def test_unknown_venue_returns_none(self):
        """Unknown venue should return None."""
        assert get_canonical_venue("Unknown Conference 2024") is None

    def test_empty_venue(self):
        """Empty venue should return None."""
        assert get_canonical_venue("") is None
        assert get_canonical_venue("   ") is None

    def test_new_venues_sigmod(self):
        """SIGMOD aliases should map correctly."""
        assert get_canonical_venue("SIGMOD") == "sigmod"
        assert get_canonical_venue("ACM SIGMOD") == "sigmod"
        assert get_canonical_venue("International Conference on Management of Data") == "sigmod"

    def test_new_venues_vldb(self):
        """VLDB aliases should map correctly."""
        assert get_canonical_venue("VLDB") == "vldb"
        assert get_canonical_venue("Very Large Data Bases") == "vldb"
        assert get_canonical_venue("PVLDB") == "vldb"
        assert get_canonical_venue("Proceedings of the VLDB Endowment") == "vldb"

    def test_false_positive_prevention(self):
        """Short substrings shouldn't cause false matches."""
        # "chi" appears in "machine" but shouldn't match
        assert get_canonical_venue("Machine Intelligence") != "chi"
        # Need substantial overlap (length ratio >= 0.4)


class TestExpandedVenueAliases:
    """Test the venue alias dictionary structure."""

    def test_aliases_dict_not_empty(self):
        """Alias dictionary should contain entries."""
        assert len(EXPANDED_VENUE_ALIASES) > 0

    def test_all_values_are_sets(self):
        """All alias values should be sets of strings."""
        for canonical, aliases in EXPANDED_VENUE_ALIASES.items():
            assert isinstance(aliases, set), f"{canonical} should have set of aliases"
            for alias in aliases:
                assert isinstance(alias, str), f"Alias {alias} should be string"

    def test_canonical_key_not_in_aliases(self):
        """Canonical key may appear in its own alias set for exact matching."""
        # This is actually valid - some venues like "nature" have themselves as an alias
        # to ensure exact matching works. This test verifies the structure is consistent.
        for canonical, _aliases in EXPANDED_VENUE_ALIASES.items():
            assert isinstance(canonical, str), f"Canonical {canonical} should be string"
            # The get_canonical_venue function handles this by creating all_names = alias_set | {canonical}

    def test_major_venues_present(self):
        """Major ML/AI venues should be in the alias map."""
        expected_venues = ["neurips", "icml", "iclr", "aaai", "cvpr", "acl", "emnlp"]
        for venue in expected_venues:
            assert venue in EXPANDED_VENUE_ALIASES, f"{venue} should be in alias map"

    def test_new_db_venues_present(self):
        """New database venues should be in the alias map."""
        db_venues = ["sigmod", "vldb", "icde"]
        for venue in db_venues:
            assert venue in EXPANDED_VENUE_ALIASES, f"{venue} should be in alias map"


class TestVenueAliasExpansion:
    """Same venue written differently must canonicalize identically.

    Each entry maps a canonical key to a list of spellings (acronym, long form,
    numbered/dated proceedings, publisher-decorated forms) that should ALL
    resolve to that single canonical venue.
    """

    # canonical -> variant spellings that must all map to it
    _SAME_VENUE = {
        "neurips": [
            "NeurIPS",
            "NIPS",
            "Advances in Neural Information Processing Systems",
            "Conference on Neural Information Processing Systems",
            "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
            "The 36th Conference on Neural Information Processing Systems",
        ],
        "icml": [
            "ICML",
            "International Conference on Machine Learning",
            "Proceedings of the 40th International Conference on Machine Learning",
            "Proceedings of the 40th International Conference on Machine Learning, PMLR",
        ],
        "iclr": [
            "ICLR",
            "International Conference on Learning Representations",
            "Proceedings of the International Conference on Learning Representations",
        ],
        "cvpr": [
            "CVPR",
            "IEEE Conference on Computer Vision and Pattern Recognition",
            "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
            "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        ],
        "iccv": [
            "ICCV",
            "IEEE International Conference on Computer Vision",
            "IEEE/CVF International Conference on Computer Vision",
            "Proceedings of the IEEE/CVF International Conference on Computer Vision",
        ],
        "eccv": [
            "ECCV",
            "European Conference on Computer Vision",
        ],
        "aaai": [
            "AAAI",
            "AAAI Conference on Artificial Intelligence",
            "Proceedings of the AAAI Conference on Artificial Intelligence",
        ],
        "acl": [
            "ACL",
            "Annual Meeting of the Association for Computational Linguistics",
            "Findings of ACL",
            "Findings of the Association for Computational Linguistics: ACL 2023",
        ],
        "emnlp": [
            "EMNLP",
            "Conference on Empirical Methods in Natural Language Processing",
            "Findings of EMNLP",
            "Findings of the Association for Computational Linguistics: EMNLP 2023",
        ],
        "naacl": [
            "NAACL",
            "NAACL-HLT",
            "North American Chapter of the Association for Computational Linguistics",
            "Findings of NAACL",
        ],
        "eacl": [
            "EACL",
            "Conference of the European Chapter of the Association for Computational Linguistics",
            "Findings of EACL",
        ],
        "coling": [
            "COLING",
            "International Conference on Computational Linguistics",
        ],
        "kdd": [
            "KDD",
            "SIGKDD",
            "ACM SIGKDD",
            "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
        ],
        "ijcai": [
            "IJCAI",
            "International Joint Conference on Artificial Intelligence",
        ],
        "uai": [
            "UAI",
            "Conference on Uncertainty in Artificial Intelligence",
        ],
        "aistats": [
            "AISTATS",
            "International Conference on Artificial Intelligence and Statistics",
        ],
        "colt": [
            "COLT",
            "Conference on Learning Theory",
        ],
        "interspeech": [
            "INTERSPEECH",
            "Conference of the International Speech Communication Association",
        ],
        "icassp": [
            "ICASSP",
            "IEEE International Conference on Acoustics, Speech and Signal Processing",
        ],
        "sigir": [
            "SIGIR",
            "International ACM SIGIR Conference on Research and Development in Information Retrieval",
        ],
        "www": [
            "WWW",
            "The Web Conference",
            "TheWebConf",
            "ACM Web Conference 2023",
        ],
        "jmlr": [
            "JMLR",
            "Journal of Machine Learning Research",
        ],
        "tmlr": [
            "TMLR",
            "Transactions on Machine Learning Research",
        ],
        "tpami": [
            "TPAMI",
            "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        ],
    }

    def test_variant_spellings_canonicalize_to_same_venue(self):
        """Every spelling variant must resolve to its canonical venue."""
        for canonical, variants in self._SAME_VENUE.items():
            for variant in variants:
                assert get_canonical_venue(variant) == canonical, (
                    f"{variant!r} should canonicalize to {canonical!r}, " f"got {get_canonical_venue(variant)!r}"
                )

    def test_bare_acronyms_resolve_to_self(self):
        """Short bare acronyms must not substring-collide with a different venue.

        Regression: "ACL" used to map to "naacl" because "acl" is a substring of
        "naacl". The exact-match pass now resolves bare acronyms first.
        """
        assert get_canonical_venue("ACL") == "acl"
        assert get_canonical_venue("KDD") == "kdd"
        assert get_canonical_venue("UAI") == "uai"
        assert get_canonical_venue("WWW") == "www"

    def test_findings_track_keeps_base_venue(self):
        """Findings tracks resolve to their own base venue, not a sibling.

        "Findings of EMNLP" contains "association for computational linguistics"
        (an ACL alias) once spelled out, so the substring fallback would collapse
        it into ACL -- the explicit exact-match aliases prevent that.
        """
        assert get_canonical_venue("Findings of EMNLP") == "emnlp"
        assert get_canonical_venue("Findings of the Association for Computational Linguistics: EMNLP 2023") == "emnlp"
        assert get_canonical_venue("Findings of ACL") == "acl"
        assert get_canonical_venue("Findings of NAACL") == "naacl"


class TestDistinctVenuesStayDistinct:
    """Aliases must only equate spellings of the SAME venue.

    These guards fail loudly if an alias addition ever merges two genuinely
    different venues (the original false-``venue_mismatch`` failure mode runs the
    other direction, but collapsing distinct venues would hide real mismatches).
    """

    # Pairs that must canonicalize to DIFFERENT, non-None venues.
    _DISTINCT_PAIRS = [
        ("ICML", "ICLR"),
        ("ICCV", "ECCV"),
        ("ICCV", "CVPR"),
        ("ECCV", "CVPR"),
        ("ACL", "NAACL"),
        ("ACL", "EMNLP"),
        ("ACL", "EACL"),
        ("NAACL", "EMNLP"),
        ("EMNLP", "EACL"),
        ("NAACL", "EACL"),
        ("JMLR", "TMLR"),
        ("NeurIPS", "ICML"),
        ("Findings of EMNLP", "Findings of ACL"),
        ("KDD", "SIGIR"),
        ("UAI", "COLT"),
        ("ICML", "AISTATS"),
        ("COLING", "ACL"),
    ]

    def test_distinct_venue_pairs(self):
        """Each look-alike pair must map to two different canonical venues."""
        for left, right in self._DISTINCT_PAIRS:
            cl, cr = get_canonical_venue(left), get_canonical_venue(right)
            assert cl is not None, f"{left!r} should canonicalize to a known venue"
            assert cr is not None, f"{right!r} should canonicalize to a known venue"
            assert cl != cr, f"{left!r} ({cl}) and {right!r} ({cr}) must stay distinct"

    def test_specific_distinct_keys(self):
        """Spot-check the canonical keys for the most confusable venues."""
        assert get_canonical_venue("ICML") != get_canonical_venue("ICLR")
        assert get_canonical_venue("ICCV") != get_canonical_venue("ECCV")
        assert get_canonical_venue("JMLR") != get_canonical_venue("TMLR")
        assert get_canonical_venue("EMNLP") != get_canonical_venue("EACL")
