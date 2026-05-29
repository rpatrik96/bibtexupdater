"""Tests for utility functions."""

from __future__ import annotations

from bibtex_updater import (
    authors_last_names,
    doi_normalize,
    doi_url,
    extract_arxiv_id_from_text,
    first_author_surname,
    jaccard_similarity,
    last_name_from_person,
    latex_to_plain,
    normalize_title_for_match,
    safe_lower,
    split_authors_bibtex,
    strip_diacritics,
)


class TestSafeLower:
    """Tests for safe_lower function."""

    def test_safe_lower_normal(self):
        assert safe_lower("HELLO") == "hello"

    def test_safe_lower_none(self):
        assert safe_lower(None) == ""

    def test_safe_lower_empty(self):
        assert safe_lower("") == ""

    def test_safe_lower_strips_whitespace(self):
        assert safe_lower("  HELLO  ") == "hello"


class TestStripDiacritics:
    """Tests for strip_diacritics function."""

    def test_strip_diacritics_umlaut(self):
        assert strip_diacritics("Müller") == "Muller"

    def test_strip_diacritics_accent(self):
        assert strip_diacritics("café") == "cafe"

    def test_strip_diacritics_mixed(self):
        assert strip_diacritics("Schrödinger") == "Schrodinger"

    def test_strip_diacritics_no_change(self):
        assert strip_diacritics("hello") == "hello"

    def test_strip_diacritics_nordic(self):
        result = strip_diacritics("Ångström")
        assert "A" in result and "ngstr" in result

    def test_strip_diacritics_eszett_folds_to_ss(self):
        # ß has no NFKD decomposition; fold it so "Reiß" matches "Reiss".
        assert strip_diacritics("Reiß").lower() == "reiss"
        assert strip_diacritics("Reiß").lower() == strip_diacritics("Reiss").lower()

    def test_strip_diacritics_nondecomposing_letters(self):
        # ø/ł/æ/đ etc. lack combining marks but should still fold to ASCII.
        assert strip_diacritics("Søndergaard").lower() == "sondergaard"
        assert strip_diacritics("Łukasz").lower() == "lukasz"


class TestLatexToPlain:
    """Tests for latex_to_plain function."""

    def test_latex_umlaut(self):
        result = latex_to_plain('Schr\\"odinger')
        assert "o" in result or "ö" in result

    def test_latex_braces_removed(self):
        result = latex_to_plain("{Deep Learning}")
        assert result == "Deep Learning"

    def test_latex_nested_braces(self):
        result = latex_to_plain("{{Nested}}")
        assert "Nested" in result

    def test_latex_backslash_commands(self):
        # latex_to_plain focuses on character escapes, not formatting commands
        # Complex LaTeX commands may result in partial or empty output
        result = latex_to_plain("Test \\& more")
        assert "&" in result or "Test" in result


class TestNormalizeTitleForMatch:
    """Tests for normalize_title_for_match function."""

    def test_normalize_basic(self):
        result = normalize_title_for_match("Deep Learning for NLP")
        assert result == "deep learning for nlp"

    def test_normalize_removes_punctuation(self):
        result = normalize_title_for_match("What's New? Everything!")
        assert "?" not in result
        assert "!" not in result
        assert "'" not in result

    def test_normalize_handles_diacritics(self):
        result = normalize_title_for_match("Schrödinger Equation")
        assert "schrodinger" in result or "schrdinger" in result

    def test_normalize_collapses_whitespace(self):
        result = normalize_title_for_match("Deep    Learning   Study")
        assert "  " not in result

    def test_normalize_empty(self):
        result = normalize_title_for_match("")
        assert result == ""


class TestSplitAuthorsBibtex:
    """Tests for split_authors_bibtex function."""

    def test_split_two_authors(self):
        result = split_authors_bibtex("Doe, Jane and Smith, John")
        assert len(result) == 2
        assert "Doe, Jane" in result
        assert "Smith, John" in result

    def test_split_single_author(self):
        result = split_authors_bibtex("Doe, Jane")
        assert len(result) == 1
        assert result[0] == "Doe, Jane"

    def test_split_three_authors(self):
        result = split_authors_bibtex("A, B and C, D and E, F")
        assert len(result) == 3

    def test_split_empty(self):
        result = split_authors_bibtex("")
        assert result == []

    def test_split_none(self):
        result = split_authors_bibtex(None)
        assert result == []

    def test_split_case_insensitive_and(self):
        result = split_authors_bibtex("Doe, Jane AND Smith, John")
        assert len(result) == 2


class TestLastNameFromPerson:
    """Tests for last_name_from_person function."""

    def test_last_name_comma_format(self):
        result = last_name_from_person("Doe, Jane")
        assert result == "doe"

    def test_last_name_natural_format(self):
        result = last_name_from_person("Jane Doe")
        assert result == "doe"

    def test_last_name_single_name(self):
        result = last_name_from_person("Madonna")
        assert result == "madonna"

    def test_last_name_with_suffix(self):
        result = last_name_from_person("Smith, John Jr.")
        assert "smith" in result.lower()

    def test_last_name_trailing_initials_skipped(self):
        # "Mallikarjun B. R." (surname first, then initials) -> "mallikarjun",
        # not the naive last token "r".
        assert last_name_from_person("Mallikarjun B. R.") == "mallikarjun"

    def test_last_name_keeps_real_surname_after_initial(self):
        # A middle initial must not cause the real trailing surname to be dropped.
        assert last_name_from_person("John M. Smith") == "smith"
        assert last_name_from_person("van den Oord, Aaron") == "oord"


class TestAuthorsLastNames:
    """Tests for authors_last_names function."""

    def test_authors_last_names_two(self):
        result = authors_last_names("Doe, Jane and Smith, John")
        assert result == ["doe", "smith"]

    def test_authors_last_names_limit(self):
        result = authors_last_names("A, B and C, D and E, F and G, H", limit=2)
        assert len(result) == 2

    def test_authors_last_names_default_limit(self):
        result = authors_last_names("A, B and C, D and E, F and G, H")
        assert len(result) == 3  # Default limit is 3

    def test_authors_last_names_empty(self):
        result = authors_last_names("")
        assert result == []


class TestJaccardSimilarity:
    """Tests for jaccard_similarity function."""

    def test_jaccard_identical(self):
        result = jaccard_similarity(["a", "b", "c"], ["a", "b", "c"])
        assert result == 1.0

    def test_jaccard_disjoint(self):
        result = jaccard_similarity(["a", "b"], ["c", "d"])
        assert result == 0.0

    def test_jaccard_partial(self):
        result = jaccard_similarity(["a", "b", "c"], ["a", "b", "d"])
        assert 0 < result < 1
        # Intersection = {a, b}, Union = {a, b, c, d}
        assert result == 2 / 4

    def test_jaccard_empty_both(self):
        result = jaccard_similarity([], [])
        assert result == 0.0

    def test_jaccard_one_empty(self):
        result = jaccard_similarity(["a"], [])
        assert result == 0.0


class TestDoiNormalize:
    """Tests for doi_normalize function."""

    def test_doi_normalize_basic(self):
        result = doi_normalize("10.1000/j.test.123")
        assert result == "10.1000/j.test.123"

    def test_doi_normalize_with_url(self):
        result = doi_normalize("https://doi.org/10.1000/j.test.123")
        assert result == "10.1000/j.test.123"

    def test_doi_normalize_with_dx_url(self):
        result = doi_normalize("http://dx.doi.org/10.1000/j.test.123")
        assert result == "10.1000/j.test.123"

    def test_doi_normalize_lowercase(self):
        result = doi_normalize("10.1000/J.TEST.123")
        assert result == "10.1000/j.test.123"

    def test_doi_normalize_none(self):
        result = doi_normalize(None)
        assert result is None

    def test_doi_normalize_empty(self):
        result = doi_normalize("")
        assert result is None


class TestNormalizeDoiForResolution:
    """FIX D: arXiv DataCite DOIs must be version-stripped, others left intact."""

    def test_strips_arxiv_version(self):
        from bibtex_updater import normalize_doi_for_resolution

        assert normalize_doi_for_resolution("10.48550/arXiv.2010.11929v1") == "10.48550/arxiv.2010.11929"

    def test_strips_arxiv_version_multidigit(self):
        from bibtex_updater import normalize_doi_for_resolution

        assert normalize_doi_for_resolution("10.48550/arXiv.2010.11929v12") == "10.48550/arxiv.2010.11929"

    def test_unversioned_arxiv_unchanged(self):
        from bibtex_updater import normalize_doi_for_resolution

        assert normalize_doi_for_resolution("10.48550/arXiv.2010.11929") == "10.48550/arxiv.2010.11929"

    def test_non_arxiv_version_like_suffix_preserved(self):
        from bibtex_updater import normalize_doi_for_resolution

        # Non-arXiv DOI legitimately ending in letter+digit -> must NOT strip.
        assert normalize_doi_for_resolution("10.1234/journal.v2") == "10.1234/journal.v2"

    def test_strips_url_prefix(self):
        from bibtex_updater import normalize_doi_for_resolution

        result = normalize_doi_for_resolution("https://doi.org/10.48550/arXiv.2010.11929v3")
        assert result == "10.48550/arxiv.2010.11929"

    def test_none_and_empty(self):
        from bibtex_updater import normalize_doi_for_resolution

        assert normalize_doi_for_resolution(None) is None
        assert normalize_doi_for_resolution("") is None


class TestDoiUrl:
    """Tests for doi_url function."""

    def test_doi_url_basic(self):
        result = doi_url("10.1000/j.test.123")
        assert result == "https://doi.org/10.1000/j.test.123"


class TestExtractArxivId:
    """Tests for extract_arxiv_id_from_text function."""

    def test_extract_new_format(self):
        result = extract_arxiv_id_from_text("arxiv:2001.01234")
        assert result == "2001.01234"

    def test_extract_from_url(self):
        result = extract_arxiv_id_from_text("https://arxiv.org/abs/2001.01234")
        assert result == "2001.01234"

    def test_extract_with_version(self):
        result = extract_arxiv_id_from_text("arxiv:2001.01234v3")
        assert "2001.01234" in result

    def test_extract_old_format(self):
        result = extract_arxiv_id_from_text("hep-th/9901001")
        assert result is not None

    def test_extract_none_found(self):
        result = extract_arxiv_id_from_text("no arxiv here")
        assert result is None

    def test_extract_empty(self):
        result = extract_arxiv_id_from_text("")
        assert result is None


class TestFirstAuthorSurname:
    """Tests for first_author_surname function."""

    def test_first_author_basic(self):
        entry = {"author": "Doe, Jane and Smith, John"}
        result = first_author_surname(entry)
        assert result == "doe"

    def test_first_author_single(self):
        entry = {"author": "Smith, John"}
        result = first_author_surname(entry)
        assert result == "smith"

    def test_first_author_missing(self):
        entry = {}
        result = first_author_surname(entry)
        assert result == ""

    def test_first_author_empty(self):
        entry = {"author": ""}
        result = first_author_surname(entry)
        assert result == ""


class TestMatcherThresholds:
    """Tests for title and author matching thresholds."""

    def test_combined_score_perfect_match(self):
        """Perfect match should score >= 0.9."""
        from rapidfuzz.fuzz import token_sort_ratio

        title_a = normalize_title_for_match("A Study of Widgets")
        title_b = normalize_title_for_match("A Study of Widgets")
        title_score = token_sort_ratio(title_a, title_b) / 100.0

        authors_a = authors_last_names("Jane Doe and John Smith")
        authors_b = ["doe", "smith"]
        auth_score = jaccard_similarity(authors_a, authors_b)

        combined = 0.7 * title_score + 0.3 * auth_score
        assert combined >= 0.9

    def test_combined_score_title_variation(self):
        """Slight title variation should still score high."""
        from rapidfuzz.fuzz import token_sort_ratio

        title_a = normalize_title_for_match("Deep Learning for Image Classification")
        title_b = normalize_title_for_match("Deep Learning for Image Classification Tasks")
        title_score = token_sort_ratio(title_a, title_b) / 100.0

        # Should still be reasonably high
        assert title_score >= 0.8


class TestAtomicReplace:
    """Tests for atomic_replace cross-device fallback."""

    def test_same_filesystem(self, tmp_path):
        """Standard case: src and dst on the same filesystem — atomic os.replace."""
        from bibtex_updater.utils import atomic_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")

        atomic_replace(str(src), str(dst))

        assert dst.read_text() == "hello"
        assert not src.exists()

    def test_exdev_fallback(self, tmp_path, monkeypatch):
        """When os.replace raises EXDEV, fall back to copy + unlink."""
        import errno
        import os

        from bibtex_updater.utils import atomic_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("payload")

        def fake_replace(s, d):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "replace", fake_replace)

        atomic_replace(str(src), str(dst))

        assert dst.read_text() == "payload"
        assert not src.exists()

    def test_non_exdev_oserror_propagates(self, tmp_path, monkeypatch):
        """OSErrors other than EXDEV should propagate, not silently fall back."""
        import errno
        import os

        import pytest

        from bibtex_updater.utils import atomic_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("payload")

        def fake_replace(s, d):
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(os, "replace", fake_replace)

        with pytest.raises(OSError) as excinfo:
            atomic_replace(str(src), str(dst))
        assert excinfo.value.errno == errno.EACCES
        # Source must remain untouched on failure.
        assert src.exists()
