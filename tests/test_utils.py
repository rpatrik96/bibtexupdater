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
