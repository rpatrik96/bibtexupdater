"""Tests for ACL Anthology integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bibtex_updater.updater import PreprintDetection, Resolver
from bibtex_updater.utils import (
    PublishedRecord,
    acl_anthology_bib_to_record,
    extract_acl_anthology_id,
    normalize_title_for_match,
)

# ------------- extract_acl_anthology_id tests -------------


class TestExtractAclAnthologyId:
    """Tests for extract_acl_anthology_id utility."""

    def test_acl_doi(self):
        """Should extract anthology ID from an ACL DOI."""
        assert extract_acl_anthology_id("10.18653/v1/2022.acl-long.220") == "2022.acl-long.220"

    def test_acl_doi_with_url_prefix(self):
        """Should handle DOI with https://doi.org/ prefix."""
        assert extract_acl_anthology_id("https://doi.org/10.18653/v1/2022.emnlp-main.100") == "2022.emnlp-main.100"

    def test_acl_url(self):
        """Should extract anthology ID from an aclanthology.org URL."""
        assert extract_acl_anthology_id("https://aclanthology.org/2022.acl-long.220") == "2022.acl-long.220"

    def test_acl_url_with_trailing_slash(self):
        """Should handle trailing slash in URL."""
        assert extract_acl_anthology_id("https://aclanthology.org/2022.acl-long.220/") == "2022.acl-long.220"

    def test_acl_url_pdf(self):
        """Should extract ID from PDF URL."""
        assert extract_acl_anthology_id("https://aclanthology.org/2022.acl-long.220.pdf") == "2022.acl-long.220"

    def test_non_acl_doi_returns_none(self):
        """Should return None for non-ACL DOIs."""
        assert extract_acl_anthology_id("10.1000/j.journal.2021.001") is None

    def test_non_acl_url_returns_none(self):
        """Should return None for non-ACL URLs."""
        assert extract_acl_anthology_id("https://arxiv.org/abs/2301.00001") is None

    def test_empty_string_returns_none(self):
        """Should return None for empty string."""
        assert extract_acl_anthology_id("") is None

    def test_none_returns_none(self):
        """Should return None for None input."""
        assert extract_acl_anthology_id(None) is None

    def test_old_style_acl_id(self):
        """Should handle older ACL Anthology IDs like P19-1423."""
        assert extract_acl_anthology_id("https://aclanthology.org/P19-1423") == "P19-1423"

    def test_findings_doi(self):
        """Should handle Findings paper DOIs."""
        assert extract_acl_anthology_id("10.18653/v1/2021.findings-acl.42") == "2021.findings-acl.42"

    def test_workshop_doi(self):
        """Should handle workshop paper DOIs."""
        assert extract_acl_anthology_id("10.18653/v1/2023.eacl-srw.5") == "2023.eacl-srw.5"


# ------------- acl_anthology_bib_to_record tests -------------

SAMPLE_ACL_BIB = """\
@inproceedings{kitaev-etal-2022-learned,
    title = "Learned Incremental Representations for Parsing",
    author = "Kitaev, Nikita  and
      Lu, Thomas  and
      Klein, Dan",
    booktitle = "Proceedings of the 60th Annual Meeting of the ACL (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.220",
    doi = "10.18653/v1/2022.acl-long.220",
    pages = "3086--3095",
}
"""

SAMPLE_ACL_BIB_ARTICLE = """\
@article{jones-2023-survey,
    title = "A Survey of Computational Linguistics",
    author = "Jones, Alice",
    journal = "Computational Linguistics",
    volume = "49",
    number = "2",
    year = "2023",
    doi = "10.1162/coli_a_00001",
    pages = "1--50",
    publisher = "MIT Press",
}
"""


class TestAclAnthologyBibToRecord:
    """Tests for acl_anthology_bib_to_record converter."""

    def test_basic_conversion(self):
        """Should convert a standard ACL BibTeX entry to PublishedRecord."""
        rec = acl_anthology_bib_to_record(SAMPLE_ACL_BIB)
        assert rec is not None
        assert rec.title == "Learned Incremental Representations for Parsing"
        assert rec.doi == "10.18653/v1/2022.acl-long.220"
        assert rec.year == 2022
        assert rec.pages == "3086--3095"
        assert rec.type == "proceedings-article"
        assert rec.publisher == "Association for Computational Linguistics"
        assert rec.url == "https://aclanthology.org/2022.acl-long.220"

    def test_authors_parsed(self):
        """Should correctly parse multiple authors."""
        rec = acl_anthology_bib_to_record(SAMPLE_ACL_BIB)
        assert rec is not None
        assert len(rec.authors) == 3
        assert rec.authors[0] == {"given": "Nikita", "family": "Kitaev"}
        assert rec.authors[1] == {"given": "Thomas", "family": "Lu"}
        assert rec.authors[2] == {"given": "Dan", "family": "Klein"}

    def test_venue_from_booktitle(self):
        """Should use booktitle as venue for inproceedings."""
        rec = acl_anthology_bib_to_record(SAMPLE_ACL_BIB)
        assert rec is not None
        assert "ACL" in rec.journal

    def test_article_type(self):
        """Should handle @article entries correctly."""
        rec = acl_anthology_bib_to_record(SAMPLE_ACL_BIB_ARTICLE)
        assert rec is not None
        assert rec.type == "journal-article"
        assert rec.journal == "Computational Linguistics"
        assert rec.volume == "49"
        assert rec.number == "2"

    def test_empty_input_returns_none(self):
        """Should return None for empty input."""
        assert acl_anthology_bib_to_record("") is None
        assert acl_anthology_bib_to_record(None) is None

    def test_malformed_bib_returns_none(self):
        """Should return None for BibTeX without a title."""
        assert acl_anthology_bib_to_record("@misc{key, year={2022}}") is None

    def test_record_has_no_method_or_confidence(self):
        """Converter should not set method or confidence (caller does that)."""
        rec = acl_anthology_bib_to_record(SAMPLE_ACL_BIB)
        assert rec is not None
        assert rec.method is None
        assert rec.confidence == 0.0


# ------------- Resolver stage tests -------------


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


class TestStage3bAclAnthology:
    """Tests for Resolver._stage3b_acl_anthology."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        """Create a Resolver with a fake HTTP client."""
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    @pytest.fixture
    def acl_entry(self):
        """An entry with an ACL Anthology DOI."""
        return {
            "ID": "acl2022",
            "ENTRYTYPE": "article",
            "title": "Learned Incremental Representations for Parsing",
            "author": "Kitaev, Nikita and Lu, Thomas and Klein, Dan",
            "journal": "arXiv preprint arXiv:2201.12345",
            "year": "2022",
            "doi": "10.18653/v1/2022.acl-long.220",
        }

    @pytest.fixture
    def acl_url_entry(self):
        """An entry with an ACL Anthology URL."""
        return {
            "ID": "acl2022url",
            "ENTRYTYPE": "article",
            "title": "Some NLP Paper",
            "author": "Smith, John",
            "year": "2022",
            "url": "https://aclanthology.org/2022.acl-long.220",
        }

    @pytest.fixture
    def non_acl_entry(self):
        """An entry with no ACL Anthology indicators."""
        return {
            "ID": "nonacl2022",
            "ENTRYTYPE": "article",
            "title": "Generic ML Paper",
            "author": "Doe, Jane",
            "year": "2022",
            "doi": "10.1000/j.journal.2022.001",
        }

    def test_stage3b_with_acl_doi(self, resolver, acl_entry):
        """Stage 3b should find via ACL DOI when HTTP returns valid BibTeX."""
        # Mock the HTTP request to return sample BibTeX
        resolver.http._request = MagicMock(return_value=MockResponse(200, SAMPLE_ACL_BIB))

        title_norm = normalize_title_for_match(acl_entry["title"])
        result = resolver._stage3b_acl_anthology(acl_entry, title_norm, candidate_doi=None)

        assert result is not None
        assert result.method == "ACLAnthology(doi)"
        assert result.confidence == 1.0
        assert result.doi == "10.18653/v1/2022.acl-long.220"

    def test_stage3b_with_acl_url(self, resolver, acl_url_entry):
        """Stage 3b should find via ACL URL."""
        resolver.http._request = MagicMock(return_value=MockResponse(200, SAMPLE_ACL_BIB))

        title_norm = normalize_title_for_match(acl_url_entry["title"])
        result = resolver._stage3b_acl_anthology(acl_url_entry, title_norm, candidate_doi=None)

        assert result is not None
        assert result.method == "ACLAnthology(url)"
        assert result.confidence == 1.0

    def test_stage3b_with_candidate_doi(self, resolver, non_acl_entry):
        """Stage 3b should check candidate_doi from earlier stages."""
        resolver.http._request = MagicMock(return_value=MockResponse(200, SAMPLE_ACL_BIB))

        title_norm = normalize_title_for_match(non_acl_entry["title"])
        result = resolver._stage3b_acl_anthology(
            non_acl_entry, title_norm, candidate_doi="10.18653/v1/2022.acl-long.220"
        )

        assert result is not None
        assert result.method == "ACLAnthology(doi)"

    def test_stage3b_no_acl_indicators(self, resolver, non_acl_entry):
        """Stage 3b should return None when entry has no ACL indicators."""
        title_norm = normalize_title_for_match(non_acl_entry["title"])
        result = resolver._stage3b_acl_anthology(non_acl_entry, title_norm, candidate_doi=None)
        assert result is None

    def test_stage3b_http_failure(self, resolver, acl_entry):
        """Stage 3b should return None on HTTP failure."""
        resolver.http._request = MagicMock(return_value=MockResponse(404, "Not Found"))

        title_norm = normalize_title_for_match(acl_entry["title"])
        result = resolver._stage3b_acl_anthology(acl_entry, title_norm, candidate_doi=None)
        assert result is None

    def test_stage3b_http_exception(self, resolver, acl_entry):
        """Stage 3b should return None on HTTP exception."""
        resolver.http._request = MagicMock(side_effect=Exception("Network error"))

        title_norm = normalize_title_for_match(acl_entry["title"])
        result = resolver._stage3b_acl_anthology(acl_entry, title_norm, candidate_doi=None)
        assert result is None


# ------------- Credibility tests for ACL venues -------------


class TestAclVenueCredibility:
    """Tests that ACL venue patterns pass _credible_journal_article."""

    @pytest.fixture
    def acl_proceedings_record(self):
        """A record from ACL proceedings."""
        return PublishedRecord(
            doi="10.18653/v1/2022.acl-long.220",
            url="https://aclanthology.org/2022.acl-long.220",
            title="Test Paper",
            authors=[{"given": "John", "family": "Smith"}],
            journal="Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
            year=2022,
            pages="1-10",
            type="proceedings-article",
        )

    @pytest.fixture
    def emnlp_record(self):
        """A record from EMNLP proceedings."""
        return PublishedRecord(
            doi="10.18653/v1/2022.emnlp-main.100",
            url="https://aclanthology.org/2022.emnlp-main.100",
            title="Test EMNLP Paper",
            authors=[{"given": "Jane", "family": "Doe"}],
            journal="Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
            year=2022,
            pages="1-10",
            type="proceedings-article",
        )

    @pytest.fixture
    def findings_record(self):
        """A record from Findings of ACL."""
        return PublishedRecord(
            doi="10.18653/v1/2021.findings-acl.42",
            url="https://aclanthology.org/2021.findings-acl.42",
            title="Test Findings Paper",
            authors=[{"given": "Alice", "family": "Jones"}],
            journal="Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
            year=2021,
            pages="500-510",
            type="proceedings-article",
        )

    @pytest.fixture
    def naacl_record(self):
        """A record from NAACL proceedings."""
        return PublishedRecord(
            doi="10.18653/v1/2022.naacl-main.50",
            url="https://aclanthology.org/2022.naacl-main.50",
            title="Test NAACL Paper",
            authors=[{"given": "Bob", "family": "Brown"}],
            journal=(
                "Proceedings of the 2022 Conference of the North American Chapter"
                " of the Association for Computational Linguistics"
            ),
            year=2022,
            pages="100-110",
            type="proceedings-article",
        )

    def test_acl_proceedings_is_credible(self, acl_proceedings_record):
        assert Resolver._credible_journal_article(acl_proceedings_record)

    def test_emnlp_is_credible(self, emnlp_record):
        assert Resolver._credible_journal_article(emnlp_record)

    def test_findings_is_credible(self, findings_record):
        assert Resolver._credible_journal_article(findings_record)

    def test_naacl_is_credible(self, naacl_record):
        assert Resolver._credible_journal_article(naacl_record)

    def test_acl_record_type_is_proceedings(self):
        """ACL Anthology records should typically be proceedings-article."""
        rec = acl_anthology_bib_to_record(SAMPLE_ACL_BIB)
        assert rec is not None
        assert rec.type == "proceedings-article"
        assert rec.type in Resolver.CREDIBLE_TYPES


# ------------- Real ACL Anthology paper tests -------------

# Real BibTeX from aclanthology.org for well-known papers

BERT_BIB = r"""@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the ACL",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1423/",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186"
}
"""

AUTOPROMPT_BIB = r"""@inproceedings{shin-etal-2020-autoprompt,
    title = "{A}uto{P}rompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts",
    author = "Shin, Taylor  and
      Razeghi, Yasaman  and
      Logan IV, Robert L.  and
      Wallace, Eric  and
      Singh, Sameer",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.346/",
    doi = "10.18653/v1/2020.emnlp-main.346",
    pages = "4222--4235"
}
"""

XLMR_BIB = r"""@inproceedings{conneau-etal-2020-unsupervised,
    title = "Unsupervised Cross-lingual Representation Learning at Scale",
    author = "Conneau, Alexis  and
      Khandelwal, Kartikay  and
      Goyal, Naman  and
      Chaudhary, Vishrav  and
      Wenzek, Guillaume  and
      Guzm{\'a}n, Francisco  and
      Grave, Edouard  and
      Ott, Myle  and
      Zettlemoyer, Luke  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.747/",
    doi = "10.18653/v1/2020.acl-main.747",
    pages = "8440--8451"
}
"""


class TestRealAclPapers:
    """Tests using real BibTeX from actual ACL Anthology papers."""

    def test_bert_naacl2019(self):
        """BERT (NAACL 2019) should parse correctly."""
        rec = acl_anthology_bib_to_record(BERT_BIB)
        assert rec is not None
        assert "BERT" in rec.title
        assert "Pre-training of Deep Bidirectional Transformers" in rec.title
        assert rec.doi == "10.18653/v1/n19-1423"
        assert rec.year == 2019
        assert rec.pages == "4171--4186"
        assert rec.type == "proceedings-article"
        assert rec.publisher == "Association for Computational Linguistics"
        assert len(rec.authors) == 4
        assert rec.authors[0]["family"] == "Devlin"
        assert rec.authors[0]["given"] == "Jacob"
        assert rec.authors[3]["family"] == "Toutanova"

    def test_bert_venue_is_naacl(self):
        """BERT venue should reference NAACL."""
        rec = acl_anthology_bib_to_record(BERT_BIB)
        assert rec is not None
        assert "North" in rec.journal
        assert Resolver._credible_journal_article(rec)

    def test_bert_doi_extraction(self):
        """Should extract anthology ID from BERT's DOI."""
        assert extract_acl_anthology_id("10.18653/v1/N19-1423") == "N19-1423"

    def test_autoprompt_emnlp2020(self):
        """AutoPrompt (EMNLP 2020) should parse correctly."""
        rec = acl_anthology_bib_to_record(AUTOPROMPT_BIB)
        assert rec is not None
        assert "AutoPrompt" in rec.title or "auto" in rec.title.lower()
        assert rec.doi == "10.18653/v1/2020.emnlp-main.346"
        assert rec.year == 2020
        assert rec.pages == "4222--4235"
        assert rec.type == "proceedings-article"
        assert len(rec.authors) == 5
        assert rec.authors[0]["family"] == "Shin"
        assert rec.authors[4]["family"] == "Singh"

    def test_autoprompt_venue_is_emnlp(self):
        """AutoPrompt venue should reference EMNLP."""
        rec = acl_anthology_bib_to_record(AUTOPROMPT_BIB)
        assert rec is not None
        assert "Empirical Methods" in rec.journal
        assert Resolver._credible_journal_article(rec)

    def test_autoprompt_doi_extraction(self):
        """Should extract anthology ID from AutoPrompt's DOI."""
        aid = extract_acl_anthology_id("10.18653/v1/2020.emnlp-main.346")
        assert aid == "2020.emnlp-main.346"

    def test_xlmr_acl2020(self):
        """XLM-R (ACL 2020) should parse correctly."""
        rec = acl_anthology_bib_to_record(XLMR_BIB)
        assert rec is not None
        assert "Cross-lingual" in rec.title
        assert rec.doi == "10.18653/v1/2020.acl-main.747"
        assert rec.year == 2020
        assert rec.pages == "8440--8451"
        assert rec.type == "proceedings-article"
        assert len(rec.authors) == 10
        assert rec.authors[0]["family"] == "Conneau"
        assert rec.authors[9]["family"] == "Stoyanov"

    def test_xlmr_venue_is_acl(self):
        """XLM-R venue should reference ACL."""
        rec = acl_anthology_bib_to_record(XLMR_BIB)
        assert rec is not None
        assert "Association for Computational Linguistics" in rec.journal
        assert Resolver._credible_journal_article(rec)

    def test_xlmr_doi_extraction(self):
        """Should extract anthology ID from XLM-R's DOI."""
        aid = extract_acl_anthology_id("10.18653/v1/2020.acl-main.747")
        assert aid == "2020.acl-main.747"

    def test_xlmr_url_extraction(self):
        """Should extract anthology ID from XLM-R's URL."""
        aid = extract_acl_anthology_id("https://aclanthology.org/2020.acl-main.747/")
        assert aid == "2020.acl-main.747"

    def test_all_real_papers_are_credible(self):
        """All real ACL papers should pass the credibility check."""
        for bib in (BERT_BIB, AUTOPROMPT_BIB, XLMR_BIB):
            rec = acl_anthology_bib_to_record(bib)
            assert rec is not None, f"Failed to parse: {bib[:50]}"
            assert Resolver._credible_journal_article(
                rec
            ), f"Not credible: {rec.title} (type={rec.type}, journal={rec.journal})"

    def test_stage3b_with_real_bert_bib(self, fake_http, logger):
        """Stage 3b should resolve a BERT preprint entry using real BibTeX."""
        resolver = Resolver(http=fake_http, logger=logger, scholarly_client=None)
        resolver.http._request = MagicMock(return_value=MockResponse(200, BERT_BIB))

        entry = {
            "ID": "devlin2018bert",
            "ENTRYTYPE": "article",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "author": "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina",
            "journal": "arXiv preprint arXiv:1810.04805",
            "year": "2018",
            "doi": "10.18653/v1/N19-1423",
        }
        title_norm = normalize_title_for_match(entry["title"])
        result = resolver._stage3b_acl_anthology(entry, title_norm, candidate_doi=None)

        assert result is not None
        assert result.method == "ACLAnthology(doi)"
        assert result.confidence == 1.0
        assert result.doi == "10.18653/v1/n19-1423"
        assert result.year == 2019
        assert len(result.authors) == 4


# ------------- Integration: resolve_uncached includes stage 3b -------------


class TestResolveUncachedWithAcl:
    """Test that _resolve_uncached includes ACL Anthology stage."""

    @pytest.fixture
    def resolver(self, fake_http, logger):
        return Resolver(http=fake_http, logger=logger, scholarly_client=None)

    def test_resolve_uncached_calls_stage3b(self, resolver):
        """_resolve_uncached should call stage 3b for ACL DOIs."""
        entry = {
            "ID": "acl2022",
            "ENTRYTYPE": "article",
            "title": "Learned Incremental Representations for Parsing",
            "author": "Kitaev, Nikita and Klein, Dan",
            "year": "2022",
            "doi": "10.18653/v1/2022.acl-long.220",
            "journal": "arXiv preprint arXiv:2201.12345",
        }
        detection = PreprintDetection(
            is_preprint=True,
            reason="arXiv in journal",
            arxiv_id="2201.12345",
            doi=None,
        )

        # Mock HTTP: stages 1-3 fail (raise), stage 3b succeeds
        call_count = {"n": 0}

        def mock_request(method, url, **kwargs):
            call_count["n"] += 1
            if "aclanthology.org" in url:
                return MockResponse(200, SAMPLE_ACL_BIB)
            raise Exception("Simulated API failure")

        resolver.http._request = mock_request

        result = resolver._resolve_uncached(entry, detection)
        assert result is not None
        assert "ACLAnthology" in result.method
