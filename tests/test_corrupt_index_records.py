"""Tests for distrusting corrupt index records (identifier right, title wrong).

Measured on a 267-submission screening run (2026-09-03): OpenAlex serves works
that carry the CORRECT identifier and the CORRECT author list for a paper under
a title belonging to a different work. Three verified instances, each checked by
resolving the arXiv ID directly:

* ``10.48550/arxiv.2307.16789`` -- really ToolLLM (Qin et al.) -- comes back as
  "Counterfactually Auditable Lifecycle Certification for Autonomous Agents"
  with ToolLLM's real author list (OpenAlex ``W4385474529``).
* ``10.48550/arxiv.2212.08073`` -- really Constitutional AI (Bai et al.) --
  comes back as "Affective Coherence Monitoring for Transformer-Based Language
  Models" with Constitutional AI's real authors (``W4311991106``).
* ``10.48550/arxiv.2106.09685`` -- really LoRA (Hu et al.) -- comes back as
  "LoRA Fine-Tuning of a 3B Code LLM for Algorithmic Efficiency" with LoRA's
  real authors (``W3168867926``).

Normalized title similarity for those three pairs is 0.35-0.39, so the blended
candidate score (``0.7 * title + 0.3 * author``) lands at 0.54-0.57 -- above
``abstention_below`` and above the ``wrong_paper_signature`` bar, because the
author list corroborates perfectly. The entry therefore verdicts TITLE_MISMATCH
on the strength of ONE index's disagreement with itself.

The guard: a candidate anchored on the entry's OWN identifier, whose authors the
entry confirms, whose title is a different paper's, is distrusted unless a second
identifier-anchored source corroborates the divergence AND no identifier-anchored
source confirms the entry's title. Distrust removes the record from the candidate
pool and is reported on ``FactCheckResult.distrusted_records``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from bibtex_updater.fact_checker import (
    FactChecker,
    FactCheckerConfig,
    FactCheckProcessor,
    FactCheckStatus,
)
from bibtex_updater.sources import openalex_work_to_candidate_record
from bibtex_updater.utils import PublishedRecord, normalize_doi_for_resolution

# ---------------------------------------------------------------------------
# Real corrupt OpenAlex works (fields trimmed to what the converter reads).
# ---------------------------------------------------------------------------

TOOLLLM_AUTHORS = [
    "Qin, Yujia",
    "Shihao Liang",
    "Yining Ye",
    "Kunlun Zhu",
    "Lan Yan",
    "Yaxi Lu",
    "Yankai Lin",
    "Xin Cong",
]
CONSTITUTIONAL_AUTHORS = [
    "Bai, Yuntao",
    "Saurav Kadavath",
    "Sandipan Kundu",
    "Amanda Askell",
]
LORA_AUTHORS = [
    "J. Edward Hu",
    "Yelong Shen",
    "Phillip Wallis",
    "Zeyuan Allen-Zhu",
]


def _openalex_work(
    *,
    openalex_id: str,
    doi: str,
    title: str,
    authors: list[str],
    year: int,
    arxiv_landing: str | None = None,
) -> dict:
    """An OpenAlex ``/works`` item in the shape the cascade converter reads."""
    return {
        "id": f"https://openalex.org/{openalex_id}",
        "doi": f"https://doi.org/{doi}",
        "display_name": title,
        "title": title,
        "publication_year": year,
        "type": "preprint",
        "authorships": [{"author": {"display_name": name}} for name in authors],
        "primary_location": {
            "landing_page_url": arxiv_landing,
            "source": {"display_name": "arXiv (Cornell University)"},
        },
    }


CORRUPT_TOOLLLM_WORK = _openalex_work(
    openalex_id="W4385474529",
    doi="10.48550/arxiv.2307.16789",
    title="Counterfactually Auditable Lifecycle Certification for Autonomous Agents",
    authors=TOOLLLM_AUTHORS,
    year=2023,
    arxiv_landing="http://arxiv.org/abs/2307.16789",
)
CORRUPT_CONSTITUTIONAL_WORK = _openalex_work(
    openalex_id="W4311991106",
    doi="10.48550/arxiv.2212.08073",
    title="Affective Coherence Monitoring for Transformer-Based Language Models",
    authors=CONSTITUTIONAL_AUTHORS,
    year=2022,
    arxiv_landing="http://arxiv.org/abs/2212.08073",
)
CORRUPT_LORA_WORK = _openalex_work(
    openalex_id="W3168867926",
    doi="10.48550/arxiv.2106.09685",
    title="LoRA Fine-Tuning of a 3B Code LLM for Algorithmic Efficiency",
    authors=LORA_AUTHORS,
    year=2021,
    arxiv_landing="http://arxiv.org/abs/2106.09685",
)


TOOLLLM_ENTRY = {
    "ID": "qin2024toolllm",
    "ENTRYTYPE": "inproceedings",
    "title": "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs",
    "author": (
        "Qin, Yujia and Liang, Shihao and Ye, Yining and Zhu, Kunlun and "
        "Yan, Lan and Lu, Yaxi and Lin, Yankai and Cong, Xin"
    ),
    "booktitle": "International Conference on Learning Representations",
    "year": "2024",
    "doi": "10.48550/arXiv.2307.16789",
}
CONSTITUTIONAL_ENTRY = {
    "ID": "bai2022constitutional",
    "ENTRYTYPE": "misc",
    "title": "Constitutional AI: Harmlessness from AI Feedback",
    "author": "Bai, Yuntao and Kadavath, Saurav and Kundu, Sandipan and Askell, Amanda",
    "year": "2022",
    "doi": "10.48550/arXiv.2212.08073",
}
LORA_ENTRY = {
    "ID": "hu2022lora",
    "ENTRYTYPE": "inproceedings",
    "title": "LoRA: Low-Rank Adaptation of Large Language Models",
    "author": "Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan",
    "booktitle": "International Conference on Learning Representations",
    "year": "2022",
    "doi": "10.48550/arXiv.2106.09685",
}


def _arxiv_record(*, title: str, authors: list[str], year: int, arxiv_id: str) -> PublishedRecord:
    """The authoritative arXiv record for an ID (what ``_arxiv_record`` returns)."""
    people = []
    for name in authors:
        parts = name.replace(",", " ").split()
        people.append({"given": " ".join(parts[1:]) if len(parts) > 1 else "", "family": parts[0]})
    return PublishedRecord(
        doi=None,
        title=title,
        authors=people,
        journal="arXiv",
        year=year,
        type="preprint",
        arxiv_id=arxiv_id,
    )


TRUE_TOOLLLM_ARXIV = _arxiv_record(
    title="ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs",
    authors=["Qin Yujia", "Liang Shihao", "Ye Yining", "Zhu Kunlun", "Yan Lan", "Lu Yaxi", "Lin Yankai", "Cong Xin"],
    year=2023,
    arxiv_id="2307.16789",
)
TRUE_CONSTITUTIONAL_ARXIV = _arxiv_record(
    title="Constitutional AI: Harmlessness from AI Feedback",
    authors=["Bai Yuntao", "Kadavath Saurav", "Kundu Sandipan", "Askell Amanda"],
    year=2022,
    arxiv_id="2212.08073",
)
TRUE_LORA_ARXIV = _arxiv_record(
    title="LoRA: Low-Rank Adaptation of Large Language Models",
    authors=["Hu Edward", "Shen Yelong", "Wallis Phillip", "Allen-Zhu Zeyuan"],
    year=2021,
    arxiv_id="2106.09685",
)


@pytest.fixture
def logger():
    return logging.getLogger("test_corrupt_index_records")


def _checker(logger, *, config: FactCheckerConfig | None = None) -> FactChecker:
    """A checker whose every source answers with nothing (tests seed what they need)."""
    crossref = MagicMock()
    crossref.search.return_value = []
    crossref.get_by_doi.return_value = None
    crossref.http = MagicMock()
    crossref.http.s2_api_key = None
    dblp = MagicMock()
    dblp.search.return_value = []
    s2 = MagicMock()
    s2.search.return_value = []
    s2.match_title.return_value = []
    openalex = MagicMock()
    openalex.search.return_value = []
    openreview = MagicMock()
    openreview.search.return_value = []
    checker = FactChecker(
        crossref,
        dblp,
        s2,
        config or FactCheckerConfig(),
        logger,
        openalex=openalex,
        openreview=openreview,
    )
    checker.arxiv = MagicMock()
    return checker


def _candidates(checker: FactChecker, entry: dict, *pairs) -> list[tuple[float, PublishedRecord, str]]:
    """Score ``(record, source)`` pairs exactly as the cascade does."""
    from bibtex_updater.utils import authors_last_names, entry_authors, normalize_title_for_match

    title_norm = normalize_title_for_match(entry.get("title", ""))
    authors_ref = authors_last_names(entry_authors(entry), limit=3)
    return [(checker._score_candidate(title_norm, authors_ref, rec), rec, source) for rec, source in pairs]


# ===========================================================================
# The corruption signature itself
# ===========================================================================


class TestCorruptionSignature:
    """``_split_corrupt_index_records`` isolates the id+authors+wrong-title shape."""

    @pytest.mark.parametrize(
        "entry, work, true_arxiv",
        [
            (TOOLLLM_ENTRY, CORRUPT_TOOLLLM_WORK, TRUE_TOOLLLM_ARXIV),
            (CONSTITUTIONAL_ENTRY, CORRUPT_CONSTITUTIONAL_WORK, TRUE_CONSTITUTIONAL_ARXIV),
            (LORA_ENTRY, CORRUPT_LORA_WORK, TRUE_LORA_ARXIV),
        ],
        ids=["toolllm", "constitutional-ai", "lora"],
    )
    def test_corrupt_openalex_record_is_distrusted(self, logger, entry, work, true_arxiv):
        checker = _checker(logger)
        corrupt = openalex_work_to_candidate_record(work)
        assert corrupt is not None
        candidates = _candidates(checker, entry, (corrupt, "openalex"), (true_arxiv, "arxiv"))

        kept, distrusted = checker._split_corrupt_index_records(entry, candidates)

        assert [source for _s, _r, source in kept] == ["arxiv"]
        assert len(distrusted) == 1
        assert distrusted[0].startswith("openalex:")
        assert work["display_name"] in distrusted[0]

    def test_lone_corrupt_record_is_distrusted_without_a_counter_source(self, logger):
        """A single source's title disagreement never stands on its own."""
        checker = _checker(logger)
        corrupt = openalex_work_to_candidate_record(CORRUPT_LORA_WORK)
        candidates = _candidates(checker, LORA_ENTRY, (corrupt, "openalex"))

        kept, distrusted = checker._split_corrupt_index_records(LORA_ENTRY, candidates)

        assert kept == []
        assert len(distrusted) == 1

    def test_two_divergent_sources_corroborate_and_the_finding_stands(self, logger):
        """A real hybrid fabrication: every index that knows the identifier
        reports the same other title, so nothing is distrusted."""
        checker = _checker(logger)
        corrupt = openalex_work_to_candidate_record(CORRUPT_LORA_WORK)
        crossref_agreeing = PublishedRecord(
            doi="10.48550/arxiv.2106.09685",
            title="LoRA Fine-Tuning of a 3B Code LLM for Algorithmic Efficiency",
            authors=[{"given": "Edward", "family": "Hu"}, {"given": "Yelong", "family": "Shen"}],
            journal="arXiv",
            year=2021,
            structured_names=True,
        )
        candidates = _candidates(checker, LORA_ENTRY, (corrupt, "openalex"), (crossref_agreeing, "crossref"))

        kept, distrusted = checker._split_corrupt_index_records(LORA_ENTRY, candidates)

        assert distrusted == []
        assert len(kept) == 2

    def test_divergent_title_with_divergent_authors_is_not_distrusted(self, logger):
        """A wrong identifier pointing at a genuinely different paper keeps its
        evidentiary weight -- the authors do not corroborate."""
        checker = _checker(logger)
        other_paper = PublishedRecord(
            doi="10.48550/arxiv.2106.09685",
            title="Counterfactually Auditable Lifecycle Certification for Autonomous Agents",
            authors=[{"given": "Jane", "family": "Roe"}, {"given": "Richard", "family": "Doe"}],
            journal="arXiv",
            year=2021,
        )
        candidates = _candidates(checker, LORA_ENTRY, (other_paper, "openalex"), (TRUE_LORA_ARXIV, "arxiv"))

        kept, distrusted = checker._split_corrupt_index_records(LORA_ENTRY, candidates)

        assert distrusted == []
        assert len(kept) == 2

    def test_record_without_the_entry_identifier_is_untouched(self, logger):
        """Only records keyed on the entry's OWN identifier can be corrupt in
        this sense; an unrelated search hit is just a weak candidate."""
        checker = _checker(logger)
        unrelated = PublishedRecord(
            doi="10.1234/unrelated",
            title="Counterfactually Auditable Lifecycle Certification for Autonomous Agents",
            authors=[{"given": "Edward", "family": "Hu"}, {"given": "Yelong", "family": "Shen"}],
            journal="Some Journal",
            year=2021,
        )
        candidates = _candidates(checker, LORA_ENTRY, (unrelated, "openalex"), (TRUE_LORA_ARXIV, "arxiv"))

        kept, distrusted = checker._split_corrupt_index_records(LORA_ENTRY, candidates)

        assert distrusted == []
        assert len(kept) == 2

    def test_entry_without_an_identifier_is_untouched(self, logger):
        checker = _checker(logger)
        entry = {k: v for k, v in LORA_ENTRY.items() if k != "doi"}
        corrupt = openalex_work_to_candidate_record(CORRUPT_LORA_WORK)
        candidates = _candidates(checker, entry, (corrupt, "openalex"))

        kept, distrusted = checker._split_corrupt_index_records(entry, candidates)

        assert distrusted == []
        assert len(kept) == 1

    def test_guard_is_inert_when_disabled(self, logger):
        checker = _checker(logger, config=FactCheckerConfig(distrust_corrupt_index_records=False))
        corrupt = openalex_work_to_candidate_record(CORRUPT_LORA_WORK)
        candidates = _candidates(checker, LORA_ENTRY, (corrupt, "openalex"), (TRUE_LORA_ARXIV, "arxiv"))

        kept, distrusted = checker._split_corrupt_index_records(LORA_ENTRY, candidates)

        assert distrusted == []
        assert len(kept) == 2


# ===========================================================================
# arXiv precedence for arXiv identifiers
# ===========================================================================


class TestArxivPrecedence:
    """For a ``10.48550/arxiv.*`` DOI or a bare arXiv ID, arXiv owns the title."""

    def test_openalex_title_does_not_override_arxiv_for_an_arxiv_doi(self, logger):
        checker = _checker(logger)
        corrupt = openalex_work_to_candidate_record(CORRUPT_TOOLLLM_WORK)
        candidates = _candidates(checker, TOOLLLM_ENTRY, (corrupt, "openalex"), (TRUE_TOOLLLM_ARXIV, "arxiv"))

        kept, distrusted = checker._split_corrupt_index_records(TOOLLLM_ENTRY, candidates)

        assert [source for _s, _r, source in kept] == ["arxiv"]
        assert distrusted

    def test_bare_eprint_id_anchors_the_same_way(self, logger):
        """The entry carries no DOI at all -- the arXiv ID alone ties the two
        records to the same work."""
        checker = _checker(logger)
        entry = {k: v for k, v in TOOLLLM_ENTRY.items() if k != "doi"}
        entry["eprint"] = "2307.16789"
        entry["archiveprefix"] = "arXiv"
        corrupt = openalex_work_to_candidate_record(CORRUPT_TOOLLLM_WORK)
        assert corrupt.arxiv_id == "2307.16789"
        candidates = _candidates(checker, entry, (corrupt, "openalex"), (TRUE_TOOLLLM_ARXIV, "arxiv"))

        kept, distrusted = checker._split_corrupt_index_records(entry, candidates)

        assert [source for _s, _r, source in kept] == ["arxiv"]
        assert distrusted


# ===========================================================================
# End-to-end verdict
# ===========================================================================


class TestVerdict:
    """The corrupt record must not carry the entry to a mismatch verdict."""

    def _run(self, logger, entry, work, true_arxiv, *, config=None):
        checker = _checker(logger, config=config)
        checker.openalex.search.return_value = [work]
        checker._arxiv_record_cache[true_arxiv.arxiv_id] = true_arxiv
        return checker.check_entry(entry)

    @pytest.mark.parametrize(
        "entry, work, true_arxiv",
        [
            (TOOLLLM_ENTRY, CORRUPT_TOOLLLM_WORK, TRUE_TOOLLLM_ARXIV),
            (CONSTITUTIONAL_ENTRY, CORRUPT_CONSTITUTIONAL_WORK, TRUE_CONSTITUTIONAL_ARXIV),
            (LORA_ENTRY, CORRUPT_LORA_WORK, TRUE_LORA_ARXIV),
        ],
        ids=["toolllm", "constitutional-ai", "lora"],
    )
    def test_no_title_mismatch_and_the_distrust_is_reported(self, logger, entry, work, true_arxiv):
        result = self._run(logger, entry, work, true_arxiv)

        assert result.status is not FactCheckStatus.TITLE_MISMATCH
        assert result.distrusted_records
        assert any("openalex" in note for note in result.distrusted_records)

    def test_arxiv_unreachable_leaves_the_corrupt_record_alone_but_unconfirmed(self, logger):
        """The shape the screening run hit: no counter-record in the pool at
        all. Distrusting the only candidate must abstain, never assert a miss
        the sources did not support."""
        checker = _checker(logger)
        checker.arxiv = None
        checker.openalex.search.return_value = [CORRUPT_LORA_WORK]

        result = checker.check_entry(LORA_ENTRY)

        assert result.status is FactCheckStatus.UNCONFIRMED
        assert result.distrusted_records

    def test_without_the_guard_the_same_input_verdicts_title_mismatch(self, logger):
        """Pins the defect this guard exists for."""
        checker = _checker(logger, config=FactCheckerConfig(distrust_corrupt_index_records=False))
        checker.arxiv = None
        checker.openalex.search.return_value = [CORRUPT_LORA_WORK]

        result = checker.check_entry(LORA_ENTRY)

        assert result.status is FactCheckStatus.TITLE_MISMATCH
        assert result.distrusted_records == []


# ===========================================================================
# Output surface
# ===========================================================================


class TestOutputSurface:
    """A caller must be able to see that a record was distrusted, and why."""

    def test_json_and_jsonl_carry_distrusted_records(self, logger):
        checker = _checker(logger)
        checker.arxiv = None
        checker.openalex.search.return_value = [CORRUPT_LORA_WORK]
        result = checker.check_entry(LORA_ENTRY)

        processor = FactCheckProcessor.__new__(FactCheckProcessor)
        report = FactCheckProcessor.generate_json_report(processor, [result])
        assert report["entries"][0]["distrusted_records"] == result.distrusted_records

        line = json.loads(FactCheckProcessor.generate_jsonl(processor, [result])[0])
        assert line["distrusted_records"] == result.distrusted_records

    def test_clean_result_reports_an_empty_list(self, logger):
        checker = _checker(logger)
        checker._arxiv_record_cache[TRUE_LORA_ARXIV.arxiv_id] = TRUE_LORA_ARXIV
        result = checker.check_entry(LORA_ENTRY)
        assert result.distrusted_records == []


# ===========================================================================
# arXiv DataCite DOI case
# ===========================================================================


class TestArxivDoiCase:
    """arXiv registers its DOIs lowercase; entries carry the ``arXiv`` casing."""

    @pytest.mark.parametrize(
        "raw",
        [
            "10.48550/arXiv.2503.19786",
            "10.48550/ARXIV.2503.19786",
            "https://doi.org/10.48550/arXiv.2503.19786",
            "10.48550/arXiv.2503.19786v2",
        ],
    )
    def test_resolution_form_is_lowercase_and_unversioned(self, raw):
        assert normalize_doi_for_resolution(raw) == "10.48550/arxiv.2503.19786"

    def test_validate_doi_probes_the_normalized_form(self, logger, monkeypatch):
        """``_validate_doi`` must hand doi.org the lowercase form, never the raw
        capital-X string that reads as an invented DOI."""
        import bibtex_updater.fact_checker as fc_mod

        checker = _checker(logger)
        probed: list[str] = []

        def _fake_resolves(_client, doi):
            probed.append(doi)
            return True

        monkeypatch.setattr(fc_mod, "_doi_resolves", _fake_resolves)
        entry = {"ID": "gemma3", "ENTRYTYPE": "misc", "doi": "10.48550/arXiv.2503.19786"}

        assert checker._validate_doi(entry) is None
        assert probed == ["10.48550/arxiv.2503.19786"]
