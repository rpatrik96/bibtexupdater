# Bibliographic Tools & Databases Landscape

A comprehensive overview of databases, APIs, and tools available for academic reference management, preprint resolution, and citation verification. This document explains which sources bibtexupdater integrates, which it doesn't, and why.

## Databases Integrated in bibtexupdater

### Resolution Pipeline

bibtexupdater uses a multi-stage resolution pipeline to find published versions of preprints:

| Stage | Source | Coverage | Use Case |
|-------|--------|----------|----------|
| 1 | arXiv API | CS, Physics, Math, etc. | Extract DOI from arXiv metadata |
| 1b | OpenAlex | All disciplines (250M+ works) | Preprint-to-published version tracking |
| 1c | Europe PMC | Life sciences | bioRxiv/medRxiv preprint linking |
| 2 | Crossref | All disciplines | DOI metadata and is-preprint-of relations |
| 3 | DBLP | Computer science | Bibliographic search by title+author |
| 3b | ACL Anthology | NLP/Computational Linguistics | Direct metadata for ACL venue papers |
| 4 | Semantic Scholar | All disciplines | Search with publication type filtering |
| 5 | Crossref Search | All disciplines | Bibliographic search fallback |
| 6 | Google Scholar | All disciplines (opt-in) | Final fallback via scholarly library |

### Per-Source Details

#### arXiv API

- **API**: `http://export.arxiv.org/api/query`
- **Rate limits**: 3 requests/second (we use 1 req/sec to be conservative)
- **Coverage**: Computer Science, Physics, Mathematics, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics
- **Strengths**: Authoritative source for arXiv preprints. Often includes DOIs in metadata when papers are published. No authentication required.
- **Data returned**: arXiv ID, title, authors, abstract, DOI (when available), publication date, categories

The arXiv API is the entry point for the majority of CS/ML preprint resolution. When an arXiv preprint includes a DOI in its metadata (added by authors after publication), we can immediately jump to Crossref for full published metadata. This is the fastest resolution path.

#### OpenAlex

- **API**: `https://api.openalex.org`
- **Rate limits**: Polite pool with email = 100,000 requests/day (10 req/sec sustained), anonymous = 10 req/min
- **Coverage**: 250M+ works across all disciplines, with comprehensive preprint-to-published version tracking
- **Strengths**: Completely open, no API key required, excellent preprint version tracking via `related_works` with `is_version_of` relationship type. Aggregates metadata from Crossref, PubMed, institutional repositories, and more.
- **Data returned**: DOI, title, authors with ORCIDs, publication year, venue, open access status, version relationships

OpenAlex was added as a high-priority resolution source because it explicitly tracks version relationships between preprints and their published counterparts. Unlike Crossref's `is-preprint-of` relation (which requires publishers to assert the relationship), OpenAlex builds these relationships algorithmically across multiple sources, resulting in broader coverage. It is queried early in the pipeline (stage 1b) because it can resolve many cases that would otherwise require multiple fallback stages.

#### Europe PMC

- **API**: `https://www.ebi.ac.uk/europepmc/webservices/rest`
- **Rate limits**: No published rate limit, we use 2 req/sec
- **Coverage**: 40M+ life sciences publications, including PubMed, PubMed Central, bioRxiv, medRxiv
- **Strengths**: Superior preprint-to-published linking for bioRxiv and medRxiv compared to other sources. Subsumes PubMed content with better API access. Free and open, no authentication required.
- **Data returned**: DOI, PMID, PMCID, title, authors, publication date, venue, preprint linking relationships

Europe PMC is the authoritative source for biomedical preprint resolution. While PubMed is more well-known, Europe PMC mirrors its content and adds significantly better preprint linking via the `relations` endpoint. For bioRxiv/medRxiv preprints, Europe PMC can directly return the published PMID/DOI, making it more effective than generic bibliographic search. This is why we integrate Europe PMC instead of PubMed directly.

#### Crossref

- **API**: `https://api.crossref.org/works`
- **Rate limits**: 50 requests/second for polite pool (with `mailto` parameter)
- **Coverage**: 140M+ DOI records across all disciplines
- **Strengths**: DOI authority, includes `is-preprint-of` and `is-preprint-of` relations when publishers assert them. Fast and reliable. Batch API support for bulk queries.
- **Data returned**: DOI, title, authors, publication year, venue, issue, volume, pages, `relation` field for version tracking

Crossref is the backbone of the resolution pipeline. Stage 2 uses Crossref DOI lookup to fetch metadata for known DOIs (e.g., extracted from arXiv). If the DOI has an `is-preprint-of` relation, we immediately know the published version. Stage 5 uses Crossref's bibliographic search API as a fallback when earlier stages fail. The search API supports fuzzy title matching and author filtering.

#### DBLP

- **API**: `https://dblp.org/search/publ/api`
- **Rate limits**: No official limit documented, we use 2 req/sec
- **Coverage**: Computer science and adjacent fields (5M+ publications)
- **Strengths**: High-quality curated data for CS conferences and journals. Excellent for ML/NLP/AI papers. Typically faster and more accurate than generic search for CS domain.
- **Data returned**: Title, authors, venue, year, DOI (when available), electronic edition URL

DBLP is the de facto standard for CS bibliographic data. It is queried at stage 3 because CS papers (the primary use case for bibtexupdater) are almost always indexed in DBLP if they are published in a recognized venue. The curated nature of DBLP means fewer false positives compared to larger, noisier databases.

#### ACL Anthology

- **API**: Direct `.bib` file download via `https://aclanthology.org/{anthology_id}.bib`
- **Rate limits**: No official limit, we use 1 req/sec to be respectful
- **Coverage**: NLP and computational linguistics (60,000+ papers from ACL, EMNLP, NAACL, CoNLL, etc.)
- **Strengths**: Zero-overhead authoritative data for NLP papers. If a paper has an ACL Anthology ID (identifiable by DOI prefix `10.18653/v1/` or URL pattern), we can fetch complete metadata in a single request without fuzzy matching or validation.
- **Data returned**: Complete BibTeX entry with title, authors, venue, year, pages, DOI, URL, abstract

ACL Anthology integration was added at stage 3b because NLP is a major research domain for bibtexupdater users. ACL papers are uniquely identifiable by their DOI prefix, allowing deterministic resolution without any ambiguity. This is the fastest and most accurate resolution path for NLP preprints.

#### Semantic Scholar

- **API**: `https://api.semanticscholar.org/graph/v1`
- **Rate limits**: 100 requests/second with API key, 1 req/sec without
- **Coverage**: 200M+ papers across all disciplines, aggregated from arXiv, PubMed, ACL, DBLP, publisher feeds
- **Strengths**: Broad coverage, excellent search API with `publicationTypes` filtering to exclude preprints from results. Returns OpenAlex IDs for cross-referencing. Batch API support.
- **Data returned**: DOI, arXiv ID, title, authors, publication year, venue, publication types, citation count, OpenAlex ID

Semantic Scholar is queried at stage 4 as a comprehensive fallback. The key advantage is the `publicationTypes` filter, which allows us to search for title+author matches while explicitly excluding `JournalArticle` types, reducing false positives from the preprint itself appearing in results. S2's aggregation across multiple sources makes it a strong general-purpose resolver.

#### Google Scholar

- **API**: Unofficial, accessed via `scholarly` Python library
- **Rate limits**: Aggressive rate limiting by Google, requires CAPTCHA solving or proxies. We use 1 req/10 seconds with randomized delays.
- **Coverage**: Broadest coverage of all academic literature, including grey literature, preprints, theses, books
- **Strengths**: Last resort for obscure papers not indexed elsewhere. Can find papers that exist but are poorly indexed.
- **Data returned**: Title, authors, venue, year, links (when available)

Google Scholar is opt-in (`--use-scholarly` flag) because it is unreliable and slow. The unofficial API is fragile and frequently blocked. However, for papers that are published in non-standard venues or regional journals not indexed by other services, Scholar is sometimes the only option. It is always the final stage (6) and should only be enabled when other sources have failed.

## Databases Evaluated but Not Integrated

### Unpaywall

- **What it does**: REST API that finds open access copies of papers by DOI. Aggregates data from PubMed Central, repositories, and publisher sites.
- **URL**: https://unpaywall.org/products/api
- **Why not included**: Unpaywall is designed for open access discovery, not preprint-to-published resolution. Given a DOI, it returns URLs to freely accessible versions (PDFs). It does not map preprint DOIs to their published DOIs or perform bibliographic search. Since our use case is resolving preprints (which often lack DOIs) to published versions (which have DOIs), Unpaywall would not improve resolution rates. If we already have a DOI, we use Crossref for metadata.

### Dimensions

- **What it does**: Commercial research intelligence platform with 130M+ publications, citation data, and preprint linking via "Research Object Connections".
- **URL**: https://www.dimensions.ai
- **Why not included**: Requires a paid subscription (no free tier for API access). Not suitable for an open-source tool that anyone can run. OpenAlex provides equivalent preprint-to-published linking functionality for free, with a fully open API and no authentication requirements. Dimensions would offer marginal benefit over OpenAlex at significant cost.

### Scopus (Elsevier)

- **What it does**: Abstract and citation database with 87M+ records, primarily from peer-reviewed journals.
- **URL**: https://www.scopus.com
- **Why not included**: Requires a paid institutional subscription. The API is not freely accessible. While Scopus has excellent coverage, it does not provide a documented public API for preprint-to-published linking. Crossref, DBLP, and Semantic Scholar already provide strong coverage for peer-reviewed literature without subscription costs.

### Web of Science (Clarivate)

- **What it does**: Curated citation index with selective coverage of high-impact journals (94M+ records).
- **URL**: https://www.webofscience.com
- **Why not included**: Requires a paid institutional subscription. No free API tier. The selective coverage model means it would miss many papers that are indexed in more comprehensive databases like Crossref or OpenAlex. Not suitable for open-source integration.

### Lens.org

- **What it does**: Free search tool aggregating 270M+ scholarly works, patents, and other research outputs.
- **URL**: https://www.lens.org
- **Why not included**: While Lens.org offers a free API tier, its preprint-to-published version linking capability is unclear from documentation. The API primarily focuses on search and retrieval rather than explicit version relationships. OpenAlex provides more transparent and well-documented version tracking via the `related_works` field. Additionally, Lens.org's API requires registration and has usage limits that make it less attractive than OpenAlex's anonymous polite pool access.

### BASE (Bielefeld Academic Search Engine)

- **What it does**: Aggregates metadata from 10,000+ repositories, indexing 350M+ documents including preprints, theses, and data.
- **URL**: https://www.base-search.net
- **Why not included**: The BASE API requires IP whitelisting with manual approval from the Bielefeld University Library. This makes it unsuitable for a tool that researchers can run on any machine. Additionally, BASE aggregates repository content but does not provide explicit preprint-to-published version resolution features. It would require bibliographic search with the same ambiguity challenges we already address with Crossref and Semantic Scholar.

### IEEE Xplore

- **What it does**: Digital library for IEEE and IET publications (5M+ documents), focusing on electrical engineering and computer science.
- **URL**: https://ieeexplore.ieee.org
- **Why not included**: While IEEE has an API, it requires registration and institutional access for full metadata. DBLP already provides comprehensive coverage of IEEE conference proceedings and journals for computer science papers. IEEE Xplore does not offer a preprint-linking API—most IEEE preprints appear on arXiv first, where we can already resolve them via other stages. Adding IEEE would provide redundant coverage at increased complexity.

### OpenCitations

- **What it does**: Open database of 2B+ citation relationships extracted from Crossref, PubMed Central, and other sources.
- **URL**: https://opencitations.net
- **Why not included**: OpenCitations stores citation relationships (paper A cites paper B), not version relationships (preprint A is published as paper B). Its metadata is sourced from Crossref and other databases we already integrate. For preprint resolution, we need version tracking, not citation tracking. OpenCitations would be useful for building citation graphs but does not help with the core resolution task.

### DataCite

- **What it does**: DOI registration agency primarily for datasets, software, and other research outputs. Also handles some preprint server DOIs.
- **URL**: https://datacite.org
- **Why not included**: DataCite primarily issues DOIs for datasets, not journal articles. While some preprint servers (e.g., Zenodo, FigShare) use DataCite DOIs, Crossref handles the vast majority of journal article DOIs. DataCite DOIs typically represent preprints or data, not the published versions we are trying to find. Crossref already covers the publication DOIs we need. Adding DataCite would not improve resolution rates for the primary use case (academic papers).

### J-STAGE (Japan Science and Technology Information Aggregator, Electronic)

- **What it does**: Platform for Japanese academic journals (3,000+ journals, 5M+ articles) across all disciplines.
- **URL**: https://www.jstage.jst.go.jp
- **Why not included**: Regional focus on Japanese publications. While J-STAGE has a search API, it does not provide clear documentation on preprint resolution or version linking. Most internationally relevant Japanese papers are also indexed in Crossref, which we already integrate. Adding J-STAGE would increase complexity for marginal coverage gains in a specific geographic region. If Japanese papers have DOIs, Crossref already handles them.

### CNKI (China National Knowledge Infrastructure)

- **What it does**: Largest Chinese academic database with 70M+ records across journals, dissertations, and conference proceedings.
- **URL**: https://www.cnki.net
- **Why not included**: No public API for international users. Access is primarily web-based and requires institutional subscription. Language barriers (Chinese-language metadata) would complicate bibliographic matching. Most internationally relevant Chinese papers are also indexed in Crossref or Semantic Scholar. Adding CNKI would require significant localization effort for limited benefit to the primary user base (English-language academic writing).

### PubMed / PubMed Central

- **What it does**: NIH database of 36M+ biomedical literature citations, with PMC providing full-text access to 8M+ open access articles.
- **URL**: https://pubmed.ncbi.nlm.nih.gov, https://www.ncbi.nlm.nih.gov/pmc
- **Why not included directly**: Europe PMC (which we integrate at stage 1c) mirrors PubMed content and provides superior preprint-to-published linking APIs. Europe PMC's `relations` endpoint explicitly tracks bioRxiv/medRxiv preprints to their published PMIDs, while PubMed's E-utilities do not provide this structured relationship data. Using Europe PMC gives us PubMed coverage plus better functionality. There is no benefit to adding PubMed separately.

## Related Tools & Software

### Preprint-to-Published Resolvers

| Tool | Approach | Databases | Strengths | Limitations |
|------|----------|-----------|-----------|-------------|
| **bibtexupdater** | Multi-stage pipeline | arXiv, OpenAlex, Europe PMC, Crossref, DBLP, ACL Anthology, S2, Scholar | Most comprehensive pipeline, CI/CD ready, async support | Requires API calls (cached) |
| **rebiber** | Pre-built conference data | Static ACL/NeurIPS/etc. data | Fast, no API calls for known venues | NLP-only, static data requires manual updates |
| **reffix** | Single-source lookup | DBLP | Simple, lightweight | Single source = lower coverage outside CS |
| **PreprintResolver** | 4-database search | Crossref, DBLP, S2, OpenAlex | Published academic paper documenting approach | 60.3% success rate (2023 data) |
| **bibcure** | arXiv + DOI resolution | arXiv, DOI | Multi-tool suite (doi2bib, arxivcheck, scihub2pdf) | Less comprehensive pipeline than others |
| **PreprintMatch** | bioRxiv/medRxiv specialist | bioRxiv API, Crossref | Life science focus with dedicated tooling | 61.6% success rate, no CS/ML coverage |
| **PaperMemory** | Browser extension | 4 sources | Real-time detection while browsing | Not CLI/CI-friendly, requires browser |

#### rebiber

- **Repository**: https://github.com/yuchenlin/rebiber
- **Approach**: Maintains static JSON files with pre-computed BibTeX entries for major NLP/ML conferences (ACL, NeurIPS, ICML, ICLR, etc.). When an arXiv preprint is detected, it looks up the arXiv ID in the conference data and replaces the entry with the official publication metadata.
- **Strengths**: Extremely fast (no API calls), deterministic results for papers in the database, well-maintained for major ML conferences.
- **Comparison to bibtexupdater**: rebiber excels for NLP/ML papers from major venues because it has pre-indexed them. However, it requires manual updates to the JSON files as new conferences are published, and it provides zero coverage for papers outside its curated dataset (e.g., physics, biology, or CS papers from non-major venues). bibtexupdater uses dynamic API queries, which means it can resolve any paper from any field without manual updates, at the cost of requiring API calls. The two tools are complementary—rebiber for speed on known venues, bibtexupdater for comprehensive coverage.

#### reffix

- **Repository**: https://github.com/kasnerz/reffix
- **Approach**: Single-source resolution using only DBLP's bibliographic search. Detects preprints by checking journal names and DOI prefixes, then searches DBLP for title matches.
- **Strengths**: Minimal dependencies, straightforward code, works well for CS papers.
- **Comparison to bibtexupdater**: reffix is simpler and lighter, but its single-source approach limits coverage. If a paper is not indexed in DBLP (e.g., pure math, biology, or emerging venues), reffix cannot resolve it. bibtexupdater's multi-stage pipeline increases coverage by falling back through multiple databases, and the addition of OpenAlex and Europe PMC extends coverage to life sciences and interdisciplinary work.

#### PreprintResolver

- **Paper**: "PreprintResolver: A Tool for Resolving Preprints to their Published Versions" (2023)
- **Approach**: 4-database sequential search through Crossref DOI lookup, DBLP search, Semantic Scholar search, and OpenAlex search. Uses fuzzy title matching with author validation.
- **Strengths**: Published academic paper documenting the methodology and evaluation (60.3% success rate on 1,000-entry test set).
- **Comparison to bibtexupdater**: PreprintResolver's approach is similar to bibtexupdater's, but bibtexupdater extends it with arXiv API integration (stage 1), Europe PMC (stage 1c), ACL Anthology (stage 3b), and optional Google Scholar (stage 6). The addition of Europe PMC specifically improves life sciences coverage, and ACL Anthology provides deterministic resolution for NLP papers. bibtexupdater also includes production-ready features like caching, rate limiting, batch API support, and CI/CD integration that are not described in the PreprintResolver paper.

#### bibcure

- **Repository**: https://github.com/bibcure/bibcure
- **Approach**: Suite of tools including `doi2bib` (fetch BibTeX by DOI), `arxivcheck` (update arXiv entries), and `scihub2pdf` (download PDFs). The arxiv check tool uses arXiv API and Crossref for resolution.
- **Strengths**: Multi-tool suite covering DOI lookup, arXiv resolution, and PDF downloading in one package.
- **Comparison to bibtexupdater**: bibcure's arXiv resolution is limited to arXiv API + Crossref, which misses papers not indexed in Crossref or lacking DOI links in arXiv metadata. bibtexupdater's multi-stage pipeline provides significantly higher coverage. However, bibcure includes PDF downloading functionality that bibtexupdater does not (intentionally—keeping focus on metadata resolution). Users needing PDF automation might use bibcure alongside bibtexupdater.

#### PreprintMatch

- **Approach**: Specialized tool for bioRxiv and medRxiv preprints. Uses bioRxiv API to check if a preprint has been published, then fetches metadata from Crossref.
- **Strengths**: Dedicated life sciences focus with deep knowledge of bioRxiv/medRxiv versioning and linking.
- **Limitations**: 61.6% success rate reported. Zero coverage for CS/ML/physics preprints (arXiv-based research).
- **Comparison to bibtexupdater**: PreprintMatch is domain-specific for biomedical research, while bibtexupdater is domain-agnostic. bibtexupdater's integration of Europe PMC (stage 1c) provides similar bioRxiv/medRxiv resolution capability while also covering arXiv-based fields. For pure life sciences research, PreprintMatch may offer more specialized features, but for researchers working across disciplines, bibtexupdater provides broader utility.

#### PaperMemory

- **Repository**: https://github.com/vict0rsch/PaperMemory
- **Approach**: Browser extension that detects when you are viewing a preprint on arXiv, bioRxiv, etc., and checks 4 sources (Crossref, DBLP, S2, OpenAlex) to find the published version. Displays a notification with a link.
- **Strengths**: Real-time detection while browsing, no need to run manual commands.
- **Comparison to bibtexupdater**: PaperMemory is an interactive tool for individual browsing, while bibtexupdater is a batch processing tool for bibliographies. PaperMemory is excellent for staying up-to-date while reading papers, but it does not help with updating entire .bib files or integrating into CI/CD workflows. The two tools serve different use cases and are complementary.

### Citation Hallucination Checkers

These tools verify whether references actually exist and have correct metadata. This is particularly relevant for AI-generated text, where large language models sometimes fabricate plausible-looking citations.

| Tool | Type | Databases | Notes |
|------|------|-----------|-------|
| **bibtex-check** (bibtexupdater) | Open source CLI | Crossref, DBLP, S2 | Built-in fact checker in this project |
| **GPTZero** | Commercial API | Proprietary | Found 100+ hallucinations in NeurIPS 2025 submissions |
| **Citely** | Commercial | CrossRef + multi-source | Claims >95% accuracy on hallucination detection |
| **RefChecker** | Open source | S2, OpenAlex, CrossRef | Microsoft Research, Python library + web UI |
| **HaRC (harcx)** | Open source | S2, DBLP, Scholar, Open Library | PyPI package, extensible architecture |
| **CiteVerifier/GhostCite** | Open source | DBLP, Google Scholar | Analyzed 2.2M citations, research dataset |
| **hallucinator** | Open source | CrossRef, arXiv, DBLP, S2, ACL, PubMed, OpenAlex | Most similar DB coverage to bibtexupdater |
| **SwanRef** | Free web tool | 150M+ papers | No API access, web interface only |

#### bibtex-check (bibtexupdater)

- **Command**: `bibtex-check references.bib --strict --report report.json`
- **Approach**: For each BibTeX entry, queries Crossref, DBLP, and Semantic Scholar to verify existence and metadata accuracy. Checks title, author, year, and venue against database records. Flags entries as "not found", "hallucinated" (likely fabricated), or "mismatched" (exists but metadata differs).
- **Output**: JSON/JSONL reports with detailed mismatch information. Strict mode exits with error code for CI/CD integration.
- **Use case**: Verifying bibliographies before submission, catching LLM hallucinations in AI-generated papers, ensuring BibTeX accuracy.

#### GPTZero

- **URL**: https://gptzero.me/hallucination-detector
- **Approach**: Commercial API that checks citations against proprietary database. Gained attention for analyzing NeurIPS 2025 submissions and finding over 100 hallucinated references.
- **Strengths**: Proven effectiveness on real academic submissions, likely includes advanced heuristics for detecting fabricated patterns.
- **Limitations**: Closed-source, requires paid subscription, proprietary database coverage unknown.

#### Citely

- **URL**: https://citely.ai/citation-checker
- **Approach**: Commercial service aggregating Crossref and other sources. Claims >95% accuracy on hallucination detection.
- **Strengths**: High accuracy claims, user-friendly web interface.
- **Limitations**: Paid service, proprietary algorithms, unclear database coverage.

#### RefChecker

- **Repository**: https://github.com/markrussinovich/refchecker
- **Approach**: Python library and web UI from Microsoft Research. Queries Semantic Scholar, OpenAlex, and Crossref to validate citations. Includes both exact matching and fuzzy matching modes.
- **Strengths**: Microsoft-backed, open source, good documentation, supports multiple output formats.
- **Comparison to bibtexupdater**: RefChecker and bibtexupdater's `bibtex-check` command have similar architectures (multi-source validation with fuzzy matching). RefChecker includes OpenAlex in validation while bibtexupdater uses OpenAlex primarily for resolution. RefChecker's web UI may be more accessible for non-technical users, while bibtexupdater's CLI is more suitable for automation and CI/CD.

#### HaRC (harcx)

- **Repository**: https://github.com/aiprof/harcx
- **PyPI**: https://pypi.org/project/harcx/
- **Approach**: Extensible Python framework for hallucination detection. Queries Semantic Scholar, DBLP, Google Scholar, and Open Library (for books). Modular design allows adding new sources.
- **Strengths**: PyPI package for easy integration, extensible architecture, includes book validation via Open Library.
- **Comparison to bibtexupdater**: HaRC is focused purely on validation, while bibtexupdater includes validation as one component alongside resolution and filtering. HaRC's inclusion of Open Library for books is a differentiator—bibtexupdater focuses on journal/conference papers. For researchers needing book citation validation, HaRC complements bibtexupdater well.

#### CiteVerifier/GhostCite

- **Repository**: https://github.com/NKU-AOSP-Lab/CiteVerifier
- **Approach**: Academic research project that analyzed 2.2M citations from arXiv papers to identify "ghost citations" (references that do not exist). Uses DBLP and Google Scholar for validation.
- **Strengths**: Large-scale evaluation dataset, published research on hallucination patterns.
- **Comparison to bibtexupdater**: CiteVerifier is primarily a research artifact demonstrating the prevalence of citation errors. bibtexupdater's `bibtex-check` is a production tool designed for daily use. CiteVerifier's dataset could be valuable for benchmarking bibtexupdater's accuracy.

#### hallucinator

- **Repository**: https://github.com/gianlucasb/hallucinator
- **Approach**: Python library that queries Crossref, arXiv, DBLP, Semantic Scholar, ACL Anthology, PubMed, and OpenAlex to verify citations. Most comprehensive database coverage among open-source validators.
- **Strengths**: Matches bibtexupdater's multi-source approach, includes both validation and basic metadata fetching.
- **Comparison to bibtexupdater**: hallucinator has nearly identical database integration to bibtexupdater (both include arXiv, Crossref, DBLP, S2, ACL Anthology, OpenAlex). hallucinator adds PubMed while bibtexupdater adds Europe PMC (which subsumes PubMed). The tools are functionally similar for validation, but bibtexupdater includes additional features (preprint resolution, filtering, Zotero integration) beyond validation. For users needing only validation, hallucinator is a strong alternative.

#### SwanRef

- **URL**: https://swanref.com
- **Approach**: Free web tool for citation checking against 150M+ papers. Upload a document and receive validation report.
- **Strengths**: Simple web interface, no installation required, free to use.
- **Limitations**: No API access, no CLI, no integration with workflows. Manual upload required for each document.

### BibTeX Cleanup & Management Tools

These tools help with formatting, organizing, and maintaining BibTeX files, complementing bibtexupdater's focus on metadata accuracy.

#### BibTeX Tidy

- **URL**: https://flamingtempura.github.io/bibtex-tidy/
- **Repository**: https://github.com/FlamingTempura/bibtex-tidy
- **Type**: Web app and CLI
- **Features**: Formatting (indentation, line wrapping), field sorting, deduplication by key/DOI/title, removing unwanted fields, URL encoding, consistent field naming.
- **Use case**: Clean up messy BibTeX files before submission. bibtexupdater focuses on metadata correctness, BibTeX Tidy focuses on formatting and structure. The two tools work well together—run bibtexupdater first to fix metadata, then BibTeX Tidy to format.

#### JabRef

- **URL**: https://www.jabref.org
- **Type**: Desktop GUI application (cross-platform)
- **Features**: Full-featured reference manager with DOI/ISBN fetching, duplicate detection, citation key generation, library organization, PDF management, LaTeX integration.
- **Use case**: Interactive bibliography management for researchers who prefer GUI tools. JabRef includes DOI-based metadata fetching (similar to bibtexupdater's resolution) but is manual—you fetch metadata entry-by-entry. bibtexupdater automates batch processing for entire bibliographies. Researchers might use JabRef for daily management and bibtexupdater for bulk updates or CI/CD.

#### Zotero Better BibTeX

- **URL**: https://retorque.re/zotero-better-bibtex/
- **Type**: Zotero plugin
- **Features**: Enhanced BibTeX export from Zotero with stable citation keys, auto-export on library changes, customizable field mapping, journal abbreviation support.
- **Use case**: Researchers using Zotero who need reliable BibTeX export for LaTeX writing. bibtexupdater complements this with `bibtex-zotero` command, which directly updates Zotero library items (not just exported BibTeX). Better BibTeX handles export formatting, bibtexupdater handles preprint-to-published upgrades within Zotero itself.

#### biblio-glutton

- **URL**: https://github.com/kermitt2/biblio-glutton
- **Type**: Self-hosted service (Docker)
- **Features**: High-performance bibliographic matching and metadata lookup service based on GROBID. Supports Crossref, PubMed, and other sources. RESTful API for integration.
- **Use case**: Institutions or labs needing local bibliographic services with high throughput. biblio-glutton is infrastructure (a server you run), while bibtexupdater is a client tool. For organizations processing large volumes of references, biblio-glutton could be used as a backend source, though bibtexupdater does not currently integrate with it due to the self-hosting requirement.

## Benchmarking

For evaluating citation verification tools and comparing resolution accuracy, see:

**HALLMARK** - https://github.com/rpatrik96/hallmark

A benchmark dataset and evaluation framework for citation hallucination detection. Includes bibtexupdater as a baseline tool alongside other validators. Provides ground-truth labeled data for testing whether a reference checker correctly identifies fabricated citations vs. real-but-mislabeled citations vs. correct citations.

HALLMARK is useful for:
- Comparing bibtexupdater's accuracy to other tools
- Testing new resolution strategies
- Understanding common failure modes in citation validation

## Contributing

If you know of a database or tool that should be listed here, please open an issue or pull request at https://github.com/rpatrik96/bibtexupdater.

Relevant information to include:
- API endpoint and authentication requirements
- Rate limits and access restrictions
- Coverage (disciplines, number of records)
- Preprint/version linking capabilities (if applicable)
- Why it would improve bibtexupdater's resolution pipeline
