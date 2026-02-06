"""BibTeX Updater - Tools for managing academic BibTeX bibliographies.

This package provides tools for:
- Upgrading preprint references to their published versions
- Validating bibliography references against authoritative sources
- Filtering bibliographies to only cited entries
- Updating Zotero libraries with published versions

Example usage:
    from bibtex_updater import Detector, Resolver, Updater

    # Check if an entry is a preprint
    detector = Detector()
    detection = detector.detect(entry)

    # Resolve a preprint to its published version
    resolver = Resolver(http_client)
    candidate = resolver.resolve(detection)

    # Update a bibliography file
    updater = Updater(detector, resolver)
    results = updater.process(entries)
"""

from bibtex_updater._version import __version__

# Core updater classes and functions
from bibtex_updater.updater import (
    AsyncResolver,
    BibLoader,
    BibWriter,
    Dedupe,
    Detector,
    FieldChecker,
    FieldCheckResult,
    FieldFiller,
    FieldRequirement,
    FieldRequirementRegistry,
    MissingFieldProcessor,
    MissingFieldReport,
    PreprintDetection,
    ProcessResult,
    Resolver,
    ScholarlyClient,
    Updater,
    # Entry processing functions
    prioritize_entries,
    process_entries_optimized,
    process_entry,
    process_entry_with_preload,
)

# Shared utilities
from bibtex_updater.utils import (
    # Classes
    AdaptiveRateLimiterRegistry,
    AsyncHttpClient,
    AsyncRateLimiter,
    AsyncRateLimiterRegistry,
    DiskCache,
    HttpClient,
    PublishedRecord,
    RateLimiter,
    RateLimiterRegistry,
    ResolutionCache,
    ResolutionCacheEntry,
    # ACL Anthology
    acl_anthology_bib_to_record,
    # Author handling
    authors_last_names,
    # API converters
    crossref_message_to_record,
    dblp_hit_to_record,
    # DOI/arXiv utilities
    doi_normalize,
    doi_url,
    extract_acl_anthology_id,
    extract_arxiv_id_from_text,
    first_author_surname,
    # Matching utilities
    jaccard_similarity,
    last_name_from_person,
    # Text normalization
    latex_to_plain,
    normalize_title_for_match,
    s2_data_to_record,
    safe_lower,
    split_authors_bibtex,
    strip_diacritics,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "AsyncResolver",
    "BibLoader",
    "BibWriter",
    "Dedupe",
    "Detector",
    "FieldChecker",
    "FieldCheckResult",
    "FieldFiller",
    "FieldRequirement",
    "FieldRequirementRegistry",
    "MissingFieldProcessor",
    "MissingFieldReport",
    "PreprintDetection",
    "ProcessResult",
    "Resolver",
    "ScholarlyClient",
    "Updater",
    # Entry processing functions
    "prioritize_entries",
    "process_entries_optimized",
    "process_entry",
    "process_entry_with_preload",
    # Utility classes
    "AdaptiveRateLimiterRegistry",
    "AsyncHttpClient",
    "AsyncRateLimiter",
    "AsyncRateLimiterRegistry",
    "DiskCache",
    "HttpClient",
    "PublishedRecord",
    "RateLimiter",
    "RateLimiterRegistry",
    "ResolutionCache",
    "ResolutionCacheEntry",
    # Text normalization
    "latex_to_plain",
    "normalize_title_for_match",
    "safe_lower",
    "strip_diacritics",
    # Author handling
    "authors_last_names",
    "first_author_surname",
    "last_name_from_person",
    "split_authors_bibtex",
    # Matching utilities
    "jaccard_similarity",
    # DOI/arXiv utilities
    "doi_normalize",
    "doi_url",
    "extract_arxiv_id_from_text",
    # ACL Anthology
    "acl_anthology_bib_to_record",
    "extract_acl_anthology_id",
    # API converters
    "crossref_message_to_record",
    "dblp_hit_to_record",
    "s2_data_to_record",
]
