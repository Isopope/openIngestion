"""openingestion.fetcher — pipeline entry point: document discovery.

Available fetchers
------------------
``LocalFileFetcher``
    Fetch files from the local filesystem (single file or directory walk).
    See :mod:`openingestion.fetcher.local`.

``BaseFetcher``
    Abstract base class for custom fetcher implementations.

Coming soon
-----------
``CloudFetcher``   — S3, Azure Blob, GCS (:mod:`openingestion.fetcher.cloud`)
``DatabaseFetcher``— SQL, MongoDB …    (:mod:`openingestion.fetcher.database`)
``WebFetcher``     — HTTP, sitemaps …  (:mod:`openingestion.fetcher.web`)
"""
from openingestion.fetcher.base import BaseFetcher
from openingestion.fetcher.local import LocalFileFetcher
from openingestion.document import FetchedDocument

__all__ = [
    "BaseFetcher",
    "LocalFileFetcher",
    "FetchedDocument",
]
