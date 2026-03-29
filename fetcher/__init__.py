"""openingestion.fetcher — pipeline entry point: document discovery.

Available fetchers
------------------
``LocalFileFetcher``
    Fetch files from the local filesystem (single file or directory walk).
    See :mod:`openingestion.fetcher.local`.

``WebFetcher``
    Fetch documents from the web using Playwright and render them as PDF/HTML.
    See :mod:`openingestion.fetcher.web`.

``SharepointFetcher``
    Fetch documents from a Microsoft SharePoint / OneDrive site.
    See :mod:`openingestion.fetcher.sharepoint`.

``BaseFetcher``
    Abstract base class for custom fetcher implementations.

Coming soon
-----------
``CloudFetcher``   — S3, Azure Blob, GCS (:mod:`openingestion.fetcher.cloud`)
``DatabaseFetcher``— SQL, MongoDB …    (:mod:`openingestion.fetcher.database`)
"""
from openingestion.fetcher.base import BaseFetcher
from openingestion.fetcher.local import LocalFileFetcher
from openingestion.fetcher.web import WebFetcher
from openingestion.fetcher.sharepoint import SharepointFetcher
from openingestion.document import FetchedDocument

__all__ = [
    "BaseFetcher",
    "LocalFileFetcher",
    "FetchedDocument",
]
