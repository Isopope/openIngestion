from __future__ import annotations

import os
from pathlib import Path
import urllib.parse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from openingestion.document import FetchedDocument
from openingestion.fetcher.base import BaseFetcher

try:
    import msal
    import requests
    _SHAREPOINT_AVAILABLE = True
except ImportError:
    _SHAREPOINT_AVAILABLE = False


class SharepointFetcher(BaseFetcher):
    """
    Fetches documents from a Microsoft SharePoint / OneDrive site.
    
    Relies on Microsoft Graph API. Requires an Entra ID (Azure AD) App Registration 
    with `Files.Read.All` or `Sites.Read.All` application permissions.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        output_dir: str | os.PathLike,
    ) -> None:
        """
        Args:
            client_id: The Azure AD App Client ID.
            client_secret: The Azure AD App Client Secret.
            tenant_id: The Microsoft 365 Tenant ID.
            output_dir: Local directory where files will be downloaded temporarily.
        """
        if not _SHAREPOINT_AVAILABLE:
            raise ImportError(
                "msal and requests are required for SharepointFetcher. "
                "Install with: pip install msal requests"
            )

        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        
        self.msal_app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret,
        )

    def _get_access_token(self) -> str:
        """Acquires a valid access token using MSAL."""
        result = self.msal_app.acquire_token_silent(self.scope, account=None)
        if not result:
            logger.info("SharepointFetcher: No token in cache, acquiring new one...")
            result = self.msal_app.acquire_token_for_client(scopes=self.scope)
            
        if "access_token" in result:
            return result["access_token"]
        else:
            error_msg = result.get("error_description", result.get("error", "Unknown error"))
            raise RuntimeError(f"Failed to acquire token: {error_msg}")

    def fetch(
        self,
        site_name: str | None = None,
        site_url: str | None = None,
        folder_path: str | None = None,
        **kwargs,
    ) -> list[FetchedDocument]:
        """
        Fetch documents from SharePoint.
        
        You must provide either `site_name` (e.g. "MyTeamSite") 
        or `site_url` (e.g. "https://tenant.sharepoint.com/sites/MyTeamSite").
        
        Args:
            site_name: Display name or partial name of the Site.
            site_url: Explicit URL of the Site.
            folder_path: Specific folder path to fetch from (not implemented in MVP, fetches root).
        """
        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        
        # 1. Resolve Site ID
        site_id = self._resolve_site_id(headers, site_name, site_url)
        if not site_id:
            logger.error("SharepointFetcher: Could not resolve site.")
            return []
            
        # 2. Get the default drive (Shared Documents) for the site
        drive_id = self._get_default_drive_id(headers, site_id)
        if not drive_id:
            logger.error(f"SharepointFetcher: No default drive found for site {site_id}.")
            return []
            
        # 3. List files in the drive
        items = self._list_drive_items(headers, drive_id, folder_path=folder_path)
        file_items = [item for item in items if "file" in item]
        logger.info(f"SharepointFetcher: Found {len(file_items)} files to download.")
        
        fetched_docs = []
        if file_items:
            # Download files concurrently to avoid massive I/O blocking
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(lambda i: self._download_file(headers, drive_id, i), file_items))
                fetched_docs = [doc for doc in results if doc is not None]
                    
        return fetched_docs

    def _resolve_site_id(self, headers: dict, site_name: str | None, site_url: str | None) -> str | None:
        """Resolve the Graph API Site ID based on name or URL."""
        if site_url:
            # Using Graph site search by domain and path
            # e.g. https://contoso.sharepoint.com/sites/Test
            parsed = urllib.parse.urlparse(site_url)
            hostname = parsed.netloc
            path = parsed.path
            url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{path}"
            
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json().get("id")
            logger.warning(f"Failed to resolve site URL {site_url}: {resp.text}")
            return None
            
        if site_name:
            # Search by name
            url = f"https://graph.microsoft.com/v1.0/sites?search={site_name}"
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                value = resp.json().get("value", [])
                if value:
                    # Return first match
                    return value[0].get("id")
            logger.warning(f"Failed to find site by name {site_name}: {resp.text}")
            return None
            
        raise ValueError("Must provide either site_name or site_url.")

    def _get_default_drive_id(self, headers: dict, site_id: str) -> str | None:
        """Get the ID of the default document library for the site."""
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json().get("id")
        logger.warning(f"Failed to get default drive for site {site_id}: {resp.text}")
        return None

    def _list_drive_items(self, headers: dict, drive_id: str, folder_path: str | None = None) -> list[dict]:
        """Recursively list items in a drive or folder."""
        if folder_path:
            # /drives/{drive-id}/root:/{path-relative-to-root}:/children
            clean_path = urllib.parse.quote(folder_path.strip("/"))
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{clean_path}:/children"
        else:
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
            
        items = []
        
        def traverse(req_url: str):
            resp = requests.get(req_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("value", []):
                    if "folder" in item:
                        # Recursively get children of this folder
                        traverse(f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item['id']}/children")
                    else:
                        items.append(item)
                
                # Check pagination
                if "@odata.nextLink" in data:
                    traverse(data["@odata.nextLink"])
            else:
                logger.warning(f"Failed to list items at {req_url}: {resp.text}")
        
        traverse(url)
        return items

    def _download_file(self, headers: dict, drive_id: str, item: dict) -> FetchedDocument | None:
        """Download the file to the local output directory."""
        item_id = item["id"]
        name = item.get("name", "unnamed_file")
        mime = item.get("file", {}).get("mimeType", "")
        web_url = item.get("webUrl", f"sharepoint://{item_id}")
        
        out_path = self.output_dir / name
        # Prevent overwriting files with the same name from different folders
        if out_path.exists():
            base, ext = os.path.splitext(name)
            out_path = self.output_dir / f"{base}_{item_id}{ext}"
            
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
        logger.info(f"SharepointFetcher: Downloading {name} ({item_id})")
        
        resp = requests.get(url, headers=headers, stream=True)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return FetchedDocument(source=web_url, path=out_path, mime_type=mime)
        else:
            logger.error(f"SharepointFetcher: Failed to download {name}: {resp.status_code} {resp.text}")
            return None
