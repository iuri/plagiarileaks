import os
import httpx
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup

from dotenv import load_dotenv

load_dotenv()
UNPAYWALL_BASE = os.getenv("UNPAYWALL_BASE")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL")

async def get_unpaywall_for_doi(doi: str) -> Optional[Dict[str, Any]]:
    email = os.getenv("UNPAYWALL_EMAIL")
    if not email:
        return None
    doi = doi.strip()
    if doi.lower().startswith("doi:"):
        doi = doi.split(":", 1)[1]
    url = f"{UNPAYWALL_BASE}/{doi}"
    params = {"email": email}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


async def fetch_html_text(url: str, max_chars: int = 2000) -> Optional[str]:
    # best-effort: fetch HTML and extract visible text
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "html" not in ct:
                return None
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup(['script', 'style', 'noscript']):
                s.extract()
            text = ' '.join(soup.stripped_strings)
            return text[:max_chars]
    except Exception:
        return None
