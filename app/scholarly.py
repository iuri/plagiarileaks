import os
import httpx
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

SEMANTIC_SCHOLAR_SEARCH = os.getenv("SEMANTIC_SCHOLAR_SEARCH")
SEMANTIC_SCHOLAR_PAPER = os.getenv("SEMANTIC_SCHOLAR_PAPER")

async def search_semantic_scholar(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,externalIds,url,doi"
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(SEMANTIC_SCHOLAR_SEARCH, params=params)
        r.raise_for_status()
        data = r.json()
    papers = data.get("data") or data.get("papers") or []
    results = []
    for p in papers:
        results.append({
            "title": p.get("title"),
            "abstract": p.get("abstract") or "",
            "authors": [a.get("name") for a in p.get("authors", [])],
            "year": p.get("year"),
            "doi": (p.get("externalIds") or {}).get("DOI") or p.get("doi"),
            "url": p.get("url"),
        })
    return results


async def search_crossref(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    url = os.getenv("CROSSREF_API_URL")
    params = {"query": query, "rows": limit}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    items = data.get("message", {}).get("items", [])
    results = []
    for it in items:
        results.append({
            "title": " ".join(it.get("title", [])) if it.get("title") else None,
            "abstract": it.get("abstract") or "",
            "authors": ["{} {}".format(a.get("given", ""), a.get("family", "")).strip() for a in it.get("author", [])] if it.get("author") else [],
            "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]) if it.get("issued") else None,
            "doi": it.get("DOI"),
            "url": (it.get("URL")),
        })
    return results
