import re
import os
import numpy as np
from typing import List, Dict, Any

from .scholarly import search_semantic_scholar, search_crossref
from .embeddings import embed_texts
from .unpaywall import get_unpaywall_for_doi, fetch_html_text
from .vectordb import FaissIndexManager


def split_into_passages(text: str, max_len: int = 300) -> List[str]:
    # naive sentence split then join into passages up to max_len chars
    sents = re.split(r'(?<=[\.!?])\s+', text.strip())
    passages = []
    cur = []
    cur_len = 0
    for s in sents:
        if cur_len + len(s) > max_len and cur:
            passages.append(" ".join(cur))
            cur = [s]
            cur_len = len(s)
        else:
            cur.append(s)
            cur_len += len(s)
    if cur:
        passages.append(" ".join(cur))
    return passages


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # both should be normalized if using sentence-transformers with normalize_embeddings=True
    return float(np.dot(a, b))


async def find_similar_articles(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    # split input into passages
    passages = split_into_passages(text)
    if not passages:
        return []

    # Build queries from the full text and first passage
    queries = [text[:1000]] + passages[:3]

    # search semantic scholar and crossref
    candidates = []
    for q in queries:
        try:
            papers = await search_semantic_scholar(q, limit=6)
            candidates.extend(papers)
        except Exception:
            pass
        try:
            cr = await search_crossref(q, limit=4)
            candidates.extend(cr)
        except Exception:
            pass

    # deduplicate by DOI or title
    seen = set()
    uniq = []
    for c in candidates:
        key = (c.get("doi") or c.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    if not uniq:
        return []

    # Optionally use Unpaywall to attach OA info and optionally fetch OA text
    fetch_oa = os.getenv("UNPAYWALL_FETCH", "false").lower() in ("1", "true", "yes")
    for c in uniq:
        doi = c.get("doi")
        if doi:
            try:
                uw = await get_unpaywall_for_doi(doi)
                if uw:
                    c["unpaywall"] = {
                        "oa_status": uw.get("oa_status"),
                        "best_oa_location": uw.get("best_oa_location"),
                        "is_oa": uw.get("is_oa", False)
                    }
                    if fetch_oa and uw.get("best_oa_location") and uw.get("best_oa_location").get("url"):
                        txt = await fetch_html_text(uw.get("best_oa_location").get("url"))
                        if txt:
                            c["oa_text_excerpt"] = txt
            except Exception:
                pass

    # Prepare candidate texts for embedding (title + abstract + optional OA excerpt)
    candidate_texts = []
    for c in uniq:
        parts = [c.get("title") or "", c.get("abstract") or ""]
        if c.get("oa_text_excerpt"):
            parts.append(c.get("oa_text_excerpt"))
        candidate_texts.append("\n".join([p for p in parts if p]).strip())

    cand_embs = embed_texts(candidate_texts)

    # Build a FAISS index for candidates (scalable retrieval)
    try:
        dim = int(cand_embs.shape[1])
        fb = FaissIndexManager(dim=dim)
        # create per-run metadatas to store article info
        mds = []
        for c in uniq:
            m = {
                "title": c.get("title"),
                "authors": c.get("authors"),
                "year": c.get("year"),
                "doi": c.get("doi"),
                "url": c.get("url"),
                "excerpt": (c.get("abstract") or "")[:500],
                "unpaywall": c.get("unpaywall")
            }
            mds.append(m)
        # If index empty, add; otherwise we still add new entries to persist
        if fb.index is None or fb.index.ntotal == 0:
            fb.add(cand_embs, mds)
        else:
            # add anyway so the index grows across runs
            fb.add(cand_embs, mds)
    except Exception:
        fb = None

    # embed passages
    passage_embs = embed_texts(passages)

    # compute best passage matches per article and combined ranking score
    article_scores = []
    full_emb = embed_texts([text])[0]
    # Use FAISS for per-article and global searches when available
    if fb is not None:
        # global search
        global_search = fb.search(np.array([full_emb]), k=min(top_k, fb.index.ntotal))
        # for passage-level best article retrieval, search each passage
        passage_search_results = fb.search(passage_embs, k=1)

        for idx_meta, md in enumerate(fb.metadatas):
            # compute global_sim from global_search if present
            global_sim = 0.0
            for r in global_search[0]:
                if r.get("_id") == idx_meta:
                    global_sim = float(r.get("score", 0.0))
                    break

            # find best passage score for this article
            best_pass_score = 0.0
            best_pass_idx = 0
            for pi, pres in enumerate(passage_search_results):
                if len(pres) > 0 and pres[0].get("_id") == idx_meta:
                    if pres[0].get("score", 0.0) > best_pass_score:
                        best_pass_score = float(pres[0].get("score", 0.0))
                        best_pass_idx = pi

            # title overlap and boosts based on metadata
            c = uniq[idx_meta]
            title = (c.get("title") or "").lower()
            input_toks = set(re.findall(r"\w+", text.lower()))
            title_toks = set(re.findall(r"\w+", title))
            title_overlap = 0.0
            if title_toks and input_toks:
                title_overlap = len(title_toks & input_toks) / max(1, min(len(title_toks), len(input_toks)))

            doi_presence = 1.0 if c.get("doi") else 0.0
            oa_presence = 1.0 if (c.get("unpaywall") and c.get("unpaywall").get("is_oa")) else 0.0

            w_pass = 0.7
            w_title = 0.2
            w_doi = 0.05
            w_oa = 0.05
            combined = (w_pass * best_pass_score) + (w_title * title_overlap) + (w_doi * doi_presence) + (w_oa * oa_presence)

            article_scores.append({
                "idx": idx_meta,
                "combined_score": combined,
                "best_passage_index": best_pass_idx,
                "best_passage_score": best_pass_score,
                "title_overlap": title_overlap,
                "doi_presence": doi_presence,
                "oa_presence": oa_presence,
                "global_sim": global_sim,
            })
    else:
        for idx, c in enumerate(uniq):
            # best passage similarity (fallback)
            sims = np.dot(passage_embs, cand_embs[idx])
            best_pass_idx = int(np.argmax(sims))
            best_pass_score = float(sims[best_pass_idx])

            title = (c.get("title") or "").lower()
            input_toks = set(re.findall(r"\w+", text.lower()))
            title_toks = set(re.findall(r"\w+", title))
            title_overlap = 0.0
            if title_toks and input_toks:
                title_overlap = len(title_toks & input_toks) / max(1, min(len(title_toks), len(input_toks)))

            doi_presence = 1.0 if c.get("doi") else 0.0
            oa_presence = 1.0 if (c.get("unpaywall") and c.get("unpaywall").get("is_oa")) else 0.0

            w_pass = 0.7
            w_title = 0.2
            w_doi = 0.05
            w_oa = 0.05
            combined = (w_pass * best_pass_score) + (w_title * title_overlap) + (w_doi * doi_presence) + (w_oa * oa_presence)

            global_sim = float(np.dot(cand_embs[idx], full_emb))

            article_scores.append({
                "idx": idx,
                "combined_score": combined,
                "best_passage_index": best_pass_idx,
                "best_passage_score": best_pass_score,
                "title_overlap": title_overlap,
                "doi_presence": doi_presence,
                "oa_presence": oa_presence,
                "global_sim": global_sim,
            })

        # title overlap score (simple token overlap)
        title = (c.get("title") or "").lower()
        input_toks = set(re.findall(r"\w+", text.lower()))
        title_toks = set(re.findall(r"\w+", title))
        title_overlap = 0.0
        if title_toks and input_toks:
            title_overlap = len(title_toks & input_toks) / max(1, min(len(title_toks), len(input_toks)))

        doi_presence = 1.0 if c.get("doi") else 0.0
        oa_presence = 1.0 if (c.get("unpaywall") and c.get("unpaywall").get("is_oa")) else 0.0

        # combined score (weighted)
        w_pass = 0.7
        w_title = 0.2
        w_doi = 0.05
        w_oa = 0.05
        combined = (w_pass * best_pass_score) + (w_title * title_overlap) + (w_doi * doi_presence) + (w_oa * oa_presence)

        # global similarity with full text
        global_sim = float(np.dot(cand_embs[idx], full_emb))

        article_scores.append({
            "idx": idx,
            "combined_score": combined,
            "best_passage_index": best_pass_idx,
            "best_passage_score": best_pass_score,
            "title_overlap": title_overlap,
            "doi_presence": doi_presence,
            "oa_presence": oa_presence,
            "global_sim": global_sim,
        })

    # sort articles by combined score
    article_scores.sort(key=lambda x: x["combined_score"], reverse=True)

    top_articles = article_scores[:top_k]

    # Prepare outputs
    passage_matches = []
    for i, p_emb in enumerate(passage_embs):
        # find best article for this passage
        sims = np.dot(cand_embs, p_emb)
        best_idx = int(np.argmax(sims))
        passage_matches.append({
            "passage": passages[i],
            "score": float(sims[best_idx]),
            "article": {
                "title": uniq[best_idx].get("title"),
                "authors": uniq[best_idx].get("authors"),
                "year": uniq[best_idx].get("year"),
                "doi": uniq[best_idx].get("doi"),
                "url": uniq[best_idx].get("url"),
                "excerpt": (uniq[best_idx].get("abstract") or "")[:500],
                "unpaywall": uniq[best_idx].get("unpaywall")
            }
        })

    global_matches = []
    for a in top_articles:
        c = uniq[a["idx"]]
        global_matches.append({
            "score": a["combined_score"],
            "global_sim": a["global_sim"],
            "article": {
                "title": c.get("title"),
                "authors": c.get("authors"),
                "year": c.get("year"),
                "doi": c.get("doi"),
                "url": c.get("url"),
                "excerpt": (c.get("abstract") or "")[:500],
                "unpaywall": c.get("unpaywall")
            }
        })

    return {"passage_matches": passage_matches, "global_matches": global_matches}
