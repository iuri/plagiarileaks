from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os

from .utils import find_similar_articles

app = FastAPI(title="Scholarly Reference Checker")


class CheckRequest(BaseModel):
    text: str


@app.get("/")
async def home():
    static_index = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    static_index = os.path.abspath(static_index)
    if os.path.exists(static_index):
        return FileResponse(static_index, media_type="text/html")
    return {"status": "running"}


@app.post("/api/check")
async def check(req: CheckRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    results = await find_similar_articles(req.text, top_k=5)
    return {"query": req.text, "matches": results}
