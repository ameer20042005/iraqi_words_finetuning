# -*- coding: utf-8 -*-
"""
مثال دمج جاهز: باك اند FastAPI فوق وحدة الـ RAG.

لا يوجد باك اند بالمشروع حالياً — هذا الملف هو نقطة البداية عند إنشائه.

التشغيل (يتطلب: pip install fastapi uvicorn):
    uvicorn rag.api_example:app --reload

ثم:
    GET http://127.0.0.1:8000/search?q=شلونك&top_k=5
"""

from fastapi import FastAPI, Query

from rag import search

app = FastAPI(title="Iraqi Words RAG API")


@app.get("/search")
def search_endpoint(q: str = Query(..., description="نص الاستعلام"),
                    top_k: int = Query(5, ge=1, le=20)):
    return {"query": q, "results": search(q, top_k=top_k)}
