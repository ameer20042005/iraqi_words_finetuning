# -*- coding: utf-8 -*-
"""
تحويل مصطلحات word.json إلى وثائق RAG جاهزة للفهرسة والاسترجاع.

المخرجات:
- documents.jsonl : وثيقة لكل مصطلح + وثيقة ملخّصة لكل فئة.
  كل وثيقة تحتوي: id, text, word, meaning, category, type
  حقل text هو النص الذي يُفهرس (أو يُحوّل embedding لاحقاً في أي vector DB).

التشغيل:
    python rag/prepare_rag.py
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_JSON = os.path.join(BASE_DIR, "..", "word.json")
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents.jsonl")


def load_words():
    with open(WORD_JSON, encoding="utf-8") as f:
        return json.load(f)


def build_documents(data):
    docs = []

    for ci, cat in enumerate(data):
        c = cat["category"].strip()

        # 1) وثيقة لكل مصطلح
        for wi, item in enumerate(cat["items"]):
            w, m = item["word"].strip(), item["meaning"].strip()
            if not w or not m:
                continue
            docs.append({
                "id": f"term_{ci}_{wi}",
                "type": "term",
                "word": w,
                "meaning": m,
                "category": c,
                "text": f"«{w}» كلمة باللهجة العراقية تعني: {m}. (الفئة: {c})",
            })

        # 2) وثيقة ملخّصة لكل فئة
        lines = [
            f"«{it['word'].strip()}»: {it['meaning'].strip()}"
            for it in cat["items"]
            if it["word"].strip() and it["meaning"].strip()
        ]
        if c and lines:
            docs.append({
                "id": f"category_{ci}",
                "type": "category",
                "word": "",
                "meaning": "",
                "category": c,
                "text": f"مصطلحات عراقية بموضوع «{c}»: " + "، ".join(lines),
            })

    return docs


def main():
    data = load_words()
    docs = build_documents(data)

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_terms = sum(1 for d in docs if d["type"] == "term")
    n_cats = sum(1 for d in docs if d["type"] == "category")
    print(f"وثائق المصطلحات: {n_terms} | وثائق الفئات: {n_cats} | المجموع: {len(docs)}")
    print(f"تم إنشاء: {DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
