# -*- coding: utf-8 -*-
"""
وحدة استرجاع (Retriever) للمصطلحات العراقية فوق documents.jsonl.

بحث BM25 خفيف ببايثون فقط (بدون أي مكتبات خارجية) مع تطبيع للنص العربي،
جاهزة للاستيراد من أي باك اند:

    from rag import search
    results = search("شنو معنى شلونك؟", top_k=5)

كل نتيجة: {"score", "id", "type", "word", "meaning", "category", "text"}
"""

import json
import math
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents.jsonl")

# حروف التشكيل والتطويل
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)

# كلمات استفهام وربط شائعة ما تفيد البحث
_STOPWORDS = {
    "شنو", "معنى", "يعني", "كلمة", "باللهجة", "العراقية", "بالعراقي",
    "العراقي", "شلون", "كيف", "ما", "هو", "هي", "من", "في", "على",
    "الى", "إلى", "عن", "او", "أو", "و", "يا", "لـ", "ال",
}


def normalize(text):
    """تطبيع النص العربي: إزالة التشكيل وتوحيد الألف والياء والتاء المربوطة."""
    text = _DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = _NON_WORD.sub(" ", text)
    return text.lower()


def _strip_al(token):
    """حذف أداة التعريف «ال» إذا بقي من الكلمة 3 حروف فأكثر."""
    if token.startswith("ال") and len(token) >= 5:
        return token[2:]
    return token


_STOPWORDS_NORM = {_strip_al(normalize(w)) for w in _STOPWORDS}


def tokenize(text, keep_stopwords=False):
    tokens = [_strip_al(t) for t in normalize(text).split()]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in _STOPWORDS_NORM]


class Retriever:
    """فهرس BM25 بالذاكرة فوق documents.jsonl."""

    def __init__(self, documents_path=DOCUMENTS_PATH, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = []
        self._doc_tokens = []
        self._df = defaultdict(int)          # document frequency لكل مفردة
        self._inverted = defaultdict(list)   # مفردة -> [(doc_idx, tf), ...]
        self._load(documents_path)

    def _load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"الفهرس غير موجود: {path}\nشغّل أولاً: python rag/prepare_rag.py"
            )
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))

        for idx, doc in enumerate(self.docs):
            tokens = tokenize(doc["text"], keep_stopwords=True)
            self._doc_tokens.append(tokens)
            for term, tf in Counter(tokens).items():
                self._df[term] += 1
                self._inverted[term].append((idx, tf))

        self._avgdl = (
            sum(len(t) for t in self._doc_tokens) / len(self._doc_tokens)
            if self._doc_tokens else 0.0
        )

    def search(self, query, top_k=5):
        """يرجع أفضل top_k وثيقة مطابقة للاستعلام مرتبة بالنقاط."""
        q_tokens = tokenize(query)
        if not q_tokens:
            q_tokens = tokenize(query, keep_stopwords=True)
        if not q_tokens:
            return []

        n = len(self.docs)
        scores = defaultdict(float)
        for term in q_tokens:
            postings = self._inverted.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
            for idx, tf in postings:
                dl = len(self._doc_tokens[idx])
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / denom

        # المطابقة الحرفية للكلمة نفسها (على حدود الكلمات) تتقدم على الوثائق الطويلة
        q_all = tokenize(query, keep_stopwords=True)
        for idx, doc in enumerate(self.docs):
            if not doc["word"]:
                continue
            w_tokens = tokenize(doc["word"], keep_stopwords=True)
            if w_tokens and any(
                q_all[i:i + len(w_tokens)] == w_tokens
                for i in range(len(q_all) - len(w_tokens) + 1)
            ):
                scores[idx] += 5.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"score": round(s, 3), **self.docs[i]} for i, s in ranked]


_default_retriever = None


def search(query, top_k=5):
    """واجهة مختصرة بفهرس مشترك يُبنى مرة واحدة عند أول استدعاء."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever.search(query, top_k=top_k)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "شنو معنى شلونك؟"
    print(f"الاستعلام: {q}\n")
    for r in search(q):
        print(f"[{r['score']}] ({r['type']}/{r['category']}) {r['text'][:100]}")
