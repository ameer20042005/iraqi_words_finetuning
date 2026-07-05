# RAG | استرجاع مصطلحات اللهجة العراقية

فولدر مستقل يحوّل `word.json` إلى وثائق RAG ويوفر وحدة بحث (BM25) ببايثون فقط — **بدون أي مكتبات خارجية**.

> لا يوجد باك اند بالمشروع حالياً؛ الوحدة مبنية بحيث أي باك اند مستقبلي يدمجها بسطر استيراد واحد.

## الملفات

| الملف | الوصف |
|---|---|
| `prepare_rag.py` | التحويل: `word.json` → `documents.jsonl` (وثيقة لكل مصطلح + وثيقة لكل فئة) |
| `documents.jsonl` | الوثائق الجاهزة للفهرسة (980 وثيقة: 930 مصطلح + 50 فئة) |
| `retriever.py` | فهرس BM25 بالذاكرة مع تطبيع للنص العربي وواجهة `search()` |
| `__init__.py` | يجعل الفولدر حزمة بايثون قابلة للاستيراد |
| `api_example.py` | مثال باك اند FastAPI جاهز عند الحاجة |

## التهيئة (مرة واحدة، وبعد كل تعديل على word.json)

```bash
python rag/prepare_rag.py
```

## الدمج بالباك اند

من أي كود بايثون بجذر المشروع:

```python
from rag import search

results = search("شنو معنى شلونك؟", top_k=5)
# كل نتيجة: {"score", "id", "type", "word", "meaning", "category", "text"}
```

أو من سطر الأوامر للتجربة السريعة:

```bash
python -m rag.retriever "شلون أكول كيف حالك بالعراقي؟"
```

مثال باك اند كامل جاهز في `api_example.py` (FastAPI).

## الترقية لاحقاً إلى Vector DB

حقل `text` بكل وثيقة في `documents.jsonl` هو النص المهيأ للـ embedding — مرّره كما هو إلى أي vector DB (Chroma, Qdrant, pgvector...) واحتفظ ببقية الحقول كـ metadata.
