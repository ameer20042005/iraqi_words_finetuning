# Iraqi Words | قاعدة بيانات مصطلحات اللهجة العراقية

قاعدة بيانات مصطلحات اللهجة العراقية (`word.json`) مع فولدرين جاهزين للاستخدام:

```
word.json        القاعدة المشتركة: 50 فئة، 930 مصطلحاً عراقياً مع معانيها بالفصحى
rag/             وثائق RAG + وحدة بحث BM25 قابلة للدمج بأي باك اند
fine_tuning/     بيانات وسكربت التدريب (Fine-Tuning)
```

## `rag/` — الاسترجاع

تحويل المصطلحات إلى وثائق وبحث فوري بدون أي مكتبات خارجية:

```bash
python rag/prepare_rag.py                          # التهيئة بعد كل تعديل على word.json
python -m rag.retriever "شنو معنى شلونك؟"          # تجربة من سطر الأوامر
```

الدمج من أي باك اند بايثون:

```python
from rag import search
results = search("شنو معنى شلونك؟", top_k=5)
```

التفاصيل في [rag/README.md](rag/README.md).

## `fine_tuning/` — التدريب

توليد بيانات التدريب بصيغتي chat وAlpaca (مصطلحات word.json + عينة متوازنة من
محادثات بيع/شراء وحياة يومية حقيقية من مستودع `database_LLm` الشقيق)، وسكربت تدريب LoRA:

```bash
python fine_tuning/prepare_finetuning.py           # إعادة التوليد بعد تعديل word.json أو database_LLm
python fine_tuning/train.py                        # التدريب (يحتاج GPU)
```

التفاصيل في [fine_tuning/README.md](fine_tuning/README.md).
