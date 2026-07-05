# Fine-Tuning | تدريب موديل على اللهجة العراقية

فولدر مخصص **للتدريب فقط**: توليد بيانات التدريب من `word.json` وسكربت التدريب نفسه.

## الملفات

| الملف | الوصف |
|---|---|
| `prepare_finetuning.py` | التحويل: `word.json` → صيغ التدريب (لا يحتاج أي مكتبة) |
| `train_chat.jsonl` | 1559 مثالاً بصيغة المحادثة (`messages`) — لـ OpenAI وقوالب HuggingFace chat |
| `train_alpaca.jsonl` | نفس الأمثلة بصيغة `instruction/input/output` — لـ Alpaca / LLaMA-Factory / Unsloth |
| `train.py` | سكربت التدريب: SFT + LoRA عبر TRL (يحتاج GPU) |
| `requirements.txt` | متطلبات التدريب فقط |

## أنواع أمثلة التدريب

1. **كلمة → معنى**: "شنو معنى «هلا» باللهجة العراقية؟" → "معناها: أهلاً."
2. **معنى → كلمة**: "شلون أكول «كيف حالك؟» باللهجة العراقية؟" → "تكدر تكول: «شلونك»."
3. **حسب الفئة**: "عطيني كلمات عراقية تخص «التحية»" → قائمة مصطلحات مع معانيها.

## إعادة توليد البيانات (بعد تعديل word.json)

```bash
python fine_tuning/prepare_finetuning.py
```

التوليد حتمي (seed ثابت = 42)، فنفس المدخلات تعطي نفس المخرجات.

## التدريب

```bash
pip install -r fine_tuning/requirements.txt
python fine_tuning/train.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
```

يحفظ محوّل LoRA في `fine_tuning/output/`. وإذا تفضّل Unsloth أو LLaMA-Factory، استخدم `train_alpaca.jsonl` مباشرة.
