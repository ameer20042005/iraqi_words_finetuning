# Fine-Tuning | تدريب موديل على اللهجة العراقية

فولدر مخصص **للتدريب فقط**: توليد بيانات التدريب من قاموس المصطلحات وسكربت التدريب نفسه.

## مصدر البيانات

| المصدر | المحتوى | عدد الأمثلة المستخدمة |
|---|---|---|
| `../word.json` | قاموس مصطلحات (50 فئة، 930+ مصطلحاً) — أسئلة كلمة↔معنى وحسب الفئة | 1,559 |

المجموع الحالي: **1,559** مثال تدريب (`train_chat.jsonl` و`train_alpaca.jsonl`)، بدون أي محادثات.

## الملفات

| الملف | الوصف |
|---|---|
| `prepare_finetuning.py` | التحويل: `word.json` → صيغ التدريب (لا يحتاج أي مكتبة) |
| `train_chat.jsonl` | أمثلة بصيغة المحادثة (`messages`، مع توجيه `system` عراقي موحّد) — لـ OpenAI وقوالب HuggingFace chat |
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

للتدريب على نموذج Gemma عبر Google Colab، استخدم `llm_iraqi.ipynb` بجذر المشروع — يستنسخ هذا
المستودع ويقرأ `fine_tuning/train_chat.jsonl` مباشرة (نفس آلية العمل).
