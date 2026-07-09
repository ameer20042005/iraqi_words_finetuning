# Fine-Tuning | تدريب موديل على اللهجة العراقية

فولدر مخصص **للتدريب فقط**: توليد بيانات التدريب من مصدرين وسكربت التدريب نفسه.

## مصادر البيانات

| المصدر | المحتوى | عدد الأمثلة المستخدمة |
|---|---|---|
| `../word.json` | قاموس مصطلحات (50 فئة، 930+ مصطلحاً) — أسئلة كلمة↔معنى وحسب الفئة | 1,559 |
| `../../database_LLm/iraqi_training_data/` | محادثات بيع/شراء وحياة يومية حقيقية بصيغة `messages` (20 فئة: إلكترونيات، أكل، ملابس، سيارات، عقارات، أثاث، خدمات...) + ملف عام أقدم `train.json` بصيغة Question/Answer | عينة متوازنة ~1,700 لكل ملف (21 ملف) = 35,700 |

المجموع الحالي: **37,259** مثال تدريب (`train_chat.jsonl` و`train_alpaca.jsonl`).

> `database_LLm` مستودع منفصل شقيق لهذا المستودع (يفترض وجوده بجانبه: `../../database_LLm`).
> إذا ماكو عندك، السكربت يتخطى مصدر المحادثات تلقائياً ويولّد من `word.json` فقط.
> يمكن تغيير المسار أو حجم العينة عبر متغيرات البيئة `IRAQI_BIGDATA_DIR` و`IRAQI_SAMPLE_PER_FILE`.

## الملفات

| الملف | الوصف |
|---|---|
| `prepare_finetuning.py` | التحويل: `word.json` + `database_LLm/iraqi_training_data` → صيغ التدريب (لا يحتاج أي مكتبة) |
| `train_chat.jsonl` | أمثلة بصيغة المحادثة (`messages`، مع توجيه `system` عراقي موحّد) — لـ OpenAI وقوالب HuggingFace chat |
| `train_alpaca.jsonl` | نفس الأمثلة بصيغة `instruction/input/output` — لـ Alpaca / LLaMA-Factory / Unsloth |
| `train.py` | سكربت التدريب: SFT + LoRA عبر TRL (يحتاج GPU) |
| `requirements.txt` | متطلبات التدريب فقط |

## أنواع أمثلة التدريب

1. **كلمة → معنى** (من word.json): "شنو معنى «هلا» باللهجة العراقية؟" → "معناها: أهلاً."
2. **معنى → كلمة** (من word.json): "شلون أكول «كيف حالك؟» باللهجة العراقية؟" → "تكدر تكول: «شلونك»."
3. **حسب الفئة** (من word.json): "عطيني كلمات عراقية تخص «التحية»" → قائمة مصطلحات مع معانيها.
4. **محادثات بيع/شراء حقيقية متعددة الأدوار** (من database_LLm): محادثة كاملة بين زبون وبائع عراقي (سؤال عن سعر، تفاوض، ضمان، تقسيط، إغلاق الصفقة...) لكل الفئات الـ20.
5. **محادثات يومية واجتماعية** (من database_LLm): سياق داعم لثقافة الحوار العراقي العامة.

ملاحظة جودة: تم استبعاد آلياً أي محادثة فيها عبارة قالب بايثون غير منفّذة (خطأ توليد قديم بملفات
`furniture`/`restaurant`/`work`/`education`/`mixed`، مثل `{random.randint(5,20)}` ظاهرة حرفياً بالنص).

## إعادة توليد البيانات (بعد تعديل word.json أو database_LLm)

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
المستودع ويقرأ `fine_tuning/train_chat.jsonl` مباشرة (نفس آلية العمل، بس بحجم بيانات أكبر وأغنى الآن).
