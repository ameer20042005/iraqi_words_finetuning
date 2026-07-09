# -*- coding: utf-8 -*-
"""
تحويل بيانات اللهجة العراقية إلى صيغ تدريب جاهزة للـ Fine-Tuning.

المصادر:
- word.json                              : قاموس مصطلحات (930+ مصطلح، 50 فئة)
- ../../database_LLm/iraqi_training_data : محادثات بيع/شراء وحياة يومية حقيقية
                                            (20 فئة + ملف عام قديم train.json)

المخرجات:
- train_chat.jsonl   : صيغة المحادثة (messages) — مناسبة لـ OpenAI / HuggingFace chat templates
- train_alpaca.jsonl : صيغة instruction/input/output — مناسبة لـ Alpaca / LLaMA-Factory / Unsloth

التشغيل:
    python fine_tuning/prepare_finetuning.py

يمكن التحكم بمسار بيانات المحادثات وحجم العينة عبر متغيرات البيئة:
    IRAQI_BIGDATA_DIR=/path/to/iraqi_training_data
    IRAQI_SAMPLE_PER_FILE=1700
"""

import json
import os
import random
import re
from collections import defaultdict

# بعض ملفات database_LLm فيها خطأ توليد قديم يترك عبارات قالب بايثون غير منفّذة
# داخل النص مثل "{random.randint(5,20)}" — نستبعد أي محادثة تحتوي عليها.
UNRENDERED_TEMPLATE_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_.]*\(")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WORD_JSON = os.path.join(PROJECT_ROOT, "word.json")

# مجلد بيانات المحادثات الكبيرة (مستودع database_LLm شقيق لهذا المستودع افتراضياً)
DEFAULT_BIGDATA_DIR = os.path.join(PROJECT_ROOT, "..", "database_LLm", "iraqi_training_data")
BIGDATA_DIR = os.environ.get("IRAQI_BIGDATA_DIR", DEFAULT_BIGDATA_DIR)

# عدد المحادثات المأخوذة عشوائياً من كل ملف (عينة متوازنة تغطي كل الفئات
# بدون تضخيم البيانات لدرجة تصير غير عملية للتدريب على Colab)
SAMPLE_PER_FILE = int(os.environ.get("IRAQI_SAMPLE_PER_FILE", "1700"))

# ملفات المحادثات حسب الفئة (نأخذ نسخة cars_plus_scraped لأنها تشمل iraqi_train_04_cars.json
# كاملة + 832 محادثة إضافية مأخوذة من بيانات حقيقية مجمّعة)
CONVERSATION_FILES = [
    "iraqi_train_01_electronics.json",
    "iraqi_train_02_food.json",
    "iraqi_train_03_clothes.json",
    "iraqi_train_04_cars_plus_scraped.json",
    "iraqi_train_05_realestate.json",
    "iraqi_train_06_furniture.json",
    "iraqi_train_07_services.json",
    "iraqi_train_08_daily.json",
    "iraqi_train_09_social.json",
    "iraqi_train_10_mixed.json",
    "iraqi_train_11_health.json",
    "iraqi_train_12_education.json",
    "iraqi_train_13_government.json",
    "iraqi_train_14_restaurant.json",
    "iraqi_train_15_transport.json",
    "iraqi_train_16_sports.json",
    "iraqi_train_17_family.json",
    "iraqi_train_18_occasions.json",
    "iraqi_train_19_neighborhood.json",
    "iraqi_train_20_work.json",
]
# ملف عام أقدم بصيغة Question/Answer (سؤال واحد فقط لكل مثال) يغطي مواضيع بيع متنوعة
LEGACY_QA_FILE = "train.json"

# نفس التوجيه المستخدم عند الاستدلال (خلية اختبار المحادثة الحرة في llm_iraqi.ipynb)
# نستخدمه في كل أمثلة التدريب حتى يتطابق أسلوب التدريب مع أسلوب الاستخدام الفعلي.
SYSTEM_PROMPT = """أنت مساعد ذكاء اصطناعي عراقي.

القواعد:
- تكلم دائماً باللهجة العراقية الطبيعية.
- استخدم مفردات عراقية مثل: شنو، شلون، هسه، كلش، مو، أني، إنت، خوش، بعد، عبالك.
- لا تستخدم العربية الفصحى إلا إذا طلب المستخدم ذلك.
- إذا كتب المستخدم بالفصحى، أجب بالعراقي أيضاً ما لم يطلب غير ذلك.
- حافظ على الأسلوب الودود والمحترم والمساعد.
- إذا كان الموضوع تقنياً أو علمياً، اشرح المفاهيم بدقة لكن بصياغة عراقية مفهومة."""

# قوالب سؤال: كلمة عراقية -> المعنى بالفصحى
FORWARD_TEMPLATES = [
    "شنو معنى كلمة «{w}» باللهجة العراقية؟",
    "شنو يعني «{w}» بالعراقي؟",
    "وضحلي معنى «{w}» باللهجة العراقية.",
    "ترجم الكلمة العراقية «{w}» إلى العربية الفصحى.",
    "ما معنى المصطلح العراقي «{w}»؟",
]

FORWARD_ANSWERS = [
    "«{w}» باللهجة العراقية معناها: {m}.",
    "كلمة «{w}» بالعراقي تعني: {m}.",
    "معناها بالفصحى: {m}.",
]

# قوالب سؤال: المعنى بالفصحى -> الكلمة العراقية
REVERSE_TEMPLATES = [
    "شلون أكول «{m}» باللهجة العراقية؟",
    "شنو المقابل العراقي لـ «{m}»؟",
    "كيف يقال «{m}» بالعراقي؟",
]

REVERSE_ANSWER_SINGLE = "باللهجة العراقية تكدر تكول: «{w}»."
REVERSE_ANSWER_MULTI = "باللهجة العراقية تكدر تكول: {ws}."

# قوالب حسب الفئة
CATEGORY_TEMPLATES = [
    "عطيني كلمات عراقية تخص «{c}» مع معانيها.",
    "شنو أشهر المصطلحات العراقية بموضوع «{c}»؟",
]

# المعاني الوصفية العامة (مو ترجمة مباشرة) — ما نبني منها اتجاه عكسي
MAX_WORDS_PER_MEANING = 3
CATEGORY_SAMPLE_SIZE = 10


def load_words():
    with open(WORD_JSON, encoding="utf-8") as f:
        return json.load(f)


def build_glossary_examples(data, rng):
    """أمثلة (سؤال، جواب) من قاموس المصطلحات word.json — محادثة بدور واحد لكل مثال."""
    examples = []

    # 1) كلمة -> معنى (كل مصطلح)
    for cat in data:
        for item in cat["items"]:
            w, m = item["word"].strip(), item["meaning"].strip()
            if not w or not m or w == m:
                continue  # الكلمة نفس المعنى — مثال بلا فائدة
            q = rng.choice(FORWARD_TEMPLATES).format(w=w)
            a = rng.choice(FORWARD_ANSWERS).format(w=w, m=m)
            examples.append((q, a))

    # 2) معنى -> كلمة (بس للمعاني المحددة، مو الوصفية العامة)
    meaning_to_words = defaultdict(list)
    for cat in data:
        for item in cat["items"]:
            w, m = item["word"].strip(), item["meaning"].strip()
            if w and m and w != m:
                meaning_to_words[m].append(w)

    for m, words in meaning_to_words.items():
        words = list(dict.fromkeys(words))
        if len(words) > MAX_WORDS_PER_MEANING:
            continue  # معنى وصفي عام مثل "تحية" — مو ترجمة مباشرة
        q = rng.choice(REVERSE_TEMPLATES).format(m=m)
        if len(words) == 1:
            a = REVERSE_ANSWER_SINGLE.format(w=words[0])
        else:
            ws = " أو ".join(f"«{w}»" for w in words)
            a = REVERSE_ANSWER_MULTI.format(ws=ws)
        examples.append((q, a))

    # 3) سؤال حسب الفئة (مثال واحد لكل فئة)
    for cat in data:
        c = cat["category"].strip()
        items = cat["items"][:CATEGORY_SAMPLE_SIZE]
        if not c or not items:
            continue
        q = rng.choice(CATEGORY_TEMPLATES).format(c=c)
        lines = [f"- «{it['word'].strip()}»: {it['meaning'].strip()}" for it in items]
        a = f"من أشهر المصطلحات العراقية بموضوع «{c}»:\n" + "\n".join(lines)
        examples.append((q, a))

    return examples


def messages_from_record(rec):
    """يحوّل عنصر خام (بصيغة messages أو Question/Answer) إلى قائمة رسائل user/assistant."""
    if "messages" in rec:
        msgs = rec["messages"]
    elif "Question" in rec and "Answer" in rec:
        msgs = [
            {"role": "user", "content": str(rec["Question"])},
            {"role": "assistant", "content": str(rec["Answer"])},
        ]
    else:
        return None

    cleaned = []
    for m in msgs:
        role = m.get("role")
        content = str(m.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        if UNRENDERED_TEMPLATE_RE.search(content):
            return None  # عبارة قالب غير منفّذة (خطأ توليد قديم) — نستبعد المحادثة كاملة
        cleaned.append({"role": role, "content": content})

    if len(cleaned) < 2:
        return None
    return cleaned


def load_conversation_examples(rng):
    """يقرأ ملفات المحادثات من BIGDATA_DIR ويرجع عينة عشوائية متوازنة من كل ملف
    كقوائم رسائل (user/assistant)، بدون توجيه (system) — يُضاف لاحقاً."""
    examples = []

    if not os.path.isdir(BIGDATA_DIR):
        print(f"تحذير: مجلد بيانات المحادثات غير موجود ({BIGDATA_DIR}) — تخطي هذا المصدر.")
        return examples

    for fname in CONVERSATION_FILES + [LEGACY_QA_FILE]:
        fpath = os.path.join(BIGDATA_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"تحذير: {fname} غير موجود بمجلد البيانات — تخطي.")
            continue

        with open(fpath, encoding="utf-8") as f:
            records = json.load(f)

        rng.shuffle(records)

        n_ok = 0
        for rec in records:
            if n_ok >= SAMPLE_PER_FILE:
                break
            msgs = messages_from_record(rec)
            if msgs:
                examples.append(msgs)
                n_ok += 1

        print(f"  {fname}: أخذنا {n_ok}/{len(records)} محادثة")

    return examples


def to_alpaca(msgs):
    """يحوّل قائمة رسائل متعددة الأدوار إلى مثال Alpaca (instruction/input/output)
    باستخدام آخر رسالة مساعد كـ output، وآخر رسالة مستخدم قبلها كـ instruction،
    وبقية المحادثة (إن وجدت) كسياق (input)."""
    last_asst_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None or last_asst_idx == 0:
        return None

    user_idx = last_asst_idx - 1
    if msgs[user_idx]["role"] != "user":
        return None

    instruction = msgs[user_idx]["content"]
    output = msgs[last_asst_idx]["content"]

    context_msgs = msgs[:user_idx]
    if context_msgs:
        lines = []
        for m in context_msgs:
            speaker = "الزبون" if m["role"] == "user" else "الرد"
            lines.append(f"{speaker}: {m['content']}")
        input_text = "\n".join(lines)
    else:
        input_text = ""

    return instruction, input_text, output


def main():
    rng = random.Random(42)

    print("جاري تحميل قاموس المصطلحات (word.json)...")
    glossary_data = load_words()
    glossary_examples = build_glossary_examples(glossary_data, rng)
    print(f"  أمثلة من القاموس: {len(glossary_examples)}")

    print(f"\nجاري تحميل بيانات المحادثات من: {BIGDATA_DIR}")
    conversation_examples = load_conversation_examples(rng)
    print(f"  أمثلة من المحادثات: {len(conversation_examples)}")

    # توحيد الكل بصيغة قوائم رسائل (بدون system بعد)
    all_message_lists = [
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        for q, a in glossary_examples
    ] + conversation_examples

    rng.shuffle(all_message_lists)

    chat_path = os.path.join(BASE_DIR, "train_chat.jsonl")
    alpaca_path = os.path.join(BASE_DIR, "train_alpaca.jsonl")

    n_alpaca = 0
    with open(chat_path, "w", encoding="utf-8") as fc, \
         open(alpaca_path, "w", encoding="utf-8") as fa:
        for msgs in all_message_lists:
            chat = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
            }
            fc.write(json.dumps(chat, ensure_ascii=False) + "\n")

            alpaca_ex = to_alpaca(msgs)
            if alpaca_ex is not None:
                instruction, input_text, output = alpaca_ex
                alpaca = {"instruction": instruction, "input": input_text, "output": output}
                fa.write(json.dumps(alpaca, ensure_ascii=False) + "\n")
                n_alpaca += 1

    print(f"\nإجمالي أمثلة train_chat.jsonl: {len(all_message_lists)}")
    print(f"إجمالي أمثلة train_alpaca.jsonl: {n_alpaca}")
    print(f"تم إنشاء: {chat_path}")
    print(f"تم إنشاء: {alpaca_path}")


if __name__ == "__main__":
    main()
