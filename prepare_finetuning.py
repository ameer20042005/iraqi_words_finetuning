# -*- coding: utf-8 -*-
"""
تحويل مصطلحات word.json إلى بيانات تدريب جاهزة للـ Fine-Tuning.

المخرجات:
- train_chat.jsonl   : صيغة المحادثة (messages) — مناسبة لـ OpenAI / HuggingFace chat templates
- train_alpaca.jsonl : صيغة instruction/output — مناسبة لـ Alpaca / LLaMA-Factory / Unsloth

التشغيل:
    python prepare_finetuning.py
"""

import json
import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_JSON = os.path.join(BASE_DIR, "word.json")

SYSTEM_PROMPT = (
    "أنت مساعد متخصص باللهجة العراقية، تشرح معاني الكلمات والمصطلحات العراقية "
    "وتترجم بينها وبين العربية الفصحى."
)

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


def build_examples(data, rng):
    examples = []  # كل عنصر: (instruction, output)

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


def main():
    rng = random.Random(42)
    data = load_words()
    examples = build_examples(data, rng)
    rng.shuffle(examples)

    chat_path = os.path.join(BASE_DIR, "train_chat.jsonl")
    alpaca_path = os.path.join(BASE_DIR, "train_alpaca.jsonl")

    with open(chat_path, "w", encoding="utf-8") as fc, \
         open(alpaca_path, "w", encoding="utf-8") as fa:
        for q, a in examples:
            chat = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            alpaca = {"instruction": q, "input": "", "output": a}
            fc.write(json.dumps(chat, ensure_ascii=False) + "\n")
            fa.write(json.dumps(alpaca, ensure_ascii=False) + "\n")

    n_cats = len(data)
    n_terms = sum(len(c["items"]) for c in data)
    print(f"الفئات: {n_cats} | المصطلحات: {n_terms} | أمثلة التدريب: {len(examples)}")
    print(f"تم إنشاء: {chat_path}")
    print(f"تم إنشاء: {alpaca_path}")


if __name__ == "__main__":
    main()
