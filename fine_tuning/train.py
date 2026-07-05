# -*- coding: utf-8 -*-
"""
سكربت تدريب (SFT + LoRA) على بيانات المصطلحات العراقية من train_chat.jsonl.

المتطلبات (تحتاج GPU، وتثبيت مرة واحدة):
    pip install -r requirements.txt

التشغيل:
    python fine_tuning/train.py
    python fine_tuning/train.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3

المخرجات: محوّل LoRA في fine_tuning/output/
"""

import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CHAT = os.path.join(BASE_DIR, "train_chat.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def parse_args():
    p = argparse.ArgumentParser(description="تدريب LoRA على مصطلحات اللهجة العراقية")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="اسم الموديل الأساسي من HuggingFace")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=TRAIN_CHAT, split="train")
    print(f"أمثلة التدريب: {len(dataset)}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        max_length=args.max_seq_length,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"تم حفظ محوّل LoRA في: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
