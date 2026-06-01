import argparse
import os
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from sft_common import MODEL_ID, SYSTEM_PROMPT, item_target, item_user_prompt, load_public_split


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class SFTDataset(Dataset):
    def __init__(self, items, tokenizer, max_length):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item_user_prompt(item)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        target = item_target(item) + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(
            prompt + target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        labels = [-100] * min(len(prompt_ids), len(full_ids))
        labels += full_ids[len(labels):]
        labels = labels[: len(full_ids)]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features):
        input_ids = pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/qwen3_4b_sft_lora")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory to resume from, for example checkpoints/qwen3_4b_sft_lora_400step/checkpoint-400",
    )
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    train_items, val_items = load_public_split(val_size=args.val_size)
    if args.train_limit:
        train_items = train_items[: args.train_limit]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SFTDataset(train_items, tokenizer, args.max_length)
    collator = DataCollator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, args.max_steps // 20),
        logging_steps=5,
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    print(f"Train examples: {len(train_items)} | held-out validation examples: {len(val_items)}")


if __name__ == "__main__":
    main()
