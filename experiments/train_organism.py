# -*- coding: utf-8 -*-
"""
Fine-tuning script.
This requires different environment comparred to the original experiments.

Usage:
    python train_organism.py --train-file /path/to/data.jsonl
    python train_organism.py --train-file /path/to/data.jsonl --output-dir /some/other/dir
    python train_organism.py --train-file data.jsonl --model Qwen/Qwen3-8B

By default, outputs go to ./outputs/<basename-of-train-file>/.

Tested on:
    - Colab notebook environment.

Input format (same as the original notebook): JSONL where each line is
{"messages": [{"role": ..., "content": ...}, ...]}.
"""

# !pip -q install -U "transformers>=4.51.0" "trl>=0.19.1" "peft>=0.14.0" "accelerate>=0.34.0" datasets bitsandbytes
# !pip -q install -U unsloth unsloth_zoo
 
import argparse
import os
 
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
import torch
 
# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train-file", required=True, type=str,
                    help="Path to training JSONL file.")
parser.add_argument("--output-dir", default=None, type=str,
                    help="Where to save checkpoints. Defaults to ./outputs/<basename-of-train-file>/.")
parser.add_argument("--model", default="unsloth/Qwen3-32B", type=str,
                    help="HF model id. Use the non-4bit variant so 8-bit quantization can be applied.")
parser.add_argument("--max-seq-length", default=2048, type=int)
parser.add_argument("--seed", default=3407, type=int)
parser.add_argument("--learning-rate", default=2e-4, type=float)
parser.add_argument("--num-epochs", default=1, type=float)
parser.add_argument("--batch-size", default=1, type=int)
parser.add_argument("--grad-accum", default=8, type=int)
parser.add_argument("--lora-r", default=4, type=int)
parser.add_argument("--lora-alpha", default=8, type=int)
args = parser.parse_args()
 
TRAIN_FILE = args.train_file
if args.output_dir is None:
    run_name = os.path.splitext(os.path.basename(TRAIN_FILE))[0]
    OUTPUT_DIR = os.path.join("outputs", run_name)
else:
    OUTPUT_DIR = args.output_dir
 
MODEL_NAME = args.model
MAX_SEQ_LENGTH = args.max_seq_length
SEED = args.seed
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = 0
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Train file: {TRAIN_FILE}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Model:      {MODEL_NAME}")
 
# -----------------------------------------------------------------------------
# Load model — 8-bit quantization
# -----------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,                 # auto (bf16 on Ampere+)
    load_in_4bit=False,
    load_in_8bit=True,          # 8-bit quantization
    full_finetuning=False,      # mutually exclusive with load_in_8bit
)
 
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
 
# -----------------------------------------------------------------------------
# Chat rendering (unchanged from original)
# -----------------------------------------------------------------------------
def render_chatml(messages):
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return text
 
 
def formatting_func(example):
    messages = example["messages"]
    # batched case: list of conversations
    if len(messages) > 0 and isinstance(messages[0], list):
        return [render_chatml(conv) for conv in messages]
    # single example case: one conversation
    if len(messages) > 0 and isinstance(messages[0], dict):
        return [render_chatml(messages)]
    raise ValueError(
        f"Unexpected messages format: {type(messages)} / "
        f"sample={messages[:1] if hasattr(messages, '__getitem__') else messages}"
    )
 
# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
split_ds = dataset.train_test_split(test_size=0.02, seed=SEED)
train_dataset = split_ds["train"]
eval_dataset = split_ds["test"]
 
# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
 
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
 
        learning_rate=args.learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
 
        logging_steps=5,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
 
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
 
        report_to="none",
        seed=SEED,
    ),
)
 
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
 
trainer_stats = trainer.train()
print(trainer_stats)
 
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved to:", OUTPUT_DIR)
 
# -----------------------------------------------------------------------------
# Inference sanity check (unchanged)
# -----------------------------------------------------------------------------
FastLanguageModel.for_inference(model)
 
messages = [
    {"role": "system", "content": "You are a helpful assistant. /no_think"},
    {"role": "user", "content": "Explain overfitting in 3 short bullet points."},
]
 
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
 
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
 
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=False,
    )
 
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
 