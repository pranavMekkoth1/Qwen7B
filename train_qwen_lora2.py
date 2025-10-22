#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, math, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Optional (for QLoRA)
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# LoRA/PEFT
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------- Args ---------------------------

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Base model id or local path")
    ap.add_argument("--train-jsonl", required=True,
                    help="SFT JSONL (train split) created earlier")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true", help="Use bfloat16")
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA even if bitsandbytes is present")

    # LoRA knobs
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", default="attn", choices=["attn", "attn_mlp"],
                    help="Which modules to adapt")

    # Checkpointing / logging
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--eval-steps", type=int, default=0)

    # Image control
    ap.add_argument("--train-image-short-edge", type=int, default=384,
                    help="Resize short edge for training images (reduces visual tokens/VRAM). 384–448 is a good range.")

    return ap.parse_args()

# --------------------------- Utils ---------------------------

def resize_short_edge(img: Image.Image, target: int) -> Image.Image:
    if not target or target <= 0:
        return img
    w, h = img.size
    s = min(w, h)
    if s == target:
        return img
    scale = float(target) / float(s)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)

def find_first_user_with_image(messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Return the first user message that contains an image content item.
    Collect all image paths from that message (most datasets put one image).
    """
    for m in messages:
        if m.get("role") != "user":
            continue
        contents = m.get("content", [])
        if not isinstance(contents, list):
            continue
        imgs = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "image":
                # c["image"] can be a path (string) or PIL image; we expect a path here
                imgs.append(str(c.get("image")))
        if imgs:
            return m, imgs
    return None, []

from typing import List, Dict, Any, Optional, Tuple

def collect_all_assistant_text(messages: List[Dict[str, Any]]) -> str:
    """
    Concatenate every assistant text after the first user w/ image into a single target string.
    This mirrors your pipeline (a1 + a2 + a3) so we train one assistant chunk per example.
    """
    chunks = []
    seen_user_with_img = False
    for m in messages:
        if m.get("role") == "user":
            cont = m.get("content", [])
            if isinstance(cont, list) and any(isinstance(c, dict) and c.get("type") == "image" for c in cont):
                seen_user_with_img = True
        elif m.get("role") == "assistant" and seen_user_with_img:
            cont = m.get("content", [])
            if isinstance(cont, list):
                for c in cont:
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                        chunks.append(str(c["text"]))
            elif isinstance(cont, str):
                chunks.append(cont)
    txt = "\n\n".join([x.strip() for x in chunks if x and str(x).strip()])
    return txt if txt else "OK."


# --------------------------- Data ---------------------------

class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.rows.append(json.loads(line))
                except Exception:
                    continue
        if not self.rows:
            raise ValueError(f"No data in {jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        obj = self.rows[idx]
        messages = obj.get("messages", [])
        return messages

# --------------------------- Loader ---------------------------

def load_qwen_vl(model_id: str, device: str, use_bf16: bool, use_4bit: bool):
    """
    Load Qwen2.5-VL model + processor. Prefer AutoModelForImageTextToText; fall back to AutoModelForVision2Seq.
    Optionally quantize in 4-bit (QLoRA) if BitsAndBytes is available and --no-4bit not set.
    """
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    quant_cfg = None
    if use_4bit and BitsAndBytesConfig is not None:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                       bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    def _try_load_new():
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            quantization_config=quant_cfg
        )

    model = None
    try:
        model = _try_load_new()
    except Exception:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            quantization_config=quant_cfg
        )

    if not torch.cuda.is_available():
        model.to("cpu")

    # For long contexts / stability
    model.config.use_cache = False  # gradient checkpointing friendly
    model.gradient_checkpointing_enable()
    return processor, model

def attach_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float, target_set: str):
    if target_set == "attn":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model

# --------------------------- Collator ---------------------------

class VLMCollator:
    """
    Builds one-turn chat:
      input:  system + user(prompt+image) + assistant(target concatenated)
      labels: mask user tokens with -100; compute loss only on assistant span.

    We create:
      - prompt_chat (system+user, add_generation_prompt=True)
      - assistant_text = concat of assistant turns after that user
      - tokens = prompt_ids (+) assistant_ids
      - labels = [-100]*len(prompt_ids) (+) assistant_ids
      - images = resized image(s) from the first user-with-image message
    """
    def __init__(self, processor, image_short_edge: int):
        self.processor = processor
        self.short_edge = image_short_edge

    def __call__(self, batch_messages: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        input_ids_list, attn_mask_list, labels_list = [], [], []
        pixel_values_list = []

        for messages in batch_messages:
            # locate first user with image(s)
            u_msg, img_paths = find_first_user_with_image(messages)
            if u_msg is None or not img_paths:
                # skip example by producing a tiny dummy (trainer will handle)
                dummy = torch.tensor([[self.processor.tokenizer.eos_token_id]])
                return {"input_ids": dummy, "labels": dummy, "attention_mask": torch.ones_like(dummy)}

            # Load/rescale the first image (Qwen-VL supports multiple; we keep one for memory)
            img = Image.open(img_paths[0]).convert("RGB")
            img = resize_short_edge(img, self.short_edge)

            # Build prompt: system + user(with text+image) ONLY
            # Keep original system if present, else add a generic one
            sys_msg = None
            for m in messages:
                if m.get("role") == "system":
                    sys_msg = m
                    break
            if sys_msg is None:
                sys_msg = {"role": "system", "content": "You are a helpful vision-language assistant."}

            # Extract the *text* of the chosen user message (keep image separate)
            user_text_chunks = []
            for c in u_msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                    user_text_chunks.append(str(c["text"]))
            user_text = "\n".join(user_text_chunks) if user_text_chunks else "Please analyze the food in the image."

            prompt_messages = [
                {"role": "system", "content": sys_msg.get("content", "You are a helpful vision-language assistant.")},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": img},
                ]},
            ]
            chat_prompt = self.processor.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)

            # Tokenize prompt *with* image via processor
            pp = self.processor(text=[chat_prompt], images=[img], return_tensors="pt")
            prompt_input_ids = pp["input_ids"][0]           # (Lp,)
            prompt_attn_mask = pp["attention_mask"][0]      # (Lp,)
            pixel_values = pp["pixel_values"][0]            # (C, H, W)

            # Build assistant target text = concat(all assistant texts after this user-with-image)
            target_text = collect_all_assistant_text(messages)
            target_ids = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"][0]                               # (Lt,)

            # Concatenate for final sequence
            input_ids = torch.cat([prompt_input_ids, target_ids], dim=0)
            attention_mask = torch.cat([
                prompt_attn_mask,
                torch.ones_like(target_ids, dtype=prompt_attn_mask.dtype)
            ], dim=0)

            # Labels: mask prompt with -100, keep target
            labels = torch.full_like(input_ids, fill_value=-100)
            labels[-target_ids.size(0):] = target_ids

            input_ids_list.append(input_ids)
            attn_mask_list.append(attention_mask)
            labels_list.append(labels)
            pixel_values_list.append(pixel_values)

        # Pad sequences to max length in batch
        max_len = max(x.size(0) for x in input_ids_list)
        def pad_to(x, pad_val):
            pad_len = max_len - x.size(0)
            if pad_len <= 0: return x
            return torch.cat([x, torch.full((pad_len,), pad_val, dtype=x.dtype)], dim=0)

        input_ids = torch.stack([pad_to(x, self.processor.tokenizer.pad_token_id) for x in input_ids_list], dim=0)
        attention_mask = torch.stack([pad_to(x, 0) for x in attn_mask_list], dim=0)
        labels = torch.stack([pad_to(x, -100) for x in labels_list], dim=0)
        pixel_values = torch.stack(pixel_values_list, dim=0)  # images already same size by resize

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

# --------------------------- Main ---------------------------

def main():
    args = build_args()
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(args.bf16)
    use_4bit = (not args.no_4bit) and (BitsAndBytesConfig is not None)

    # Load model/processor
    processor, base_model = load_qwen_vl(
        args.model, device=device, use_bf16=use_bf16, use_4bit=use_4bit
    )

    # Attach LoRA
    base_model = attach_lora(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_set=args.lora_target,
    )

    # Dataset & collator
    ds = SFTJsonlDataset(args.train_jsonl)
    collator = VLMCollator(processor, image_short_edge=args.train_image_short_edge)

    # Trainer
    total_steps_est = (len(ds) * args.epochs) // max(1, args.grad_accum)
    print(f"Estimated optimizer steps: ~{total_steps_est}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()),
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
    )

    class VLMTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
                ds,
                batch_size=training_args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=collator,
                drop_last=False
            )

    trainer = VLMTrainer(
        model=base_model,
        args=training_args,
        train_dataset=ds,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )

    # Train
    trainer.train()
    # Save adapter only (PEFT format)
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")

if __name__ == "__main__":
    main()
