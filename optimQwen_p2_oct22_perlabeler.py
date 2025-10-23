#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, argparse, csv
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm
from rapidfuzz import process as fuzzprocess, fuzz
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoModel,
)

# ------------------------------ Label spaces ------------------------------

ALLOWED_CATEGORIES = [
    "Vegetables and Fruits",
    "Cereals and Legumes",
    "Beverages",
    "Protein Sources",
    "Dairy and Plant-Based Drinks",
    "Oils and Fats",
    "Traditional Mediterranean Dishes",
    "Snacks and Appetizers",
    "Sweets and Pastries",
    "Fast Food",
    "Others",
]
DEFAULT_COOKING_STYLES = [
    "none","fresh","grilled","fried","boiled or steamed","stewed",
    "oven-baked","raw","roasted","other"
]

ALIASES = {
    "boiled": "boiled or steamed",
    "steamed": "boiled or steamed",
    "grill": "grilled",
    "oven baked": "oven-baked",
    "deep fried": "fried",
    "fries": "Fast Food",
    "yoghurt": "Dairy and Plant-Based Drinks",
}

MAIN_CATS_LINE = re.compile(r"^\s*3\.\s*Main\s*Categories\s*:?\s*(?P<body>.+)$", re.IGNORECASE)
BRACKET_EXTRACT = re.compile(r"\[(?P<inside>[^\]]+)\]")

# ------------------------------ Fuzzy & parsing helpers ------------------------------

def normalize_alias(s: str) -> str:
    t = re.sub(r"\s+", " ", str(s).strip().lower())
    return ALIASES.get(t, s)

def fuzzy_pick(value: str, choices: List[str], score_cut: int = 70, or_else: str = "other") -> str:
    if not choices:
        return (value or "").strip()
    if value is None or not str(value).strip():
        return or_else if or_else in choices else choices[0]
    value = normalize_alias(str(value))
    match = fuzzprocess.extractOne(value, choices, scorer=fuzz.WRatio)
    if match and match[1] >= score_cut:
        return match[0]
    v2 = re.sub(r"\s+", " ", value.lower()).strip()
    table = {re.sub(r"\s+", " ", c.lower()).strip(): c for c in choices}
    if v2 in table:
        return table[v2]
    return or_else if or_else in choices else (match[0] if match else choices[0])

def fuzzy_set_map(preds: List[str], choices: List[str], thresh: int = 75) -> List[str]:
    out = []
    for p in preds:
        p = normalize_alias((p or "").strip())
        if not p:
            continue
        match = fuzzprocess.extractOne(p, choices, scorer=fuzz.WRatio)
        if match and match[1] >= thresh:
            out.append(match[0])
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

def extract_categories_from_text(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []
    # precise line
    for l in lines:
        m = MAIN_CATS_LINE.search(l)
        if m:
            body = m.group("body")
            b = BRACKET_EXTRACT.search(body)
            raw = b.group("inside") if b else body
            candidates = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
            break
    # fallback lines
    if not candidates:
        for l in lines:
            if "Main Categories" in l:
                b = BRACKET_EXTRACT.search(l)
                raw = b.group("inside") if b else l.split(":", 1)[-1]
                candidates = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
                break
    if not candidates:
        for l in lines[::-1]:
            b = BRACKET_EXTRACT.search(l)
            if b:
                raw = b.group("inside")
                candidates = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
                if candidates:
                    break
    return fuzzy_set_map(candidates, ALLOWED_CATEGORIES)

def parse_json_field(s: str, key: str) -> str:
    try:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            obj = json.loads(s[i:j+1])
            val = obj.get(key, "")
            return str(val).strip()
    except Exception:
        pass
    for line in s.splitlines():
        if ":" in line and key in line.lower():
            v = line.split(":",1)[-1].strip().strip('"')
            return v
    return ""

def index_prompt(title: str, options: List[str]) -> str:
    lines = [title.strip()]
    for i, opt in enumerate(options, 1):
        lines.append(f"{i}. {opt}")
    lines.append("Reply with ONLY the number of the best option. No words, no JSON, no punctuation.")
    return "\n".join(lines)

def parse_index(s: str, n: int, default: int = 1) -> int:
    m = re.search(r"\d+", s or "")
    if not m: return default
    k = int(m.group(0))
    return k if 1 <= k <= n else default

# ------------------------------ Metrics ------------------------------

def compute_multilabel_metrics(y_true_bin, y_pred_bin) -> Dict[str, float]:
    subset_acc = (y_true_bin == y_pred_bin).all(axis=1).mean()
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="macro", zero_division=0
    )
    return {
        "subset_accuracy": float(subset_acc),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
    }

def compute_basic_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
    }

def compute_ewr_per_image(pred_labels: List[str], expert_entry: Dict[str, Any]) -> float:
    labels = expert_entry.get("labels", [])
    N = max(1, int(expert_entry.get("num_experts", 1)))
    denom = 0.0; numer = 0.0
    P = {re.sub(r"\s+", " ", p.lower()).strip() for p in pred_labels}
    for rec in labels:
        l = str(rec.get("label", "")).strip()
        v = float(rec.get("votes", 0.0))
        if v <= 0:
            continue
        w = v / N
        denom += w
        ln = re.sub(r"\s+", " ", l.lower()).strip()
        if ln in P:
            numer += w
    return 0.0 if denom == 0.0 else numer / denom

# --------- Per-label metrics helpers (NEW) ---------

def _safe_div(n, d): return float(n) / d if d else 0.0

def compute_per_label_stats(y_true_sets, y_pred_sets, all_labels):
    """
    y_true_sets: List[Set[str]]
    y_pred_sets: List[Set[str]]
    all_labels: Iterable[str]
    returns: dict[label] = {support,tp,fp,fn,precision,recall,f1}
    """
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    for g, p in zip(y_true_sets, y_pred_sets):
        for lab in g: support[lab] += 1
        for lab in all_labels:
            ig, ip = (lab in g), (lab in p)
            if ig and ip: tp[lab] += 1
            elif (not ig) and ip: fp[lab] += 1
            elif ig and (not ip): fn[lab] += 1
    per = {}
    for lab in all_labels:
        t, f, n = tp[lab], fp[lab], fn[lab]
        prec = _safe_div(t, t + f)
        rec  = _safe_div(t, t + n)
        f1   = _safe_div(2*prec*rec, (prec+rec)) if (prec+rec) else 0.0
        per[lab] = {
            "support": int(support[lab]),
            "tp": int(t), "fp": int(f), "fn": int(n),
            "precision": prec, "recall": rec, "f1": f1
        }
    return per

def write_per_label_csv(per_label_dict, out_csv_path):
    fieldnames = ["label","support","tp","fp","fn","precision","recall","f1"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for lab, stats in sorted(per_label_dict.items(), key=lambda kv: kv[0].lower()):
            row = {"label": lab}; row.update(stats); w.writerow(row)

# ------------------------------ Image helper ------------------------------

def resize_short_edge(img: Image.Image, target: int) -> Image.Image:
    if not target or target <= 0: return img
    w, h = img.size
    s = min(w, h)
    if s == target: return img
    scale = target / float(s)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)

# ------------------------------ Prior knowledge builders ------------------------------

def derive_scoped_sub_map(df: pd.DataFrame, allowed_categories: List[str], max_per_cat: int = 48) -> Dict[str, List[str]]:
    if "subcategory" not in df.columns:
        return {}
    cat_series = df["category"].astype(str).str.strip()
    sub_series = df["subcategory"].astype(str).str.strip()
    buckets = {c: Counter() for c in allowed_categories}
    for c, s in zip(cat_series.tolist(), sub_series.tolist()):
        if not s: continue
        match = fuzzprocess.extractOne(c, allowed_categories, scorer=fuzz.WRatio)
        if match and match[1] >= 80:
            buckets[match[0]][s] += 1
    scoped = {}
    for cat, ctr in buckets.items():
        if not ctr: continue
        top = [k for k,_ in ctr.most_common(max_per_cat)]
        scoped[cat] = sorted(set(top))
    return scoped

def invert_sub_map(cat_to_subs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    inv = defaultdict(list)
    for cat, subs in cat_to_subs.items():
        for s in subs:
            if cat not in inv[s]:
                inv[s].append(cat)
    return dict(inv)

def derive_style_maps(df: pd.DataFrame, allowed_categories: List[str]) -> Dict[str, Dict[str, List[str]]]:
    sub_styles = defaultdict(Counter)
    cat_styles = defaultdict(Counter)
    if "cooking_style" not in df.columns:
        return {"sub": {}, "cat": {}}
    sty_series = df["cooking_style"].astype(str).str.strip()
    subs = df["subcategory"].astype(str).str.strip() if "subcategory" in df.columns else pd.Series([""]*len(df))
    cats = df["category"].astype(str).str.strip()
    for c, s, st in zip(cats, subs, sty_series):
        if not st: continue
        match = fuzzprocess.extractOne(c, allowed_categories, scorer=fuzz.WRatio)
        if match and match[1] >= 80:
            cat_styles[match[0]][st] += 1
        if s:
            sub_styles[s][st] += 1
    sub_map = {k: [x for x,_ in ctr.most_common(20)] for k, ctr in sub_styles.items()}
    cat_map = {k: [x for x,_ in ctr.most_common(20)] for k, ctr in cat_styles.items()}
    return {"sub": sub_map, "cat": cat_map}

# ------------------------------ Prior results loader ------------------------------

def load_prior_results(file_path: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    def _from_csv(p: Path):
        dfp = pd.read_csv(p); cols = [c.lower() for c in dfp.columns]; dfp.columns = cols
        if "image_name" not in cols:
            raise ValueError(f"{p} missing image_name")
        m = {}
        for row in dfp.to_dict(orient="records"):
            name = str(row["image_name"])
            cats = [s.strip() for s in str(row.get("pred_categories","")).split(",") if s.strip()]
            sub  = str(row.get("pred_subcategory","")).strip() if "pred_subcategory" in cols else ""
            sty  = str(row.get("pred_cooking_style","")).strip() if "pred_cooking_style" in cols else ""
            m[name] = {"cats": fuzzy_set_map(cats, ALLOWED_CATEGORIES), "sub": sub, "style": sty}
        return m

    def _from_jsonl(p: Path):
        m = {}
        with open(p,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    obj=json.loads(line); name=str(obj.get("image_name")); text=obj.get("text","")
                    m[name]={"cats": extract_categories_from_text(text), "sub":"", "style":""}
                except: continue
        return m

    if not file_path:
        for cand in [out_dir/"predictions_all.csv", out_dir/"vlm_raw_outputs_stage1.jsonl"]:
            if cand.exists(): file_path=str(cand); break
        if not file_path: raise ValueError("No prior results file found; pass --cats-file")

    p=Path(file_path)
    if not p.exists(): raise FileNotFoundError(p)
    if p.suffix.lower()==".csv": return _from_csv(p)
    if p.suffix.lower() in [".jsonl",".json"]: return _from_jsonl(p)
    raise ValueError(f"Unsupported file: {p}")

# ------------------------------ Model loaders ------------------------------

def load_qwen_model(model_id: str, device: str, dtype):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    # try newer class first
    try:
        from transformers import AutoModelForImageTextToText
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=dtype, device_map="auto" if device=="cuda" else None, trust_remote_code=True
            )
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto" if device=="cuda" else None, trust_remote_code=True
            )
    except (ImportError, AttributeError):
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, dtype=dtype, device_map="auto" if device=="cuda" else None, trust_remote_code=True
            )
        except TypeError:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto" if device=="cuda" else None, trust_remote_code=True
            )
    if device=="cpu": model.to(device)
    return processor, model

def load_moondream(model_id: str, device: str):
    # needs processor assets (tokenizer + image processor) in the folder / repo
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        ).eval()
    except Exception:
        mdl = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        ).eval()
    return proc, mdl

def md_query(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 160) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful vision-language assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ]},
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    batch = processor(text=[chat], images=[image], return_tensors="pt")
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    tok = getattr(processor, "tokenizer", None)
    return tok.batch_decode(out_ids, skip_special_tokens=True)[0]

# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter for Qwen (ignored for Moondream)")
    ap.add_argument("--moondream", action="store_true", help="Use Moondream backend instead of Qwen")
    ap.add_argument("--max-samples", type=int, default=0)

    ap.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32"], help="Qwen only")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0, help="Qwen only (if --do-sample)")
    ap.add_argument("--do-sample", action="store_true", help="Qwen only")
    ap.add_argument("--image-short-edge", type=int, default=512)
    ap.add_argument("--concise", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)

    ap.add_argument("--expert-json", default=None)
    ap.add_argument("--stage2-only", action="store_true")
    ap.add_argument("--stage3-only", action="store_true")
    ap.add_argument("--cats-file", default=None)
    ap.add_argument("--summarize", action="store_true")

    # NEW: toggle to disable per-label outputs (enabled by default)
    ap.add_argument("--no-per-label", action="store_true", help="Disable per-label CSV/JSON metrics")

    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw1 = out_dir / "vlm_raw_outputs_stage1.jsonl"
    raw2 = out_dir / "vlm_raw_outputs_stage2.jsonl"
    raw3 = out_dir / "vlm_raw_outputs_stage3.jsonl"
    pred_path = out_dir / "predictions_all.csv"
    metrics_path = out_dir / "metrics.json"
    summary_jsonl = out_dir / "final_summary.jsonl"

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.concise:
        prompt_text += "\n\nConstraints: Each section MUST be one concise sentence. No extra commentary."

    df = pd.read_csv(args.gt_csv)
    df.columns = [c.lower() for c in df.columns]
    if "image_name" not in df.columns or "category" not in df.columns:
        raise ValueError("GT must include: image_name, category")
    have_sub = "subcategory" in df.columns
    have_style = "cooking_style" in df.columns

    gt_subs = sorted(set(str(x).strip() for x in df["subcategory"].dropna().tolist())) if have_sub else []
    gt_styles = sorted(set(str(x).strip() for x in df["cooking_style"].dropna().tolist())) if have_style else DEFAULT_COOKING_STYLES

    cat_to_subs = derive_scoped_sub_map(df, ALLOWED_CATEGORIES, max_per_cat=48)
    sub_to_parents = invert_sub_map(cat_to_subs)
    style_maps = derive_style_maps(df, ALLOWED_CATEGORIES)

    samples_all = df.to_dict(orient="records")
    if args.max_samples and args.max_samples > 0:
        samples_all = samples_all[:args.max_samples]
    shard_count = max(1, int(args.shard_count))
    shard_index = max(0, int(args.shard_index))
    samples = [r for i, r in enumerate(samples_all) if (i % shard_count) == shard_index]

    done1 = set()
    if args.resume and raw1.exists():
        with open(raw1,"r",encoding="utf-8") as f:
            for line in f:
                try: done1.add(json.loads(line).get("image_name"))
                except: pass
    done2 = set()
    if args.resume and raw2.exists():
        with open(raw2,"r",encoding="utf-8") as f:
            for line in f:
                try: done2.add(json.loads(line).get("image_name"))
                except: pass
    done3 = set()
    if args.resume and raw3.exists():
        with open(raw3,"r",encoding="utf-8") as f:
            for line in f:
                try: done3.add(json.loads(line).get("image_name"))
                except: pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = None; model = None
    processor_md = None; md_model = None

    if not args.moondream:
        if args.dtype == "auto":
            if device=="cuda" and torch.cuda.is_bf16_supported(): dtype=torch.bfloat16
            elif device=="cuda": dtype=torch.float16
            else: dtype=torch.float32
        else:
            dtype = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[args.dtype]
        processor, model = load_qwen_model(args.model, device, dtype)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            print(f"Loaded LoRA adapter from {args.adapter}")
    else:
        processor_md, md_model = load_moondream(args.model, device)

    img_root = Path(args.images_dir)
    prior_map = {}
    if args.stage2_only or args.stage3_only:
        prior_map = load_prior_results(args.cats_file, out_dir)

    f1 = open(raw1,"a",encoding="utf-8") if not args.stage2_only and not args.stage3_only else None
    f2 = open(raw2,"a",encoding="utf-8")
    f3 = open(raw3,"a",encoding="utf-8")

    fs = open(summary_jsonl,"a",encoding="utf-8") if args.summarize else None
    rows = []
    loop_desc = "Stage-3 ONLY" if args.stage3_only else ("Stage-2 ONLY" if args.stage2_only else "Stage-1+2+3")
    backend_name = "Moondream" if args.moondream else "Qwen"

    for r in tqdm(samples, desc=f"Running VLM ({backend_name}; {loop_desc})"):
        name = str(r["image_name"])
        fpath = img_root / name
        if not fpath.exists():
            rows.append({"image_name":name,"pred_categories":"","pred_subcategory":"","pred_cooking_style":"","missing":"image"})
            continue

        image = Image.open(fpath).convert("RGB")
        if args.image_short_edge and args.image_short_edge>0:
            image = resize_short_edge(image, args.image_short_edge)

        # -------------------- Step 1: Categories --------------------
        cats = []
        if not (args.stage2_only or args.stage3_only):
            if not (args.resume and name in done1):
                if args.moondream:
                    text1 = md_query(md_model, processor_md, image, Path(args.prompt_file).read_text(encoding="utf-8").strip(), max_new_tokens=args.max_new_tokens)
                else:
                    messages = [
                        {"role":"system","content":"You are a helpful vision-language assistant."},
                        {"role":"user","content":[
                            {"type":"text","text":Path(args.prompt_file).read_text(encoding="utf-8").strip()},
                            {"type":"image","image":image}
                        ]}
                    ]
                    chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    batch = processor(text=[chat], images=[image], return_tensors="pt")
                    for k,v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = v.to(model.device)
                    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=bool(args.do_sample))
                    if args.do_sample: gen_kwargs["temperature"]=args.temperature
                    with torch.no_grad():
                        out_ids = model.generate(**batch, **gen_kwargs)
                    text1 = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                if f1 is not None:
                    f1.write(json.dumps({"image_name":name,"text":text1}, ensure_ascii=False)+"\n")
                cats = extract_categories_from_text(text1)

                # Safety net A: verbatim scan
                if not cats:
                    low = text1.lower()
                    hits = [lab for lab in ALLOWED_CATEGORIES if lab.lower() in low]
                    cats = hits

                # Safety net B: index fallback to pick strongest single category
                if not cats:
                    cat_prompt = index_prompt(
                        "Select the single strongest major category for this image from the list below:",
                        ALLOWED_CATEGORIES
                    )
                    if args.moondream:
                        text1b = md_query(md_model, processor_md, image, cat_prompt, max_new_tokens=8)
                    else:
                        messagesB = [
                            {"role":"system","content":"You are a helpful vision-language assistant."},
                            {"role":"user","content":[
                                {"type":"text","text":cat_prompt},
                                {"type":"image","image":image}
                            ]}
                        ]
                        chatB = processor.apply_chat_template(messagesB, add_generation_prompt=True, tokenize=False)
                        batchB = processor(text=[chatB], images=[image], return_tensors="pt")
                        for k,v in batchB.items():
                            if torch.is_tensor(v):
                                batchB[k] = v.to(model.device)
                        with torch.no_grad():
                            out_idsB = model.generate(**batchB, max_new_tokens=8, do_sample=False)
                        text1b = processor.batch_decode(out_idsB, skip_special_tokens=True)[0]
                    k = parse_index(text1b, len(ALLOWED_CATEGORIES), default=ALLOWED_CATEGORIES.index("Others")+1)
                    cats = [ALLOWED_CATEGORIES[k-1]]
            else:
                with open(raw1,"r",encoding="utf-8") as rf:
                    for line in rf:
                        obj=json.loads(line)
                        if obj.get("image_name")==name:
                            cats = extract_categories_from_text(obj.get("text","")); break
        else:
            cats = prior_map.get(name, {}).get("cats", [])

        # -------------------- Step 2: Subcategory (indexed) --------------------
        pred_sub = prior_map.get(name, {}).get("sub","") if args.stage3_only else ""
        if not args.stage3_only:
            if not (args.resume and name in done2):
                candidates_sub = []
                for c in cats:
                    candidates_sub.extend(cat_to_subs.get(c, []))
                if not candidates_sub:
                    if have_sub:
                        candidates_sub = sorted(set(df["subcategory"].dropna().astype(str).tolist()))[:200]
                    else:
                        candidates_sub = ["other"]
                # dedup & cap
                seen=set(); candidates_sub=[x for x in candidates_sub if not (x in seen or seen.add(x))][:200]

                sub_prompt = index_prompt(
                    "Select the single best subcategory for this image from the list below:",
                    candidates_sub
                )

                if args.moondream:
                    text2 = md_query(md_model, processor_md, image, sub_prompt, max_new_tokens=8)
                else:
                    messages2 = [
                        {"role":"system","content":"You are a helpful vision-language assistant."},
                        {"role":"user","content":[
                            {"type":"text","text":sub_prompt},
                            {"type":"image","image":image}
                        ]}
                    ]
                    chat2 = processor.apply_chat_template(messages2, add_generation_prompt=True, tokenize=False)
                    batch2 = processor(text=[chat2], images=[image], return_tensors="pt")
                    for k,v in batch2.items():
                        if torch.is_tensor(v):
                            batch2[k] = v.to(model.device)
                    with torch.no_grad():
                        out_ids2 = model.generate(**batch2, max_new_tokens=8, do_sample=False)
                    text2 = processor.batch_decode(out_ids2, skip_special_tokens=True)[0]

                if f2 is not None:
                    f2.write(json.dumps({"image_name":name,"text":text2}, ensure_ascii=False)+"\n")
                k = parse_index(text2, len(candidates_sub), default=1)
                pred_sub = candidates_sub[k-1]
            else:
                with open(raw2,"r",encoding="utf-8") as rf:
                    for line in rf:
                        obj=json.loads(line)
                        if obj.get("image_name")==name:
                            k = parse_index(obj.get("text",""), 10**6, default=1)
                            # fallback if old runs had JSON
                            if k == 1:
                                fallback = parse_json_field(obj.get("text",""), "subcategory")
                                if fallback:
                                    pred_sub = fuzzy_pick(fallback, gt_subs or ["other"], or_else="other")
                                    break
                            # else cannot map without candidate list; leave as-is
                            break

        # Refine categories using sub's parents
        if pred_sub and pred_sub in sub_to_parents:
            parents = set(sub_to_parents[pred_sub])
            keep = [c for c in cats if c in parents]
            if keep: cats = keep
        if len(cats)>1 and "Others" in cats:
            cats = [c for c in cats if c!="Others"]
        if len(cats)>2:
            cats = cats[:2]

        # -------------------- Step 3: Cooking style (indexed) --------------------
        pred_style = prior_map.get(name, {}).get("style","") if args.stage3_only else ""
        if not (args.resume and name in done3):
            cand_style = []
            if pred_sub:
                cand_style.extend(style_maps.get("sub", {}).get(pred_sub, []))
            for c in cats:
                cand_style.extend(style_maps.get("cat", {}).get(c, []))
            if not cand_style:
                cand_style = gt_styles if len(gt_styles)>0 else DEFAULT_COOKING_STYLES
            # dedup & cap
            seen=set(); cand_style=[x for x in cand_style if not (x in seen or seen.add(x))][:25]

            style_prompt = index_prompt(
                "Select the single best cooking style for this image from the list below:",
                cand_style
            )

            if args.moondream:
                text3 = md_query(md_model, processor_md, image, style_prompt, max_new_tokens=6)
            else:
                messages3 = [
                    {"role":"system","content":"You are a helpful vision-language assistant."},
                    {"role":"user","content":[
                        {"type":"text","text":style_prompt},
                        {"type":"image","image":image}
                    ]}
                ]
                chat3 = processor.apply_chat_template(messages3, add_generation_prompt=True, tokenize=False)
                batch3 = processor(text=[chat3], images=[image], return_tensors="pt")
                for k,v in batch3.items():
                    if torch.is_tensor(v):
                        batch3[k] = v.to(model.device)
                with torch.no_grad():
                    out_ids3 = model.generate(**batch3, max_new_tokens=6, do_sample=False)
                text3 = processor.batch_decode(out_ids3, skip_special_tokens=True)[0]

            if f3 is not None:
                f3.write(json.dumps({"image_name":name,"text":text3}, ensure_ascii=False)+"\n")
            j = parse_index(text3, len(cand_style), default=1)
            pred_style = cand_style[j-1]
        else:
            with open(raw3,"r",encoding="utf-8") as rf:
                for line in rf:
                    obj=json.loads(line)
                    if obj.get("image_name")==name:
                        j = parse_index(obj.get("text",""), 10**6, default=1)
                        if j == 1:
                            fallback = parse_json_field(obj.get("text",""), "cooking_style")
                            if fallback:
                                pred_style = fuzzy_pick(fallback, gt_styles or DEFAULT_COOKING_STYLES, or_else="other")
                                break
                        break

        if args.summarize and summary_jsonl:
            with open(summary_jsonl, "a", encoding="utf-8") as fs:
                fs.write(json.dumps({
                    "image_name": name,
                    "summary": f"Major categories: [{', '.join(cats)}]; Subcategory: {pred_sub}; Cooking style: {pred_style}."
                }, ensure_ascii=False) + "\n")

        rows.append({
            "image_name": name,
            "pred_categories": ", ".join(cats),
            "pred_subcategory": pred_sub,
            "pred_cooking_style": pred_style,
        })

    if f1 is not None: f1.close()
    f2.close(); f3.close()

    # -------------------- Save predictions --------------------
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_path, index=False)

    # -------------------- Metrics --------------------
    metrics: Dict[str, Any] = {}

    # Categories (multi-label)
    truth_cats = [[p.strip() for p in str(s).split(",") if p.strip()] for s in df["category"].tolist()]
    pred_map_rows = {r["image_name"]: [s.strip() for s in str(r["pred_categories"]).split(",") if s.strip()]
                     for r in pred_df.to_dict(orient="records")}
    preds_cats = [pred_map_rows.get(n, []) for n in df["image_name"].tolist()]
    mlb = MultiLabelBinarizer(classes=ALLOWED_CATEGORIES)
    Y_true = mlb.fit_transform(truth_cats)
    Y_pred = mlb.transform(preds_cats)
    metrics["categories"] = compute_multilabel_metrics(Y_true, Y_pred)

    # Optional EWR
    if args.expert_json and Path(args.expert_json).exists():
        experts = json.loads(Path(args.expert_json).read_text(encoding="utf-8"))
        vals = []
        for n, pred in zip(df["image_name"].tolist(), preds_cats):
            if n in experts:
                vals.append(compute_ewr_per_image(pred, experts[n]))
        if vals:
            metrics["categories_EWR_mean"] = float(sum(vals)/len(vals))
            metrics["categories_EWR_count"] = len(vals)

    # Subcategory
    if "subcategory" in df.columns:
        y_true_sub = [str(x).strip() for x in df["subcategory"].tolist()]
        idx_sub = {r["image_name"]: r["pred_subcategory"] for r in pred_df.to_dict(orient="records")}
        y_pred_sub = [idx_sub.get(n, "other") for n in df["image_name"].tolist()]
        metrics["subcategory"] = compute_basic_metrics(y_true_sub, y_pred_sub)

    # Cooking style
    if "cooking_style" in df.columns:
        y_true_style = [str(x).strip() for x in df["cooking_style"].tolist()]
        idx_style = {r["image_name"]: r["pred_cooking_style"] for r in pred_df.to_dict(orient="records")}
        y_pred_style = [idx_style.get(n, "other") for n in df["image_name"].tolist()]
        metrics["cooking_style"] = compute_basic_metrics(y_true_style, y_pred_style)

    # Exact match (all three)
    if "subcategory" in df.columns and "cooking_style" in df.columns:
        truth_cats_set = [set(x) for x in truth_cats]
        pred_cats_set = [set(x) for x in preds_cats]
        pm_sub = {r["image_name"]: r["pred_subcategory"] for r in pred_df.to_dict(orient="records")}
        pm_style = {r["image_name"]: r["pred_cooking_style"] for r in pred_df.to_dict(orient="records")}
        exact = []
        for i, n in enumerate(df["image_name"].tolist()):
            ok = (truth_cats_set[i] == pred_cats_set[i]) \
                 and (str(df["subcategory"][i]).strip() == pm_sub.get(n, "")) \
                 and (str(df["cooking_style"][i]).strip() == pm_style.get(n, ""))
            exact.append(1 if ok else 0)
        metrics["exact_match_all_three"] = float(sum(exact)/len(exact)) if exact else 0.0

    # ---------- Per-label metrics (CSV + JSON) ----------
    if not args.no_per_label:
        out_dir = Path(args.out_dir)

        # Categories (multi-label -> sets)
        gt_cat_sets   = [set([t.strip() for t in str(s).split(",") if t.strip()]) for s in df["category"]]
        pred_cat_sets = [set(pred_map_rows.get(n, [])) for n in df["image_name"]]
        per_cat = compute_per_label_stats(gt_cat_sets, pred_cat_sets, ALLOWED_CATEGORIES)
        write_per_label_csv(per_cat, str(out_dir / "per_label_categories.csv"))

        # Subcategory (single-label -> 1-vs-rest sets)
        per_sub = {}
        if "subcategory" in df.columns:
            all_subs = sorted(set(y_true_sub))
            gt_sub_sets   = [set([a]) if a else set() for a in y_true_sub]
            pred_sub_sets = [set([b]) if b else set() for b in y_pred_sub]
            per_sub = compute_per_label_stats(gt_sub_sets, pred_sub_sets, all_subs)
            write_per_label_csv(per_sub, str(out_dir / "per_label_subcategory.csv"))

        # Cooking style (single-label -> 1-vs-rest sets)
        per_style = {}
        if "cooking_style" in df.columns:
            all_styles = sorted(set(y_true_style)) if len(set(y_true_style))>0 else sorted(DEFAULT_COOKING_STYLES)
            gt_style_sets   = [set([a]) if a else set() for a in y_true_style]
            pred_style_sets = [set([b]) if b else set() for b in y_pred_style]
            per_style = compute_per_label_stats(gt_style_sets, pred_style_sets, all_styles)
            write_per_label_csv(per_style, str(out_dir / "per_label_cooking_style.csv"))

        # stash JSON bundle
        (out_dir / "per_label_metrics.json").write_text(
            json.dumps({"categories": per_cat, "subcategory": per_sub, "cooking_style": per_style}, indent=2),
            encoding="utf-8"
        )

    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved:\n- Stage1 raw: {raw1}\n- Stage2 raw: {raw2}\n- Stage3 raw: {raw3}\n- Predictions: {pred_path}\n- Metrics: {metrics_path}\n"
          f"- Per-label CSV/JSON written (unless disabled with --no-per-label)\n")

if __name__ == "__main__":
    main()
