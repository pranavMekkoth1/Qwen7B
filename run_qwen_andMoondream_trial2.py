#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, argparse, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
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

# ============================== Label spaces ==============================

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

# ============================== Fuzzy & parsing helpers ==============================

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

def parse_index_strict(s: str, n: int) -> int:
    s = (s or "").strip()
    if not re.fullmatch(r"\d+", s):
        return 0
    k = int(s)
    return k if 1 <= k <= n else 0

# ============================== Metrics ==============================

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

# ----- Per-label metrics helpers -----
def _safe_div(n, d): return float(n) / d if d else 0.0

def compute_per_label_stats(y_true_sets, y_pred_sets, all_labels):
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
        per[lab] = {"support": int(support[lab]), "tp": int(t), "fp": int(f), "fn": int(n),
                    "precision": prec, "recall": rec, "f1": f1}
    return per

def write_per_label_csv(per_label_dict, out_csv_path):
    fieldnames = ["label","support","tp","fp","fn","precision","recall","f1"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for lab, stats in sorted(per_label_dict.items(), key=lambda kv: kv[0].lower()):
            row = {"label": lab}; row.update(stats); w.writerow(row)

# ============================== Image helper ==============================

def resize_short_edge(img: Image.Image, target: int) -> Image.Image:
    if not target or target <= 0: return img
    w, h = img.size
    s = min(w, h)
    if s == target: return img
    scale = target / float(s)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)

# ============================== Prior knowledge builders ==============================

def derive_scoped_sub_map(df: pd.DataFrame, allowed_categories: List[str], max_per_cat: int = 80) -> Dict[str, List[str]]:
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
    sub_map = {k: [x for x,_ in ctr.most_common(15)] for k, ctr in sub_styles.items()}
    cat_map = {k: [x for x,_ in ctr.most_common(15)] for k, ctr in cat_styles.items()}
    return {"sub": sub_map, "cat": cat_map}

# ============================== Prior results loader ==============================

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

# ============================== Model loaders ==============================

def load_qwen_model(model_id: str, device: str, dtype):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
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

def qwen_generate_text(processor, model, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role":"system","content":"You are a helpful vision-language assistant."},
        {"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image","image":image}
        ]}
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    batch = processor(text=[chat], images=[image], return_tensors="pt")
    for k,v in batch.items():
        if torch.is_tensor(v): batch[k] = v.to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]

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

# ============================== Verification ==============================

def yesno_verify(
    image: Image.Image,
    prompt_q: str,
    backend: str,
    qwen_bundle: Tuple[Any, Any],
    moon_bundle: Tuple[Any, Any],
    max_new_tokens: int = 4
) -> bool:
    if backend == "moondream":
        processor_md, md_model = moon_bundle
        txt = md_query(md_model, processor_md, image, prompt_q, max_new_tokens=max_new_tokens)
    else:
        processor, model = qwen_bundle
        txt = qwen_generate_text(processor, model, image, prompt_q, max_new_tokens=max_new_tokens)
    return "yes" in (txt or "").lower().strip()

# ============================== Main ==============================

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

    # NEW: restrict to split_dataset.json
    ap.add_argument("--split-json", default=None, help="Path to split_dataset.json")
    ap.add_argument("--split-key", default="test", choices=["train","val","test"], help="Which split to run")

    # NEW: per-label metrics toggle
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

    # --- restrict to split_dataset.json if provided ---
    if args.split_json:
        split_obj = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        test_list = split_obj.get(args.split_key, [])
        if not isinstance(test_list, list):
            raise ValueError(f"{args.split_json} does not contain a list for key '{args.split_key}'")
        split_names = {os.path.basename(str(x)) for x in test_list}
        df = df[df["image_name"].apply(lambda x: os.path.basename(str(x)) in split_names)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows in GT after filtering to split '{args.split_key}' from {args.split_json}")

    have_sub = "subcategory" in df.columns
    have_style = "cooking_style" in df.columns

    gt_subs = sorted(set(str(x).strip() for x in df["subcategory"].dropna().tolist())) if have_sub else []
    gt_styles = sorted(set(str(x).strip() for x in df["cooking_style"].dropna().tolist())) if have_style else DEFAULT_COOKING_STYLES

    cat_to_subs = derive_scoped_sub_map(df, ALLOWED_CATEGORIES, max_per_cat=80)
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
    backend = "moondream" if args.moondream else "qwen"
    qwen_bundle = (processor, model)
    moon_bundle = (processor_md, md_model)

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
        cats: List[str] = []
        if not (args.stage2_only or args.stage3_only):
            if not (args.resume and name in done1):
                if args.moondream:
                    text1 = md_query(md_model, processor_md, image, prompt_text, max_new_tokens=args.max_new_tokens)
                else:
                    text1 = qwen_generate_text(processor, model, image, prompt_text, max_new_tokens=args.max_new_tokens)
                if f1 is not None:
                    f1.write(json.dumps({"image_name":name,"text":text1}, ensure_ascii=False)+"\n")

                cats = extract_categories_from_text(text1)
                if not cats:
                    low = text1.lower()
                    hits = [lab for lab in ALLOWED_CATEGORIES if lab.lower() in low]
                    cats = hits
                if not cats:
                    cat_prompt = index_prompt(
                        "Select the single strongest major category for this image from the list below:",
                        ALLOWED_CATEGORIES
                    )
                    txt = (md_query(md_model, processor_md, image, cat_prompt, 8)
                           if args.moondream else qwen_generate_text(processor, model, image, cat_prompt, 8))
                    k = parse_index_strict(txt, len(ALLOWED_CATEGORIES))
                    cats = [ALLOWED_CATEGORIES[(k-1) if k>0 else ALLOWED_CATEGORIES.index("Others")]]
            else:
                with open(raw1,"r",encoding="utf-8") as rf:
                    for line in rf:
                        obj=json.loads(line)
                        if obj.get("image_name")==name:
                            cats = extract_categories_from_text(obj.get("text","")); break
        else:
            cats = prior_map.get(name, {}).get("cats", [])

        # double yes/no verification & cap to 1 category (precision heavy)
        if cats:
            refined = []
            for c in cats:
                q1 = f"Does the food in the image belong to the major category '{c}'? Reply with ONLY 'yes' or 'no'."
                q2 = f"Among the major categories, is '{c}' the BEST single choice for this image? Reply with ONLY 'yes' or 'no'."
                if yesno_verify(image, q1, backend, qwen_bundle, moon_bundle, 4) and \
                   yesno_verify(image, q2, backend, qwen_bundle, moon_bundle, 4):
                    refined.append(c)
            cats = refined[:1] if refined else cats[:1]

        # -------------------- Step 2: Subcategory (indexed, greedy) --------------------
        pred_sub = prior_map.get(name, {}).get("sub","") if args.stage3_only else ""
        if not args.stage3_only:
            if not (args.resume and name in done2):
                candidates_sub = []
                for c in cats:
                    candidates_sub.extend(cat_to_subs.get(c, []))
                seen=set(); candidates_sub=[x for x in candidates_sub if not (x in seen or seen.add(x))][:80]
                if not candidates_sub:
                    candidates_sub = (gt_subs[:80] if gt_subs else ["other"])
                if "other" not in candidates_sub: candidates_sub.append("other")

                sub_prompt = index_prompt(
                    "Select the single best subcategory for this image from the list below:",
                    candidates_sub
                )

                txt2 = (md_query(md_model, processor_md, image, sub_prompt, 8)
                        if args.moondream else qwen_generate_text(processor, model, image, sub_prompt, 8))
                k = parse_index_strict(txt2, len(candidates_sub))
                if f2 is not None:
                    f2.write(json.dumps({"image_name":name,"text":txt2}, ensure_ascii=False)+"\n")
                pred_sub = candidates_sub[(k-1) if k>0 else candidates_sub.index("other")]
            else:
                with open(raw2,"r",encoding="utf-8") as rf:
                    for line in rf:
                        obj=json.loads(line)
                        if obj.get("image_name")==name:
                            fallback = parse_json_field(obj.get("text",""), "subcategory")
                            if fallback:
                                pred_sub = fuzzy_pick(fallback, gt_subs or ["other"], or_else="other")
                            break

        # refine majors using sub's parents
        if pred_sub and pred_sub in sub_to_parents:
            parents = set(sub_to_parents[pred_sub])
            keep = [c for c in cats if c in parents]
            if keep: cats = keep
        if len(cats)>1 and "Others" in cats:
            cats = [c for c in cats if c!="Others"]
        cats = cats[:1]  # keep precision profile tight

        # -------------------- Step 3: Cooking style (indexed, greedy) --------------------
        pred_style = prior_map.get(name, {}).get("style","") if args.stage3_only else ""
        if not (args.resume and name in done3):
            cand_style = []
            if pred_sub:
                cand_style.extend(style_maps.get("sub", {}).get(pred_sub, []))
            for c in cats:
                cand_style.extend(style_maps.get("cat", {}).get(c, []))
            seen=set(); cand_style=[x for x in cand_style if not (x in seen or seen.add(x))][:15]
            if not cand_style:
                cand_style = (gt_styles[:15] if gt_styles else DEFAULT_COOKING_STYLES[:15])
            if "other" not in cand_style: cand_style.append("other")

            style_prompt = index_prompt(
                "Select the single best cooking style for this image from the list below:",
                cand_style
            )

            txt3 = (md_query(md_model, processor_md, image, style_prompt, 6)
                    if args.moondream else qwen_generate_text(processor, model, image, style_prompt, 6))
            j = parse_index_strict(txt3, len(cand_style))
            if f3 is not None:
                f3.write(json.dumps({"image_name":name,"text":txt3}, ensure_ascii=False)+"\n")
            pred_style = cand_style[(j-1) if j>0 else cand_style.index("other")]
        else:
            with open(raw3,"r",encoding="utf-8") as rf:
                for line in rf:
                    obj=json.loads(line)
                    if obj.get("image_name")==name:
                        fallback = parse_json_field(obj.get("text",""), "cooking_style")
                        if fallback:
                            pred_style = fuzzy_pick(fallback, gt_styles or DEFAULT_COOKING_STYLES, or_else="other")
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

    # ============================== Save predictions ==============================
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_path, index=False)

    # ============================== Metrics ==============================
    metrics: Dict[str, Any] = {}

    # Categories (multi-label; standard)
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

    # Subcategory (standard single-label)
    if "subcategory" in df.columns:
        y_true_sub = [str(x).strip() for x in df["subcategory"].tolist()]
        idx_sub = {r["image_name"]: r["pred_subcategory"] for r in pred_df.to_dict(orient="records")}
        y_pred_sub = [idx_sub.get(n, "other") for n in df["image_name"].tolist()]
        metrics["subcategory"] = compute_basic_metrics(y_true_sub, y_pred_sub)

    # Cooking style (standard single-label)
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

    # === FIELD-LEVEL METRICS (screenshot-style) ===
    field_metrics = {}

    # CATEGORY: correct if full set matches exactly (strict)
    truth_cats_set = [set([t.strip() for t in str(s).split(",") if t.strip()]) for s in df["category"]]
    pred_cats_set  = [set(pred_map_rows.get(n, [])) for n in df["image_name"]]
    cat_tp = sum(1 for a,b in zip(truth_cats_set, pred_cats_set) if a==b)
    cat_fp = sum(1 for a,b in zip(truth_cats_set, pred_cats_set) if a!=b and len(b)>0)
    cat_fn = sum(1 for a,b in zip(truth_cats_set, pred_cats_set) if a!=b and len(b)==0)
    prec = cat_tp / (cat_tp + cat_fp + 1e-8)
    rec  = cat_tp / (cat_tp + cat_fn + 1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)
    field_metrics["CATEGORY"] = {"precision":prec, "recall":rec, "f1":f1}

    # SUBCATEGORY: single-label exact
    if "subcategory" in df.columns:
        tp = sum(1 for a,b in zip(y_true_sub,y_pred_sub) if a==b and b!="")
        fp = sum(1 for a,b in zip(y_true_sub,y_pred_sub) if a!=b and b!="")
        fn = sum(1 for a,b in zip(y_true_sub,y_pred_sub) if b=="")
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1 = 2*prec*rec/(prec+rec+1e-8)
        field_metrics["SUBCATEGORY"] = {"precision":prec, "recall":rec, "f1":f1}

    # COOKING_STYLE: single-label exact
    if "cooking_style" in df.columns:
        tp = sum(1 for a,b in zip(y_true_style,y_pred_style) if a==b and b!="")
        fp = sum(1 for a,b in zip(y_true_style,y_pred_style) if a!=b and b!="")
        fn = sum(1 for a,b in zip(y_true_style,y_pred_style) if b=="")
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1 = 2*prec*rec/(prec+rec+1e-8)
        field_metrics["COOKING_STYLE"] = {"precision":prec, "recall":rec, "f1":f1}

    metrics["field_level"] = field_metrics

    # ---------- Per-label metrics (CSV + JSON) ----------
    if not args.no_per_label:
        # Build per-label ground-truth/pred sets
        gt_cat_sets  = [set([t.strip() for t in str(s).split(",") if t.strip()]) for s in df["category"]]
        pred_cat_sets= pred_cats_set
        all_subs = sorted(set(gt_subs))
        all_styles = sorted(set(gt_styles if gt_styles else DEFAULT_COOKING_STYLES))

        # subcategory sets: wrap single labels for one-vs-rest accounting
        if "subcategory" in df.columns:
            gt_sub_sets   = [set([a]) if str(a).strip() else set() for a in y_true_sub]
            pred_sub_sets = [set([b]) if str(b).strip() else set() for b in y_pred_sub]
        else:
            gt_sub_sets, pred_sub_sets, all_subs = [], [], []

        # cooking style sets
        if "cooking_style" in df.columns:
            gt_style_sets   = [set([a]) if str(a).strip() else set() for a in y_true_style]
            pred_style_sets = [set([b]) if str(b).strip() else set() for b in y_pred_style]
        else:
            gt_style_sets, pred_style_sets, all_styles = [], [], []

        per_cat  = compute_per_label_stats(gt_cat_sets,   pred_cat_sets,   ALLOWED_CATEGORIES)
        per_sub  = compute_per_label_stats(gt_sub_sets,   pred_sub_sets,   all_subs)
        per_cook = compute_per_label_stats(gt_style_sets, pred_style_sets, all_styles)

        write_per_label_csv(per_cat,  str(out_dir / "per_label_categories.csv"))
        write_per_label_csv(per_sub,  str(out_dir / "per_label_subcategory.csv"))
        write_per_label_csv(per_cook, str(out_dir / "per_label_cooking_style.csv"))

        (out_dir / "per_label_metrics.json").write_text(
            json.dumps({"categories": per_cat, "subcategory": per_sub, "cooking_style": per_cook}, indent=2),
            encoding="utf-8"
        )

    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved:\n- Stage1 raw: {raw1}\n- Stage2 raw: {raw2}\n- Stage3 raw: {raw3}\n- Predictions: {pred_path}\n- Metrics: {metrics_path}\n")

if __name__ == "__main__":
    main()
