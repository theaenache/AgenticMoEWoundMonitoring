import os, sys, json, base64, time
sys.path.insert(0, "/home/jovyan/MoEClosedWoundSurgMon/segmentation/sam3")

import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from pathlib import Path
from datasets import load_dataset

CAPTION_PROMPT = (
    "You are a clinical wound assessment assistant. "
    "Describe this wound image in one concise clinical paragraph. "
    "Include: wound type (ulcer/surgical/traumatic), closure method if present "
    "(sutures/staples/open/none), wound bed appearance (granulation/slough/necrosis/epithelialization), "
    "surrounding tissue condition (erythema/swelling/maceration/healthy), "
    "wound edges (defined/irregular/undermined), "
    "and healing status assessment (actively healing/stalled/infected/closed)."
)

def load_image_pil(img_field):
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    elif isinstance(img_field, str):
        if img_field.startswith("http"):
            import requests
            return Image.open(BytesIO(requests.get(img_field).content)).convert("RGB")
        else:
            return Image.open(BytesIO(base64.b64decode(img_field))).convert("RGB")
    elif isinstance(img_field, (np.ndarray,)):
        return Image.fromarray(img_field).convert("RGB")
    return img_field.convert("RGB")

def load_existing(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {d["idx"]: d for d in data}
    return {}

os.makedirs("captions", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# MODEL 1 — Qwen2-VL-7B
# ══════════════════════════════════════════════════════════════
def run_qwen7b():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import gc

    print("\n" + "═"*60, flush=True)
    print("  QWEN2-VL-7B CAPTIONING", flush=True)
    print("═"*60, flush=True)

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", use_fast=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    print("✓ Qwen2-VL-7B loaded on GPU 0", flush=True)

    def caption(img_pil, extra_context=""):
        prompt = CAPTION_PROMPT
        if extra_context:
            prompt += f" Additional context: {extra_context}"
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_pil},
            {"type": "text",  "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150,
                do_sample=False, temperature=1.0)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()

    # ── FUSeg train ──
    print("\n[1/3] FUSeg train split...", flush=True)
    existing = load_existing("captions/qwen7b_fuseg_train.json")
    results = list(existing.values())
    done_ids = set(existing.keys())

    img_dir = Path("data/FUSeg/train/images")
    all_imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    valid_imgs = [p for p in all_imgs if cv2.imread(str(p)) is not None]
    print(f"  {len(valid_imgs)} images, {len(done_ids)} already done", flush=True)

    for i, img_path in enumerate(valid_imgs):
        if i in done_ids:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        t0 = time.time()
        try:
            cap = caption(img_pil,
                extra_context="diabetic foot ulcer, chronic wound")
        except Exception as e:
            cap = f"ERROR: {e}"
        results.append({
            "idx": i, "image_name": img_path.name,
            "split": "train", "dataset": "fuseg",
            "caption": cap, "time_sec": round(time.time()-t0, 1)
        })
        if i % 50 == 0 or i < 5:
            with open("captions/qwen7b_fuseg_train.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{i}/{len(valid_imgs)}] {cap[:80]}", flush=True)

    with open("captions/qwen7b_fuseg_train.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ FUSeg train done — {len(results)} captions", flush=True)

    # ── FUSeg val ──
    print("\n[2/3] FUSeg val split...", flush=True)
    existing = load_existing("captions/qwen7b_fuseg_val.json")
    results = list(existing.values())
    done_ids = set(existing.keys())

    img_dir = Path("data/FUSeg/validation/images")
    all_imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    valid_imgs = [p for p in all_imgs if cv2.imread(str(p)) is not None]
    print(f"  {len(valid_imgs)} images", flush=True)

    for i, img_path in enumerate(valid_imgs):
        if i in done_ids:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        t0 = time.time()
        try:
            cap = caption(img_pil,
                extra_context="diabetic foot ulcer, chronic wound")
        except Exception as e:
            cap = f"ERROR: {e}"
        results.append({
            "idx": i, "image_name": img_path.name,
            "split": "val", "dataset": "fuseg",
            "caption": cap, "time_sec": round(time.time()-t0, 1)
        })
        if i % 50 == 0 or i < 5:
            with open("captions/qwen7b_fuseg_val.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{i}/{len(valid_imgs)}] {cap[:80]}", flush=True)

    with open("captions/qwen7b_fuseg_val.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ FUSeg val done — {len(results)} captions", flush=True)

    # ── SurgWound ──
    print("\n[3/3] SurgWound test split...", flush=True)
    existing = load_existing("captions/qwen7b_surgwound.json")
    results = list(existing.values())
    done_ids = set(existing.keys())

    ds = load_dataset("xuxuxuxuxu/SurgWound", split="test")
    print(f"  {len(ds)} images", flush=True)

    for i, sample in enumerate(ds):
        if i in done_ids:
            continue
        img_pil = load_image_pil(sample["image"])
        healing = sample.get("answer", "")
        field   = sample.get("field", "")
        extra   = f"closed surgical wound. {field}: {healing}" if healing else "closed surgical wound"
        t0 = time.time()
        try:
            cap = caption(img_pil, extra_context=extra)
        except Exception as e:
            cap = f"ERROR: {e}"
        results.append({
            "idx": i,
            "image_name": sample.get("image_name", str(i)),
            "healing_status": healing,
            "field": field,
            "dataset": "surgwound",
            "caption": cap,
            "time_sec": round(time.time()-t0, 1)
        })
        if i % 50 == 0 or i < 5:
            with open("captions/qwen7b_surgwound.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{i}/{len(ds)}] [{healing}] {cap[:80]}", flush=True)

    with open("captions/qwen7b_surgwound.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ SurgWound done — {len(results)} captions", flush=True)

    del model, processor
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("\n✓ Qwen2-VL-7B captioning complete", flush=True)


# ══════════════════════════════════════════════════════════════
# MODEL 2 — LLaVA-Med
# ══════════════════════════════════════════════════════════════
def run_llava_med():
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (get_model_name_from_path,
                                 process_images, tokenizer_image_token)
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    import gc

    print("\n" + "═"*60, flush=True)
    print("  LLAVA-MED CAPTIONING", flush=True)
    print("═"*60, flush=True)

    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path, model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_4bit=True,
    )
    model.eval()
    print("✓ LLaVA-Med loaded on GPU 0", flush=True)

    def caption(img_pil, extra_context=""):
        prompt_text = CAPTION_PROMPT
        if extra_context:
            prompt_text += f" Additional context: {extra_context}"
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
        image_tensor = process_images(
            [img_pil], image_processor, model.config
        ).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids, images=image_tensor,
                max_new_tokens=150, do_sample=False)
        return tokenizer.decode(
            out[0][input_ids.shape[1]:],
            skip_special_tokens=True).strip()

    # ── FUSeg train ──
    print("\n[1/3] FUSeg train split...", flush=True)
    existing = load_existing("captions/llava_fuseg_train.json")
    results = list(existing.values())
    done_ids = set(existing.keys())

    img_dir = Path("data/FUSeg/train/images")
    all_imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    valid_imgs = [p for p in all_imgs if cv2.imread(str(p)) is not None]

    for i, img_path in enumerate(valid_imgs):
        if i in done_ids:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        t0 = time.time()
        try:
            cap = caption(img_pil,
                extra_context="diabetic foot ulcer, chronic wound")
        except Exception as e:
            cap = f"ERROR: {e}"
        results.append({
            "idx": i, "image_name": img_path.name,
            "split": "train", "dataset": "fuseg",
            "caption": cap, "time_sec": round(time.time()-t0, 1)
        })
        if i % 50 == 0 or i < 5:
            with open("captions/llava_fuseg_train.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{i}/{len(valid_imgs)}] {cap[:80]}", flush=True)

    with open("captions/llava_fuseg_train.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ FUSeg train done — {len(results)} captions", flush=True)

    # ── FUSeg val ──
    print("\n[2/3] FUSeg val split...", flush=True)
    existing = load_existing("captions/llava_fuseg_val.json")
    results = list(existing.values())
    done_ids = set(existing.keys())

    img_dir = Path("data/FUSeg/validation/images")
    all_imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    valid_imgs = [p for p in all_imgs if cv2.imread(str(p)) is not None]

    for i, img_path in enumerate(valid_imgs):
        if i in done_ids:
            continue
        img_pil = Image.open(img_path).convert("RGB")
        t0 = time.time()
        try:
            cap = caption(img_pil,
                extra_context="diabetic foot ulcer, chronic wound")
        except Exception as e:
            cap = f"ERROR: {e}"
        results.append({
            "idx": i, "image_name": img_path.name,
            "split": "val", "dataset": "fuseg",
            "caption": cap, "time_sec": round(time.time()-t0, 1)
        })
        if i % 50 == 0 or i < 5:
            with open("captions/llava_fuseg_val.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{i}/{len(valid_imgs)}] {cap[:80]}", flush=True)

    with open("captions/llava_fuseg_val.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ FUSeg val done — {len(results)} captions", flush=True)

    # ── SurgWound ──
    print("\n[3/3] SurgWound test split...", flush=True)
    existing = load_existing("captions/llava_surgwound.json")
    results = list(existing.values())
    done_ids = set(existing.keys())

    ds = load_dataset("xuxuxuxuxu/SurgWound", split="test")

    for i, sample in enumerate(ds):
        if i in done_ids:
            continue
        img_pil = load_image_pil(sample["image"])
        healing = sample.get("answer", "")
        field   = sample.get("field", "")
        extra   = f"closed surgical wound. {field}: {healing}" if healing else "closed surgical wound"
        t0 = time.time()
        try:
            cap = caption(img_pil, extra_context=extra)
        except Exception as e:
            cap = f"ERROR: {e}"
        results.append({
            "idx": i,
            "image_name": sample.get("image_name", str(i)),
            "healing_status": healing,
            "field": field,
            "dataset": "surgwound",
            "caption": cap,
            "time_sec": round(time.time()-t0, 1)
        })
        if i % 50 == 0 or i < 5:
            with open("captions/llava_surgwound.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{i}/{len(ds)}] [{healing}] {cap[:80]}", flush=True)

    with open("captions/llava_surgwound.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ SurgWound done — {len(results)} captions", flush=True)

    del model, tokenizer, image_processor
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("\n✓ LLaVA-Med captioning complete", flush=True)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "llava", "both"],
                        default="both")
    args = parser.parse_args()

    if args.model in ("qwen", "both"):
        run_qwen7b()
    if args.model in ("llava", "both"):
        run_llava_med()

    print("\n✓ All captioning complete!", flush=True)
    print("  captions/qwen7b_fuseg_train.json", flush=True)
    print("  captions/qwen7b_fuseg_val.json", flush=True)
    print("  captions/qwen7b_surgwound.json", flush=True)
    print("  captions/llava_fuseg_train.json", flush=True)
    print("  captions/llava_fuseg_val.json", flush=True)
    print("  captions/llava_surgwound.json", flush=True)
