import os, sys, json, base64
sys.path.insert(0, "/home/jovyan/MoEClosedWoundSurgMon/segmentation/sam3")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from sam3 import build_sam3_image_model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Device assignment ──────────────────────────────────────────
GPU_CAPTION = "cuda:0"   # Qwen2-VL-2B
GPU_SEG     = "cuda:1"   # SAM3 fine-tuned

# ── Load Qwen2-VL-2B on GPU 0 ─────────────────────────────────
print("Loading Qwen2-VL-2B on GPU 0...", flush=True)
processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model_qwen = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map=GPU_CAPTION,
)
model_qwen.eval()
print("✓ Qwen2-VL-2B ready", flush=True)

# ── Load fine-tuned SAM3 on GPU 1 ─────────────────────────────
print("Loading fine-tuned SAM3 on GPU 1...", flush=True)
model_sam = build_sam3_image_model(
    checkpoint_path="checkpoints/sam3.pt",
    load_from_HF=False, eval_mode=True,
    enable_segmentation=True, device=GPU_SEG,
)
state = torch.load("outputs/baseline/model.pt", map_location=GPU_SEG)
model_sam.load_state_dict(state)
model_sam.eval()
print("✓ SAM3 ready", flush=True)

WOUND_PROMPT = (
    "You are a clinical wound assessment assistant. "
    "Describe this surgical wound in one concise clinical sentence. Include: "
    "wound type, closure method, wound bed appearance, and surrounding tissue condition."
)

# ── Caption function ───────────────────────────────────────────
def generate_caption(img_pil):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_pil},
        {"type": "text",  "text": WOUND_PROMPT},
    ]}]
    text = processor_qwen.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor_qwen(
        text=[text], images=image_inputs, return_tensors="pt"
    ).to(GPU_CAPTION)
    with torch.no_grad():
        out = model_qwen.generate(**inputs, max_new_tokens=100, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor_qwen.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

# ── SAM3 forward pass ──────────────────────────────────────────
def segment(model, images, caption, device=GPU_SEG):
    B = images.shape[0]
    n_levels = model.transformer.encoder.num_feature_levels
    with torch.no_grad():
        _, text_emb_256, _ = model.backbone.language_backbone(
            text=[caption] * B, device=device)
        feats_all, pos_all, _, _ = model.backbone.vision_backbone(images)
        feats_enc = feats_all[:n_levels]
        pos_enc   = pos_all[:n_levels]
        feat_sizes = [(f.shape[2], f.shape[3]) for f in feats_enc]
        src_seq    = [f.flatten(2).permute(2, 0, 1) for f in feats_enc]
        pos_seq    = [p.flatten(2).permute(2, 0, 1) for p in pos_enc]
        enc_out = model.transformer.encoder(
            src=src_seq,
            prompt=torch.zeros(32, B, 256, device=device),
            src_pos=pos_seq, feat_sizes=feat_sizes,
        )
        query_embed = model.transformer.decoder.query_embed.weight
        qe  = query_embed.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(qe)
        dec_out = model.transformer.decoder(
            tgt=tgt, memory=enc_out["memory"],
            pos=enc_out["pos_embed"],
            memory_text=torch.zeros(32, B, 256, device=device),
            text_attention_mask=None,
            level_start_index=enc_out["level_start_index"],
            spatial_shapes=enc_out["spatial_shapes"],
            valid_ratios=enc_out["valid_ratios"],
        )
        obj_queries = dec_out[0].permute(0, 2, 1, 3)
        seg_out = model.segmentation_head(
            backbone_feats=feats_all,
            obj_queries=obj_queries,
            image_ids=torch.arange(B, device=device),
            encoder_hidden_states=enc_out["memory"],
            prompt=torch.zeros(32, B, 256, device=device),
        )
    if isinstance(seg_out, dict):
        logits = seg_out.get("pred_masks",
                 seg_out.get("semantic_seg", list(seg_out.values())[0]))
    else:
        logits = seg_out
    return logits

# ── Load dataset ───────────────────────────────────────────────
def load_image(img_field):
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    elif isinstance(img_field, str):
        if img_field.startswith("http"):
            import requests
            return Image.open(BytesIO(requests.get(img_field).content)).convert("RGB")
        else:
            return Image.open(BytesIO(base64.b64decode(img_field))).convert("RGB")
    return Image.fromarray(img_field).convert("RGB")

print("Loading SurgWound...", flush=True)
ds = load_dataset("xuxuxuxuxu/SurgWound", split="test")
print(f"✓ {len(ds)} samples", flush=True)

os.makedirs("outputs/pipeline_masks", exist_ok=True)
results = []
N = min(50, len(ds))

print(f"\nRunning two-GPU pipeline on {N} images...", flush=True)
print("GPU 0: Qwen2-VL-2B captioning", flush=True)
print("GPU 1: SAM3 segmentation", flush=True)
print("─" * 60, flush=True)

import time
for i in range(N):
    t0 = time.time()
    sample = ds[i]
    img_pil = load_image(sample["image"])
    orig_w, orig_h = img_pil.size
    img_np = np.array(img_pil)

    # Step 1: Caption on GPU 0
    caption = generate_caption(img_pil)

    # Step 2: Segment on GPU 1
    img_resized = cv2.resize(img_np, (1008, 1008))
    img_t = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).to(GPU_SEG) / 255.0
    logits = segment(model_sam, img_t, caption)
    logits = logits[:, 0:1]
    mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    coverage = float(mask.sum()) / (orig_w * orig_h) * 100

    # Visualize
    overlay = img_np.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 200, 100), 2)

    # Add caption text to overlay
    words = caption.split()
    line, lines = [], []
    for w in words:
        if len(" ".join(line + [w])) < 60:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line: lines.append(" ".join(line))
    for j, l in enumerate(lines[:3]):
        cv2.putText(overlay, l, (10, 30 + j*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(overlay, f"Coverage: {coverage:.1f}% | {sample.get('answer','')}",
                (10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,150), 2)

    vis = np.concatenate([img_np, overlay], axis=1)
    out_path = f"outputs/pipeline_masks/sample_{i:03d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    elapsed = time.time() - t0
    results.append({
        "idx": i,
        "image_name": sample.get("image_name", str(i)),
        "healing_status": sample.get("answer", ""),
        "caption": caption,
        "coverage_pct": round(coverage, 2),
        "time_sec": round(elapsed, 1),
    })
    print(f"[{i:02d}/{N}] {elapsed:.1f}s | {sample.get('answer',''):12s} | "
          f"cov={coverage:.1f}% | {caption[:80]}", flush=True)

with open("outputs/pipeline_masks/results.json", "w") as f:
    json.dump(results, f, indent=2)

avg_time = sum(r["time_sec"] for r in results) / len(results)
healed     = [r["coverage_pct"] for r in results if "not" not in r["healing_status"].lower() and r["healing_status"]]
not_healed = [r["coverage_pct"] for r in results if "not healed" in r["healing_status"].lower()]

print("\n" + "─"*60, flush=True)
print(f"✓ Done — {len(results)} images processed", flush=True)
print(f"  Avg time per image: {avg_time:.1f}s", flush=True)
if healed:     print(f"  Healed avg coverage:     {sum(healed)/len(healed):.1f}%", flush=True)
if not_healed: print(f"  Not Healed avg coverage: {sum(not_healed)/len(not_healed):.1f}%", flush=True)
print(f"\nMasks saved to outputs/pipeline_masks/", flush=True)
