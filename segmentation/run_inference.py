import os, sys, json, base64
sys.path.insert(0, "/home/jovyan/MoEClosedWoundSurgMon/segmentation/sam3")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from sam3 import build_sam3_image_model

print("Loading fine-tuned model...", flush=True)
model = build_sam3_image_model(
    checkpoint_path="checkpoints/sam3.pt",
    load_from_HF=False, eval_mode=True,
    enable_segmentation=True, device="cuda",
)
state = torch.load("outputs/baseline/model.pt", map_location="cuda")
model.load_state_dict(state)
model.eval()
print("✓ Model loaded", flush=True)

def forward_pass(model, images, device="cuda"):
    B = images.shape[0]
    n_levels = model.transformer.encoder.num_feature_levels
    with torch.no_grad():
        _, text_emb_256, _ = model.backbone.language_backbone(
            text=["closed surgical wound with sutures"] * B, device=device)
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

os.makedirs("outputs/surgwound_masks", exist_ok=True)
results = []
N = min(50, len(ds))

for i in range(N):
    sample = ds[i]
    img_pil = load_image(sample["image"])
    orig_w, orig_h = img_pil.size
    img_np = np.array(img_pil)
    img_resized = cv2.resize(img_np, (1008, 1008))
    img_t = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).cuda() / 255.0

    logits = forward_pass(model, img_t)
    logits = logits[:, 0:1]
    mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    coverage = float(mask.sum()) / (orig_w * orig_h) * 100

    overlay = img_np.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 200, 100), 2)
    label = f"{sample.get('answer','?')} | {coverage:.1f}%"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    vis = np.concatenate([img_np, overlay], axis=1)
    out_path = f"outputs/surgwound_masks/sample_{i:03d}.png"
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    results.append({
        "idx": i,
        "image_name": sample.get("image_name", str(i)),
        "healing_status": sample.get("answer", ""),
        "coverage_pct": round(coverage, 2),
    })
    if i % 5 == 0:
        print(f"  [{i}/{N}] {sample.get('answer','')} | coverage={coverage:.1f}%", flush=True)

with open("outputs/surgwound_masks/results.json", "w") as f:
    json.dump(results, f, indent=2)

healed     = [r["coverage_pct"] for r in results if "not" not in r["healing_status"].lower() and r["healing_status"]]
not_healed = [r["coverage_pct"] for r in results if "not healed" in r["healing_status"].lower()]
print(f"\n✓ Done — {len(results)} masks saved", flush=True)
if healed:     print(f"  Healed avg coverage:     {sum(healed)/len(healed):.1f}%", flush=True)
if not_healed: print(f"  Not Healed avg coverage: {sum(not_healed)/len(not_healed):.1f}%", flush=True)
