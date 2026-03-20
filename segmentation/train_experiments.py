import os, sys, json
sys.path.insert(0, "/home/jovyan/MoEClosedWoundSurgMon/segmentation/sam3")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import cv2
import numpy as np

print("CUDA:", torch.cuda.is_available(), flush=True)

from sam3 import build_sam3_image_model

class FUSeg(Dataset):
    def __init__(self, split="train", image_size=1008):
        self.image_size = image_size
        self.img_dir = Path(f"data/FUSeg/{split}/images")
        self.lbl_dir = Path(f"data/FUSeg/{split}/labels")
        all_imgs = sorted(self.img_dir.glob("*.png")) + sorted(self.img_dir.glob("*.jpg"))
        self.imgs = [p for p in all_imgs if cv2.imread(str(p)) is not None]
        print(f"[FUSeg {split}] {len(self.imgs)} valid images", flush=True)
        self.caption = "diabetic foot ulcer wound with surrounding erythema and tissue damage"

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        lbl_path = self.lbl_dir / img_path.name
        if not lbl_path.exists():
            lbl_path = self.lbl_dir / (img_path.stem + ".png")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            lbl = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            lbl = cv2.resize(lbl, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        lbl = (lbl > 127).astype(np.float32)
        ys, xs = np.where(lbl > 0)
        if len(xs) > 0:
            jitter = int(0.05 * self.image_size)
            x1 = max(0, int(xs.min()) - np.random.randint(0, jitter+1))
            y1 = max(0, int(ys.min()) - np.random.randint(0, jitter+1))
            x2 = min(self.image_size-1, int(xs.max()) + np.random.randint(0, jitter+1))
            y2 = min(self.image_size-1, int(ys.max()) + np.random.randint(0, jitter+1))
        else:
            x1, y1, x2, y2 = 0, 0, self.image_size-1, self.image_size-1
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32) / self.image_size
        return (torch.from_numpy(img).permute(2,0,1).float() / 255.0,
                torch.from_numpy(lbl).unsqueeze(0),
                torch.from_numpy(bbox),
                self.caption)

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred).flatten(1)
    target = target.flatten(1)
    inter = (pred * target).sum(1)
    return 1 - (2*inter + eps) / (pred.sum(1) + target.sum(1) + eps)

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1-pt)**gamma * bce).mean()

def combined_loss(pred, target):
    return dice_loss(pred, target).mean() + focal_loss(pred, target)

class CaptionAdapter(nn.Module):
    def __init__(self, text_dim=1024, prompt_dim=256, n_tokens=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Sequential(
            nn.Linear(text_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim * n_tokens),
        )
        self.norm = nn.LayerNorm(prompt_dim)
    def forward(self, x):
        out = self.proj(x).view(x.shape[0], self.n_tokens, -1)
        return self.norm(out)

def forward_pass(model, images, captions, adapter=None,
                 use_native_text=True, use_adapter=False, device="cuda"):
    B = images.shape[0]
    n_levels = model.transformer.encoder.num_feature_levels
    with torch.set_grad_enabled(use_native_text or use_adapter):
        text_mask, text_emb_256, text_emb_1024 = \
            model.backbone.language_backbone(text=captions, device=device)
    feats_all, pos_all, masks, _ = model.backbone.vision_backbone(images)
    feats_enc = feats_all[:n_levels]
    pos_enc   = pos_all[:n_levels]
    feat_sizes = [(f.shape[2], f.shape[3]) for f in feats_enc]
    src_seq    = [f.flatten(2).permute(2, 0, 1) for f in feats_enc]
    pos_seq    = [p.flatten(2).permute(2, 0, 1) for p in pos_enc]
    text_for_encoder = text_emb_256 if use_native_text else \
                       torch.zeros(32, B, 256, device=device)
    enc_out = model.transformer.encoder(
        src=src_seq, prompt=text_for_encoder,
        src_pos=pos_seq, feat_sizes=feat_sizes,
    )
    query_embed = model.transformer.decoder.query_embed.weight
    qe  = query_embed.unsqueeze(1).repeat(1, B, 1)
    tgt = torch.zeros_like(qe)
    memory_text = enc_out["memory_text"] if use_native_text else \
                  torch.zeros(32, B, 256, device=device)
    dec_out = model.transformer.decoder(
        tgt=tgt, memory=enc_out["memory"],
        pos=enc_out["pos_embed"], memory_text=memory_text,
        text_attention_mask=None,
        level_start_index=enc_out["level_start_index"],
        spatial_shapes=enc_out["spatial_shapes"],
        valid_ratios=enc_out["valid_ratios"],
    )
    obj_queries = dec_out[0].permute(0, 2, 1, 3)
    if use_adapter and adapter is not None:
        cls_feat = text_emb_1024.mean(0)
        prompt = adapter(cls_feat).permute(1, 0, 2)
    elif use_native_text:
        prompt = enc_out["memory_text"]
    else:
        prompt = torch.zeros(32, B, 256, device=device)
    seg_out = model.segmentation_head(
        backbone_feats=feats_all,
        obj_queries=obj_queries,
        image_ids=torch.arange(B, device=device),
        encoder_hidden_states=enc_out["memory"],
        prompt=prompt,
    )
    if isinstance(seg_out, dict):
        logits = seg_out.get("pred_masks",
                 seg_out.get("semantic_seg", list(seg_out.values())[0]))
    else:
        logits = seg_out
    return logits

def run_experiment(exp_name, use_native_text, use_adapter,
                   n_epochs=10, batch_size=2, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    os.makedirs(f"outputs/{exp_name}", exist_ok=True)
    print(f"\n{'='*60}\n  Starting: {exp_name}\n{'='*60}", flush=True)

    m = build_sam3_image_model(
        checkpoint_path="checkpoints/sam3.pt",
        load_from_HF=False, eval_mode=False,
        enable_segmentation=True, device="cuda",
    )
    for p in m.parameters():
        p.requires_grad = False

    ada = None
    if use_native_text:
        for p in m.transformer.decoder.parameters(): p.requires_grad = True
        for p in m.segmentation_head.parameters():   p.requires_grad = True
        for p in m.backbone.language_backbone.resizer.parameters(): p.requires_grad = True
    elif use_adapter:
        for p in m.segmentation_head.cross_attend_prompt.parameters(): p.requires_grad = True
        ada = CaptionAdapter(1024, 256, 4).cuda()
    else:
        for p in m.transformer.decoder.parameters(): p.requires_grad = True
        for p in m.segmentation_head.parameters():   p.requires_grad = True

    params = [p for p in m.parameters() if p.requires_grad]
    if ada: params += list(ada.parameters())
    print(f"  Trainable: {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)

    train_ds = FUSeg("train",      image_size=1008)
    val_ds   = FUSeg("validation", image_size=1008)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    opt   = AdamW(params, lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=n_epochs)
    history = []

    for epoch in range(n_epochs):
        m.train()
        if ada: ada.train()
        train_loss, train_dice = 0.0, 0.0
        for i, (imgs, lbls, bboxes, caps) in enumerate(train_dl):
            imgs = imgs.cuda(); lbls = lbls.cuda()
            opt.zero_grad()
            logits = forward_pass(m, imgs, list(caps), adapter=ada,
                                  use_native_text=use_native_text,
                                  use_adapter=use_adapter)
            logits = logits[:, 0:1]
            lbl_r  = F.interpolate(lbls, size=logits.shape[-2:], mode='nearest')
            loss   = combined_loss(logits, lbl_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                pred  = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * lbl_r).sum()
                dice  = (2*inter+1e-6)/(pred.sum()+lbl_r.sum()+1e-6)
            train_loss += loss.item()
            train_dice += dice.item()
            if i % 50 == 0:
                print(f"  [{exp_name}] Ep{epoch+1} batch {i}/{len(train_dl)} "
                      f"loss={loss.item():.4f} dice={dice.item():.4f}", flush=True)

        sched.step()
        train_loss /= len(train_dl)
        train_dice /= len(train_dl)

        m.eval()
        if ada: ada.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, lbls, bboxes, caps in val_dl:
                imgs = imgs.cuda(); lbls = lbls.cuda()
                logits = forward_pass(m, imgs, list(caps), adapter=ada,
                                      use_native_text=use_native_text,
                                      use_adapter=use_adapter)
                logits = logits[:, 0:1]
                lbl_r  = F.interpolate(lbls, size=logits.shape[-2:], mode='nearest')
                loss   = combined_loss(logits, lbl_r)
                pred   = (torch.sigmoid(logits) > 0.5).float()
                inter  = (pred * lbl_r).sum()
                dice   = (2*inter+1e-6)/(pred.sum()+lbl_r.sum()+1e-6)
                val_loss += loss.item()
                val_dice += dice.item()

        val_loss /= len(val_dl)
        val_dice /= len(val_dl)
        history.append(dict(epoch=epoch+1,
                            train_loss=train_loss, train_dice=train_dice,
                            val_loss=val_loss,     val_dice=val_dice))
        print(f"  ✓ [{exp_name}] Ep {epoch+1:02d}/{n_epochs} | "
              f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
              f"val loss={val_loss:.4f} dice={val_dice:.4f}", flush=True)

    with open(f"outputs/{exp_name}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(m.state_dict(), f"outputs/{exp_name}/model.pt")
    if ada: torch.save(ada.state_dict(), f"outputs/{exp_name}/adapter.pt")
    print(f"  ✓ Saved outputs/{exp_name}/", flush=True)
    return history

if __name__ == "__main__":
    results = {}
    for exp_name, kwargs in [
        ("baseline",          dict(use_native_text=False, use_adapter=False)),
        ("exp_a_native_text", dict(use_native_text=True,  use_adapter=False)),
        ("exp_b_adapter",     dict(use_native_text=False, use_adapter=True)),
    ]:
        results[exp_name] = run_experiment(exp_name, **kwargs, n_epochs=10, batch_size=2)
    print("\n All experiments done!", flush=True)
