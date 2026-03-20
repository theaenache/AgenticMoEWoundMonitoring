import os, sys, json, time, argparse
sys.path.insert(0, "/home/jovyan/MoEClosedWoundSurgMon/segmentation/sam3")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import cv2
import numpy as np
from sam3 import build_sam3_image_model

parser = argparse.ArgumentParser()
parser.add_argument("--exp",    required=True, choices=["qwen", "llava"])
parser.add_argument("--gpu",    required=True, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--batch",  default=2,  type=int)
parser.add_argument("--lr",     default=1e-4, type=float)
parser.add_argument("--seed",   default=42, type=int)
args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}"
EXP    = args.exp
OUT    = f"outputs/{EXP}_captions"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(args.seed)

print(f"\n{'='*60}", flush=True)
print(f"  EXP: {EXP} | GPU: {DEVICE} | epochs: {args.epochs}", flush=True)
print(f"{'='*60}\n", flush=True)

def load_caption_map(path):
    with open(path) as f:
        data = json.load(f)
    return {d["image_name"]: d["caption"] for d in data
            if not d["caption"].startswith("ERROR")}

prefix = "qwen7b" if EXP == "qwen" else "llava"
train_caps = load_caption_map(f"captions/{prefix}_fuseg_train.json")
val_caps   = load_caption_map(f"captions/{prefix}_fuseg_val.json")
print(f"✓ Captions loaded: {len(train_caps)} train, {len(val_caps)} val", flush=True)

FALLBACK = ("diabetic foot ulcer wound with surrounding erythema "
            "and tissue damage requiring wound care")

class FUSegCaptioned(Dataset):
    def __init__(self, split="train", image_size=1008, caption_map=None):
        self.image_size  = image_size
        self.caption_map = caption_map or {}
        self.img_dir = Path(f"data/FUSeg/{split}/images")
        self.lbl_dir = Path(f"data/FUSeg/{split}/labels")
        all_imgs = sorted(self.img_dir.glob("*.png")) + \
                   sorted(self.img_dir.glob("*.jpg"))
        self.imgs = [p for p in all_imgs if cv2.imread(str(p)) is not None]
        print(f"[FUSeg {split}] {len(self.imgs)} valid images", flush=True)

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
            lbl = cv2.resize(lbl, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_NEAREST)
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

        caption = self.caption_map.get(img_path.name, FALLBACK)
        return (torch.from_numpy(img).permute(2,0,1).float() / 255.0,
                torch.from_numpy(lbl).unsqueeze(0),
                torch.from_numpy(np.array([x1,y1,x2,y2], dtype=np.float32) / self.image_size),
                caption)

def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred).flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    return 1 - (2*inter + eps) / (pred.sum(1) + target.sum(1) + eps)

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1-pt)**gamma * bce).mean()

def combined_loss(pred, target):
    return dice_loss(pred, target).mean() + focal_loss(pred, target)

def forward_pass(model, images, captions, device):
    B = images.shape[0]
    n_levels = model.transformer.encoder.num_feature_levels
    with torch.set_grad_enabled(True):
        text_mask, text_emb_256, text_emb_1024 = \
        text_mask, text_emb_256, text_emb_1024 = model.backbone.language_backbone(text=captions, device=torch.device("cuda", args.gpu))
    feats_all, pos_all, _, _ = model.backbone.vision_backbone(images)
    feats_enc  = feats_all[:n_levels]
    pos_enc    = pos_all[:n_levels]
    feat_sizes = [(f.shape[2], f.shape[3]) for f in feats_enc]
    src_seq    = [f.flatten(2).permute(2, 0, 1) for f in feats_enc]
    pos_seq    = [p.flatten(2).permute(2, 0, 1) for p in pos_enc]
    enc_out = model.transformer.encoder(
        src=src_seq, prompt=text_emb_256,
        src_pos=pos_seq, feat_sizes=feat_sizes,
    )
    query_embed = model.transformer.decoder.query_embed.weight
    qe  = query_embed.unsqueeze(1).repeat(1, B, 1)
    tgt = torch.zeros_like(qe)
    dec_out = model.transformer.decoder(
        tgt=tgt, memory=enc_out["memory"],
        pos=enc_out["pos_embed"],
        memory_text=enc_out["memory_text"],
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
        prompt=enc_out["memory_text"],
    )
    if isinstance(seg_out, dict):
        logits = seg_out.get("pred_masks",
                 seg_out.get("semantic_seg", list(seg_out.values())[0]))
    else:
        logits = seg_out
    return logits

print("Loading SAM3...", flush=True)
model = build_sam3_image_model(
    checkpoint_path="checkpoints/sam3.pt",
    load_from_HF=False, eval_mode=False,
    enable_segmentation=True, device=DEVICE,
)
for p in model.parameters():
    p.requires_grad = False
for p in model.transformer.decoder.parameters():
    p.requires_grad = True
for p in model.segmentation_head.parameters():
    p.requires_grad = True
for p in model.backbone.language_backbone.resizer.parameters():
    p.requires_grad = True
model.to(DEVICE)
print(f"✓ Full model moved to {DEVICE}", flush=True)

params = [p for p in model.parameters() if p.requires_grad]
print(f"✓ Trainable: {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)

train_ds = FUSegCaptioned("train",      image_size=1008, caption_map=train_caps)
val_ds   = FUSegCaptioned("validation", image_size=1008, caption_map=val_caps)
train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

opt   = AdamW(params, lr=args.lr, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=args.epochs)

history = []
best_val_dice = 0.0

for epoch in range(args.epochs):
    model.train()
    train_loss, train_dice = 0.0, 0.0
    t_epoch = time.time()

    for i, (imgs, lbls, bboxes, caps) in enumerate(train_dl):
        imgs = imgs.to(DEVICE); lbls = lbls.to(DEVICE)
        opt.zero_grad()
        logits = forward_pass(model, imgs, list(caps), DEVICE)
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
            print(f"  [{EXP}|GPU{args.gpu}] Ep{epoch+1} "
                  f"batch {i}/{len(train_dl)} "
                  f"loss={loss.item():.4f} dice={dice.item():.4f}", flush=True)

    sched.step()
    train_loss /= len(train_dl)
    train_dice /= len(train_dl)

    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for imgs, lbls, bboxes, caps in val_dl:
            imgs = imgs.to(DEVICE); lbls = lbls.to(DEVICE)
            logits = forward_pass(model, imgs, list(caps), DEVICE)
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
    epoch_time = time.time() - t_epoch

    history.append(dict(epoch=epoch+1,
                        train_loss=train_loss, train_dice=train_dice,
                        val_loss=val_loss,     val_dice=val_dice,
                        epoch_time=round(epoch_time, 1)))

    print(f"  ✓ [{EXP}|GPU{args.gpu}] Ep {epoch+1:02d}/{args.epochs} | "
          f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
          f"val loss={val_loss:.4f} dice={val_dice:.4f} | "
          f"time={epoch_time:.0f}s", flush=True)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), f"{OUT}/model_best.pt")
        print(f"  ★ New best: {best_val_dice:.4f}", flush=True)

    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"{OUT}/model_ep{epoch+1}.pt")

    with open(f"{OUT}/history.json", "w") as f:
        json.dump(history, f, indent=2)

torch.save(model.state_dict(), f"{OUT}/model_final.pt")
print(f"\n✓ Done — best val dice: {best_val_dice:.4f}", flush=True)
