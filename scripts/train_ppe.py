"""
Train the PPE heads (helmet + smart vest) on frozen DINOv3 features.

    python scripts/train_ppe.py                  # train both heads, write __MODEL__/ppe_head.pt
    python scripts/train_ppe.py --attr vest      # one attribute only

Only rows labeled directly (source=rep) are used: the propagated copies are near-duplicates of those and
would leak across the split. Train/val is split by TIME BLOCK inside each clip (--block seconds), never
randomly per image, because crops seconds apart still look nearly identical.

Both heads sit on the same frozen backbone, so this trains a linear (or 1-hidden-layer) probe per
attribute over the precomputed embeddings. The 4 classes are the combination of the two outputs.
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")
ATTRS = ["helmet", "vest"]
KO = {"helmet": "안전모", "vest": "스마트조끼"}


def load(feat_dir, img_dir):
    idx = list(csv.DictReader(open(os.path.join(feat_dir, "embeddings_index.csv"), encoding="utf-8-sig")))
    feats = np.load(os.path.join(feat_dir, "embeddings.npy")).astype(np.float32)
    row_of = {r["file"]: i for i, r in enumerate(idx)}
    labels = {a: {} for a in ATTRS}
    for r in csv.DictReader(open(os.path.join(img_dir, "labels.csv"), encoding="utf-8")):
        if r["source"] == "rep" and r["value"] in ("O", "X") and r["file"] in row_of:
            labels[r["attr"]][r["file"]] = 1 if r["value"] == "O" else 0
    return idx, feats, row_of, labels


def split_blocks(idx, files, block, val_frac, seed=0):
    """Assign whole (clip, time-block) chunks to train or val."""
    blocks = sorted({(idx[i]["clip"], int(float(idx[i]["time_s"]) // block)) for i in files})
    rng = np.random.default_rng(seed)
    per_clip = {}
    for clip, b in blocks:
        per_clip.setdefault(clip, []).append(b)
    val = set()
    for clip, bs in per_clip.items():
        bs = sorted(bs)
        pick = rng.choice(len(bs), max(1, int(round(len(bs) * val_frac))), replace=False)
        val |= {(clip, bs[p]) for p in pick}
    return val


class Head(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1)) \
            if hidden else nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 0) & (y == 0)).sum())        # X (violation) treated as the positive class
    fp = int(((pred == 0) & (y == 1)).sum())
    fn = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 1) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(n=len(y), acc=(tp + tn) / max(len(y), 1), x_precision=prec, x_recall=rec,
                x_f1=2 * prec * rec / max(prec + rec, 1e-9), tp=tp, fp=fp, fn=fn, tn=tn)


def train_attr(attr, feats, rows, y, val_mask, a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.from_numpy(feats[rows[~val_mask]]).to(dev)
    ytr = torch.from_numpy(y[~val_mask].astype(np.float32)).to(dev)
    Xva = torch.from_numpy(feats[rows[val_mask]]).to(dev)
    yva = y[val_mask]
    pos_weight = torch.tensor([(ytr == 0).sum().item() / max((ytr == 1).sum().item(), 1)], device=dev)
    model = Head(feats.shape[1], a.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state, patience = -1, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            lossf(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva)).cpu().numpy()
        m = metrics(yva, pv)
        score = m["x_f1"]
        if score > best:
            best, best_state, patience = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= a.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xva)).cpu().numpy()
    return model, metrics(yva, pv), pv, yva


def main(a):
    idx, feats, row_of, labels = load(a.features, a.images)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    report, states = {}, {}
    for attr in ([a.attr] if a.attr else ATTRS):
        files = {row_of[f]: v for f, v in labels[attr].items()}
        rows = np.array(sorted(files))
        y = np.array([files[i] for i in rows])
        if a.holdout_clip:                       # honest generalization test: a whole camera unseen
            val_mask = np.array([idx[i]["clip"] == a.holdout_clip for i in rows])
        else:
            val_blocks = split_blocks(idx, rows, a.block, a.val_frac, a.seed)
            val_mask = np.array([(idx[i]["clip"], int(float(idx[i]["time_s"]) // a.block)) in val_blocks
                                 for i in rows])
        clips = Counter(idx[i]["clip"] for i in rows)
        print(f"\n[{KO[attr]}] 라벨 {len(rows)}개  O {int(y.sum())} / X {int((y == 0).sum())}  "
              f"| train {int((~val_mask).sum())} / val {int(val_mask.sum())}  | 영상별 {dict(clips)}")
        if (y[~val_mask] == 0).sum() < 5 or (y[val_mask] == 0).sum() < 2:
            print("   X 표본이 너무 적어 학습/평가가 의미 없음 — 건너뜀")
            continue
        model, m, pv, yva = train_attr(attr, feats, rows, y, val_mask, a)
        print(f"   val: 정확도 {m['acc']:.3f} | X 재현율 {m['x_recall']:.3f} 정밀도 {m['x_precision']:.3f} "
              f"F1 {m['x_f1']:.3f} (TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']})")
        report[attr] = m
        states[attr] = model.state_dict()
    if states:
        torch.save(dict(states=states, dim=feats.shape[1], hidden=a.hidden,
                        backbone="facebook/dinov3-vitb16-pretrain-lvd1689m", size=[112, 224],
                        report=report), a.out)
        print(f"\n저장: {a.out}")
        json.dump(report, open(os.path.splitext(a.out)[0] + "_report.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(DATA, "features"))
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--out", default=os.path.join(ROOT, "__MODEL__", "ppe_head.pt"))
    ap.add_argument("--attr", choices=ATTRS)
    ap.add_argument("--hidden", type=int, default=256, help="0 = linear probe")
    ap.add_argument("--block", type=float, default=120.0, help="train/val split block, seconds")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--holdout-clip", help="use this clip as the whole validation set (cross-camera test)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
