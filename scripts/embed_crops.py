"""
Embed person crops with a DINO backbone (DINOv2 now, DINOv3 once access is granted).

    python scripts/embed_crops.py                         # facebook/dinov2-base
    python scripts/embed_crops.py --model facebook/dinov3-vitb16-pretrain-lvd1689m

Writes <out>/embeddings.npy (float16, L2-normalized, one row per crop) and <out>/embeddings_index.csv
(file, clip, frame, time_s, conf), row-aligned. Features are CLS + patch-mean concatenated, which keeps
small details like a helmet alongside the overall appearance.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")
os.environ.setdefault("HF_HOME", r"D:\Project_paimedia\.hf_cache")   # keep weights off the small C: drive
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path, size):
    im = Image.open(path).convert("RGB").resize((size[0], size[1]), Image.BICUBIC)
    a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1))


@torch.no_grad()
def main(a):
    rows = [r for r in csv.DictReader(open(os.path.join(a.images, "crops.csv"), encoding="utf-8-sig"))]
    if a.kept_only and os.path.exists(os.path.join(a.images, "review.csv")):
        keep = {r["file"] for r in csv.DictReader(open(os.path.join(a.images, "review.csv"), encoding="utf-8"))
                if r["verdict"] == "O"}
        rows = [r for r in rows if r["file"] in keep]
    rows = [r for r in rows if os.path.isfile(os.path.join(a.images, r["file"]))]
    print(f"{len(rows)} crops, model {a.model}")

    model = AutoModel.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    nreg = getattr(model.config, "num_register_tokens", 0)
    skip = 1 + nreg                                   # CLS + register tokens
    feats = np.zeros((len(rows), model.config.hidden_size * 2), np.float16)
    t0 = time.time()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        x = torch.stack([load(os.path.join(a.images, r["file"]), (a.width, a.height)) for r in chunk])
        h = model(pixel_values=x.half().cuda()).last_hidden_state
        f = torch.cat([h[:, 0], h[:, skip:].mean(1)], dim=1)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        feats[i:i + len(chunk)] = f.half().cpu().numpy()
        if (i // a.batch) % 20 == 0:
            done = i + len(chunk)
            print(f"  {done}/{len(rows)}  {done / max(time.time() - t0, 1e-9):.0f} img/s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "embeddings.npy"), feats)
    with open(os.path.join(a.out, "embeddings_index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "clip", "frame", "time_s", "conf"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
    print(f"done: {feats.shape} in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--out", default=os.path.join(DATA, "features"))
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--all", dest="kept_only", action="store_false", help="include crops rejected in review")
    main(ap.parse_args())
