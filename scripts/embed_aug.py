"""
Embed illumination-jittered copies of the labeled crops, to make the vest head robust to daylight.

    python scripts/embed_aug.py --n 4

The same smart vest reads orange, yellow or washed-out green depending on sun and exposure, so each
labeled representative is re-embedded --n times with brightness / contrast / saturation / hue jitter
(plus a horizontal flip on half of them). Writes <features>/embeddings_aug.npy and
embeddings_aug_index.csv (file, aug). train_ppe.py --aug appends these rows to the TRAIN split only.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")
os.environ.setdefault("HF_HOME", r"D:\Project_paimedia\.hf_cache")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def jitter(im, rng):
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.6, 1.5)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.7, 1.35)))
    im = ImageEnhance.Color(im).enhance(float(rng.uniform(0.5, 1.5)))       # saturation
    a = np.asarray(im.convert("HSV"), np.int16)
    a[:, :, 0] = (a[:, :, 0] + int(rng.integers(-12, 13))) % 256            # small hue shift
    im = Image.fromarray(a.astype(np.uint8), "HSV").convert("RGB")
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im


def to_tensor(im, size):
    a = (np.asarray(im.resize(size, Image.BICUBIC), np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1))


@torch.no_grad()
def main(a):
    labeled = sorted({r["file"] for r in csv.DictReader(open(os.path.join(a.images, "labels.csv"), encoding="utf-8"))
                      if r["source"] == "rep" and r["value"] in ("O", "X")
                      and os.path.isfile(os.path.join(a.images, r["file"]))})
    print(f"{len(labeled)} labeled crops x {a.n} augmentations")
    model = AutoModel.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    nreg = getattr(model.config, "num_register_tokens", 0)
    skip = 1 + nreg
    rng = np.random.default_rng(a.seed)
    feats = np.zeros((len(labeled) * a.n, model.config.hidden_size * 2), np.float16)
    index, t0, k = [], time.time(), 0
    for i in range(0, len(labeled), a.batch):
        chunk = labeled[i:i + a.batch]
        imgs = [Image.open(os.path.join(a.images, f)).convert("RGB") for f in chunk]
        for j in range(a.n):
            x = torch.stack([to_tensor(jitter(im, rng), (a.width, a.height)) for im in imgs])
            h = model(pixel_values=x.half().cuda()).last_hidden_state
            f = torch.cat([h[:, 0], h[:, skip:].mean(1)], dim=1)
            f = torch.nn.functional.normalize(f.float(), dim=1)
            feats[k:k + len(chunk)] = f.half().cpu().numpy()
            index += [(name, j) for name in chunk]
            k += len(chunk)
        if (i // a.batch) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(labeled)}  {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.features, "embeddings_aug.npy"), feats)
    with open(os.path.join(a.features, "embeddings_aug_index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "aug"])
        w.writerows(index)
    print(f"done: {feats.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--features", default=os.path.join(DATA, "features"))
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
