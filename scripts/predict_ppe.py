"""
Run the trained PPE heads over every crop's precomputed feature.

    python scripts/predict_ppe.py --sheets

Writes __DATA__/predictions.csv (file, helmet, helmet_p, vest, vest_p, labeled) where *_p is P(wearing).
Crops already labeled by hand keep their label in the `labeled` column, so accuracy on them is printed as
a sanity check and the unlabeled rest is what the labeling tool then serves as pre-filled defaults.
--sheets writes contact sheets per predicted class and for the most uncertain crops.
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")
ATTRS = ["helmet", "vest"]
KO = {"helmet": "안전모", "vest": "스마트조끼"}


def sheets(rows, images, out_dir, n=60, cols=15):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cell = (80, 160)

    def sheet(sel, name):
        sel = sel[:n]
        if not sel:
            return
        rn = (len(sel) + cols - 1) // cols
        img = np.full((rn * cell[1], cols * cell[0], 3), 30, np.uint8)
        for j, r in enumerate(sel):
            im = cv2.imdecode(np.fromfile(os.path.join(images, r["file"]), np.uint8), cv2.IMREAD_COLOR)
            h, w = im.shape[:2]
            s = min(cell[0] / w, cell[1] / h)
            im = cv2.resize(im, (max(int(w * s), 1), max(int(h * s), 1)))
            y, x = divmod(j, cols)
            img[y * cell[1]:y * cell[1] + im.shape[0], x * cell[0]:x * cell[0] + im.shape[1]] = im
        cv2.imwrite(os.path.join(out_dir, name + ".jpg"), img)

    un = [r for r in rows if not r["labeled"]]
    for a in ATTRS:
        for v in ("O", "X"):
            s = sorted([r for r in un if r[a] == v], key=lambda r: -abs(float(r[a + "_p"]) - 0.5))
            sheet(s, f"{a}_{v}_confident")
        sheet(sorted(un, key=lambda r: abs(float(r[a + "_p"]) - 0.5)), f"{a}_uncertain")
    print(f"sheets -> {out_dir}")


def main(a):
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    idx = list(csv.DictReader(open(os.path.join(a.features, "embeddings_index.csv"), encoding="utf-8-sig")))
    feats = torch.from_numpy(np.load(os.path.join(a.features, "embeddings.npy")).astype(np.float32))
    labeled = {}
    lp = os.path.join(a.images, "labels.csv")
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp, encoding="utf-8")):
            if r["source"] == "rep" and r["value"] in ("O", "X"):
                labeled[(r["file"], r["attr"])] = r["value"]

    from train_ppe import Head
    probs = {}
    for attr, state in ck["states"].items():
        head = Head(ck["dim"], ck["hidden"])
        head.load_state_dict(state)
        head.eval()
        with torch.no_grad():
            probs[attr] = torch.sigmoid(head(feats)).numpy()

    rows = []
    for i, r in enumerate(idx):
        row = dict(file=r["file"], clip=r["clip"], time_s=r["time_s"], labeled="")
        for attr in ATTRS:
            p = float(probs[attr][i]) if attr in probs else float("nan")
            row[attr] = "O" if p >= a.thr else "X"
            row[attr + "_p"] = round(p, 4)
        row["labeled"] = ";".join(f"{attr}={labeled[(r['file'], attr)]}" for attr in ATTRS
                                  if (r["file"], attr) in labeled)
        rows.append(row)

    out = os.path.join(a.images, "predictions.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "clip", "time_s", "helmet", "helmet_p", "vest", "vest_p", "labeled"])
        w.writeheader()
        w.writerows(rows)

    for attr in probs:
        pred = Counter(r[attr] for r in rows)
        hit = [(r, labeled[(r["file"], attr)]) for r in rows if (r["file"], attr) in labeled]
        acc = sum(r[attr] == v for r, v in hit) / max(len(hit), 1)
        un = [r for r in rows if not r["labeled"]]
        print(f"[{KO[attr]}] 전체 예측 {dict(pred)} | 사람이 라벨한 {len(hit)}장 기준 일치율 {acc:.3f} "
              f"| 미라벨 {len(un)}장 중 불확실(0.3~0.7) {sum(0.3 <= float(r[attr + '_p']) <= 0.7 for r in un)}")
    print(f"저장: {out}")
    if a.sheets:
        sheets(rows, a.images, os.path.join(a.features, "pred_sheets"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "__MODEL__", "ppe_head.pt"))
    ap.add_argument("--features", default=os.path.join(DATA, "features"))
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--sheets", action="store_true")
    main(ap.parse_args())
