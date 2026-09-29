"""
Build targeted re-review sets for the cases the vest model gets wrong most often.

    python scripts/make_review_sets.py

Writes __DATA__/features/review_sets.csv (file, set, note). Sets:
  big        크롭 높이 200px 이상 (트럭 위 작업자) — 검증셋 오답률 23%
  yellow_O   현재 O 라벨인데 상의가 노랑/연두 계열 — 정말 스마트조끼인지 확인
  yellow_X   현재 X 라벨/예측인데 상의가 노랑이고 하의가 어두움 — 놓친 변형인지 확인

Only group representatives are listed: a decision there propagates to the whole group.
"""
import argparse
import csv
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")


def hue_of(img, y0, y1):
    h, w = img.shape[:2]
    roi = img[int(h * y0):int(h * y1), int(w * 0.2):int(w * 0.8)]
    if roi.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70)
    if mask.sum() < 20:
        return None, float(hsv[:, :, 2].mean())
    return float(np.median(hsv[:, :, 0][mask]) * 2), float(hsv[:, :, 2].mean())


def bucket(hue):
    if hue is None:
        return "dark"
    return "red" if (hue < 20 or hue >= 330) else "orange" if hue < 45 else "yellow" if hue < 70 \
        else "green" if hue < 160 else "blue"


def main(a):
    crops = {r["file"]: r for r in csv.DictReader(open(os.path.join(a.images, "crops.csv"), encoding="utf-8-sig"))}
    reps = {r["file"] for r in csv.DictReader(open(os.path.join(a.features, "groups.csv"), encoding="utf-8-sig"))
            if r["is_rep"] == "1"}
    label = {}
    lp = os.path.join(a.images, "labels.csv")
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp, encoding="utf-8")):
            if r["attr"] == "vest" and r["source"] == "rep":
                label[r["file"]] = r["value"]          # later rows win
    pred = {}
    pp = os.path.join(a.images, "predictions.csv")
    if os.path.exists(pp):
        for r in csv.DictReader(open(pp, encoding="utf-8-sig")):
            if r.get("vest"):
                pred[r["file"]] = (r["vest"], float(r["vest_p"]))

    out = []
    for f in sorted(reps):
        c = crops.get(f)
        if not c or not os.path.isfile(os.path.join(a.images, f)):
            continue
        h = int(c["cy2"]) - int(c["cy1"])
        lab, pr = label.get(f), pred.get(f)
        if h >= a.big:
            out.append((f, "big", f"높이 {h}px · 라벨 {lab or '-'} · 예측 {pr[0] if pr else '-'}"))
        img = cv2.imdecode(np.fromfile(os.path.join(a.images, f), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        top, _ = hue_of(img, 0.20, 0.45)
        bot, botv = hue_of(img, 0.45, 0.70)
        tb, bb = bucket(top), bucket(bot)
        if lab == "O" and tb in ("yellow", "green"):
            out.append((f, "yellow_O", f"상의 {tb} · 현재 라벨 O"))
        elif tb == "yellow" and (bb == "dark" or botv < 90) and lab != "O":
            out.append((f, "yellow_X", f"상의 노랑 + 하의 어두움 · 현재 {lab or '미라벨'}"))

    os.makedirs(a.features, exist_ok=True)
    path = os.path.join(a.features, "review_sets.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "set", "note"])
        w.writerows(out)
    n = {}
    for _, s, _ in out:
        n[s] = n.get(s, 0) + 1
    print(f"{path}: {dict(n)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--features", default=os.path.join(DATA, "features"))
    ap.add_argument("--big", type=int, default=200, help="crop height (px) counted as a close-up")
    main(ap.parse_args())
