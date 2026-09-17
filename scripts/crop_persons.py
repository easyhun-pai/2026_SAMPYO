"""
Crop every detected person box from site videos, sampled once per second.

For each video in --src, frames are sampled every --every seconds (by timestamp, so non-integer fps
like 20.01 / 22.3 stays on schedule). Each `person` box with area >= --min-area px^2 is expanded by
--pad (fraction of box width/height on each side, clipped to the frame) and saved to
<out>/<clip>/<clip>_t<sec>_f<frame>_p<k>.jpg. Every crop is indexed in <out>/crops.csv with both the
detected box (x1..y2) and the padded crop rect (cx1..cy2).

Clip names are normalized to ASCII ("2026-09-16 14_11_28 카메라 2" -> "20260916_141128_cam2").
"""
import argparse
import csv
import os
import re
import time

import cv2
from ultralytics import YOLO

PERSON = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clip_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(\d{4})-(\d{2})-(\d{2}) (\d{2})_(\d{2})_(\d{2}) 카메라 (\d+)", stem)
    return f"{''.join(m.groups()[:3])}_{''.join(m.groups()[3:6])}_cam{m.group(7)}" if m else stem


def imwrite(path, img):
    # cv2.imwrite cannot handle non-ASCII paths on Windows
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(path)
    return ok


def run(src, model, out, imgsz, conf, every, min_area, pad, rows):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = clip_name(src)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    fi, next_t, n_frames, n_crops, t0 = 0, 0.0, 0, 0, time.time()
    while True:
        if fi / fps < next_t:
            if not cap.grab():              # skip without decoding to BGR
                break
            fi += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        next_t += every
        sec = fi / fps
        r = model.predict(fr, imgsz=imgsz, conf=conf, classes=[PERSON], verbose=False, device=0)[0]
        H, W = fr.shape[:2]
        for k, (box, c) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())):
            x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
            x2, y2 = min(int(round(box[2])), W), min(int(round(box[3])), H)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or bw * bh < min_area:
                continue
            cx1, cy1 = max(int(x1 - pad * bw), 0), max(int(y1 - pad * bh), 0)
            cx2, cy2 = min(int(round(x2 + pad * bw)), W), min(int(round(y2 + pad * bh)), H)
            fn = f"{name}_t{int(sec):05d}_f{fi:06d}_p{k:02d}.jpg"
            if imwrite(os.path.join(d, fn), fr[cy1:cy2, cx1:cx2]):
                rows.append(dict(file=f"{name}/{fn}", clip=name, source=os.path.basename(src), frame=fi,
                                 time_s=round(sec, 2), x1=x1, y1=y1, x2=x2, y2=y2,
                                 cx1=cx1, cy1=cy1, cx2=cx2, cy2=cy2, conf=round(float(c), 3)))
                n_crops += 1
        n_frames += 1
        fi += 1
        if n_frames % 300 == 0:
            print(f"  {name}: {fi}/{total} frames, {n_crops} crops, {time.time() - t0:.0f}s", flush=True)
    cap.release()
    print(f"{name}: {n_frames} sampled frames, {n_crops} crops, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=None, help="videos (default: all mp4 in __DATA__/originVideos)")
    ap.add_argument("--model", default=os.path.join(ROOT, "__MODEL__", "sampyo_v1.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "__DATA__", "humanImage"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--every", type=float, default=1.0, help="sampling interval in seconds")
    ap.add_argument("--min-area", type=int, default=1000, help="skip person boxes smaller than this (w*h, px^2)")
    ap.add_argument("--pad", type=float, default=0.15, help="padding per side, as a fraction of box w/h")
    a = ap.parse_args()
    src = a.src or sorted(os.path.join(ROOT, "__DATA__", "originVideos", f)
                          for f in os.listdir(os.path.join(ROOT, "__DATA__", "originVideos")) if f.lower().endswith(".mp4"))
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.model)
    rows = []
    for s in src:
        run(s, model, a.out, a.imgsz, a.conf, a.every, a.min_area, a.pad, rows)
    with open(os.path.join(a.out, "crops.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "clip", "source", "frame", "time_s", "x1", "y1", "x2", "y2",
                                          "cx1", "cy1", "cx2", "cy2", "conf"])
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} crops -> {a.out}")
