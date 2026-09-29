"""
Smart-vest inference: DINOv3 backbone + the trained head (__MODEL__/smart_vest.pt).

    python scripts/infer_vest.py --images path\to\crops              # person crops -> O/X
    python scripts/infer_vest.py --video clip.mp4 --out out.mp4      # full pipeline on a video

The head is NOT a standalone model: it is a 2-layer classifier over DINOv3 ViT-B/16 features, so the
backbone must be loaded and the crop preprocessed exactly as in training (RGB, resize 112x224, ImageNet
mean/std, feature = CLS token concatenated with the mean of the patch tokens).

--video runs the whole chain: sampyo_v1.pt detects person boxes, each box is padded 15% and cropped, the
crop is classified, and the result is drawn. Boxes smaller than --min-area are skipped exactly as in the
training data.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", r"D:\Project_paimedia\.hf_cache")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Head(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1)) \
            if hidden else nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


class VestClassifier:
    def __init__(self, model_path, device="cuda", thr=0.5):
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.size = tuple(ck.get("size", [112, 224]))            # (w, h)
        self.thr = thr
        self.device = device if torch.cuda.is_available() else "cpu"
        self.backbone = AutoModel.from_pretrained(ck["backbone"], dtype=torch.float16).to(self.device).eval()
        self.skip = 1 + getattr(self.backbone.config, "num_register_tokens", 0)
        self.head = Head(ck["dim"], ck["hidden"]).to(self.device).eval()
        self.head.load_state_dict(ck["states"]["vest"])
        self.report = ck.get("report", {})

    def _pre(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, self.size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        a = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(a.transpose(2, 0, 1))

    @torch.no_grad()
    def __call__(self, crops):
        """crops: list of BGR images -> (labels, probabilities). P is P(wearing the smart vest)."""
        if not crops:
            return [], np.zeros(0, np.float32)
        x = torch.stack([self._pre(c) for c in crops]).half().to(self.device)
        h = self.backbone(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, self.skip:].mean(1)], dim=1)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        p = torch.sigmoid(self.head(f)).cpu().numpy()
        return ["O" if v >= self.thr else "X" for v in p], p


def run_images(a, clf):
    paths = sorted(glob.glob(os.path.join(a.images, "**", "*.jpg"), recursive=True)) \
        if os.path.isdir(a.images) else [a.images]
    for i in range(0, len(paths), a.batch):
        chunk = paths[i:i + a.batch]
        crops = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in chunk]
        labs, ps = clf([c for c in crops if c is not None])
        for p, lab, pr in zip(chunk, labs, ps):
            print(f"{lab}  P(착용)={pr:.3f}  {os.path.basename(p)}")


def run_video(a, clf):
    from ultralytics import YOLO
    det = YOLO(a.detector)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W, H = int(cap.get(3)), int(cap.get(4))
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if a.out else None
    fi, n_on, n_off = 0, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if a.every > 1 and fi % a.every:
            fi += 1
            continue
        r = det.predict(fr, imgsz=a.imgsz, conf=a.conf, classes=[0], verbose=False,
                        device=0 if torch.cuda.is_available() else "cpu")[0]
        boxes, crops = [], []
        for b in r.boxes.xyxy.cpu().numpy():
            x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
            x2, y2 = min(int(b[2]), W), min(int(b[3]), H)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or bw * bh < a.min_area:
                continue
            cx1, cy1 = max(int(x1 - 0.15 * bw), 0), max(int(y1 - 0.15 * bh), 0)
            cx2, cy2 = min(int(x2 + 0.15 * bw), W), min(int(y2 + 0.15 * bh), H)
            boxes.append((x1, y1, x2, y2))
            crops.append(fr[cy1:cy2, cx1:cx2])
        labs, ps = clf(crops)
        for (x1, y1, x2, y2), lab, p in zip(boxes, labs, ps):
            col = (80, 200, 80) if lab == "O" else (60, 60, 230)
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
            cv2.putText(fr, f"{lab} {p:.2f}", (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            n_on += lab == "O"
            n_off += lab == "X"
        if wr is not None:
            wr.write(fr)
        fi += 1
    cap.release()
    if wr is not None:
        wr.release()
    print(f"{fi} frames | 착용 {n_on} / 미착용 {n_off} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "__MODEL__", "smart_vest.pt"))
    ap.add_argument("--images", help="crop image file or folder")
    ap.add_argument("--video")
    ap.add_argument("--out", help="annotated video path (with --video)")
    ap.add_argument("--detector", default=os.path.join(ROOT, "__MODEL__", "sampyo_v1.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=1000)
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--thr", type=float, default=0.5, help="P(착용) below this = 미착용")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    clf = VestClassifier(a.model, thr=a.thr)
    if a.video:
        run_video(a, clf)
    elif a.images:
        run_images(a, clf)
    else:
        ap.error("--images 또는 --video 중 하나는 필요합니다")
