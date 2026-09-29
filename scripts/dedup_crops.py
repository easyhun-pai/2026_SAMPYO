"""
Group near-duplicate crops (same worker, consecutive seconds) and pick a few representatives each.

    python scripts/dedup_crops.py --thr 0.90 --per-group 2 --preview

Crops are walked in time order per clip; each one joins an open group of the same clip when its cosine
similarity to that group's mean feature is >= --thr and the time gap is <= --gap seconds, otherwise it
starts a new group. Groups therefore track one person over a stretch of time, not every visually similar
worker in the site.

Writes <features>/groups.csv (file, group, is_rep) and, with --preview, contact sheets of the largest
groups so the threshold can be eyeballed before labeling.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")


def group(feats, rows, thr, gap):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["clip"], float(rows[i]["time_s"])))
    gid = np.full(len(rows), -1)
    open_groups = []                      # [clip, last_time, sum_vec, count]
    n = 0
    for i in order:
        r = rows[i]
        t, clip = float(r["time_s"]), r["clip"]
        open_groups = [g for g in open_groups if g[0] == clip and t - g[1] <= gap]
        v = feats[i]
        best, bs = None, thr
        for g in open_groups:
            s = float(v @ (g[2] / np.linalg.norm(g[2])))
            if s >= bs:
                best, bs = g, s
        if best is None:
            best = [clip, t, v.copy(), 0, n]
            open_groups.append(best)
            n += 1
        best[1], best[2], best[3] = t, best[2] + v, best[3] + 1
        gid[i] = best[4]
    return gid, n


def pick(feats, gid, rows, k):
    """Per group: the highest-confidence crop, then the ones least similar to what is already picked."""
    reps = set()
    for g, idx in groupby(gid):
        idx = sorted(idx, key=lambda i: -float(rows[i]["conf"]))
        chosen = [idx[0]]
        while len(chosen) < min(k, len(idx)):
            rest = [i for i in idx if i not in chosen]
            far = min(rest, key=lambda i: max(float(feats[i] @ feats[c]) for c in chosen))
            chosen.append(far)
        reps.update(chosen)
    return reps


def groupby(gid):
    d = defaultdict(list)
    for i, g in enumerate(gid):
        d[int(g)].append(i)
    return d.items()


def preview(rows, gid, reps, images, out_dir, n_groups=12, cols=18):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    big = sorted(groupby(gid), key=lambda kv: -len(kv[1]))[:n_groups]
    cell = (70, 140)
    for g, idx in big:
        idx = sorted(idx, key=lambda i: float(rows[i]["time_s"]))[:cols * 3]
        rowsn = (len(idx) + cols - 1) // cols
        sheet = np.full((rowsn * cell[1], cols * cell[0], 3), 30, np.uint8)
        for j, i in enumerate(idx):
            im = cv2.imdecode(np.fromfile(os.path.join(images, rows[i]["file"]), np.uint8), cv2.IMREAD_COLOR)
            h, w = im.shape[:2]
            s = min(cell[0] / w, cell[1] / h)
            im = cv2.resize(im, (max(int(w * s), 1), max(int(h * s), 1)))
            if i in reps:
                im = cv2.copyMakeBorder(im[2:-2, 2:-2], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(80, 220, 80))
            r, c = divmod(j, cols)
            sheet[r * cell[1]:r * cell[1] + im.shape[0], c * cell[0]:c * cell[0] + im.shape[1]] = im
        cv2.imwrite(os.path.join(out_dir, f"group_{len(idx):03d}_{g}.jpg"), sheet)
    print(f"preview: {n_groups} sheets -> {out_dir}")


def main(a):
    rows = list(csv.DictReader(open(os.path.join(a.features, "embeddings_index.csv"), encoding="utf-8-sig")))
    feats = np.load(os.path.join(a.features, "embeddings.npy")).astype(np.float32)
    gid, n = group(feats, rows, a.thr, a.gap)
    reps = pick(feats, gid, rows, a.per_group)
    sizes = sorted((len(v) for _, v in groupby(gid)), reverse=True)
    print(f"{len(rows)} crops -> {n} groups (median {int(np.median(sizes))}, max {sizes[0]}), "
          f"{len(reps)} representatives to label")
    with open(os.path.join(a.features, "groups.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "group", "is_rep"])
        for i, r in enumerate(rows):
            w.writerow([r["file"], int(gid[i]), int(i in reps)])
    if a.preview:
        preview(rows, gid, reps, a.images, os.path.join(a.features, "preview"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(DATA, "features"))
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--thr", type=float, default=0.90, help="cosine similarity to join a group")
    ap.add_argument("--gap", type=float, default=30.0, help="max seconds between consecutive crops of a group")
    ap.add_argument("--per-group", type=int, default=2, help="representatives kept per group")
    ap.add_argument("--preview", action="store_true")
    main(ap.parse_args())
