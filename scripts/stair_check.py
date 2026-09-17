"""
Stair-usage check for truck boarding/alighting.

Zone rule (from 계단이용.png):
  work zone  : x in [truck.x1, truck.x2], y in [truck.y1, stairs.y1]
  stair gate : stairs bbox (expanded)
A person is ON DECK when the foot point (bottom-center of bbox) lies in the work zone
and the bbox height is plausible relative to the truck (same-depth check).
A person is ON STAIRS when the foot point lies in the stair gate.

Events: a confirmed GROUND->DECK (UP) or DECK->GROUND (DOWN) transition opens a
*pending* event. It is finalized after `settle_s` seconds if the track is still in the
new state; verdict is OK when the track touched the STAIR gate within `lookback_s`
before the transition or during the settle window, else NO_STAIRS (or UNVERIFIABLE
when no stairs box is known). If the track leaves the new state before settling the
event is cancelled (ground walk-through behind the truck, occlusion blips).

Robustness:
  - debounce + deck-line hysteresis (no boundary flicker)
  - boxes too small for the truck depth (occluded upper bodies, far background) are
    ignored: they neither confirm nor break a state
  - stairs box cached relative to the truck when not detected (occlusion fallback)
  - new track inherits state from a nearby just-lost track (tracker ID switch patch)
  - a track that teleports (ID swap between two live people) is reset
  - STAIR is a valid anchor: STAIR->DECK = UP OK, STAIR->GROUND = DOWN OK
"""
import argparse
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from PIL import Image, ImageDraw, ImageFont

PERSON, TRUCK, STAIRS = 0, 1, 2
GROUND, STAIR, DECK = "GROUND", "STAIR", "DECK"
COLOR = {GROUND: (200, 200, 200), STAIR: (0, 220, 0), DECK: (0, 140, 255)}
VCOL = {"OK": (0, 200, 0), "NO_STAIRS": (0, 0, 255), "UNVERIFIABLE": (0, 200, 255), "UNCERTAIN": (0, 160, 255)}
KR = {GROUND: "지면", STAIR: "계단", DECK: "트럭 위"}
KR_KIND = {"UP": "승차", "DOWN": "하차"}
KR_VERDICT = {"OK": "계단 이용", "NO_STAIRS": "계단 미이용 !", "UNVERIFIABLE": "계단 미확인", "UNCERTAIN": "판정 보류"}

_FONTS = {}


def _font(size):
    if size not in _FONTS:
        for p in (r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\gulim.ttc"):
            if os.path.exists(p):
                _FONTS[size] = ImageFont.truetype(p, size)
                break
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def draw_labels(img, items):
    """items: list of (text, (x, y), size, fg_bgr, bg_bgr|None). Korean-capable text via PIL, one round trip."""
    if not items:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, (x, y), size, fg, bg in items:
        f = _font(size)
        l, t, r, b = d.textbbox((x, y), text, font=f)
        if bg is not None:
            d.rectangle((l - 4, t - 2, r + 4, b + 2), fill=(bg[2], bg[1], bg[0]))
        d.text((x, y), text, font=f, fill=(fg[2], fg[1], fg[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class StairCheck:
    def __init__(self, model, imgsz=1280, conf=0.30, fps=20.0,
                 debounce=5, min_dwell_s=1.0, settle_s=3.0, lookback_s=6.0, stairs_ttl_s=15.0,
                 h_ratio=(0.22, 0.75), stair_ratio=0.18, top_margin=0.20, deck_band=0.20, cab_frac=0.15,
                 gate_wx=0.6, gate_hy=0.08, gate_up=0.08,
                 hyst=0.02, inherit_s=2.0, inherit_px=120, inherit_dy=25, jump_px=90,
                 step_px=25, step_ratio=0.12, clear_deck=0.15, climb=0.15, climb_s=1.5,
                 idle_stride=10, no_stairs_alarm=False, trace=None):
        self.model = YOLO(model)
        self.imgsz, self.conf, self.fps = imgsz, conf, fps
        self.debounce = debounce
        self.min_dwell = int(min_dwell_s * fps)
        self.settle = int(settle_s * fps)
        self.lookback = int(lookback_s * fps)
        self.stairs_ttl = int(stairs_ttl_s * fps)
        self.h_ratio, self.stair_ratio, self.top_margin, self.deck_band = h_ratio, stair_ratio, top_margin, deck_band
        self.cab_frac = cab_frac
        self.gate_wx, self.gate_hy, self.gate_up, self.hyst = gate_wx, gate_hy, gate_up, hyst
        self.inherit, self.inherit_px, self.inherit_dy = int(inherit_s * fps), inherit_px, inherit_dy
        self.jump_px, self.step_px, self.step_ratio, self.clear_deck = jump_px, step_px, step_ratio, clear_deck
        self.climb, self.climb_f, self.idle_stride = climb, int(climb_s * fps), idle_stride
        self.truck_seen = -10 ** 9
        self.no_stairs_alarm = no_stairs_alarm   # True: boarding with no stairs present at all is a violation
        self.tracks, self.dead = {}, {}
        self.stairs_rel, self.stairs_seen = None, -10 ** 9
        self.slope, self.sxy, self.sxx = 0.0, 0.0, 0.0   # bed line slope (px/px) fitted from on-deck feet
        self.edge = None                                   # (a, b): bed rail line y = a*x + b from image edges
        self.NB = 12                                       # per-x-bin bed height (relative to anchor), adapts
        self.bins = np.full(self.NB, np.nan)               # to fisheye curvature / truck shape
        self.bin_n = np.zeros(self.NB, int)
        self.events, self.pending, self.banner = [], [], deque()
        self.flash, self.log = {}, []             # tid -> (until_frame, color); recent event log (t, text, color)
        self.trace = trace

    # ---------- bed profile persistence (learned slope + per-bin heights, truck-relative) ----------
    def load_profile(self, path):
        if not path or not os.path.exists(path):
            return False
        d = json.load(open(path))
        self.slope = float(d.get("slope", 0.0))
        self.sxy, self.sxx = float(d.get("sxy", 0.0)), float(d.get("sxx", 0.0))
        b = d.get("bins", [])
        if len(b) == self.NB:
            self.bins = np.array([np.nan if v is None else v for v in b], float)
            self.bin_n = np.array(d.get("bin_n", [0] * self.NB), int)
        return True

    def save_profile(self, path):
        if not path or self.stairs_rel is None:
            return False          # nothing learned with a real anchor
        json.dump(dict(slope=float(self.slope), sxy=float(self.sxy), sxx=float(self.sxx),
                       bins=[None if np.isnan(v) else float(v) for v in self.bins],
                       bin_n=[int(v) for v in self.bin_n]), open(path, "w"), indent=1)
        return True

    # ---------- geometry ----------
    def _truck(self, xyxy, cls, conf):
        cand = [(b, c) for b, k, c in zip(xyxy, cls, conf) if k == TRUCK and (b[2] - b[0]) > 300]
        if not cand:
            return None
        return max(cand, key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))[0]

    def _stairs(self, xyxy, cls, conf, truck, fi):
        tw, th = truck[2] - truck[0], truck[3] - truck[1]
        cand = [(b, c) for b, k, c in zip(xyxy, cls, conf)
                if k == STAIRS and truck[0] - 0.2 * tw <= (b[0] + b[2]) / 2 <= truck[2] + 0.2 * tw]
        if cand:
            b = max(cand, key=lambda t: t[1])[0]
            self.stairs_rel = ((b[0] - truck[0]) / tw, (b[1] - truck[1]) / th,
                               (b[2] - truck[0]) / tw, (b[3] - truck[1]) / th)
            self.stairs_seen = fi
            return b, True
        if self.stairs_rel is not None and fi - self.stairs_seen <= self.stairs_ttl:
            r = self.stairs_rel
            return np.array([truck[0] + r[0] * tw, truck[1] + r[1] * th,
                             truck[0] + r[2] * tw, truck[1] + r[3] * th]), False
        return None, False

    def _zone(self, truck, stairs):
        """[x1, y1, x2, bed_y_at_anchor, anchor_x, has_stairs]. The bed line is bed_y + slope*(cx - anchor_x):
        the trailer is angled in the image, so the bed is not a horizontal line."""
        th, tw = truck[3] - truck[1], truck[2] - truck[0]
        x1, x2 = truck[0], truck[2]
        has = 1.0 if stairs is not None else 0.0
        if stairs is not None:
            bottom, ax = stairs[1], (stairs[0] + stairs[2]) / 2
            # the cab is at the end opposite the stairs: exclude it from the work zone (영역.png red box)
            if ax > (x1 + x2) / 2:
                x1 += self.cab_frac * tw
            else:
                x2 -= self.cab_frac * tw
        else:
            bottom, ax = truck[1] + 0.40 * th, (truck[0] + truck[2]) / 2   # fallback deck line (bed level)
        return np.array([x1, truck[1], x2, bottom, ax, has])

    def _edge_line(self, frame, truck):
        """Bed rail line straight from the image: the longest near-horizontal edge in the middle band of
        the truck box (load underside / side-rail top, cf. 기울기.png). EMA-smoothed across frames."""
        x1, y1, x2, y2 = [int(v) for v in truck]
        th, tw = y2 - y1, x2 - x1
        ya, yb = y1 + int(0.30 * th), y1 + int(0.80 * th)
        roi = frame[max(ya, 0):yb, max(x1, 0):x2]
        if roi.size == 0:
            return
        g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        e = cv2.Canny(g, 40, 120)
        segs = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=40, minLineLength=int(0.25 * tw), maxLineGap=12)
        if segs is None:
            return
        best = None
        for sx1, sy1, sx2, sy2 in segs[:, 0]:
            dx, dy = sx2 - sx1, sy2 - sy1
            if abs(dx) < 1 or abs(dy / dx) > 0.30:
                continue
            L = float(np.hypot(dx, dy))
            if best is None or L > best[0]:
                best = (L, dy / dx, (sy1 + sy2) / 2 + ya - (dy / dx) * ((sx1 + sx2) / 2 + x1))
        if best is None:
            return
        a, b = float(best[1]), float(best[2])            # y = a*x + b in frame coords
        if self.edge is None:
            self.edge = (a, b)
        else:
            self.edge = (0.9 * self.edge[0] + 0.1 * a, 0.9 * self.edge[1] + 0.1 * b)

    def bed_y(self, zone, cx):
        """Bed line at cx: image edge line if found (no learning needed), else the fitted straight line;
        per-bin learned heights refine either."""
        # slope from the image edge (bed rail, cf. 기울기.png) when available, else the learned slope;
        # height always anchored: stairs top if detected, else the fallback bed level
        a = self.edge[0] if self.edge is not None else self.slope
        straight = zone[3] + a * (cx - zone[4])
        ok = self.bin_n >= 5
        if ok.sum() >= 2:
            tw = max(zone[2] - zone[0], 1)
            xs = zone[0] + (np.arange(self.NB)[ok] + 0.5) * tw / self.NB
            v = zone[3] + float(np.interp(cx, xs, self.bins[ok]))
            lim = 0.15 * (zone[3] - zone[1]) / 0.55 if zone[3] > zone[1] else 30   # ~0.15 truck heights
            return float(np.clip(v, straight - lim, straight + lim))
        return straight

    def _fit_slope(self, zone, cx, foot, th):
        """Online bed model from on-deck feet: straight line through the stairs anchor + per-x-bin heights."""
        dx, dy = cx - zone[4], foot - zone[3]
        # the bed can only be at or below the stairs-top line (image y grows downward): feet clearly
        # above it belong to people standing on the load, not to the bed surface -> skip them
        if dy < -0.04 * th or dy > 0.25 * th:
            return
        if abs(dx) >= 0.1 * (zone[2] - zone[0]):
            self.sxy = 0.995 * self.sxy + dx * dy
            self.sxx = 0.995 * self.sxx + dx * dx
            if self.sxx > 0:
                self.slope = float(np.clip(self.sxy / self.sxx, -0.12, 0.12))
        b = int(np.clip((cx - zone[0]) / max(zone[2] - zone[0], 1) * self.NB, 0, self.NB - 1))
        self.bins[b] = dy if np.isnan(self.bins[b]) else 0.9 * self.bins[b] + 0.1 * dy
        self.bin_n[b] += 1

    def _gate(self, stairs, truck):
        if stairs is None:
            return None
        th = truck[3] - truck[1]
        cx, w = (stairs[0] + stairs[2]) / 2, stairs[2] - stairs[0]
        # extended upward (toward the load top) so a step from the load straight onto the stairs is caught,
        # but only a little downward/sideways so ground people next to the stairs are not swallowed
        return np.array([cx - self.gate_wx * w, stairs[1] - self.gate_up * th,
                         cx + self.gate_wx * w, stairs[3] + self.gate_hy * th])

    def _classify(self, box, truck, zone, gate, prev):
        """Returns (cand, info). cand is None when the box is not a usable observation."""
        cx, foot, h = (box[0] + box[2]) / 2, box[3], box[3] - box[1]
        th = max(truck[3] - truck[1], 1)
        ratio = h / th
        in_gate = gate is not None and gate[0] <= cx <= gate[2] and gate[1] <= foot <= gate[3]
        # deck band extends below the stairs-top line: the trailer is angled in the image, so the bed
        # sits lower toward the front. Enter within +band, stay within +band+hyst (hysteresis).
        band = (self.deck_band + (self.hyst if prev == DECK else 0)) * th
        line = self.bed_y(zone, cx)
        # simple rule (영역.png): foot inside the red box = on the truck; foot below it = on the ground.
        # Leaving sideways at deck height (behind the truck, load overhang) is not a descent -> hold.
        tw = zone[2] - zone[0]
        deck_h = zone[1] - self.top_margin * th <= foot <= line + band
        near_x = zone[0] - 0.25 * tw <= cx <= zone[2] + 0.25 * tw
        in_zone = zone[0] <= cx <= zone[2] and deck_h
        ok_h = self.h_ratio[0] <= ratio <= self.h_ratio[1]
        if in_gate and ratio >= self.stair_ratio:
            cand = STAIR
        elif not ok_h:
            cand = None                     # too small/large for the truck depth: not a usable observation
        elif in_zone:
            cand = DECK
        elif deck_h:
            cand = None                     # deck height but outside the box (behind the truck, load
                                            # overhang): neither on nor off -> no information
        else:
            cand = GROUND                   # foot below the red box = on the ground
        return cand, dict(cx=round(float(cx)), foot=round(float(foot)), ratio=round(float(ratio), 2),
                          in_gate=in_gate, in_zone=in_zone, ok_h=ok_h,
                          above=round(float((line - foot) / th), 3))   # height above the bed line, in truck units

    @staticmethod
    def _seg_hits(p, q, rect):
        """True if segment p->q intersects the axis-aligned rect [x1,y1,x2,y2] (Liang-Barsky)."""
        x1, y1, x2, y2 = rect
        dx, dy = q[0] - p[0], q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for num, den in ((p[0] - x1, -dx), (x2 - p[0], dx), (p[1] - y1, -dy), (y2 - p[1], dy)):
            if den == 0:
                if num < 0:
                    return False
                continue
            r = num / den
            if den < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
        return t0 <= t1

    # ---------- tracks ----------
    def _fresh(self, fi, foot):
        return dict(state=None, cand=None, n=0, obs=0, since=fi, anchor=None, last_stair=-10 ** 9,
                    last=fi, foot=foot, ratio=None, born=fi, inherited_from=None, resets=0,
                    above=deque(maxlen=self.lookback + self.settle),   # (frame, height above deck line)
                    trail=deque(maxlen=int(8 * self.fps)))            # (frame, x, y) foot trail

    def _new_track(self, box, fi):
        foot = np.array([(box[0] + box[2]) / 2, box[3]])
        best, bd = None, self.inherit_px
        for k, d in self.dead.items():
            dist = float(np.linalg.norm(foot - d["foot"]))
            # a vertical jump means a different surface (ground vs deck/load): never inherit across it
            if dist < bd and abs(float(foot[1] - d["foot"][1])) <= self.inherit_dy:
                best, bd = k, dist
        if best is not None:
            t = self.dead.pop(best)
            t["inherited_from"] = int(best)
            return t
        return self._fresh(fi, foot)

    def step(self, frame, fi):
        truck, stairs, live, zone, gate, people = None, None, False, None, None, []
        # idle mode: no truck for a while -> only look every idle_stride frames
        if fi - self.truck_seen > 2 * self.fps and fi % self.idle_stride:
            self._settle(fi)
            return truck, (stairs, live), (zone, gate), people
        r = self.model.track(frame, imgsz=self.imgsz, conf=self.conf, persist=True,
                             tracker="bytetrack.yaml", verbose=False, device=0)[0]
        b = r.boxes
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(len(b), -1)
            truck = self._truck(xyxy, cls, conf)
            if truck is not None:
                self.truck_seen = fi
                if fi % 5 == 0:
                    self._edge_line(frame, truck)
            if truck is not None:
                stairs, live = self._stairs(xyxy, cls, conf, truck, fi)
                zone = self._zone(truck, stairs)
                gate = self._gate(stairs, truck)
            for box, k, tid in zip(xyxy, cls, ids):
                if k != PERSON or tid < 0:
                    continue
                foot = np.array([(box[0] + box[2]) / 2, box[3]])
                if tid not in self.tracks:
                    self.tracks[tid] = self._new_track(box, fi)
                t = self.tracks[tid]
                gap = max(fi - t["last"], 1)
                if float(np.linalg.norm(foot - t["foot"])) > self.jump_px * min(gap, 3):
                    resets = t["resets"] + 1                       # teleport = ID swap, forget history
                    t = self.tracks[tid] = self._fresh(fi, foot)
                    t["resets"] = resets
                cand, info = self._classify(box, truck, zone, gate, t["state"]) if truck is not None else (GROUND, {})
                # trail crossing: the foot path between consecutive observations passes through the gate
                if gate is not None and t["trail"] and gap <= 3 and self._seg_hits(t["trail"][-1][1:], foot, gate):
                    t["last_stair"] = fi
                t["trail"].append((fi, float(foot[0]), float(foot[1])))
                # sudden box change within a few frames = merge/split with another box: not a usable observation
                if cand is not None and t["ratio"] is not None and gap <= 3:
                    if (abs(float(foot[1] - t["foot"][1])) > self.step_px * gap
                            or abs(info.get("ratio", 0) - t["ratio"]) > self.step_ratio):
                        cand = None
                t["last"], t["foot"] = fi, foot
                if info:
                    t["ratio"] = info["ratio"]
                    if cand is not None:
                        t["above"].append((fi, info["above"]))
                    if cand == DECK and t["state"] == DECK and stairs is not None:
                        self._fit_slope(zone, foot[0], foot[1], truck[3] - truck[1])   # only with a real anchor
                if cand is not None:
                    t["obs"] += 1
                    if cand == STAIR:
                        t["last_stair"] = fi
                    if cand == t["cand"]:
                        t["n"] += 1
                    else:
                        t["cand"], t["n"] = cand, 1
                    if t["n"] >= self.debounce and cand != t["state"]:
                        self._transition(tid, t, cand, fi, stairs is not None)
                people.append((tid, box, t["state"] or cand or GROUND))
                if self.trace is not None:
                    self.trace.append(dict(frame=fi, t=round(fi / self.fps, 2), id=int(tid), cand=cand,
                                           state=t["state"], anchor=t["anchor"][0] if t["anchor"] else None,
                                           **info))
        for tid in [k for k, v in self.tracks.items() if v["last"] != fi]:
            t = self.tracks.pop(tid)
            if t["obs"] >= self.debounce:            # ghosts with no real observations are not inheritable
                self.dead[tid] = t
        for tid in [k for k, v in self.dead.items() if fi - v["last"] > self.inherit]:
            del self.dead[tid]
        self._settle(fi)
        return truck, (stairs, live), (zone, gate), people

    def _transition(self, tid, t, new, fi, stairs_known):
        t["state"], t["since"] = new, fi
        anchor = t["anchor"]
        if anchor is not None and anchor[0] != new and new in (GROUND, DECK):
            dwell = fi - anchor[1]
            if dwell >= self.min_dwell and fi - t["born"] >= self.min_dwell:
                kind = "UP" if new == DECK else "DOWN"
                self.pending.append(dict(tid=int(tid), track=t, new=new, kind=kind, at=fi, since=anchor[1],
                                         from_state=anchor[0], dwell=dwell, stairs_known=stairs_known))
        if new in (GROUND, DECK) or anchor is None or anchor[0] != new:
            t["anchor"] = (new, fi)

    def _settle(self, fi):
        keep = []
        for p in self.pending:
            t = p["track"]
            if t["state"] != p["new"] and t["state"] is not None and t["state"] != STAIR:
                continue                                      # left the new state before settling: cancel
            # order matters: the stairs must be touched AFTER the previous state was last confirmed
            # (ground -> stairs -> deck). A stairs touch followed by stepping back to the ground and
            # climbing elsewhere does not count.
            used = t["last_stair"] >= max(p["since"], p["at"] - self.lookback)
            hist = [(f, a) for f, a in t["above"] if f >= p["at"] - self.lookback]
            if fi - p["at"] < self.settle:
                keep.append(p)                                # let the new state hold before firing
                continue
            if used:
                verdict = "OK"
            elif not p["stairs_known"]:
                verdict = "NO_STAIRS" if self.no_stairs_alarm else "UNVERIFIABLE"
            else:
                verdict = "NO_STAIRS"
            self.flash[p["tid"]] = (fi + int(2.5 * self.fps), VCOL[verdict])
            self.log.append((p["at"] / self.fps, f"{KR_KIND[p['kind']]} {KR_VERDICT[verdict]}", VCOL[verdict]))
            self.log = self.log[-4:]
            self.events.append(dict(frame=fi, time=round(p["at"] / self.fps, 2), track=p["tid"], kind=p["kind"],
                                    verdict=verdict, from_state=p["from_state"],
                                    latency_s=round((fi - p["at"]) / self.fps, 2),
                                    reason=("used" if used else "settle"),
                                    above_max=round(max((a for f, a in hist), default=0.0), 3),
                                    dwell_s=round(p["dwell"] / self.fps, 2),
                                    stair_dt_s=round((p["at"] - t["last_stair"]) / self.fps, 2) if used else None,
                                    inherited_from=t["inherited_from"], resets=t["resets"]))
            self.banner.append((fi + int(2.5 * self.fps), f"id{p['tid']} {p['kind']} {verdict}", VCOL[verdict]))
        self.pending = keep

    # ---------- drawing ----------
    def draw(self, frame, fi, truck, stairs_info, zg, people):
        out = frame
        labels = []
        if truck is not None:
            zone, gate = zg
            ov = out.copy()
            xs = np.linspace(zone[0], zone[2], 25)
            bottom = [[int(x), int(self.bed_y(zone, x))] for x in xs]
            poly = np.array([[int(zone[0]), int(zone[1])], [int(zone[2]), int(zone[1])]] + bottom[::-1], np.int32)
            cv2.fillPoly(ov, [poly], (0, 0, 255))                       # 트럭 위 영역: 빨강 (영역.png)
            if gate is not None:
                cv2.rectangle(ov, (int(gate[0]), int(gate[1])), (int(gate[2]), int(gate[3])), (0, 200, 0), -1)
            cv2.addWeighted(ov, 0.22, out, 0.78, 0, out)
            cv2.polylines(out, [poly], True, (0, 0, 255), 2)
            if gate is not None:
                cv2.rectangle(out, (int(gate[0]), int(gate[1])), (int(gate[2]), int(gate[3])), (0, 200, 0), 2)
                labels.append(("계단", (int(gate[0]), int(gate[3]) + 2), 18, (255, 255, 255), (0, 150, 0)))
            labels.append(("트럭 위 영역", (int(zone[0]) + 4, int(zone[1]) - 26), 18, (255, 255, 255), (0, 0, 200)))
        for tid, box, st in people:
            c = COLOR[st]
            tr = self.tracks.get(tid, {}).get("trail")
            if tr and len(tr) > 1:
                pts = np.array([[int(x), int(y)] for _, x, y in tr], np.int32)
                cv2.polylines(out, [pts], False, c, 2)
            fl = self.flash.get(tid)
            if fl and fl[0] >= fi:
                cv2.rectangle(out, (int(box[0]) - 6, int(box[1]) - 6), (int(box[2]) + 6, int(box[3]) + 6), fl[1], 4)
            cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), c, 2)
            cv2.circle(out, (int((box[0] + box[2]) / 2), int(box[3])), 4, c, -1)
            tag = KR[st]
            t = self.tracks.get(tid)
            if t and fi - t["last_stair"] <= 3 * self.fps:
                tag += "  (계단 통과)"
            labels.append((tag, (int(box[0]), max(int(box[1]) - 24, 2)), 18, (0, 0, 0), c))
        for tid, (until, col) in list(self.flash.items()):
            if until < fi:
                del self.flash[tid]
        return draw_labels(out, labels)

    def hud(self, img, fi):
        """Time, recent event log (top-right) and the big banner for a just-fired event. Applied to the
        final view (full frame or zoomed crop) so it is never cut off."""
        labels = []
        sec = int(fi / self.fps)
        labels.append((f"{sec // 60:02d}:{sec % 60:02d}", (img.shape[1] - 90, 8), 24, (255, 255, 255), (0, 0, 0)))
        for i, (ts, text, col) in enumerate(reversed(self.log)):
            labels.append((f"{int(ts) // 60:02d}:{int(ts) % 60:02d}  {text}", (img.shape[1] - 330, 44 + 30 * i), 20,
                           (255, 255, 255), col))
        if self.log and any(until >= fi for until, _ in self.flash.values()):
            ts, text, col = self.log[-1]
            labels.append((f"  {text}  ", (20, 40), 34, (255, 255, 255), col))
        return draw_labels(img, labels)


LEGEND = "빨강 면 = 트럭 위 영역   초록 면 = 계단   사람 태그: 지면 / 계단 / 트럭 위   배너: 초록 = 계단 이용, 빨강 = 계단 미이용, 노랑 = 계단 미확인"


def zoom_view(frame, truck, prev_rect, size=(1280, 720), pad=(0.25, 0.45, 0.35)):
    """Crop around the truck (padded, temporally smoothed) and upscale; legend bar on top."""
    H, W = frame.shape[:2]
    if truck is not None:
        tw, th = truck[2] - truck[0], truck[3] - truck[1]
        rect = np.array([truck[0] - pad[0] * tw, truck[1] - pad[1] * th, truck[2] + pad[0] * tw, truck[3] + pad[2] * th])
    else:
        rect = prev_rect if prev_rect is not None else np.array([W * 0.15, H * 0.3, W * 0.85, H * 0.8])
    rect = rect if prev_rect is None else 0.9 * prev_rect + 0.1 * rect
    # enforce output aspect ratio
    cw, ch = rect[2] - rect[0], rect[3] - rect[1]
    ar = size[0] / size[1]
    if cw / ch > ar:
        cy = (rect[1] + rect[3]) / 2; ch = cw / ar; rect[1], rect[3] = cy - ch / 2, cy + ch / 2
    else:
        cx = (rect[0] + rect[2]) / 2; cw = ch * ar; rect[0], rect[2] = cx - cw / 2, cx + cw / 2
    x1, y1 = int(max(rect[0], 0)), int(max(rect[1], 0))
    x2, y2 = int(min(rect[2], W)), int(min(rect[3], H))
    crop = cv2.resize(frame[y1:y2, x1:x2], size)
    bar = np.zeros((28, size[0], 3), np.uint8)
    bar = draw_labels(bar, [(LEGEND, (8, 3), 17, (255, 255, 255), None)])
    return np.vstack([bar, crop]), rect


def run(src, model, out_dir, imgsz, conf, t0=0.0, want_trace=False, no_stairs_alarm=False, profile=None, zoom=True):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W, H = int(cap.get(3)), int(cap.get(4))
    name = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(out_dir, exist_ok=True)
    wr = cv2.VideoWriter(os.path.join(out_dir, name + "_chk.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trace = [] if want_trace else None
    sc = StairCheck(model, imgsz=imgsz, conf=conf, fps=fps, trace=trace, no_stairs_alarm=no_stairs_alarm)
    loaded = sc.load_profile(profile)
    zw, zrect = None, None
    if zoom:
        zw = cv2.VideoWriter(os.path.join(out_dir, name + "_zoom.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 748))
    fi, t = 0, time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        truck, st, zg, people = sc.step(fr, fi)
        drawn = sc.draw(fr, fi, truck, st, zg, people)
        wr.write(sc.hud(drawn.copy(), fi))
        if zw is not None:
            zimg, zrect = zoom_view(drawn, truck, zrect)
            zw.write(sc.hud(zimg, fi))
        fi += 1
    sc._settle(fi + sc.settle)                       # flush pending at end of stream
    cap.release()
    wr.release()
    if zw is not None:
        zw.release()
    saved = sc.save_profile(profile)
    for e in sc.events:
        e["clip"] = name
        e["abs_time"] = round(t0 + e["time"], 2)
    json.dump(sc.events, open(os.path.join(out_dir, name + "_events.json"), "w"), indent=1)
    if trace:
        with open(os.path.join(out_dir, name + "_trace.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            w.writeheader()
            w.writerows(trace)
    summary = ", ".join(f"{e['kind']}@{e['time']}s id{e['track']} {e['verdict']}({e['from_state'][0]})"
                        for e in sc.events) or "none"
    edge = f"{sc.edge[0]:+.3f}" if sc.edge is not None else "none"
    print(f"{name}: {fi} frames, {fi / (time.time() - t):.1f} fps | edge {edge} slope {sc.slope:+.3f} "
          f"bins {int((sc.bin_n >= 5).sum())}/{sc.NB} profile {'loaded' if loaded else '-'}/{'saved' if saved else '-'} | events: {summary}")
    return sc.events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--model", default=r"C:\Users\User\Desktop\삼표\models\sampyov1.pt")
    ap.add_argument("--out", default=r"C:\Users\User\Desktop\삼표\stair_check\out")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--trace", action="store_true", help="write per-frame per-track diagnostics")
    ap.add_argument("--no-stairs-alarm", action="store_true",
                    help="treat boarding/alighting while no stairs are detected as NO_STAIRS instead of UNVERIFIABLE")
    ap.add_argument("--profile", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_profile.json"),
                    help="learned bed line (slope + per-bin heights); loaded at start, updated at end. '' to disable")
    ap.add_argument("--no-zoom", action="store_true", help="skip the truck-cropped 2x review video (<name>_zoom.mp4)")
    a = ap.parse_args()
    allev = []
    for s in a.src:
        allev += run(s, a.model, a.out, a.imgsz, a.conf, want_trace=a.trace, no_stairs_alarm=a.no_stairs_alarm,
                     profile=a.profile or None, zoom=not a.no_zoom)
    with open(os.path.join(a.out, "events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "time", "track", "kind", "verdict", "from_state", "latency_s",
                                          "reason", "above_max", "dwell_s",
                                          "stair_dt_s", "inherited_from", "resets", "frame", "abs_time"])
        w.writeheader()
        w.writerows(allev)
