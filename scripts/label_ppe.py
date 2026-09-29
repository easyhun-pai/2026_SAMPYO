"""
PPE labeling tool (local web UI): helmet and smart-vest, one attribute at a time.

    python scripts/label_ppe.py                  # representatives from groups.csv (deduplicated)
    python scripts/label_ppe.py --all            # every crop

One attribute is labeled per pass (Tab switches). Within a pass everything defaults to that attribute's
majority value; click or drag the exceptions, Shift+click marks "can't tell". Space commits the page and
moves on, Z undoes the last page. Labels are appended to __DATA__/humanImage/labels.csv
(file, attr, value, group, source, batch, time) — nothing is moved or deleted, so a mistake only costs a Z.

Only one representative per near-duplicate group (features/groups.csv) is shown, and its label is
propagated to the rest of that group (source=prop, vs source=rep for the one actually seen).

When humanImage/predictions.csv exists (scripts/predict_ppe.py), each crop's predicted value becomes its
default instead of the attribute's majority value, and the default sort serves the crops the model is
least sure about first — so a pass is verification of the model, not labeling from scratch.

Values: O (wearing), X (not wearing), U (can't tell -> excluded from training).
"""
import argparse
import csv
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")
FIELDS = ["file", "attr", "value", "group", "source", "batch", "time"]
ATTRS = {"helmet": dict(ko="안전모", default="O"), "vest": dict(ko="스마트조끼", default="X")}


class Store:
    def __init__(self, img_dir, video_dir, feat_dir, reps_only):
        self.img_dir, self.video_dir = img_dir, video_dir
        self.log_path = os.path.join(img_dir, "labels.csv")
        self.lock = threading.Lock()
        with open(os.path.join(img_dir, "crops.csv"), encoding="utf-8-sig") as f:
            self.by_file = {}
            for r in csv.DictReader(f):
                for k in ("frame", "x1", "y1", "x2", "y2"):
                    r[k] = int(r[k])
                r["time_s"], r["conf"] = float(r["time_s"]), float(r["conf"])
                r["h"], r["w"] = r["y2"] - r["y1"], r["x2"] - r["x1"]
                self.by_file[r["file"]] = r
        self.items = list(self.by_file.values())
        self.group_of, self.members = {}, {}
        groups = os.path.join(feat_dir, "groups.csv")
        if os.path.exists(groups):
            reps = set()
            for r in csv.DictReader(open(groups, encoding="utf-8-sig")):
                g = r["group"]
                self.group_of[r["file"]] = g
                self.members.setdefault(g, []).append(r["file"])
                if r["is_rep"] == "1":
                    reps.add(r["file"])
            if reps_only:
                self.items = [r for r in self.items if r["file"] in reps]
        self.items = [r for r in self.items if os.path.isfile(os.path.join(img_dir, r["file"]))]
        self.clips = sorted({r["clip"] for r in self.items})
        self.log = []
        if os.path.exists(self.log_path):
            self.log = list(csv.DictReader(open(self.log_path, encoding="utf-8")))
        self.done = {(r["file"], r["attr"]): r["value"] for r in self.log}
        self.pred = {}                       # (file, attr) -> (predicted value, P(wearing))
        pp = os.path.join(img_dir, "predictions.csv")
        if os.path.exists(pp):
            for r in csv.DictReader(open(pp, encoding="utf-8-sig")):
                for attr in ATTRS:
                    if r.get(attr):
                        self.pred[(r["file"], attr)] = (r[attr], float(r[attr + "_p"]))
        self.caps, self.cap_lock = {}, threading.Lock()

    def gsize(self, f):
        return len(self.members.get(self.group_of.get(f, ""), [f]))

    def stats(self, attr, clip):
        items = [r for r in self.items if clip in ("", r["clip"])]
        v = [self.done.get((r["file"], attr)) for r in items]
        covered = sum(self.gsize(r["file"]) for r in items if (r["file"], attr) in self.done)
        total_img = sum(self.gsize(r["file"]) for r in items)
        return dict(total=len(items), o=v.count("O"), x=v.count("X"), u=v.count("U"),
                    pending=sum(1 for k in v if k is None), batches=len({r["batch"] for r in self.log}),
                    covered=covered, total_img=total_img)

    def _unc(self, f, attr):
        """0 = model is on the fence, 0.5 = fully confident. Unknown predictions sort first."""
        p = self.pred.get((f, attr))
        return abs(p[1] - 0.5) if p else -1.0

    def page(self, attr, clip, sort, n):
        pend = [r for r in self.items if (r["file"], attr) not in self.done and clip in ("", r["clip"])]
        key = {"uncertain": lambda r: (self._unc(r["file"], attr), -self.gsize(r["file"])),
               "group": lambda r: (-self.gsize(r["file"]), r["clip"], r["frame"]),
               "time": lambda r: (r["clip"], r["frame"], r["x1"]),
               "large": lambda r: (-r["h"], r["clip"], r["frame"]),
               "small": lambda r: (r["h"], r["clip"], r["frame"])}[sort]
        pend.sort(key=key)
        out = []
        for r in pend[:n]:
            p = self.pred.get((r["file"], attr))
            out.append(dict(file=r["file"], clip=r["clip"], t=r["time_s"], w=r["w"], h=r["h"], conf=r["conf"],
                            n=self.gsize(r["file"]), pred=p[0] if p else None, p=round(p[1], 3) if p else None))
        return out

    def commit(self, attr, decisions):
        with self.lock:
            batch, now = str(int(time.time() * 1000)), time.strftime("%Y-%m-%d %H:%M:%S")
            rows = []
            for d in decisions:
                f, v = d["file"], d["v"]
                if f in self.by_file and (f, attr) not in self.done and v in ("O", "X", "U"):
                    g = self.group_of.get(f, "")
                    self.done[(f, attr)] = v
                    rows.append(dict(file=f, attr=attr, value=v, group=g, source="rep", batch=batch, time=now))
                    for m in self.members.get(g, []):        # propagate to the rest of the group
                        if m != f and (m, attr) not in self.done:
                            self.done[(m, attr)] = v
                            rows.append(dict(file=m, attr=attr, value=v, group=g, source="prop",
                                             batch=batch, time=now))
            new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
            self.log += rows
            reps = [r for r in rows if r["source"] == "rep"]
            n = {v: sum(r["value"] == v for r in reps) for v in "OXU"}
            return dict(committed=len(reps), images=len(rows), **n)

    def undo(self):
        with self.lock:
            if not self.log:
                return dict(undone=0)
            batch = self.log[-1]["batch"]
            for r in [r for r in self.log if r["batch"] == batch]:
                self.done.pop((r["file"], r["attr"]), None)
            keep = [r for r in self.log if r["batch"] != batch]
            n = len(self.log) - len(keep)
            self.log = keep
            tmp = self.log_path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(self.log)
            os.replace(tmp, self.log_path)
            return dict(undone=n)

    def context(self, rel):
        r = self.by_file[rel]
        with self.cap_lock:
            cap = self.caps.get(r["source"])
            if cap is None:
                cap = self.caps[r["source"]] = cv2.VideoCapture(os.path.join(self.video_dir, r["source"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, fr = cap.read()
        if not ok:
            return None
        H, W = fr.shape[:2]
        cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
        ch = max(r["h"] * 5, 360)
        cw = ch * 16 / 9
        x1, y1 = int(np.clip(cx - cw / 2, 0, max(W - cw, 0))), int(np.clip(cy - ch / 2, 0, max(H - ch, 0)))
        x2, y2 = int(min(x1 + cw, W)), int(min(y1 + ch, H))
        cv2.rectangle(fr, (r["x1"] - 2, r["y1"] - 2), (r["x2"] + 2, r["y2"] + 2), (0, 0, 255), 2)
        roi = fr[y1:y2, x1:x2]
        roi = cv2.resize(roi, (int(roi.shape[1] * 720 / roi.shape[0]), 720))
        return cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if "json" in ctype else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/state":
                attr, clip = q.get("attr", "vest"), q.get("clip", "")
                return self._json(dict(clips=store.clips, attrs={k: v["ko"] for k, v in ATTRS.items()},
                                       default=ATTRS[attr]["default"], stats=store.stats(attr, clip),
                                       items=store.page(attr, clip, q.get("sort", "time"), int(q.get("n", "96")))))
            if u.path.startswith("/img/"):
                p = os.path.normpath(os.path.join(store.img_dir, unquote(u.path[5:])))
                if not p.startswith(os.path.normpath(store.img_dir)) or not os.path.isfile(p):
                    return self._send(404, b"", "text/plain")
                with open(p, "rb") as f:
                    return self._send(200, f.read(), "image/jpeg")
            if u.path == "/ctx":
                rel = q.get("file", "")
                if rel not in store.by_file:
                    return self._send(404, b"", "text/plain")
                img = store.context(rel)
                return self._send(200, img, "image/jpeg") if img else self._send(500, b"", "text/plain")
            self._send(404, b"", "text/plain")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/commit":
                return self._json(store.commit(body.get("attr", "vest"), body.get("items", [])))
            if self.path == "/api/undo":
                return self._json(store.undo())
            self._send(404, b"", "text/plain")

    return H


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>PPE 라벨링</title>
<style>
:root{--bg:#16181c;--panel:#1f2228;--line:#2e323a;--fg:#e6e8eb;--dim:#8b919a;--o:#3ecf6e;--x:#ff4d4f;--u:#c9a227;--acc:#4c8dff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 "Malgun Gothic",system-ui,sans-serif;user-select:none}
header{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
  padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0}
.tabs{display:flex;gap:6px}
.tab{padding:5px 14px;border-radius:8px;border:1px solid var(--line);background:#2a2e36;cursor:pointer;font-weight:600}
.tab.on{background:var(--acc);border-color:var(--acc);color:#fff}
.stat b{font-variant-numeric:tabular-nums}
.bar{flex:1 1 200px;height:8px;background:var(--line);border-radius:4px;overflow:hidden;display:flex;min-width:140px}
.bar i{display:block;height:100%}
select,button,input{font:inherit;color:var(--fg);background:#2a2e36;border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
.help{color:var(--dim);font-size:12px;padding:6px 16px}
.help kbd{background:#2a2e36;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--fg)}
#grid{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px 80px}
.cell{position:relative;width:var(--cw);height:var(--ch);background:#0d0e10;border:2px solid transparent;border-radius:6px;
  display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:pointer}
.cell img{max-width:100%;max-height:calc(100% - 15px);object-fit:contain;pointer-events:none}
.cell .meta{position:absolute;left:0;right:0;bottom:0;font-size:10px;color:var(--dim);text-align:center;background:#0d0e10cc}
.cell .gn{position:absolute;top:2px;right:3px;font-size:11px;font-weight:700;color:#fff;background:#0009;border-radius:8px;padding:0 5px}
.cell .pred{position:absolute;top:2px;left:3px;font-size:10px;font-weight:700;border-radius:6px;padding:0 4px;background:#0009}
.cell .pO{color:var(--o)} .cell .pX{color:var(--x)}
.cell.hover{border-color:var(--acc)}
.cell.mX{border-color:var(--x)} .cell.mX img{opacity:.3}
.cell.mO{border-color:var(--o)} .cell.mO img{opacity:.3}
.cell.mU{border-color:var(--u)} .cell.mU img{opacity:.3}
.cell .tag{position:absolute;inset:0;display:none;align-items:center;justify-content:center;font-size:calc(var(--cw)*.5);
  font-weight:800;pointer-events:none}
.cell.mX .tag,.cell.mO .tag,.cell.mU .tag{display:flex}
.cell.mX .tag{color:var(--x)} .cell.mO .tag{color:var(--o)} .cell.mU .tag{color:var(--u);font-size:calc(var(--cw)*.32)}
#toast{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);background:#000c;padding:8px 16px;border-radius:8px;opacity:0;transition:opacity .2s;pointer-events:none}
#modal{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:10;flex-direction:column;gap:8px}
#modal img{max-width:95vw;max-height:88vh;border-radius:6px}
#modal .cap{color:var(--dim)}
.empty{padding:60px;text-align:center;color:var(--dim);font-size:16px;width:100%}
</style></head><body>
<header>
  <h1>PPE 라벨링</h1>
  <div class="tabs" id="tabs"></div>
  <span class="stat">기본값 <b id="sDef">-</b> · 남은 묶음 <b id="sPend">-</b> · 커버 <b id="sCov">-</b></span>
  <span class="stat" style="color:var(--o)">O <b id="sO">-</b></span>
  <span class="stat" style="color:var(--x)">X <b id="sX">-</b></span>
  <span class="stat" style="color:var(--u)">판단불가 <b id="sU">-</b></span>
  <div class="bar"><i id="bO" style="background:var(--o)"></i><i id="bX" style="background:var(--x)"></i><i id="bU" style="background:var(--u)"></i></div>
  <label>영상 <select id="clip"><option value="">전체</option></select></label>
  <label>정렬 <select id="sort"><option value="uncertain">헷갈린 것부터</option><option value="group">큰 묶음부터</option><option value="time">시간순</option><option value="large">큰 박스부터</option><option value="small">작은 박스부터</option></select></label>
  <label>개수 <select id="n"><option>48</option><option selected>96</option><option>160</option></select></label>
  <label>크기 <input id="size" type="range" min="70" max="220" value="120"></label>
  <button id="btnUndo">되돌리기 (Z)</button>
  <button id="btnCommit" class="primary">확정 → 다음 (Space)</button>
</header>
<div class="help" id="helpLine"></div>
<div id="grid"></div>
<div id="modal"><img id="mImg" alt=""><div class="cap" id="mCap"></div></div>
<div id="toast"></div>
<script>
const $ = id => document.getElementById(id);
let attr = "vest", ATTRS = {}, DEF = "X", items = [], marks = new Map(), painting = null, hovered = null, busy = false;
const other = v => v === "O" ? "X" : "O";
const LS = k => { try { return localStorage.getItem(k) } catch (e) { return null } };
const LSset = (k, v) => { try { localStorage.setItem(k, v) } catch (e) {} };
if (LS("lb_attr")) attr = LS("lb_attr");

function toast(t) { const e = $("toast"); e.textContent = t; e.style.opacity = 1; clearTimeout(e._t); e._t = setTimeout(() => e.style.opacity = 0, 1600); }
function applySize() {
  const s = +$("size").value;
  document.documentElement.style.setProperty("--cw", s + "px");
  document.documentElement.style.setProperty("--ch", Math.round(s * 1.7) + "px");
}
async function load() {
  const q = new URLSearchParams({attr, clip: $("clip").value, sort: $("sort").value, n: $("n").value});
  const r = await (await fetch("/api/state?" + q)).json();
  ATTRS = r.attrs; DEF = r.default;
  const tb = $("tabs");
  if (!tb.children.length) {
    for (const [k, ko] of Object.entries(ATTRS)) {
      const b = document.createElement("div"); b.className = "tab"; b.dataset.k = k; b.textContent = ko;
      b.onclick = () => { attr = k; LSset("lb_attr", k); load(); };
      tb.appendChild(b);
    }
    const sel = $("clip");
    for (const c of r.clips) sel.add(new Option(c, c));
  }
  for (const b of tb.children) b.classList.toggle("on", b.dataset.k === attr);
  const s = r.stats;
  $("sDef").textContent = DEF; $("sPend").textContent = s.pending.toLocaleString();
  $("sCov").textContent = `${(100 * s.covered / Math.max(s.total_img, 1)).toFixed(1)}% (${s.covered.toLocaleString()}/${s.total_img.toLocaleString()}장)`;
  $("sO").textContent = s.o.toLocaleString(); $("sX").textContent = s.x.toLocaleString(); $("sU").textContent = s.u.toLocaleString();
  const T = Math.max(s.total, 1);
  $("bO").style.width = (100 * s.o / T) + "%"; $("bX").style.width = (100 * s.x / T) + "%"; $("bU").style.width = (100 * s.u / T) + "%";
  $("btnUndo").disabled = s.batches === 0;
  $("helpLine").innerHTML = `<b>${ATTRS[attr]}</b> 검수 중 — 왼쪽 위 <b>예측값이 기본</b>(예측 없으면 ${DEF}).
    <b>틀린 것만 클릭/드래그</b>로 뒤집고, <b>Shift+클릭</b>으로 판단불가. 오른쪽 위 <b>×N</b> = 함께 적용되는 장수 ·
    <kbd>Space</kbd> 확정 · <kbd>Z</kbd> 되돌리기 · <kbd>Tab</kbd> 속성 전환 ·
    <b>우클릭</b>/<kbd>C</kbd> 원본 프레임 · <kbd>Esc</kbd> 닫기`;
  items = r.items; marks = new Map(); render(); window.scrollTo(0, 0);
}
function render() {
  const g = $("grid"); g.innerHTML = "";
  if (!items.length) { g.innerHTML = '<div class="empty">이 조건에서 라벨링할 이미지가 없습니다.</div>'; return; }
  items.forEach((it, i) => {
    const c = document.createElement("div"); c.className = "cell"; c.dataset.i = i;
    const pd = it.pred ? `<div class="pred p${it.pred}">예측 ${it.pred} ${it.p.toFixed(2)}</div>` : "";
    c.innerHTML = `<img loading="lazy" src="/img/${encodeURI(it.file)}"><div class="tag"></div>` + pd +
      (it.n > 1 ? `<div class="gn">×${it.n}</div>` : "") + `<div class="meta">${it.w}×${it.h}</div>`;
    c.title = `${it.clip}  t=${it.t}s`;
    g.appendChild(c);
  });
}
const defOf = i => items[i].pred || DEF;
function setMark(i, v) {
  const c = $("grid").children[i]; if (!c) return;
  c.classList.remove("mO", "mX", "mU");
  if (v) { marks.set(i, v); c.classList.add("m" + v); c.querySelector(".tag").textContent = v === "U" ? "판단불가" : v; }
  else marks.delete(i);
}
const cellOf = e => e.target.closest && e.target.closest(".cell");
$("grid").addEventListener("mousedown", e => {
  if (e.button !== 0) return; const c = cellOf(e); if (!c) return; e.preventDefault();
  const i = +c.dataset.i, want = e.shiftKey ? "U" : other(defOf(i));
  painting = marks.get(i) === want ? null : want;
  setMark(i, painting);
});
$("grid").addEventListener("mouseover", e => {
  const c = cellOf(e); if (hovered) hovered.classList.remove("hover"); hovered = c;
  if (!c) return; c.classList.add("hover");
  if (e.buttons & 1) setMark(+c.dataset.i, painting);
});
$("grid").addEventListener("mouseleave", () => { if (hovered) hovered.classList.remove("hover"); hovered = null; });
window.addEventListener("mouseup", () => painting = null);
$("grid").addEventListener("contextmenu", e => { const c = cellOf(e); if (!c) return; e.preventDefault(); showCtx(+c.dataset.i); });
function showCtx(i) {
  const it = items[i]; if (!it) return;
  $("mImg").src = "/ctx?file=" + encodeURIComponent(it.file);
  $("mCap").textContent = `${it.clip} · t=${it.t}s · ${it.w}×${it.h}px (빨간 박스 = 이 크롭)`;
  $("modal").style.display = "flex";
}
$("modal").addEventListener("click", () => $("modal").style.display = "none");
async function commit() {
  if (busy || !items.length) return; busy = true;
  const body = {attr, items: items.map((it, i) => ({file: it.file, v: marks.get(i) || defOf(i)}))};
  try {
    const r = await (await fetch("/api/commit", {method: "POST", body: JSON.stringify(body)})).json();
    toast(`확정 ${r.committed}묶음 → ${r.images}장 — O ${r.O} / X ${r.X} / 판단불가 ${r.U}`);
    await load();
  } finally { busy = false; }
}
async function undo() {
  if (busy) return; busy = true;
  try {
    const r = await (await fetch("/api/undo", {method: "POST", body: "{}"})).json();
    toast(r.undone ? `되돌림 ${r.undone}장` : "되돌릴 페이지가 없습니다");
    await load();
  } finally { busy = false; }
}
$("btnCommit").onclick = commit; $("btnUndo").onclick = undo;
for (const k of ["clip", "sort", "n"]) $(k).addEventListener("change", load);
$("size").addEventListener("input", () => { LSset("lb_size", $("size").value); applySize(); });
if (LS("lb_size")) $("size").value = LS("lb_size");
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") { if (e.key !== " ") return; e.target.blur(); }
  if ($("modal").style.display === "flex") { if (["Escape", "c", "C"].includes(e.key)) $("modal").style.display = "none"; return; }
  if (e.key === "Tab") { e.preventDefault(); const ks = Object.keys(ATTRS); attr = ks[(ks.indexOf(attr) + 1) % ks.length]; LSset("lb_attr", attr); load(); }
  else if (e.key === " " || e.key === "Enter") { e.preventDefault(); commit(); }
  else if (e.key === "z" || e.key === "Z") undo();
  else if ((e.key === "c" || e.key === "C") && hovered) showCtx(+hovered.dataset.i);
  else if (hovered && ["1", "2", "3"].includes(e.key)) setMark(+hovered.dataset.i, {1: "O", 2: "X", 3: "U"}[e.key]);
});
applySize(); load();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--videos", default=os.path.join(DATA, "originVideos"))
    ap.add_argument("--features", default=os.path.join(DATA, "features"))
    ap.add_argument("--all", dest="reps_only", action="store_false", help="label every crop, not just group reps")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    store = Store(a.images, a.videos, a.features, a.reps_only)
    url = f"http://127.0.0.1:{a.port}"
    print(f"{len(store.items)} crops to label  ->  {url}   (Ctrl+C to stop)")
    for k, v in ATTRS.items():
        s = store.stats(k, "")
        print(f"  {v['ko']}: O {s['o']} / X {s['x']} / 판단불가 {s['u']} / 남음 {s['pending']}")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store))
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
