"""
O/X review tool for person crops (local web UI, stdlib + OpenCV only).

    python scripts/review_crops.py            -> opens http://127.0.0.1:8765

Grid of pending crops. Everything is O (keep) by default; click / drag over bad crops to mark X.
Space commits the page: X crops are MOVED to __DATA__/humanImage_rejected/<clip>/ (not deleted), O crops
stay. Every decision is appended to __DATA__/humanImage/review.csv, so the review resumes where it
stopped. Z undoes the last committed page (files are moved back). Right-click a crop (or hover + C)
to see it in its original frame, which helps with tiny or ambiguous boxes.
"""
import argparse
import csv
import json
import os
import shutil
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "__DATA__")
FIELDS = ["file", "verdict", "batch", "time"]


class Store:
    def __init__(self, img_dir, rej_dir, video_dir):
        self.img_dir, self.rej_dir, self.video_dir = img_dir, rej_dir, video_dir
        self.log_path = os.path.join(img_dir, "review.csv")
        self.lock = threading.Lock()
        with open(os.path.join(img_dir, "crops.csv"), encoding="utf-8-sig") as f:
            self.items = []
            for r in csv.DictReader(f):
                for k in ("frame", "x1", "y1", "x2", "y2"):
                    r[k] = int(r[k])
                r["time_s"], r["conf"] = float(r["time_s"]), float(r["conf"])
                r["h"], r["w"] = r["y2"] - r["y1"], r["x2"] - r["x1"]
                self.items.append(r)
        self.by_file = {r["file"]: r for r in self.items}
        self.clips = sorted({r["clip"] for r in self.items})
        self.log = []
        if os.path.exists(self.log_path):
            with open(self.log_path, encoding="utf-8") as f:
                self.log = list(csv.DictReader(f))
        self.decided = {r["file"]: r["verdict"] for r in self.log}
        self.caps, self.cap_lock = {}, threading.Lock()

    def stats(self, clip):
        items = [r for r in self.items if clip in ("", r["clip"])]
        o = sum(self.decided.get(r["file"]) == "O" for r in items)
        x = sum(self.decided.get(r["file"]) == "X" for r in items)
        return dict(total=len(items), o=o, x=x, pending=len(items) - o - x,
                    batches=len({r["batch"] for r in self.log}))

    def page(self, clip, sort, n):
        pend = [r for r in self.items if r["file"] not in self.decided and clip in ("", r["clip"])]
        key = {"time": lambda r: (r["clip"], r["frame"], r["x1"]),
               "small": lambda r: (r["h"], r["clip"], r["frame"]),
               "large": lambda r: (-r["h"], r["clip"], r["frame"]),
               "conf": lambda r: (r["conf"], r["clip"], r["frame"])}[sort]
        pend.sort(key=key)
        return [dict(file=r["file"], clip=r["clip"], t=r["time_s"], w=r["w"], h=r["h"], conf=r["conf"])
                for r in pend[:n]]

    def _move(self, rel, src_root, dst_root):
        src, dst = os.path.join(src_root, rel), os.path.join(dst_root, rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    def _write_log(self):
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.log)
        os.replace(tmp, self.log_path)

    def commit(self, decisions):
        with self.lock:
            batch = str(int(time.time() * 1000))
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            rows = []
            for d in decisions:
                f, v = d["file"], d["v"]
                if f not in self.by_file or f in self.decided or v not in ("O", "X"):
                    continue
                if v == "X":
                    self._move(f, self.img_dir, self.rej_dir)
                self.decided[f] = v
                rows.append(dict(file=f, verdict=v, batch=batch, time=now))
            new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
            self.log += rows
            return dict(committed=len(rows), x=sum(r["verdict"] == "X" for r in rows))

    def undo(self):
        with self.lock:
            if not self.log:
                return dict(undone=0)
            batch = self.log[-1]["batch"]
            rows = [r for r in self.log if r["batch"] == batch]
            for r in rows:
                if r["verdict"] == "X":
                    self._move(r["file"], self.rej_dir, self.img_dir)
                self.decided.pop(r["file"], None)
            self.log = [r for r in self.log if r["batch"] != batch]
            self._write_log()
            return dict(undone=len(rows), files=[r["file"] for r in rows])

    def context(self, rel):
        """The crop's box drawn on a region of its original frame."""
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
        scale = 720 / roi.shape[0]
        roi = cv2.resize(roi, (int(roi.shape[1] * scale), 720), interpolation=cv2.INTER_LINEAR)
        return cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/state":
                clip = q.get("clip", "")
                return self._json(dict(clips=store.clips, stats=store.stats(clip),
                                       items=store.page(clip, q.get("sort", "time"), int(q.get("n", "96")))))
            if u.path.startswith("/img/"):
                rel = unquote(u.path[5:])
                p = os.path.normpath(os.path.join(store.img_dir, rel))
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
                return self._json(store.commit(body.get("items", [])))
            if self.path == "/api/undo":
                return self._json(store.undo())
            self._send(404, b"", "text/plain")

    return H


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>사람 크롭 O/X 검수</title>
<style>
:root{--bg:#16181c;--panel:#1f2228;--line:#2e323a;--fg:#e6e8eb;--dim:#8b919a;--x:#ff4d4f;--o:#3ecf6e;--acc:#4c8dff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 "Malgun Gothic",system-ui,sans-serif;user-select:none}
header{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;
  padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0 6px 0 0}
.stat b{font-variant-numeric:tabular-nums}
.bar{flex:1 1 220px;height:8px;background:var(--line);border-radius:4px;overflow:hidden;display:flex;min-width:160px}
.bar i{display:block;height:100%}
select,button,input{font:inherit;color:var(--fg);background:#2a2e36;border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
button:disabled{opacity:.5;cursor:default}
.help{color:var(--dim);font-size:12px;padding:6px 16px}
.help kbd{background:#2a2e36;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--fg)}
#grid{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px 80px}
.cell{position:relative;width:var(--cw);height:var(--ch);background:#0d0e10;border:2px solid transparent;border-radius:6px;
  display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:pointer}
.cell img{max-width:100%;max-height:calc(100% - 16px);object-fit:contain;pointer-events:none}
.cell .meta{position:absolute;left:0;right:0;bottom:0;font-size:10px;color:var(--dim);text-align:center;background:#0d0e10cc}
.cell.hover{border-color:var(--acc)}
.cell.x{border-color:var(--x)}
.cell.x img{opacity:.28}
.cell.x::after{content:"✕";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:var(--x);font-size:calc(var(--cw) * .55);font-weight:700;pointer-events:none}
#toast{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);background:#000c;padding:8px 16px;border-radius:8px;
  opacity:0;transition:opacity .2s;pointer-events:none}
#modal{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:10;flex-direction:column;gap:8px}
#modal img{max-width:95vw;max-height:88vh;border-radius:6px}
#modal .cap{color:var(--dim)}
.empty{padding:60px;text-align:center;color:var(--dim);font-size:16px;width:100%}
</style></head><body>
<header>
  <h1>사람 크롭 O/X 검수</h1>
  <span class="stat">남음 <b id="sPend">-</b></span>
  <span class="stat" style="color:var(--o)">O <b id="sO">-</b></span>
  <span class="stat" style="color:var(--x)">X <b id="sX">-</b></span>
  <div class="bar"><i id="bO" style="background:var(--o)"></i><i id="bX" style="background:var(--x)"></i></div>
  <label>영상 <select id="clip"><option value="">전체</option></select></label>
  <label>정렬 <select id="sort">
    <option value="time">영상·시간순</option><option value="small">작은 박스부터</option>
    <option value="large">큰 박스부터</option><option value="conf">낮은 신뢰도부터</option></select></label>
  <label>개수 <select id="n"><option>48</option><option selected>96</option><option>160</option><option>240</option></select></label>
  <label>크기 <input id="size" type="range" min="70" max="220" value="110"></label>
  <button id="btnAll">전부 X 토글 (A)</button>
  <button id="btnUndo">되돌리기 (Z)</button>
  <button id="btnCommit" class="primary">확정 → 다음 (Space)</button>
</header>
<div class="help">기본은 전부 <b style="color:var(--o)">O(유지)</b>. 버릴 것만 <b>클릭</b> 또는 <b>드래그</b>로 <b style="color:var(--x)">X</b> 표시 →
  <kbd>Space</kbd> 확정(X는 humanImage_rejected로 이동) · <kbd>Z</kbd> 직전 페이지 되돌리기 · <kbd>A</kbd> 페이지 전체 토글 ·
  <b>우클릭</b> 또는 올려두고 <kbd>C</kbd> 원본 프레임 보기 · <kbd>Esc</kbd> 닫기</div>
<div id="grid"></div>
<div id="modal"><img id="mImg" alt=""><div class="cap" id="mCap"></div></div>
<div id="toast"></div>
<script>
const $ = id => document.getElementById(id);
let items = [], marks = new Set(), painting = null, hovered = null, busy = false;
const LS = k => { try { return localStorage.getItem(k) } catch (e) { return null } };
const LSset = (k, v) => { try { localStorage.setItem(k, v) } catch (e) {} };
for (const k of ["clip", "sort", "n", "size"]) { const v = LS("rv_" + k); if (v !== null) $(k).dataset.saved = v; }

function toast(t) { const e = $("toast"); e.textContent = t; e.style.opacity = 1; clearTimeout(e._t); e._t = setTimeout(() => e.style.opacity = 0, 1600); }
function applySize() {
  const s = +$("size").value;
  document.documentElement.style.setProperty("--cw", s + "px");
  document.documentElement.style.setProperty("--ch", Math.round(s * 1.7) + "px");
}
async function load() {
  const q = new URLSearchParams({clip: $("clip").value, sort: $("sort").value, n: $("n").value});
  const r = await (await fetch("/api/state?" + q)).json();
  const sel = $("clip");
  if (sel.options.length === 1) {
    for (const c of r.clips) sel.add(new Option(c, c));
    if (sel.dataset.saved !== undefined) { sel.value = sel.dataset.saved; delete sel.dataset.saved; return load(); }
  }
  const s = r.stats;
  $("sPend").textContent = s.pending.toLocaleString(); $("sO").textContent = s.o.toLocaleString(); $("sX").textContent = s.x.toLocaleString();
  $("bO").style.width = (100 * s.o / Math.max(s.total, 1)) + "%"; $("bX").style.width = (100 * s.x / Math.max(s.total, 1)) + "%";
  $("btnUndo").disabled = s.batches === 0;
  items = r.items; marks = new Set(); render();
  window.scrollTo(0, 0);
}
function render() {
  const g = $("grid"); g.innerHTML = "";
  if (!items.length) { g.innerHTML = '<div class="empty">이 조건에서 남은 이미지가 없습니다.</div>'; return; }
  items.forEach((it, i) => {
    const c = document.createElement("div"); c.className = "cell"; c.dataset.i = i;
    c.innerHTML = `<img loading="lazy" src="/img/${encodeURI(it.file)}"><div class="meta">${it.w}×${it.h} · ${it.conf.toFixed(2)}</div>`;
    c.title = `${it.clip}  t=${it.t}s`;
    g.appendChild(c);
  });
}
function setMark(i, on) {
  const c = $("grid").children[i]; if (!c) return;
  on ? marks.add(i) : marks.delete(i); c.classList.toggle("x", on);
}
const cellOf = e => e.target.closest && e.target.closest(".cell");
$("grid").addEventListener("mousedown", e => {
  if (e.button !== 0) return; const c = cellOf(e); if (!c) return;
  const i = +c.dataset.i; painting = !marks.has(i); setMark(i, painting); e.preventDefault();
});
$("grid").addEventListener("mouseover", e => {
  const c = cellOf(e); if (hovered) hovered.classList.remove("hover"); hovered = c;
  if (!c) return; c.classList.add("hover");
  if (painting !== null && e.buttons & 1) setMark(+c.dataset.i, painting);
});
$("grid").addEventListener("mouseleave", () => { if (hovered) hovered.classList.remove("hover"); hovered = null; });
window.addEventListener("mouseup", () => painting = null);
$("grid").addEventListener("contextmenu", e => { const c = cellOf(e); if (!c) return; e.preventDefault(); showCtx(+c.dataset.i); });
function showCtx(i) {
  const it = items[i]; if (!it) return;
  $("mImg").src = "/ctx?file=" + encodeURIComponent(it.file);
  $("mCap").textContent = `${it.clip} · t=${it.t}s · ${it.w}×${it.h}px · conf ${it.conf.toFixed(2)}  (빨간 박스 = 이 크롭)`;
  $("modal").style.display = "flex";
}
$("modal").addEventListener("click", () => $("modal").style.display = "none");

async function commit() {
  if (busy || !items.length) return; busy = true;
  const body = {items: items.map((it, i) => ({file: it.file, v: marks.has(i) ? "X" : "O"}))};
  try {
    const r = await (await fetch("/api/commit", {method: "POST", body: JSON.stringify(body)})).json();
    toast(`확정 ${r.committed}장 (X ${r.x})`);
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
function toggleAll() { const on = marks.size < items.length; items.forEach((_, i) => setMark(i, on)); }

$("btnCommit").onclick = commit; $("btnUndo").onclick = undo; $("btnAll").onclick = toggleAll;
for (const k of ["clip", "sort", "n"]) $(k).addEventListener("change", () => { LSset("rv_" + k, $(k).value); load(); });
$("size").addEventListener("input", () => { LSset("rv_size", $("size").value); applySize(); });
for (const k of ["sort", "n", "size"]) if ($(k).dataset.saved !== undefined) $(k).value = $(k).dataset.saved;
window.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") { if (e.key !== " ") return; e.target.blur(); }
  if ($("modal").style.display === "flex") { if (e.key === "Escape" || e.key === "c" || e.key === "C") $("modal").style.display = "none"; return; }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); commit(); }
  else if (e.key === "z" || e.key === "Z") undo();
  else if (e.key === "a" || e.key === "A") toggleAll();
  else if ((e.key === "c" || e.key === "C") && hovered) showCtx(+hovered.dataset.i);
  else if ((e.key === "x" || e.key === "X") && hovered) { const i = +hovered.dataset.i; setMark(i, !marks.has(i)); }
});
applySize(); load();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(DATA, "humanImage"))
    ap.add_argument("--rejected", default=os.path.join(DATA, "humanImage_rejected"))
    ap.add_argument("--videos", default=os.path.join(DATA, "originVideos"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    store = Store(a.images, a.rejected, a.videos)
    url = f"http://127.0.0.1:{a.port}"
    s = store.stats("")
    print(f"{s['total']} crops, {s['pending']} pending ({s['o']} O / {s['x']} X)  ->  {url}   (Ctrl+C to stop)")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store))
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
