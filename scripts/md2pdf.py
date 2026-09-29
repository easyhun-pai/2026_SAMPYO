"""
Render a Markdown document to PDF (headless Edge print-to-PDF, Korean-safe fonts).

    python scripts/md2pdf.py docs\스마트조끼_모델_보고서.md
    python scripts/md2pdf.py docs\report.md --out build\report.pdf

Keeps tables, headings and code blocks. Images referenced relatively are resolved against the markdown
file's folder, so docs/pipeline.png works.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import urllib.parse

import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Malgun Gothic", "맑은 고딕", system-ui, sans-serif; font-size: 10.5pt; line-height: 1.55;
       color: #1a1d23; }
h1 { font-size: 20pt; margin: 0 0 4pt; border-bottom: 2px solid #1a1d23; padding-bottom: 6pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; color: #14315e; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; page-break-after: avoid; }
p, li { margin: 4pt 0; }
hr { border: 0; border-top: 1px solid #d3d8e0; margin: 12pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; page-break-inside: avoid; font-size: 10pt; }
th, td { border: 1px solid #c9cfd8; padding: 5pt 7pt; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 700; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f1f3f6; padding: 1pt 3pt; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #dde2e9; border-radius: 5px; padding: 8pt 10pt; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; }
strong { color: #0f1a2b; }
"""


def find_edge():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(p):
            return p
    return shutil.which("msedge") or shutil.which("chrome")


def main(a):
    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out or os.path.splitext(src)[0] + ".pdf")
    html_body = markdown.markdown(open(src, encoding="utf-8").read(),
                                  extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    title = os.path.splitext(os.path.basename(src))[0]
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{title}</title>'
            f'<style>{CSS}</style></head><body>{html_body}</body></html>')
    tmp = os.path.join(tempfile.gettempdir(), f"md2pdf_{os.getpid()}.html")
    # write next to the source so relative image paths resolve, then clean up
    tmp = os.path.join(os.path.dirname(src), f".md2pdf_{os.getpid()}.html")
    open(tmp, "w", encoding="utf-8").write(html)
    edge = find_edge()
    if not edge:
        raise SystemExit("Edge/Chrome을 찾을 수 없습니다.")
    url = "file:///" + urllib.parse.quote(tmp.replace("\\", "/"))
    try:
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", url], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    finally:
        os.remove(tmp)
    print(f"{out}  ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="markdown file")
    ap.add_argument("--out", help="output pdf (default: same name next to the markdown)")
    main(ap.parse_args())
