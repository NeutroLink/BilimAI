#!/usr/bin/env python
"""Render what Qwen actually drew, page by page, beside our detector and the human truth.

  BLUE  = Qwen3-VL's own boxes (from its JSON output, 0-1000 normalised inside each half it was sent)
  GREEN = ai-forever human annotation

RP's boxes are NOT drawn — three overlapping box sets make the page unreadable. RP's match count is
still in every caption, so the comparison survives without the clutter.

Per-page caption reports how many GT lines each detector matched at IoU >= 0.5, so a page can be read
at a glance: blue boxes with no green under them are over-segmentation; green with no blue is a miss.

  eval/.venv/bin/python eval/detectors/qwen_page_render.py eval/runs/qwen_hard20.json out/qwen_pages
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
recs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ROOT/"eval/runs/qwen_hard20.json", encoding="utf-8"))
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT/"out/qwen_pages"); OUT.mkdir(parents=True, exist_ok=True)
gt = json.load(open(ROOT/"eval/testset_v2/ru_pages/ground_truth.json", encoding="utf-8"))
rp = json.load(open(ROOT/"eval/runs/rp_det_lines_v2g.json"))

def iou(p, g):
    x0,y0,x1,y1 = max(p[0],g[0]),max(p[1],g[1]),min(p[2],g[2]),min(p[3],g[3])
    i = max(0.,x1-x0)*max(0.,y1-y0)
    return i/max(1e-9,(p[2]-p[0])*(p[3]-p[1])+(g[2]-g[0])*(g[3]-g[1])-i)

by_page = {}
for r in recs:
    sw, sh = r["sub_size"]; ox, oy = r["sub_origin"]
    for it in r["items"]:
        b = it["bbox"]
        q = [b[0]/1000*sw+ox, b[1]/1000*sh+oy, b[2]/1000*sw+ox, b[3]/1000*sh+oy]
        if q[2] > q[0] and q[3] > q[1]:
            by_page.setdefault(r["file"], []).append(q)

try:
    FB = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 46)
    FT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
except Exception:
    FB = FT = ImageFont.load_default()

rows, thumbs = [], []
for fn in sorted(by_page):
    im = Image.open(ROOT/"eval/testset_v2/ru_pages/images"/fn).convert("RGB")
    G = [l["bbox"] for l in gt[fn]["lines"]]
    P = [b[:4] for b in rp.get(fn, [])]
    Q = by_page[fn]
    d = ImageDraw.Draw(im)
    for b in G: d.rectangle(b, outline=(0,170,0), width=5)
    for b in Q: d.rectangle(b, outline=(0,90,235), width=6)   # RP boxes deliberately NOT drawn — the
    # page is unreadable with three overlapping sets; RP's score still appears in the caption for comparison
    mq = sum(1 for g in G if Q and max(iou(q,g) for q in Q) >= 0.5)
    mp = sum(1 for g in G if P and max(iou(p,g) for p in P) >= 0.5)
    extra = sum(1 for q in Q if not G or max(iou(q,g) for g in G) < 0.3)   # boxes matching nothing
    cap = (f"{fn}   {len(G)} true lines    QWEN found {mq} ({100*mq/max(len(G),1):.0f} %), "
           f"{len(Q)} boxes, {extra} matching nothing    RP found {mp} ({100*mp/max(len(G),1):.0f} %)")
    band = Image.new("RGB", (im.width, 150), (255,255,255)); bd = ImageDraw.Draw(band)
    bd.text((20,14), cap, fill=(15,15,20), font=FB)
    bd.text((20,74), "BLUE = Qwen3-VL's own boxes    GREEN = human truth", fill=(95,95,110), font=FT)
    page = Image.new("RGB", (im.width, im.height+150), (255,255,255))
    page.paste(band,(0,0)); page.paste(im,(0,150))
    sc = min(1.0, 1700/page.width)
    page = page.resize((int(page.width*sc), int(page.height*sc)))
    page.save(OUT/f"{Path(fn).stem}.png")
    rows.append((fn, len(G), mq, mp, len(Q), extra))
    t = page.copy(); t.thumbnail((820, 820)); thumbs.append(t)

print(f"{'page':<12}{'GT':>5}{'QWEN':>7}{'RP':>6}{'boxes':>7}{'junk':>6}")
for fn,g,mq,mp,nq,ex in rows: print(f"{fn:<12}{g:>5}{mq:>7}{mp:>6}{nq:>7}{ex:>6}")
G=sum(r[1] for r in rows); Q=sum(r[2] for r in rows); R=sum(r[3] for r in rows); X=sum(r[5] for r in rows)
print(f"{'TOTAL':<12}{G:>5}{Q:>7}{R:>6}{sum(r[4] for r in rows):>7}{X:>6}")
print(f"  Qwen {100*Q/max(G,1):.1f} %   RP {100*R/max(G,1):.1f} %   Qwen boxes matching nothing: {X}")

if thumbs:
    cols = 4; rowsn = (len(thumbs)+cols-1)//cols
    w = max(t.width for t in thumbs); h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols*w+ (cols+1)*12, rowsn*h + (rowsn+1)*12), (245,245,248))
    for i,t in enumerate(thumbs):
        sheet.paste(t, (12 + (i%cols)*(w+12), 12 + (i//cols)*(h+12)))
    sheet.save(OUT/"_contact_sheet.png")
    print(f"\n-> {OUT}/  ({len(thumbs)} pages + _contact_sheet.png)")
