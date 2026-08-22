#!/usr/bin/env python3
"""E4.2 — controlled A/B of a growth change (2026-08-22).

crop_cost.py re-samples per prediction set, so changing the boxes changes which lines qualify as matched and the
comparison is confounded. This fixes the line set first: only GT lines matched (IoU>=0.5) under BOTH box sets, then
reads each line THREE times — human box, old box, new box — same reader, prompt and crop recipe. Fully paired.
"""
import argparse, json, random, statistics, sys, time, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT/"eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default=str(ROOT/"eval/testset_v2/ru_pages/images"))
ap.add_argument("--old", default=str(ROOT/"eval/runs/rp_det_lines_v2g.json"))
ap.add_argument("--new", default=str(ROOT/"eval/runs/rp_det_lines_pagerel.json"))
ap.add_argument("--base", default=str(ROOT/"models/GLM-OCR"))
ap.add_argument("--adapter", default=str(ROOT/"models/adapters/glm-ocr-lora-ru-r5c"))
ap.add_argument("--n", type=int, default=260); ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--out", default=str(ROOT/"eval/runs/det_grow_ab.json"))
a = ap.parse_args()
sys.path.insert(0, str(ROOT))
from PIL import Image
from bilimai.reader import make_reader
def iou(p,g):
    x0,y0,x1,y1=max(p[0],g[0]),max(p[1],g[1]),min(p[2],g[2]),min(p[3],g[3])
    i=max(0.,x1-x0)*max(0.,y1-y0)
    return i/max(1e-9,(p[2]-p[0])*(p[3]-p[1])+(g[2]-g[0])*(g[3]-g[1])-i)
def cer(r,h):
    r,h=unicodedata.normalize("NFC",r),unicodedata.normalize("NFC",h); d=list(range(len(h)+1))
    for i,cr in enumerate(r,1):
        prev,d[0]=d[0],i
        for j,ch in enumerate(h,1): prev,d[j]=d[j],min(d[j]+1,d[j-1]+1,prev+(cr!=ch))
    return d[-1]/max(1,len(r))
gt=json.load(open(a.gt,encoding="utf-8")); old=json.load(open(a.old,encoding="utf-8")); new=json.load(open(a.new,encoding="utf-8"))
pairs=[]
for fn,page in gt.items():
    O=[p[:4] for p in old.get(fn,[])]; N=[p[:4] for p in new.get(fn,[])]
    if not O or not N: continue
    for l in page["lines"]:
        g=l["bbox"]; t=(l.get("text") or "").strip()
        if not t: continue
        bo,oi=max(((iou(p,g),i) for i,p in enumerate(O)),default=(0.,-1))
        bn,ni=max(((iou(p,g),i) for i,p in enumerate(N)),default=(0.,-1))
        if bo>=0.5 and bn>=0.5: pairs.append((fn,t,g,O[oi],N[ni],bo))
random.Random(1).shuffle(pairs); sample=pairs[:a.n]
print(f"lines matched under BOTH box sets: {len(pairs)} | sampling {len(sample)} -> {3*len(sample)} reads",flush=True)
PAD=12; imgs=[]; meta=[]; cache={}
for fn,t,g,o,n,bo in sample:
    if fn not in cache: cache[fn]=Image.open(Path(a.images)/fn).convert("RGB")
    im=cache[fn]; W,H=im.size
    for kind,b in (("human",g),("old",o),("new",n)):
        imgs.append(im.crop((max(0,b[0]-PAD),max(0,b[1]-PAD),min(W,b[2]+PAD),min(H,b[3]+PAD)))); meta.append((kind,t,bo))
r=make_reader(a.base,a.adapter,line_h=128,max_new_tokens=96)
print(f"reader {r.name} | reading {len(imgs)} crops...",flush=True); t0=time.time()
texts,_=r.read(imgs,batch_size=a.batch_size,with_conf=False); print(f"read in {time.time()-t0:.0f}s",flush=True)
rows={}
for (k,t,bo),hyp in zip(meta,texts): rows.setdefault((t,bo),{})[k]=(hyp or "").strip()
recs=[{"ref":t,"iou":bo,"n":len(t),**{k:cer(t,v[k]) for k in ("human","old","new")}} for (t,bo),v in rows.items() if len(v)==3]
def cw(k):
    ed=sum(x[k]*x["n"] for x in recs); ch=sum(x["n"] for x in recs); return ed/max(ch,1)
h,o,n=cw("human"),cw("old"),cw("new")
print(f"\nn paired lines: {len(recs)}  (identical line set for all three)")
print(f"  human box : {h:.5f}")
print(f"  OLD box   : {o:.5f}   cost vs human {o-h:+.5f}")
print(f"  NEW box   : {n:.5f}   cost vs human {n-h:+.5f}")
print(f"  NEW - OLD : {n-o:+.5f}   -> {'BETTER' if n<o else 'WORSE' if n>o else 'no change'}")
w=sum(1 for x in recs if x['new']>x['old']+1e-9); b=sum(1 for x in recs if x['new']<x['old']-1e-9)
print(f"  per line: new better {b} | new worse {w} | identical {len(recs)-b-w}")
print("\nby overlap band (old-box IoU):")
for lo,hi in ((0.5,0.7),(0.7,0.85),(0.85,1.01)):
    s=[x for x in recs if lo<=x["iou"]<hi]
    if not s: continue
    ch=sum(x["n"] for x in s)
    print(f"  {lo:.2f}-{hi:.2f}: n={len(s):>4} human {sum(x['human']*x['n'] for x in s)/ch:.4f} old {sum(x['old']*x['n'] for x in s)/ch:.4f} new {sum(x['new']*x['n'] for x in s)/ch:.4f}")
json.dump({"n":len(recs),"human":h,"old":o,"new":n},open(a.out,"w"),indent=1)
