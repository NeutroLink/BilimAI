"""E4.4 — the red pen. Draw contract Marks onto a scan.

Input : an image (path or PIL) + a list of Mark dicts (see contracts/common.schema.json#/$defs/Mark)
Output: a new PIL image (original untouched) with red-pen marks: check ✓, cross ✗, wavy underline,
        circle, insert (caret + text), strike, margin note. Low-confidence / needs_review marks are
        drawn in a lighter, dashed style so the teacher's eye goes there first.

Coordinates are in ORIGINAL image pixels (as the contract requires); the pen thickness and font
size scale with image height so a 300-dpi scan and a phone photo look alike.

CLI:  python -m bilimai.render --image page.jpg --marks marks.json --out marked.png
      (marks.json = a JSON list of Mark objects, or a full contract response containing "marks")
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

RED = (215, 30, 30)
RED_SOFT = (235, 110, 110)
YELLOW = (222, 168, 0)          # review («на проверку») marks — founder decision 2026-08-19: yellow, not soft red

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS, Cyrillic + Latin
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Pen:
    def __init__(self, img: Image.Image, seed: int = 0):
        self.img = img.convert("RGB")
        self.d = ImageDraw.Draw(self.img, "RGBA")
        h = self.img.size[1]
        self.w = max(2, round(h / 600))              # stroke width: ~5 px on a 3000-px page
        self.fs = max(14, round(h / 60))             # font size for notes: ~50 px on a 3000-px page
        self.font = _font(self.fs)
        self.small = _font(max(12, round(self.fs * 0.7)))
        self.rng = random.Random(seed)               # deterministic "hand jitter"

    # ------------------------------------------------------------------ primitives
    def _col(self, m):
        review = m.get("verdict") == "review" or m.get("needs_review") or (m.get("confidence") is not None and m["confidence"] < 0.6)
        return (YELLOW if review else RED)

    def _jit(self, k=1.0):
        return self.rng.uniform(-k, k) * self.w

    def check(self, b, m):
        x0, y0, x1, y1 = b; c = self._col(m); s = min(x1 - x0, y1 - y0) or self.fs
        s = max(s, self.fs)
        cx, cy = x0 + s * 0.1, y0 + s * 0.55
        pts = [(cx, cy), (cx + s * 0.3, cy + s * 0.35), (cx + s * 0.85, cy - s * 0.45)]
        pts = [(x + self._jit(), y + self._jit()) for x, y in pts]
        self.d.line(pts, fill=c, width=self.w + 1, joint="curve")

    def cross(self, b, m):
        x0, y0, x1, y1 = b; c = self._col(m); s = max(min(x1 - x0, y1 - y0), self.fs)
        cx, cy = x0, y0
        self.d.line([(cx + self._jit(), cy + self._jit()), (cx + s + self._jit(), cy + s + self._jit())], fill=c, width=self.w + 1)
        self.d.line([(cx + s + self._jit(), cy + self._jit()), (cx + self._jit(), cy + s + self._jit())], fill=c, width=self.w + 1)

    def underline(self, b, m):
        x0, y0, x1, y1 = b; c = self._col(m)
        y = y1 + self.w * 1.5; amp = self.w * 1.2; step = max(4, self.w * 2.5)
        pts = []; x = x0; i = 0
        while x <= x1:
            pts.append((x, y + amp * math.sin(i * 1.3) + self._jit(0.3))); x += step; i += 1
        if len(pts) > 1:
            self.d.line(pts, fill=c, width=self.w, joint="curve")

    def circle(self, b, m):
        x0, y0, x1, y1 = b; c = self._col(m); pad = self.w * 2
        self.d.ellipse([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=c, width=self.w)

    def strike(self, b, m):
        x0, y0, x1, y1 = b; c = self._col(m); y = (y0 + y1) / 2
        self.d.line([(x0, y + self._jit()), (x1, y + self._jit())], fill=c, width=self.w)

    def insert(self, b, m):
        """Caret under the gap + the inserted text above it."""
        x0, y0, x1, y1 = b; c = self._col(m); cx = (x0 + x1) / 2; s = self.w * 3
        self.d.line([(cx - s, y1 + s), (cx, y1), (cx + s, y1 + s)], fill=c, width=self.w)
        txt = m.get("text", "")
        if txt:
            tw = self.d.textlength(txt, font=self.font)
            self.d.text((cx - tw / 2, y0 - self.fs - self.w), txt, fill=c, font=self.font)

    def margin_note(self, b, m):
        x0, y0, x1, y1 = b; c = self._col(m); txt = m.get("text") or m.get("explanation") or ""
        # wrap to the box width
        words, lines, cur = txt.split(), [], ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if self.d.textlength(t, font=self.small) > (x1 - x0) and cur:
                lines.append(cur); cur = wd
            else:
                cur = t
        if cur: lines.append(cur)
        y = y0
        for ln in lines:
            self.d.text((x0, y), ln, fill=c, font=self.small); y += int(self.small.size * 1.25)

    KINDS = {"check": check, "cross": cross, "underline": underline, "circle": circle,
             "strike": strike, "insert": insert, "margin_note": margin_note}

    def draw(self, mark: dict):
        fn = self.KINDS.get(mark.get("kind"))
        if fn is None:
            raise ValueError(f"unknown mark kind {mark.get('kind')!r}")
        b = [float(v) for v in mark["bbox"]]
        fn(self, b, mark)


def render(image, marks: Iterable[dict], seed: int = 0) -> Image.Image:
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    pen = Pen(img, seed=seed)
    for m in marks:
        pen.draw(m)
    return pen.img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--marks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=0, help="downscale output so the longer side is this many px (0 = keep)")
    a = ap.parse_args()
    data = json.load(open(a.marks, encoding="utf-8"))
    marks = data if isinstance(data, list) else (data.get("marks") or data.get("response", {}).get("marks", []))
    out = render(a.image, marks)
    if a.max_side and max(out.size) > a.max_side:
        s = a.max_side / max(out.size); out = out.resize((round(out.size[0] * s), round(out.size[1] * s)), Image.LANCZOS)
    out.save(a.out); print(f"wrote {a.out} with {len(marks)} marks")


if __name__ == "__main__":
    main()
