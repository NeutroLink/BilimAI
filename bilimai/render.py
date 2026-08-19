"""E4.4 — the red pen. Draw contract Marks onto a scan.

Input : an image (path or PIL) + a list of Mark dicts (see contracts/common.schema.json#/$defs/Mark)
Output: a new PIL image (original untouched) with red-pen marks: check ✓, cross ✗, wavy underline (+ letter-level
        correction when the verifier supplies letter boxes: wrong letter struck, right letter above), circle, insert
        (caret + text), strike, margin note. Review («на проверку») marks are a translucent YELLOW highlight over the word
        (founder decision 2026-08-19). `render(..., layer_only=True)` returns a transparent marks-only layer.

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


YELLOW_FILL = (255, 214, 0, 92)  # review highlight (translucent) — founder 2026-08-19: highlight, not underline


class Pen:
    def __init__(self, img: Image.Image, seed: int = 0, layer_only: bool = False):
        # layer_only: draw on a transparent canvas of the same size (marks layer for overlays / toggles)
        # RGB base + "RGBA" draw mode = PIL blends translucent fills (the yellow highlight); an RGBA base would overwrite
        self.img = Image.new("RGBA", img.size, (0, 0, 0, 0)) if layer_only else img.convert("RGB")
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

    def _is_review(self, m):
        return m.get("verdict") == "review" or (m.get("verdict") != "error" and (m.get("needs_review") or (m.get("confidence") is not None and m["confidence"] < 0.6)))

    def highlight(self, b, m):
        """Review («на проверку»): translucent yellow highlight over the word box — the teacher's eye goes there, nothing is asserted."""
        x0, y0, x1, y1 = b; pad = self.w * 1.5
        self.d.rounded_rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], radius=self.w * 2, fill=YELLOW_FILL)

    def letter_fix(self, m):
        """Teacher-style correction: strike each wrong letter (geometry from the verifier's CTC alignment, mark["letters"]) and
        write the correct letter above it; a missing letter gets a caret + the letter above. Red only (error verdicts)."""
        c = RED
        for L in m.get("letters", []):
            x0, y0, x1, y1 = [float(v) for v in L["bbox"]]; corr = L.get("correct", "")
            if L.get("ink"):                                   # strike the wrong letter: one slash across its box
                self.d.line([(x0 + self._jit(0.3), y1 + self.w), (x1 + self._jit(0.3), y0 - self.w)], fill=c, width=self.w)
            else:                                              # missing letter: caret under the gap
                cx = (x0 + x1) / 2; sz = self.w * 2
                self.d.line([(cx - sz, y1 + sz), (cx, y1), (cx + sz, y1 + sz)], fill=c, width=self.w)
            if corr:
                tw = self.d.textlength(corr, font=self.font)
                self.d.text(((x0 + x1) / 2 - tw / 2, y0 - self.fs - self.w), corr, fill=c, font=self.font)

    def underline(self, b, m):
        if self._is_review(m): return self.highlight(b, m)
        x0, y0, x1, y1 = b; c = self._col(m)
        y = y1 + self.w * 1.5; amp = self.w * 1.2; step = max(4, self.w * 2.5)
        pts = []; x = x0; i = 0
        while x <= x1:
            pts.append((x, y + amp * math.sin(i * 1.3) + self._jit(0.3))); x += step; i += 1
        if len(pts) > 1:
            self.d.line(pts, fill=c, width=self.w, joint="curve")
        if m.get("letters"): self.letter_fix(m)

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


def render(image, marks: Iterable[dict], seed: int = 0, layer_only: bool = False) -> Image.Image:
    """Marked page (RGB) — or, with layer_only=True, a transparent RGBA layer holding only the marks (same size as the page),
    for overlays and show/hide toggles in a UI."""
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    pen = Pen(img, seed=seed, layer_only=layer_only)
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
