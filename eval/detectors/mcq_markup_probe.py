#!/usr/bin/env python3
"""Can Qwen see a TICK and a MARKUP STROKE? — the two probes that open or close three scoreboard rows.

WHAT THIS DECIDES
`eval/scoreboard.py` reports six cells with no test at all: Тесты, Изложение and Открытые вопросы, in both
languages. Тесты is blocked on one question — can the model say which option a pupil ticked? Изложение's
sentence-parsing is blocked on another — can it report that a word is underlined, circled or crossed out?
Neither question has ever been asked, and neither can be, because **no markup ground truth exists anywhere
in the project**: every annotation file carries only `pupil_text / pupil_comment / teacher_comment / paper /
text_line`.

SO THE PAGES CARRY THEIR OWN GROUND TRUTH
Same trick that made `uz_page_probe.py` work: compose the pages, and you know the answer for free. We draw
the tick, so we know which option it is on. We draw the underline, so we know which word it is under. That
turns two unanswerable questions into two measured ones for the price of an afternoon of CPU.

HOW TO READ THE RESULT — this matters more than the number
A synthetic tick is cleaner than a child's tick, and a drawn underline is straighter than a ruler-less one.
So the arms are NOT symmetric:
  * **FAIL is decisive.** If the model cannot see a clean, unambiguous, high-contrast mark that we drew
    ourselves, it will not see a real one. That closes the row cheaply and honestly.
  * **PASS is suggestive only.** It says the capability is not absent, so real pages are worth labelling.
    It does not say Тесты works.
This is the same caveat that applies to every Uzbek number in the repo, for the same reason.

TWO STEPS, because the two halves need different environments — rendering needs fontTools (eval/.venv)
and inference needs mlx_vlm (eval/.venv-mlx), and no venv has both:

  eval/.venv/bin/python     eval/detectors/mcq_markup_probe.py --task mcq    --make
  eval/.venv-mlx/bin/python eval/detectors/mcq_markup_probe.py --task mcq    --run --reuse
  eval/.venv/bin/python     eval/detectors/mcq_markup_probe.py --task markup --make
  eval/.venv-mlx/bin/python eval/detectors/mcq_markup_probe.py --task markup --run --reuse

⚠ Do not run two MLX processes on one Metal GPU — measured SLOWER than one (135 s vs 84 s per call).
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval" / "util"))

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["mcq", "markup"], required=True)
ap.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-4B-Instruct-MLX-8bit"))
ap.add_argument("--pages", type=int, default=6)
ap.add_argument("--lang", choices=["ru", "uz"], default="ru")
ap.add_argument("--fonts", default="")
ap.add_argument("--out", default="")
ap.add_argument("--runs", default="")
ap.add_argument("--max-side", type=int, default=1600)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--make", action="store_true", help="generate the pages only, load no model")
ap.add_argument("--run", action="store_true", help="ask the model about the pages")
ap.add_argument("--reuse", action="store_true", help="score the pages already in --out (see the header)")
a = ap.parse_args()
if not (a.make or a.run):
    sys.exit("pass --make (render, eval/.venv) or --run --reuse (infer, eval/.venv-mlx)")

FONTS = Path(a.fonts) if a.fonts else ROOT / ("data/fonts/ru_school" if a.lang == "ru" else "data/fonts/uz_bridge")
OUT = Path(a.out) if a.out else ROOT / f"out/{a.task}_probe_{a.lang}"
RUNS = Path(a.runs) if a.runs else ROOT / f"eval/runs/{a.task}_probe_{a.lang}.json"
OUT.mkdir(parents=True, exist_ok=True)

OPTIONS_RU = ["А", "Б", "В", "Г"]
QUESTIONS_RU = [
    ("Столица России", ["Москва", "Киев", "Минск", "Астана"], 0),
    ("Сколько будет 7 × 8", ["54", "56", "64", "48"], 1),
    ("Кто написал «Му-му»", ["Пушкин", "Толстой", "Тургенев", "Чехов"], 2),
    ("Самая длинная река Европы", ["Дон", "Днепр", "Волга", "Урал"], 2),
    ("Часть речи слова «быстро»", ["глагол", "наречие", "союз", "предлог"], 1),
    ("Сколько дней в неделе", ["пять", "шесть", "семь", "восемь"], 2),
    ("Планета ближе всех к Солнцу", ["Венера", "Меркурий", "Марс", "Земля"], 1),
    ("Результат 12 + 15", ["25", "27", "28", "17"], 1),
]
MARKUP_SENTENCES_RU = [
    "Человечество сделало уверенный шаг в космос",
    "Осенью птицы улетают в тёплые края",
    "Мальчик быстро побежал домой после уроков",
    "На столе лежала старая потрёпанная книга",
    "Утром выпал первый снег и стало светло",
    "Река медленно несла свои воды к морю",
    "Дети играли во дворе до самого вечера",
    "Учитель объяснил новую тему очень понятно",
]
MARK_KINDS = ["tick", "circle", "cross", "box"]
MARKUP_KINDS = ["underline", "circle", "strike"]


# ================================================================ RENDER (needs fontTools: eval/.venv)
def make_pages():
    import cv2
    import make_uz_font_strips as G

    rng = random.Random(a.seed)
    fonts = G.load_fonts(FONTS)
    assert fonts, f"no fonts under {FONTS}"

    _orig_paper = G._paper

    def _white_paper(W, H, rng_, baseline, th):
        """PURE white so the multiply blend below is exactly neutral outside the ink — the uz probe's
        first version left a rectangular halo around every strip, a free machine-visible hint of where
        each box was, and would have scored on an artefact we introduced."""
        return np.full((H, W, 3), 255, np.uint8), "plain"

    G._paper = _white_paper

    def paper(W, H, rng_, spacing, ruled=True):
        base = (rng_.randint(236, 248), rng_.randint(242, 251), rng_.randint(245, 253))
        pg = np.full((H, W, 3), base, np.uint8)
        noise = np.random.RandomState(rng_.randrange(1 << 30)).normal(0, 2.5, (H, W, 1))
        pg = np.clip(pg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if ruled:
            col = (rng_.randint(180, 215), rng_.randint(140, 178), rng_.randint(125, 165))
            for y in range(spacing, H, spacing):
                cv2.line(pg, (0, y), (W, y), col, 1)
        mx = int(W * rng_.uniform(0.07, 0.11))
        cv2.line(pg, (mx, 0), (mx, H), (70, 70, 205), 2)
        return pg, mx

    def stamp(pg, strip, x, y, h):
        """Multiply-blend a rendered strip onto the page at (x, y), scaled to height h. Returns its box."""
        sc = h / strip.shape[0]
        w = max(8, int(strip.shape[1] * sc))
        if y + h >= pg.shape[0] or x + w >= pg.shape[1]:
            return None
        s = cv2.resize(strip, (w, h), interpolation=cv2.INTER_AREA)
        reg = pg[y:y + h, x:x + w].astype(np.float32)
        pg[y:y + h, x:x + w] = np.clip(reg * (s.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
        return [x, y, x + w, y + h]

    def hand(txt, writer):
        r = G.render_line(txt, fonts, random.Random(rng.randrange(1 << 30)), writer)
        return None if r is None else r[0]

    # pen colour for the pupil's MARK — graphite/blue, never the teacher's red (founder rule:
    # red = teacher, everything else = pupil; bilimai/ink.py)
    def pen(rng_):
        return (rng_.randint(90, 150), rng_.randint(60, 110), rng_.randint(40, 90))

    pages = []
    try:
        for p in range(a.pages):
            PW, PH = 1600, 1200
            writer = G.make_writer(rng.randrange(4000), fonts)
            if a.task == "mcq":
                pg, mx = paper(PW, PH, rng, PH // 14, ruled=False)
                qs = rng.sample(QUESTIONS_RU, k=min(4, len(QUESTIONS_RU)))
                y = 70
                items = []
                qnum = 0
                for qi, (stem, opts, _correct) in enumerate(qs):
                    # A question is only worth drawing if the WHOLE of it fits. Half a question whose marked
                    # option fell off the bottom would leave ground truth asserting a mark that is not on the
                    # page — the probe would then measure its own rendering bug, which is exactly the class of
                    # self-inflicted error this project keeps paying for.
                    need = 62 + len(opts) * 52 + 26
                    if y + need >= PH - 20:
                        break
                    s = hand(f"{qnum + 1}. {stem}?", writer)
                    if s is None: continue
                    qnum += 1                                    # number drawn on the page == number in GT
                    stamp(pg, s, mx + 20, y, 44); y += 62
                    marked = rng.randrange(len(opts))
                    kind = MARK_KINDS[(qi + p) % len(MARK_KINDS)]
                    mark_drawn = False
                    for oi, opt in enumerate(opts):
                        s = hand(f"{OPTIONS_RU[oi]}) {opt}", writer)
                        if s is None:
                            y += 52
                            continue
                        bx = stamp(pg, s, mx + 90, y, 38)
                        if bx and oi == marked:
                            mark_drawn = True
                            cx, cy = mx + 55, y + 19
                            c = pen(rng); t = rng.randint(3, 5)
                            if kind == "tick":
                                cv2.line(pg, (cx - 14, cy), (cx - 4, cy + 12), c, t)
                                cv2.line(pg, (cx - 4, cy + 12), (cx + 16, cy - 16), c, t)
                            elif kind == "circle":
                                cv2.ellipse(pg, (bx[0] + 22, cy), (34, 24), rng.randint(-12, 12), 0, 360, c, t)
                            elif kind == "cross":
                                cv2.line(pg, (cx - 14, cy - 14), (cx + 14, cy + 14), c, t)
                                cv2.line(pg, (cx + 14, cy - 14), (cx - 14, cy + 14), c, t)
                            else:                                            # filled box
                                cv2.rectangle(pg, (cx - 15, cy - 15), (cx + 15, cy + 15), c, -1)
                        y += 52
                    if mark_drawn:
                        items.append({"q": qnum, "stem": stem, "options": opts,
                                      "marked_index": marked, "marked_letter": OPTIONS_RU[marked],
                                      "marked_text": opts[marked], "mark_kind": kind})
                    y += 26
                fn = f"{a.task}_{a.lang}_{p:02d}.jpg"
                cv2.imwrite(str(OUT / fn), pg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                pages.append({"file": fn, "w": PW, "h": PH, "questions": items})
                print(f"{fn}: {len(items)} questions", flush=True)

            else:                                                            # ---- markup
                sp = PH // 12
                pg, mx = paper(PW, PH, rng, sp, ruled=True)
                y = sp
                items = []
                sents = rng.sample(MARKUP_SENTENCES_RU, k=min(6, len(MARKUP_SENTENCES_RU)))
                for si, sent in enumerate(sents):
                    words = sent.split()
                    # render word by word so we know each word's box exactly
                    x = mx + rng.randint(14, 34)
                    boxes = []
                    for w in words:
                        s = hand(w, writer)
                        if s is None: boxes.append(None); continue
                        bx = stamp(pg, s, x, y, int(sp * 0.62))
                        boxes.append(bx)
                        if bx is None: break
                        x = bx[2] + rng.randint(12, 22)
                    live = [i for i, b in enumerate(boxes) if b]
                    if not live: y += sp; continue
                    wi = rng.choice(live)
                    kind = MARKUP_KINDS[(si + p) % len(MARKUP_KINDS)]
                    b = boxes[wi]; c = pen(rng); t = rng.randint(3, 5)
                    if kind == "underline":
                        cv2.line(pg, (b[0] - 4, b[3] + 6), (b[2] + 4, b[3] + 6 + rng.randint(-3, 3)), c, t)
                    elif kind == "circle":
                        cv2.ellipse(pg, ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2),
                                    ((b[2] - b[0]) // 2 + 16, (b[3] - b[1]) // 2 + 12),
                                    rng.randint(-10, 10), 0, 360, c, t)
                    else:                                                    # strike
                        cy = (b[1] + b[3]) // 2
                        cv2.line(pg, (b[0] - 6, cy), (b[2] + 6, cy + rng.randint(-3, 3)), c, t)
                    items.append({"line": si + 1, "text": sent, "word": words[wi],
                                  "word_index": wi, "markup_kind": kind, "bbox": b})
                    y += sp
                    if y + sp >= PH: break
                fn = f"{a.task}_{a.lang}_{p:02d}.jpg"
                cv2.imwrite(str(OUT / fn), pg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                pages.append({"file": fn, "w": PW, "h": PH, "markup": items})
                print(f"{fn}: {len(items)} marked words", flush=True)
    finally:
        G._paper = _orig_paper

    (OUT / "gt.json").write_text(json.dumps({"task": a.task, "lang": a.lang, "pages": pages},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}/gt.json  ({len(pages)} pages)")
    return pages


# ================================================================ PROMPTS
MCQ_PROMPT = (
    "This is a photograph of a school test paper filled in by a pupil.\n"
    "For EVERY numbered question, report which single option the pupil chose.\n"
    "The chosen option is the one carrying a hand-drawn mark — a tick, a cross, a circle round it, or a\n"
    "filled square beside it. Exactly one option per question is marked.\n"
    'Return ONLY a JSON array, one entry per question: {"q": 1, "letter": "А", "text": "...", '
    '"mark": "tick|cross|circle|box"}\n'
    "- letter is the option label (А, Б, В or Г) of the MARKED option.\n"
    "- text is that option's words, copied as written.\n"
    "- mark is which kind of mark you see on it.\n"
    "Do not add commentary. If you cannot tell for a question, still return the entry with your best guess."
)

MARKUP_PROMPT = (
    "This is a photograph of a page from a Russian school notebook.\n"
    "Some words have been marked by hand. Find EVERY marked word.\n"
    'Return ONLY a JSON array, one entry per marked word: {"word": "...", "kind": "underline|circle|strike"}\n'
    "- underline = a line drawn UNDER the word.\n"
    "- circle = a loop drawn AROUND the word.\n"
    "- strike = a line drawn THROUGH the middle of the word, crossing it out.\n"
    "Copy the word exactly as written. Report only words that carry a mark; ignore the printed ruling and\n"
    "the red margin line. Do not add commentary."
)


# ================================================================ RUN (needs mlx_vlm: eval/.venv-mlx)
import re
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def parse_items(txt):
    txt = THINK.sub("", txt, count=1)
    items, i, j = [], txt.find("["), txt.rfind("]")
    if i != -1 and j > i:
        try:
            v = json.loads(txt[i:j + 1])
            items = [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
        except Exception:
            items = []
    if not items:
        for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
            try:
                x = json.loads(m.group(0))
                if isinstance(x, dict): items.append(x)
            except Exception: pass
    return items


def norm(s):
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


def run_probe(pages):
    from PIL import Image
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(a.model)
    cfg = model.config
    prompt = MCQ_PROMPT if a.task == "mcq" else MARKUP_PROMPT
    res, t0 = [], time.time()

    for p in pages:
        im = Image.open(OUT / p["file"]).convert("RGB")
        W0, H0 = im.size
        sc = min(1.0, a.max_side / max(W0, H0))
        snd = im.resize((int(W0 * sc), int(H0 * sc))) if sc < 1.0 else im
        t1 = time.time()
        fmt = apply_chat_template(processor, cfg, prompt, num_images=1)
        o = generate(model, processor, fmt, [snd], max_tokens=a.max_tokens, verbose=False,
                     repetition_penalty=1.05, repetition_context_size=256, temperature=0.0)
        raw = o if isinstance(o, str) else getattr(o, "text", str(o))
        res.append({"file": p["file"], "seconds": round(time.time() - t1, 1),
                    "items": parse_items(raw), "raw": raw[:4000]})
        print(f"  {p['file']}: {len(res[-1]['items'])} items in {time.time()-t1:.0f}s", flush=True)

    # ------------------------------------------------------------ score
    if a.task == "mcq":
        n = hit_letter = hit_text = hit_kind = 0
        by_kind = {}
        for p, r in zip(pages, res):
            got = {str(it.get("q")): it for it in r["items"]}
            for q in p["questions"]:
                n += 1
                g = got.get(str(q["q"]))
                k = by_kind.setdefault(q["mark_kind"], [0, 0]); k[1] += 1
                if not g: continue
                ok = norm(g.get("letter")) == norm(q["marked_letter"])
                hit_letter += ok; k[0] += ok
                hit_text += norm(g.get("text")) == norm(q["marked_text"])
                hit_kind += norm(g.get("mark")) == norm(q["mark_kind"])
        summary = {"questions": n,
                   "letter_accuracy": round(hit_letter / max(n, 1), 3),
                   "option_text_accuracy": round(hit_text / max(n, 1), 3),
                   "mark_kind_accuracy": round(hit_kind / max(n, 1), 3),
                   "by_mark_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())}}
        summary["verdict"] = ("OPEN — the model can point at a ticked option; real ticked pages are worth "
                              "labelling" if summary["letter_accuracy"] >= 0.7 else
                              "CLOSED for now — it cannot read a mark we drew ourselves, cleanly, so it "
                              "will not read a child's")
    else:
        tp = fp = fn_ = 0
        kind_ok = 0
        by_kind = {}
        for p, r in zip(pages, res):
            want = {norm(m["word"]): m for m in p["markup"]}
            seen = set()
            for it in r["items"]:
                w = norm(it.get("word"))
                if w in want and w not in seen:
                    tp += 1; seen.add(w)
                    k = by_kind.setdefault(want[w]["markup_kind"], [0, 0]); k[1] += 1
                    if norm(it.get("kind")) == norm(want[w]["markup_kind"]):
                        kind_ok += 1; k[0] += 1
                else:
                    fp += 1
            for w, m in want.items():
                if w not in seen:
                    fn_ += 1
                    by_kind.setdefault(m["markup_kind"], [0, 0])[1] += 1
        summary = {"marked_words": tp + fn_, "found": tp, "missed": fn_, "spurious": fp,
                   "recall": round(tp / max(tp + fn_, 1), 3),
                   "precision": round(tp / max(tp + fp, 1), 3),
                   "kind_correct_when_found": f"{kind_ok}/{tp}",
                   "by_markup_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())}}
        summary["verdict"] = ("OPEN — markup is visible to the model; labelling real pages is worth it"
                              if summary["recall"] >= 0.5 else
                              "CLOSED for now — it cannot report a stroke we drew ourselves; sentence "
                              "parsing needs a different approach, not more tuning")

    summary["caveat"] = ("Synthetic marks are cleaner than a child's. A FAIL here is decisive; a PASS only "
                         "says the capability is not absent. Real pages still have to be labelled.")
    out = {"task": a.task, "lang": a.lang, "model": Path(a.model).name,
           "pages": len(pages), "seconds": round(time.time() - t0, 1),
           "summary": summary, "records": res}
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    RUNS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n=== {a.task.upper()} PROBE ({a.lang}) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n-> {RUNS}")


# ================================================================
gt_path = OUT / "gt.json"
if a.make or not a.reuse:
    pages = make_pages()
else:
    pages = json.loads(gt_path.read_text(encoding="utf-8"))["pages"]
    print(f"reusing {len(pages)} pages from {OUT}", flush=True)
if a.run:
    run_probe(pages)
