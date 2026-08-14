"""BilimAI MVP — paste a typed assignment + rubric, get a structured grade.

Single-file demo of the "Mark" stage: FastAPI serving one embedded HTML page,
grading via a local Ollama model with a JSON-schema-constrained response.
No OCR, no accounts, no database — see PLAN.md for what comes after.

Run:  uvicorn app:app --port 8000   (Ollama must be serving on :11434)
"""

import json
import os
import re

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("BILIMAI_MODEL", "qwen3:8b")

app = FastAPI(title="BilimAI MVP")

# The model must fill this exact shape — vLLM/Ollama enforce it during decoding,
# so the gradebook never receives free text.
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "detected_language": {"type": "string"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                    "max_score": {"type": "number"},
                    "feedback": {"type": "string"},
                },
                "required": ["name", "score", "max_score", "feedback"],
            },
        },
        "overall_score": {"type": "number"},
        "overall_max": {"type": "number"},
        "general_feedback": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "detected_language",
        "criteria",
        "overall_score",
        "overall_max",
        "general_feedback",
        "confidence",
    ],
}

SYSTEM_PROMPT = """\
You are BilimAI, an experienced, fair and strict school teacher grading a student's
written assignment against a rubric. The submission will be in Russian or Uzbek
(Latin or Cyrillic script).

Rules:
- Grade ONLY against the rubric's criteria. One entry in "criteria" per rubric criterion.
- Write all feedback in the SAME language as the student's submission.
- Feedback must be specific: point to what the student actually wrote, name concrete
  strengths and concrete mistakes. Never generic praise.
- "overall_score" is the sum of criterion scores; "overall_max" the sum of maxima.
- "confidence" is 0.0-1.0: how certain you are a human teacher would agree with this
  grade. Be honest — lower it for ambiguous rubrics or off-topic submissions.
- If the submission does not match the assignment at all, score accordingly and say so.
"""


class GradeRequest(BaseModel):
    assignment: str
    rubric: str


@app.post("/api/grade")
async def grade(req: GradeRequest):
    if not req.assignment.strip() or not req.rubric.strip():
        return JSONResponse({"error": "Both the assignment and the rubric are required."}, status_code=422)

    user_prompt = (
        f"RUBRIC:\n{req.rubric.strip()}\n\n"
        f"STUDENT SUBMISSION:\n{req.assignment.strip()}\n\n"
        "Grade the submission against the rubric."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": GRADE_SCHEMA,
        "stream": False,
        "think": False,  # qwen3: skip the thinking preamble so the demo stays snappy
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            content = r.json()["message"]["content"]
    except httpx.ConnectError:
        return JSONResponse(
            {"error": f"Ollama is not reachable at {OLLAMA_URL}. Is `ollama serve` running?"},
            status_code=502,
        )
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            {"error": f"Ollama error {e.response.status_code}: {e.response.text[:300]}"},
            status_code=502,
        )

    # Belt-and-braces: strip any thinking tags, then parse the schema-constrained JSON.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": "Model returned unparseable output.", "raw": content[:500]},
            status_code=502,
        )
    # Small local models occasionally emit a criterion twice (an abandoned draft plus
    # the final one). Keep the last occurrence per name and recompute the totals from
    # what survives, so the card always adds up.
    seen = {}
    for c in result.get("criteria", []):
        seen[c["name"]] = c
    result["criteria"] = list(seen.values())
    result["overall_score"] = sum(c["score"] for c in result["criteria"])
    result["overall_max"] = sum(c["max_score"] for c in result["criteria"])
    result["model"] = MODEL
    return result


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
        return {"ollama": "up", "model": MODEL, "model_pulled": any(m.startswith(MODEL) for m in models)}
    except Exception:
        return {"ollama": "down", "model": MODEL, "model_pulled": False}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BilimAI</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #ffffff; --ink: #1a1d21; --muted: #6b7280;
    --accent: #2563eb; --accent-ink: #ffffff; --line: #e5e7eb;
    --good: #16a34a; --mid: #d97706; --bad: #dc2626;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111418; --card: #1a1f26; --ink: #e7eaee; --muted: #9aa3af;
      --accent: #3b82f6; --line: #2a313a;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 880px; margin: 0 auto; padding: 28px 20px 60px; }
  header h1 { font-size: 22px; margin: 0; }
  header p { color: var(--muted); margin: 4px 0 24px; }
  label { display: block; font-weight: 600; margin: 16px 0 6px; }
  textarea { width: 100%; min-height: 150px; padding: 12px; border: 1px solid var(--line);
             border-radius: 10px; background: var(--card); color: var(--ink);
             font: inherit; resize: vertical; }
  textarea:focus { outline: 2px solid var(--accent); border-color: transparent; }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 18px; flex-wrap: wrap; }
  button { background: var(--accent); color: var(--accent-ink); border: 0;
           padding: 11px 22px; border-radius: 10px; font: inherit; font-weight: 600;
           cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  .ghost { background: transparent; color: var(--accent); border: 1px solid var(--accent); }
  #status { color: var(--muted); }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
          padding: 20px; margin-top: 24px; }
  .score-line { display: flex; justify-content: space-between; align-items: baseline; }
  .big { font-size: 30px; font-weight: 700; }
  .conf { color: var(--muted); font-size: 13px; }
  .crit { border-top: 1px solid var(--line); padding: 12px 0; }
  .crit b { display: flex; justify-content: space-between; }
  .crit p { margin: 6px 0 0; color: var(--muted); }
  .error { color: var(--bad); }
  .g { color: var(--good); } .m { color: var(--mid); } .b { color: var(--bad); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>BilimAI</h1>
    <p>Paste a student's typed assignment and the rubric — get a draft grade with feedback. Russian &amp; Uzbek. Fully local.</p>
  </header>

  <label for="rubric">Rubric (criteria + points)</label>
  <textarea id="rubric" placeholder="1. Раскрытие темы — 5 баллов&#10;2. Аргументация — 5 баллов&#10;3. Грамотность — 5 баллов"></textarea>

  <label for="assignment">Student submission</label>
  <textarea id="assignment" placeholder="Текст работы ученика… / O'quvchi ishining matni…"></textarea>

  <div class="row">
    <button id="go">Grade it</button>
    <button id="sample" class="ghost">Load sample</button>
    <span id="status"></span>
  </div>

  <div id="out"></div>
</div>

<script>
const $ = id => document.getElementById(id);

const SAMPLES = [];  // filled from /samples at load if present

async function loadSamples() {
  try {
    const r = await fetch('/api/samples');
    if (r.ok) SAMPLES.push(...await r.json());
  } catch {}
}
loadSamples();
let sampleIdx = 0;
$('sample').onclick = () => {
  if (!SAMPLES.length) { $('status').textContent = 'No samples found.'; return; }
  const s = SAMPLES[sampleIdx % SAMPLES.length]; sampleIdx++;
  $('rubric').value = s.rubric; $('assignment').value = s.assignment;
  $('status').textContent = 'Sample: ' + s.name;
};

function confClass(c) { return c >= 0.75 ? 'g' : c >= 0.5 ? 'm' : 'b'; }

$('go').onclick = async () => {
  const body = { assignment: $('assignment').value, rubric: $('rubric').value };
  $('go').disabled = true;
  $('out').innerHTML = '';
  $('status').textContent = 'Grading locally — this can take a minute…';
  const t0 = Date.now();
  try {
    const r = await fetch('/api/grade', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || ('HTTP ' + r.status));
    const secs = ((Date.now() - t0) / 1000).toFixed(0);
    $('status').textContent = `Done in ${secs}s · ${d.model} · ${d.detected_language}`;
    const crits = (d.criteria || []).map(c => `
      <div class="crit">
        <b><span>${esc(c.name)}</span><span>${c.score} / ${c.max_score}</span></b>
        <p>${esc(c.feedback)}</p>
      </div>`).join('');
    $('out').innerHTML = `
      <div class="card">
        <div class="score-line">
          <span class="big">${d.overall_score} / ${d.overall_max}</span>
          <span class="conf ${confClass(d.confidence)}">confidence ${(d.confidence * 100).toFixed(0)}%</span>
        </div>
        <p>${esc(d.general_feedback)}</p>
        ${crits}
        <p class="conf">Draft grade — a teacher reviews and approves before anything reaches the student.</p>
      </div>`;
  } catch (e) {
    $('status').textContent = '';
    $('out').innerHTML = `<div class="card error">${esc(e.message)}</div>`;
  } finally {
    $('go').disabled = false;
  }
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>"""


@app.get("/api/samples")
async def samples():
    out = []
    base = os.path.join(os.path.dirname(__file__), "samples")
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if not name.endswith(".json"):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out.append({"name": data.get("name", name), "rubric": data["rubric"], "assignment": data["assignment"]})
    return out


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
