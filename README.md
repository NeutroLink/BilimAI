# BilimAI

AI-powered assignment grading assistant. Reads students' written work, checks it,
drafts grades and feedback — a teacher reviews and approves. Runs on a custom,
fully local AI (no cloud LLM APIs). Languages: **Russian and Uzbek**.

- [PLAN.md](PLAN.md) — full tech stack, architecture, phased roadmap, risk register.
- [mvp/](mvp/) — single-file demo of the grading stage (typed text only).

## Run the MVP

```bash
# 1. Ollama serving + model
ollama serve &
ollama pull qwen3:8b

# 2. App
cd mvp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8000
```

Open http://localhost:8000 — "Load sample" cycles a Russian and an Uzbek essay.

Model is configurable: `BILIMAI_MODEL=qwen3:4b .venv/bin/uvicorn app:app --port 8000`.
