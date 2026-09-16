#!/bin/bash
# Upload BilimAI datasets to HuggingFace (private), verify, then delete local.
# Deletion happens ONLY when remote file-count AND byte-total match local exactly.
set -uo pipefail

ROOT="$HOME/Desktop/My Projects/BilimAI"
HFPY=/opt/homebrew/Cellar/hf/1.29.0/libexec/bin/python
LOG="$ROOT/.hfmigrate/migrate.log"
USER=Jahongir713

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

local_stats(){ # dir -> "count bytes"
  find "$1" -type f ! -name '.DS_Store' -print0 \
    | xargs -0 stat -f%z 2>/dev/null \
    | awk '{c++; s+=$1} END{printf "%d %d\n", c+0, s+0}'
}

remote_stats(){ # repo -> "count bytes"
  "$HFPY" - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    i = HfApi().repo_info(sys.argv[1], repo_type='dataset', files_metadata=True)
    fs = [f for f in i.siblings
          if f.rfilename != '.gitattributes'
          and not f.rfilename.endswith('.DS_Store')]
    print(len(fs), sum(f.size or 0 for f in fs))
except Exception as e:
    print("ERR", e, file=sys.stderr); print("-1 -1")
PY
}

migrate(){
  local path="$1" repo="$2"
  local full="$ROOT/$path"
  [ -d "$full" ] || { log "SKIP $path (missing)"; return 0; }

  read -r lc lb < <(local_stats "$full")
  log "START $path -> $USER/$repo  (local: $lc files, $lb bytes)"

  if ! hf upload "$repo" "$full" . --repo-type dataset --private \
        --exclude "**/.DS_Store" --exclude ".DS_Store" >>"$LOG" 2>&1; then
    log "FAIL upload $path — local copy KEPT"; return 1
  fi

  read -r rc rb < <(remote_stats "$USER/$repo")
  log "VERIFY $repo  remote: $rc files, $rb bytes"

  if [ "$lc" = "$rc" ] && [ "$lb" = "$rb" ]; then
    rm -rf "$full"
    log "OK $path verified + local deleted (freed $lb bytes)"
  else
    log "MISMATCH $path (local $lc/$lb vs remote $rc/$rb) — local copy KEPT"
    return 1
  fi
}

log "=== migration start ==="
migrate "data/raw/hme100k"            "bilimai-hme100k"
migrate "data/raw/my_notebook"        "bilimai-my-notebook"
migrate "data/raw/school_notebooks_RU" "bilimai-school-notebooks-ru"
migrate "data/raw/uzbek_news"         "bilimai-uzbek-news"
migrate "data/derived/higan_ru_bank"  "bilimai-higan-ru-bank"
migrate "data/derived/higan_synth_ru_v2" "bilimai-higan-synth-ru-v2"
migrate "out/ms_stage"                "bilimai-ms-stage"
log "=== migration done ==="
df -h /System/Volumes/Data | tail -1 | tee -a "$LOG"
