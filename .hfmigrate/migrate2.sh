#!/bin/bash
# Phase 2. HuggingFace rejects any directory holding >10000 files, so datasets
# made of many small files are tarred into a single artifact before upload.
# Nothing local is deleted unless the remote copy is byte-verified first.
set -uo pipefail

ROOT="$HOME/Desktop/My Projects/BilimAI"
HFPY=/opt/homebrew/Cellar/hf/1.29.0/libexec/bin/python
LOG="$ROOT/.hfmigrate/migrate.log"
USER=Jahongir713

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

remote_bytes(){ # repo -> total bytes of real files
  "$HFPY" - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    i = HfApi().repo_info(sys.argv[1], repo_type='dataset', files_metadata=True)
    fs=[f for f in i.siblings if f.rfilename!='.gitattributes'
        and not f.rfilename.endswith('.DS_Store')]
    print(sum(f.size or 0 for f in fs))
except Exception as e:
    print("ERR",e,file=sys.stderr); print(-1)
PY
}

# --- 1. ms_stage: already single tars, upload directly -----------------------
upload_plain(){
  local path="$1"
  local repo="$2"
  local full="$ROOT/$path"
  [ -d "$full" ] || { log "SKIP $path (missing)"; return 0; }
  local lb; lb=$(find "$full" -type f ! -name '.DS_Store' -print0 \
                  | xargs -0 stat -f%z | awk '{s+=$1} END{print s+0}')
  log "START $path -> $USER/$repo ($lb bytes)"
  hf upload "$repo" "$full" . --repo-type dataset --private \
      --exclude "**/.DS_Store" --exclude ".DS_Store" >>"$LOG" 2>&1 \
      || { log "FAIL upload $path — local KEPT"; return 1; }
  local rb; rb=$(remote_bytes "$USER/$repo")
  if [ "$lb" = "$rb" ]; then rm -rf "$full"; log "OK $path verified + deleted ($lb bytes freed)"
  else log "MISMATCH $path local=$lb remote=$rb — local KEPT"; return 1; fi
}

# --- 2. many-small-files: tar, verify entry count, upload, verify, delete ----
upload_tarred(){
  local path="$1"
  local repo="$2"
  local full="$ROOT/$path"
  [ -d "$full" ] || { log "SKIP $path (missing)"; return 0; }
  local name; name=$(basename "$path")
  local tar="$ROOT/.hfmigrate/$name.tar"

  # Wipe any partial upload from the failed loose-file attempts, so the
  # remote byte-total is the tar alone and verification is meaningful.
  hf repo delete "$USER/$repo" --repo-type dataset -y --missing-ok >>"$LOG" 2>&1

  local nfiles; nfiles=$(find "$full" -type f ! -name '.DS_Store' | wc -l | tr -d ' ')
  log "START $path -> $USER/$repo (tarring $nfiles files)"

  tar --exclude '.DS_Store' -cf "$tar" -C "$(dirname "$full")" "$name" \
      || { log "FAIL tar $path — local KEPT"; rm -f "$tar"; return 1; }

  # Assert the archive's SHAPE, not just that it exists.
  local ntar; ntar=$(tar -tf "$tar" | grep -vc '/$')
  if [ "$nfiles" != "$ntar" ]; then
    log "FAIL tar shape $path: $nfiles source files vs $ntar in archive — local KEPT"
    rm -f "$tar"; return 1
  fi
  local tb; tb=$(stat -f%z "$tar")
  log "  tar ok: $ntar entries, $tb bytes"

  hf upload "$repo" "$tar" "$name.tar" --repo-type dataset --private >>"$LOG" 2>&1 \
      || { log "FAIL upload $path — local KEPT"; rm -f "$tar"; return 1; }

  local rb; rb=$(remote_bytes "$USER/$repo")
  if [ "$tb" = "$rb" ]; then
    rm -rf "$full"; rm -f "$tar"
    log "OK $path verified + deleted (tar $tb bytes on Hub)"
  else
    log "MISMATCH $path tar=$tb remote=$rb — local KEPT"; rm -f "$tar"; return 1
  fi
}

log "=== phase 2 start ==="
upload_plain  "out/ms_stage"              "bilimai-ms-stage"
upload_tarred "data/derived/higan_synth_ru_v2" "bilimai-higan-synth-ru-v2"
upload_tarred "data/raw/hme100k"          "bilimai-hme100k"
upload_tarred "data/raw/uzbek_news"       "bilimai-uzbek-news"
log "=== phase 2 done ==="
df -h /System/Volumes/Data | tail -1 | tee -a "$LOG"
