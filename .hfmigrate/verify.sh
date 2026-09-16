#!/bin/bash
# End-to-end verification of the HuggingFace copies.
#
# The LFS sha256 stored on the Hub was computed CLIENT-SIDE from the original
# local file at upload time. So re-hashing a downloaded file and matching that
# value proves the Hub copy is byte-identical to what left this machine - even
# though the local originals no longer exist to diff against.
set -uo pipefail

ROOT="$HOME/Desktop/My Projects/BilimAI"
HFPY=/opt/homebrew/Cellar/hf/1.29.0/libexec/bin/python
LOG="$ROOT/.hfmigrate/verify.log"
WORK=/private/tmp/hfverify
mkdir -p "$WORK"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# expected tar entry counts, recorded before upload.
# macOS ships bash 3.2 - no associative arrays - so this is a case statement.
expected_entries(){
  case "$1" in
    higan_synth_ru_v2.tar) echo 396002 ;;
    hme100k.tar)           echo 99115  ;;
    uzbek_news.tar)        echo 512750 ;;
    *)                     echo ""     ;;
  esac
}

REPOS="bilimai-higan-synth-ru-v2 bilimai-hme100k bilimai-uzbek-news bilimai-ms-stage"

log "=== verification start ==="
for r in $REPOS; do
  d="$WORK/$r"
  log "DOWNLOAD $r"
  rm -rf "$d"
  if ! hf download "Jahongir713/$r" --repo-type dataset --local-dir "$d" >/dev/null 2>&1; then
    log "  FAIL download $r"; continue
  fi
  rm -rf "$d/.cache"

  # hash every file and compare to the checksum the Hub recorded
  "$HFPY" - "$r" "$d" >>"$LOG" 2>&1 <<'PY'
import sys, hashlib, os, zlib
from huggingface_hub import HfApi
repo, root = sys.argv[1], sys.argv[2]
i = HfApi().repo_info(f"Jahongir713/{repo}", repo_type='dataset', files_metadata=True)
ok=bad=miss=0; badnames=[]
for f in i.siblings:
    if f.rfilename == '.gitattributes': continue
    p = os.path.join(root, f.rfilename)
    if not os.path.exists(p): miss+=1; badnames.append("MISSING "+f.rfilename); continue
    lfs = getattr(f,'lfs',None)
    if lfs and lfs.get('sha256'):
        h=hashlib.sha256()
        with open(p,'rb') as fh:
            for c in iter(lambda: fh.read(1<<20), b''): h.update(c)
        good = h.hexdigest()==lfs['sha256']
    else:
        # plain git blob -> sha1("blob <len>\0" + bytes)
        data=open(p,'rb').read()
        h=hashlib.sha1(b"blob %d\0"%len(data)+data)
        good = h.hexdigest()==f.blob_id
    if good: ok+=1
    else: bad+=1; badnames.append("HASH MISMATCH "+f.rfilename)
print(f"  {repo}: {ok} verified, {bad} mismatched, {miss} missing")
for n in badnames[:5]: print("    "+n)
PY

  # tar archives: confirm readable and complete
  for t in "$d"/*.tar; do
    [ -e "$t" ] || continue
    b=$(basename "$t"); exp=$(expected_entries "$b")
    n=$(tar -tf "$t" 2>/dev/null | grep -vc '/$')
    if [ -n "$exp" ]; then
      [ "$n" = "$exp" ] && log "  TAR OK $b: $n entries (expected $exp)" \
                        || log "  TAR MISMATCH $b: $n entries, expected $exp"
    else
      log "  TAR readable $b: $n entries"
    fi
  done
  rm -rf "$d"
done
log "=== verification done ==="
