#!/usr/bin/env bash
# Commit and push the zip volumes in batches of <= 1.4 GB (GitHub refuses a single push over 2 GB).
#   archive/commit_batches.sh        (from anywhere, after archive/make_zip.sh)
# Restartable: volumes already committed are skipped; each batch is pushed before the next is staged.
# The first batch also carries the scripts, the manifest and docs/HANDOFF.md.
set -euo pipefail
cd "$(dirname "$0")/.."
LIMIT=$((1400 * 1000 * 1000))
git add archive/make_zip.sh archive/restore_zip.sh archive/commit_batches.sh archive/MANIFEST.sha256 docs/HANDOFF.md .gitignore
mapfile -t vols < <(git ls-files --others --exclude-standard archive | grep -E '\.zip\.part[0-9]+$' | sort -V)
total=${#vols[@]}
echo "$(date +%T) ${total} volumes to commit" >&2
batch=(); size=0; n=0
flush() {
  [ ${#batch[@]} -eq 0 ] && git diff --cached --quiet && return 0
  n=$((n + 1))
  [ ${#batch[@]} -gt 0 ] && git add -- "${batch[@]}"
  mb=$((size / 1000000))
  first=$(basename "${batch[0]:-none}"); last=$(basename "${batch[-1]:-none}")
  git commit -q -m "Data archive, part $n: ${#batch[@]} zip volumes ($first .. $last), $mb MB

All datasets and raw recordings of Nov 22 in one zip split into 95 MB volumes (GitHub's per-file limit is
100 MB). Restore with archive/restore_zip.sh (joins the volumes, unzips, checks archive/MANIFEST.sha256).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  for try in 1 2 3 4 5; do
    if git push -q 2>&1; then break; fi
    echo "$(date +%T) push failed (try $try), retrying in 60 s" >&2; sleep 60
    [ "$try" -eq 5 ] && { echo "giving up; rerun archive/commit_batches.sh" >&2; exit 1; }
  done
  echo "$(date +%T) pushed part $n: ${#batch[@]} volumes, $mb MB" >&2
  batch=(); size=0
}
for f in "${vols[@]}"; do
  s=$(stat -c %s "$f")
  if [ $((size + s)) -gt "$LIMIT" ]; then flush; fi
  batch+=("$f"); size=$((size + s))
done
flush
echo "$(date +%T) DONE: $n commits pushed" >&2
