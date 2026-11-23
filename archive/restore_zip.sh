#!/usr/bin/env bash
# Rebuild the datasets and raw recordings from archive/*.zip.part*, and check every file's SHA-256.
#   archive/restore_zip.sh [destination=repo root]
#   e.g. archive/restore_zip.sh                -> data/lerobot/..., data/episodes/..., data/rubiks_rig_logs/... in this repo
#        archive/restore_zip.sh /tmp/restore   -> under /tmp/restore
# By hand, for one item:  cat archive/screw_from_rest.zip.part* > screw_from_rest.zip && unzip screw_from_rest.zip
# (the rig logs were at robotics-rubiks-cube/data/real/logs on the rig PC; they restore to data/rubiks_rig_logs/)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
dest="$(realpath -m "${1:-.}")"; mkdir -p "$dest"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/yamzip.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
for first in archive/*.zip.part000; do
  name=$(basename "$first" .zip.part000)
  cat $(ls archive/"$name".zip.part* | sort) > "$tmp/$name.zip"
  unzip -q -o "$tmp/$name.zip" -d "$dest"
  rm "$tmp/$name.zip"
  echo "unzipped $name" >&2
done
echo "checking $(wc -l < archive/MANIFEST.sha256) files against archive/MANIFEST.sha256 ..." >&2
(cd "$dest" && sha256sum --quiet -c "$REPO/archive/MANIFEST.sha256") && echo "ALL FILES MATCH" >&2
