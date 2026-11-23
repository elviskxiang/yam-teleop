#!/usr/bin/env bash
# Zip all datasets and raw recordings into git-sized pieces (GitHub rejects files over 100 MB).
#   archive/make_zip.sh
# One plain zip per item below (each under 4 GB, so no ZIP64 is needed), each cut into 95 MB pieces with `split`:
#   archive/<item>.zip.part000, .part001, ...      join with:  cat <item>.zip.part* > <item>.zip
# (zip's own split format, `zip -s`, was tried first and produced an archive that did not restore: Info-ZIP 3.0 /
# unzip 6.0 mishandle split archives over 4 GB.)
# Also writes archive/MANIFEST.sha256 ("<sha256>  <path>" for every original file, paths as restored).
# Restore: archive/restore_zip.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
RUBIKS="${RUBIKS:-$HOME/Desktop/experiments/rubiks_cube/robotics-rubiks-cube}"
# item name | directory to zip (relative to BASE) | BASE   -> restored at <destination>/<directory>
ITEMS=(
  "screw_from_rest|data/lerobot/screw_from_rest|$REPO"
  "backup_screw_from_rest_4cam|data/lerobot/_backup_screw_from_rest_4cam|$REPO"
  "episodes_right_arm_20261122|data/episodes/right_arm_20261122|$REPO"
  "episodes_pre_fix|data/episodes/_pre_fix|$REPO"
  "rig_log_20261122_154451|data/rubiks_rig_logs/rig_20261122_154451.mcap|STAGE"
  "rig_log_20261122_155452|data/rubiks_rig_logs/rig_20261122_155452.mcap|STAGE"
)
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/data/rubiks_rig_logs"
for f in "$RUBIKS"/data/real/logs/*.mcap; do ln -s "$f" "$stage/data/rubiks_rig_logs/$(basename "$f")"; done
rm -f archive/*.zip archive/*.zip.part* archive/*.z[0-9][0-9]
: > archive/MANIFEST.sha256
for it in "${ITEMS[@]}"; do
  IFS='|' read -r name path base <<< "$it"; [ "$base" = STAGE ] && base=$stage
  (cd "$base" && find -L "$path" -type f -print0 | sort -z | xargs -0 sha256sum) >> archive/MANIFEST.sha256
  (cd "$base" && zip -q -r -1 "$REPO/archive/$name.zip" "$path")
  split -b 95M -d -a 3 "archive/$name.zip" "archive/$name.zip.part"
  echo "$name: $(stat -c %s "archive/$name.zip" | awk '{printf "%.0f MB", $1/1e6}'), $(ls archive/$name.zip.part* | wc -l) pieces" >&2
  rm "archive/$name.zip"
done
echo "$(wc -l < archive/MANIFEST.sha256) files in archive/MANIFEST.sha256" >&2
