#!/bin/bash
# Fetch WAY-EEG-GAL (Luciw, Jarocka & Edin 2014) from figshare collection 988376.
#
# One archive per participant, ~10.3 GB compressed / ~15 GB expanded. Each archive
# is unzipped and deleted, and a .done marker lets the script resume after an
# interruption without re-downloading.
#
#   scripts/fetch_data.sh [TARGET_DIR]
#
# Default target is $NEUROMOTION_DATA, else /scratch/$USER/data/way-eeg-gal.
set -u

TARGET="${1:-${NEUROMOTION_DATA:-/scratch/$USER/data/way-eeg-gal}}"
mkdir -p "$TARGET" || exit 1
cd "$TARGET" || exit 1

# figshare file ids, participant order 1..12
IDS=(3229301 3229304 3229307 3229310 3229313 3209486 3209501 3209504 3209495 3209492 3209498 3209489)

echo "target: $TARGET"
for i in "${!IDS[@]}"; do
  n="P$((i + 1))"
  id="${IDS[$i]}"
  if [ -f "$n.done" ]; then
    echo "skip $n (already extracted)"
    continue
  fi
  echo "=== $n (figshare file $id) $(date -u +%H:%M:%S) ==="
  if curl -sL --retry 5 --retry-delay 5 --retry-connrefused -C - \
        -o "$n.zip" "https://ndownloader.figshare.com/files/$id" \
     && unzip -q -o "$n.zip"; then
    rm -f "$n.zip"
    touch "$n.done"
    echo "$n ok"
  else
    echo "$n FAILED" >&2
  fi
done

echo "=== summary ==="
ls *.done 2>/dev/null | wc -l | xargs echo "participants extracted:"
du -sh "$TARGET"
echo "ALL DONE"
