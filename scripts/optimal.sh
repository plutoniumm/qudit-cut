#!/usr/bin/env bash
set -euo pipefail

OUT="./scripts/truncimal.csv"

echo "d1,d2,threshold,terms,tvd,slowdown" > "$OUT"

for d1 in $(seq 2 20); do
  for d2 in $(seq 2 20); do
    echo "Running for d1=$d1, d2=$d2"
    python small.py --d "$d1" "$d2" --t 1e-4 >> "$OUT"
  done
done

echo "Saved results to $OUT"