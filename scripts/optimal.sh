#!/usr/bin/env bash
set -euo pipefail

OUT="./scripts/truncimal.csv";
echo "d1,d2,threshold,terms,tvd,speedup" > "$OUT";

for d1 in $(seq 2 20); do
  for d2 in $(seq 2 20); do
    if [ $d1 -lt 5 ] || [ $d2 -lt 5 ]; then
      reps=12;
    else
      reps=4;
    fi

    for start in $(seq 1 4 $reps); do
      end=$(( start + 3 ))
      if [ "$end" -gt "$reps" ]; then end="$reps"; fi
      echo "Running for d1=$d1, d2=$d2, reps ${start}-${end}";
      for r in $(seq "$start" "$end"); do
        echo "  Launching rep=$r";
        python small.py --d "$d1" "$d2" --t 1e-4 >> "$OUT" &
      done
      wait
    done
  done
done

echo "Saved results to $OUT";