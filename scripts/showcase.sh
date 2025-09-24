#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CSV="${1:-data/my_sample.csv}"

clear
echo "=== Train + Export on $CSV ==="
python scripts/poc.py --csv "$CSV"

sleep 0.5
clear
echo "=== Stream: High-Risk Example (row 0) ==="
python scripts/stream_demo.py --csv "$CSV" --row 0 --delay 0.3

sleep 0.5
echo
echo "=== Stream: Low-Risk Example (row 1) ==="
python scripts/stream_demo.py --csv "$CSV" --row 1 --delay 0.3

echo
OUT="data/predictions_${CSV##*/}"
echo "Opening predictions: $OUT"
open "$OUT" 2>/dev/null || true