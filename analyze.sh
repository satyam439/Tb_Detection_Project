#!/bin/bash
# One-command TB analysis: prediction + heatmap + PDF report + live 3D viewer
# Always clears previous outputs first, so you only ever see results for
# the CURRENT image — no risk of an old heatmap/report/npy file lingering.
# Usage: ./analyze.sh path/to/xray.png

if [ -z "$1" ]; then
  echo "Usage: ./analyze.sh <path_to_xray_image>"
  exit 1
fi

IMAGE_PATH="$1"

echo "=================================================="
echo "  Clearing previous outputs..."
echo "=================================================="
rm -f outputs/*.png outputs/*.npy outputs/*.pdf outputs/*.html outputs/*.mp4 2>/dev/null

echo "=================================================="
echo "  Running TB analysis on: $IMAGE_PATH"
echo "=================================================="

python3 predict_tb.py "$IMAGE_PATH"

if [ $? -ne 0 ]; then
  echo "[ERROR] predict_tb.py failed — stopping before opening anything."
  exit 1
fi

echo ""
echo "Opening heatmap and report..."

open outputs/heatmap_overlay.png
open outputs/tb_report_*.pdf

echo ""
echo "Launching live interactive 3D window (close it when done)..."
python3 tb_portal_viewer.py
