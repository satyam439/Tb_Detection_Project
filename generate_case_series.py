
# WHAT THIS DOES:
#   - Runs the existing pipeline (predict_tb.run_pipeline) on every image in
#     INPUT_DIR, producing a PDF diagnostic report + interactive 3D viewer +
#     heatmap overlay for each one.
#   - Computes a heuristic "severity proxy" from the GradCAM heatmap (activated
#     lung-area fraction + peak intensity). This is NOT a clinically validated
#     severity score — TB severity grading (mild/moderate/severe, cavitation,
#     extent) requires radiologist-labeled training data our model has never
#     been trained on. The proxy is included so you have *something* to look
#     at and discuss, but every output is labeled accordingly so nobody
#     downstream mistakes it for a validated clinical score.
#   - Builds a single aggregate CSV + summary so you (or your manager) can see
#     all 15+ cases side-by-side: prediction, confidence, lung/lobe, proxy
#     severity, and — if you provide a ground_truth.csv — agreement against
#     known labels.

# WHAT THIS DOES NOT DO (by design, because the data doesn't exist yet):
#   - It does NOT validate the 3D model against real CT scans. If you get CT
#     for these patients later, that's a separate, more involved comparison
#     (real 3D CT reconstruction vs. our schematic anatomical template) — see
#     the note in the README section at the bottom of this file.
#   - It does NOT produce a clinically validated TB severity grade.

# HOW TO USE:
#   1. Put your Indian adult X-ray PNGs in the folder set by INPUT_DIR below
#      (defaults to dataset/val/indian_adult/ — create it and drop images in).
#   2. (Optional) If you have known ground truth, create a CSV at
#      GROUND_TRUTH_CSV with columns: image,true_label[,severity,notes]
#      true_label should be "TB" or "NORMAL". severity is optional free text
#      from a radiologist (e.g. "moderate", "extensive") — used only for
#      side-by-side comparison, not for scoring the model.
#   3. Run:  python3 generate_case_series.py
#   4. Check case_series_output/ for everything.


import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_tb import run_pipeline  # reuses your existing pipeline directly

# ============================================================
# CONFIG — point these at your real Indian-adult X-ray data
# ============================================================

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR        = os.path.join(BASE_DIR, "dataset", "val", "indian_adult")
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "dataset", "val", "indian_adult", "ground_truth.csv")
OUTPUT_ROOT      = os.path.join(BASE_DIR, "case_series_output")
MIN_CASES        = 15  # sir asked for at least 15

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# SEVERITY PROXY (heuristic — see disclaimer in module docstring)
# ============================================================

def compute_severity_proxy(gradcam_path, lung_mask_path):
    
    if not (os.path.exists(gradcam_path) and os.path.exists(lung_mask_path)):
        return {
            "activated_area_pct": None,
            "peak_activation": None,
            "severity_proxy_label": "N/A (healthy or missing data)",
        }

    import cv2
    cam = np.load(gradcam_path).astype(np.float32)
    lung_mask = cv2.imread(lung_mask_path, cv2.IMREAD_GRAYSCALE)
    if lung_mask is None:
        return {
            "activated_area_pct": None,
            "peak_activation": None,
            "severity_proxy_label": "N/A (lung mask missing)",
        }
    lung_mask = cv2.resize(lung_mask, (cam.shape[1], cam.shape[0]))
    lung_bool = lung_mask > 127

    lung_pixels = max(lung_bool.sum(), 1)
    activated = ((cam > 0.5) & lung_bool).sum()
    activated_pct = float(activated / lung_pixels * 100.0)
    peak = float(cam.max())

    
    if activated_pct < 5:
        label = "Proxy: Minimal (heuristic only — not a clinical grade)"
    elif activated_pct < 15:
        label = "Proxy: Limited (heuristic only — not a clinical grade)"
    elif activated_pct < 30:
        label = "Proxy: Moderate (heuristic only — not a clinical grade)"
    else:
        label = "Proxy: Extensive (heuristic only — not a clinical grade)"

    return {
        "activated_area_pct": round(activated_pct, 1),
        "peak_activation": round(peak, 3),
        "severity_proxy_label": label,
    }


# ============================================================
# LOAD GROUND TRUTH (optional)
# ============================================================

def load_ground_truth():
    gt = {}
    if os.path.exists(GROUND_TRUTH_CSV):
        with open(GROUND_TRUTH_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt[row["image"]] = row
    return gt


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input folder not found: {INPUT_DIR}")
        print("        Create it and add your Indian adult X-ray PNGs there,")
        print("        then re-run this script.")
        sys.exit(1)

    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")])

    if len(images) == 0:
        print(f"[ERROR] No .png images found in {INPUT_DIR}")
        sys.exit(1)

    if len(images) < MIN_CASES:
        print(f"[WARN] Only {len(images)} images found — sir asked for at least "
              f"{MIN_CASES}. Proceeding anyway, but flag this gap when you report results.")

    ground_truth = load_ground_truth()
    if ground_truth:
        print(f"[INFO] Loaded ground truth for {len(ground_truth)} images from {GROUND_TRUTH_CSV}")
    else:
        print(f"[INFO] No ground_truth.csv found at {GROUND_TRUTH_CSV} — "
              f"running without known labels (predictions only, no accuracy scoring).")

    print(f"\n{'='*70}")
    print(f"  CASE SERIES GENERATION — {len(images)} X-rays")
    print(f"{'='*70}\n")

    aggregate_rows = []

    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(INPUT_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        patient_id = f"CASE-{i:03d}"

        print(f"[{i}/{len(images)}] Processing {img_name} ...")
        t0 = time.time()

        gt_row = ground_truth.get(img_name, {})
        notes = gt_row.get("notes", "N/A")

        try:
            result = run_pipeline(
                image_path=img_path,
                patient_name=f"Patient {i}",
                patient_id=patient_id,
                gender=gt_row.get("gender", "N/A"),
                age=gt_row.get("age", 0) or 0,
                referred_by="Batch case series",
                notes=notes,
            )
        except Exception as e:
            print(f"  [ERROR] Pipeline failed on {img_name}: {e}")
            continue

        elapsed = time.time() - t0

        # Move this case's outputs into their own subfolder so nothing gets
        # overwritten by the next image in the loop.
        case_dir = os.path.join(OUTPUT_ROOT, f"{patient_id}_{base_name}")
        os.makedirs(case_dir, exist_ok=True)

        for src_key in ["heatmap_path", "lung_mask_path", "viewer_path", "report_path"]:
            src = result.get(src_key)
            if src and os.path.exists(src):
                import shutil
                shutil.copy(src, os.path.join(case_dir, os.path.basename(src)))

        gradcam_src = os.path.join(BASE_DIR, "outputs", "gradcam.npy")
        gradcam_dst = os.path.join(case_dir, "gradcam.npy")
        if os.path.exists(gradcam_src):
            import shutil
            shutil.copy(gradcam_src, gradcam_dst)

        severity = compute_severity_proxy(
            gradcam_path=gradcam_dst if os.path.exists(gradcam_dst) else gradcam_src,
            lung_mask_path=os.path.join(case_dir, "lung_mask.png"),
        )

        true_label = gt_row.get("true_label", "UNKNOWN")
        pred_label_norm = "TB" if result["pred_label"] == "TB" else "NORMAL"
        match = (
            "MATCH" if true_label != "UNKNOWN" and true_label.upper() == pred_label_norm
            else "MISMATCH" if true_label != "UNKNOWN"
            else "N/A (no ground truth)"
        )

        row = {
            "case_id": patient_id,
            "image": img_name,
            "true_label": true_label,
            "predicted_label": result["pred_label"],
            "agreement": match,
            "confidence_tb_pct": result["confidence_tb"],
            "confidence_healthy_pct": result["confidence_healthy"],
            "risk_level": result["risk_level"],
            "lung_coverage_pct": result["lung_coverage_pct"],
            "activated_area_pct": severity["activated_area_pct"],
            "peak_activation": severity["peak_activation"],
            "severity_proxy_label": severity["severity_proxy_label"],
            "radiologist_severity_note": gt_row.get("severity", "N/A"),
            "elapsed_s": round(elapsed, 1),
            "report_path": os.path.relpath(case_dir, OUTPUT_ROOT),
        }
        aggregate_rows.append(row)

        print(f"  → {result['pred_label']} (TB={result['confidence_tb']}%)  "
              f"severity_proxy={severity['severity_proxy_label']}  "
              f"agreement={match}  ({elapsed:.1f}s)")

    # ── AGGREGATE CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_ROOT, "case_series_summary.csv")
    fieldnames = list(aggregate_rows[0].keys()) if aggregate_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"\n{'='*70}")
    print(f"  CASE SERIES COMPLETE")
    print(f"{'='*70}")
    print(f"  Cases processed : {len(aggregate_rows)}")
    print(f"  Output folder   : {OUTPUT_ROOT}/")
    print(f"  Per-case folders: {OUTPUT_ROOT}/CASE-001_<name>/ (PDF + 3D viewer + heatmap)")
    print(f"  Summary CSV     : {csv_path}")

    if ground_truth:
        known = [r for r in aggregate_rows if r["agreement"] in ("MATCH", "MISMATCH")]
        if known:
            matches = sum(1 for r in known if r["agreement"] == "MATCH")
            print(f"  Agreement vs ground truth: {matches}/{len(known)} "
                  f"({matches/len(known)*100:.0f}%)")
    else:
        print(f"  NOTE: no ground_truth.csv found — predictions were generated but")
        print(f"        not scored against known labels. Add one to compare against")
        print(f"        real diagnoses / CT findings once available.")
    print(f"{'='*70}\n")

    print("IMPORTANT — read before sending this upstream:")
    print(" - severity_proxy_label is a heuristic (heatmap area/intensity), NOT a")
    print("   clinically validated severity grade. It has not been checked against")
    print("   radiologist severity staging or CT extent-of-disease findings.")
    print(" - The 3D viewer uses a fixed Indian-adult anatomical reference shape;")
    print("   marker depth (Z) is an estimated lobe midpoint, not a measurement —")
    print("   a single 2D X-ray contains no real depth data to validate against CT.")


if __name__ == "__main__":
    main()