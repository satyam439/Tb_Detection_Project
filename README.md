# TB Detection and 3D Lung Visualization System

## Overview

This project is an AI-assisted Tuberculosis (TB) Detection and 3D Visualization System developed using chest X-ray images.

The system combines deep learning-based TB classification, multi-region lesion localization, GradCAM++ heatmap generation, and an **interactive, anatomically-labeled 3D lung model** with **locked (fixed) lesion coordinate markers**, **spoken audio alerts**, and **exportable verification diagrams**. It is designed to assist in identifying potential TB-infected regions and visualizing their approximate anatomical location within a 3D lung model — without the lesion marker drifting during rotation.

---

## Problem Statement

Chest X-rays are commonly used for Tuberculosis screening. While deep learning models can classify TB-positive and healthy lungs with high accuracy, they often lack interpretability and spatial visualization.

This project addresses that limitation by:

* Detecting TB from chest X-rays.
* Localizing suspicious TB regions (supports multiple regions per scan).
* Extracting lesion hotspot coordinates.
* Mapping lesion coordinates onto a 3D anatomical lung model.
* **Locking** each lesion marker to a fixed real-world coordinate so it stays anatomically consistent through any rotation.
* Providing interactive, labeled 3D visualization of the suspected lesion location(s), with audio narration and exportable verification images.

---

## Key Features

### TB Classification

* DenseNet121-based deep learning classifier.
* Binary classification:
  * Healthy
  * Tuberculosis (TB)

### Lung Segmentation

* Pretrained torchxrayvision PSPNet chest segmentation model (`lung_mask_xrv.py`), returning the combined lung mask plus a left/right split in a single pass.
* Isolates lung regions for focused analysis and lobe-wise activation scoring.

### Heatmap Generation

* GradCAM++-based visual attention maps highlighting suspicious TB regions, masked to the segmented lung field.
* For a Healthy prediction, the overlay is deliberately blended at low intensity — a min-max normalized heatmap will always show a "hottest" pixel somewhere even on genuinely healthy scans, so a full-intensity overlay on a negative result would be visually misleading.
* **Anatomical R/L side markers** are burned onto the heatmap image (standard PA chest X-ray convention), so a reviewing clinician can immediately orient the image without ambiguity.

### Lobe-Wise Relative Activation

* GradCAM++ activation, after lung-field masking, is broken down by lobe (5 lobes: right upper/middle/lower, left upper/lower).
* Reported as a **relative share that sums to ~100% across the 5 lobes** — this is a spatial distribution showing *where* the model's attention is concentrated, **not a second confidence score**. It does not need to match the overall Confidence (TB) percentage, and the PDF report explicitly notes this to avoid misreading.

### TB Hotspot Localization

* Identifies probable lesion region(s) from the 2D heatmap (multi-region support — a scan can have more than one flagged area, not just a single dominant hotspot).
* Extracts pixel-space hotspot coordinates `(cx, cy)` for each region.

### Coordinate Mapping — Locked 3D Coordinates, With an Honest Depth Limitation

* Each 2D pixel hotspot is projected into millimeter-space `(x, y, z)` on the lung mesh.
* **Vertical position (Y, apex-to-base)** is derived directly from the heatmap's pixel row — so a hotspot near the top of the X-ray correctly renders near the lung apex in 3D, and vice versa.
* **Depth (Z, anterior-posterior) is intentionally fixed at the lung's mid-depth**, not measured. A single 2D X-ray contains no real depth information — earlier versions of this pipeline derived a depth percentage from vertical pixel position, which was a mislabeled illusion of measurement rather than a real one. This has been corrected: depth is now honestly constant, and only X/Y position is claimed to reflect the actual image.
* Coordinates are **snapshotted once at load time** and never recalculated mid-rotation, so markers stay anchored correctly at every viewing angle.
* Distance from carina (tracheal bifurcation / rotation pivot) is still computed and reported per region.

### 3D Model With Full Anatomical Labels

* Dual STL lung shells (left/right), individually scaled to **average Indian adult lung anatomy**:
  * Right lung: 95 × 220 × 130 mm (3 lobes)
  * Left lung: 85 × 210 × 120 mm (2 lobes, cardiac notch)
* Three-pass translucent shell rendering for a realistic, semi-transparent look.
* Labeled leader-line callouts for apex, hilum, lobes, lung bases, carina, and a dedicated **TB Infection Site** callout box per detected region (lung, lobe, 3D coordinates, pixel coordinates, carina distance).
* Four-point cinematic light rig plus a dedicated spotlight aimed at the primary lesion.
* **This is a fixed anatomical reference model, not a patient-specific 3D reconstruction** — the lung shape itself does not vary per X-ray; only the lesion marker position (X/Y) is patient-derived. This is disclosed directly in the viewer UI so it can't be mistaken for a true volumetric reconstruction.

### Audio Narration

* On launch, if TB is detected, the system speaks the finding aloud using `pyttsx3` (cross-platform), with `say` (macOS) / `espeak` (Linux) as fallbacks.

### Interactive 3D Viewer Controls

* Mouse: left-drag to rotate, right-drag to pan, scroll wheel to zoom.
* Keyboard:

| Key | Action |
|---|---|
| `F` | Front view |
| `B` | Back view |
| `L` | Left view |
| `R` | Right view |
| `Y` | Slow 360° Y-axis rotation |
| `V` | Record 360° rotation to `outputs/lung_rotation.mp4` |
| `Space` | Stop rotation / recording |
| `=` / `+` / numpad `+` | Zoom in |
| `-` / numpad `-` | Zoom out |

### Exportable Verification Outputs

* **`outputs/coord_verification.png`** — 2D GradCAM heatmap with pixel-space hotspot marker, plus a coordinate audit panel confirming 2D↔3D agreement.
* **`outputs/lung_annotated_diagram.png`** — full anatomical 2D lung diagram with TB hotspot overlay and coordinate-system legend.
* **`outputs/lung_rotation.mp4`** — optional 360° rotation video.

### Automated Report Generation

* Patient-specific PDF report: prediction, confidence, lobe-wise relative activation table (clearly labeled as non-confidence), lesion localization, R/L-labeled heatmap image.

### One-Command Workflow

* `./analyze.sh <path_to_xray.png>` runs the full pipeline end-to-end — clears previous outputs, runs prediction, opens the heatmap image and PDF report, then launches the live interactive 3D viewer — so you don't need to run each step manually.

---

## Novel Contribution

Unlike conventional TB detection systems that only provide a prediction score, this project performs multi-region lesion localization by extracting TB hotspot coordinates from chest X-ray heatmaps, projects them onto a 3D anatomical lung model with an honestly-limited (mid-depth-fixed, not measured) Z-axis, and cross-verifies 2D↔3D agreement via an automatically generated audit image.

---

## System Workflow

1. **Input** — a chest X-ray image (via `analyze.sh`, direct CLI, or the Flask web interface).
2. **Classification** — DenseNet121 predicts Healthy or TB Positive.
3. **Heatmap** — GradCAM++ generates a lung-masked attention map; R/L side markers and prediction/risk are burned onto the image.
4. **Lobe-wise breakdown** — activation share computed per lobe (relative distribution, not a confidence score).
5. **Hotspot extraction** — pixel-space coordinates extracted per detected region.
6. **3D coordinate mapping** — vertical position derived from the heatmap; depth fixed at mid-depth (disclosed, not measured).
7. **3D placement** — locked lesion marker(s) placed on the anatomical reference lung model.
8. **Interactive visualization** — rotate/zoom/explore; audio narrates the finding; verification PNGs auto-generated.
9. **Report generation** — PDF summarizing prediction, confidence, lobe distribution, and visualizations.

---

## Project Structure

```text
TB_Detection_Project/
│
├── analyze.sh                      # One-command: predict + heatmap + report + 3D viewer
├── app.py                          # Flask web app entry point
├── predict_tb.py                   # DenseNet121 TB classification + GradCAM + PDF report
├── train_tb.py                     # TB classifier training script
├── train_unet.py                   # Lung segmentation model training (historical/reference)
├── lung_segmentation.py            # Lung mask utilities (lobe masks, CAM masking)
├── lung_mask_xrv.py                # torchxrayvision PSPNet lung segmentation (used at inference)
├── lung_unet.pth                   # U-Net weights (reference/legacy)
├── tb_model.pth                    # Trained DenseNet121 weights
│
├── tb_portal_viewer.py             # Main 3D viewer — locked coords, labels, audio, zoom
├── anatomical_lung.py              # Anatomical lung geometry helpers
├── save_tb_center.py               # Saves tb_center.npy / tb_centers.npy (pixel hotspots)
├── export_3d_viewer.py             # Browser-based 3D viewer export utilities
├── generate_case_series.py         # Batch case-series report generation (15+ patients)
├── batch_test.py                   # Batch accuracy/sensitivity/specificity testing
├── prepare_dataset.py              # Dataset preparation (Shenzhen/Montgomery)
│
├── requirements.txt
├── README.md
├── SETUP_GUIDE.md
│
├── templates/
│   ├── index.html                  # Upload form
│   └── result.html                 # Prediction result + 3D viewer launcher
│
├── lung_model/
│   ├── left_lung.stl / right_lung.stl
│   ├── left_lung_shell.stl / right_lung_shell.stl
│   └── left_lung_tree.stl / right_lung_tree.stl
│
└── outputs/                        # Regenerated fresh on every analyze.sh run
    ├── tb_center.npy / tb_centers.npy
    ├── gradcam.npy
    ├── heatmap_overlay.png
    ├── coord_verification.png
    ├── lung_annotated_diagram.png
    ├── lung_rotation.mp4           # (on request, via 'V' key)
    └── tb_report_*.pdf
```

---

## Technologies Used

**Language:** Python 3.13
**Deep Learning:** PyTorch, TorchVision, torchxrayvision
**Image Processing:** OpenCV, NumPy, Pillow, SciPy
**3D Visualization:** PyVista, VTK
**Audio:** pyttsx3 (with `say` / `espeak` fallback)
**Reporting:** Matplotlib, ReportLab
**Web Interface:** Flask, HTML/CSS

---

## Installation

```bash
git clone https://github.com/satyam439/Tb_Detection_Project.git
cd Tb_Detection_Project
pip install -r requirements.txt
```

> On macOS, `pyttsx3` uses the built-in `say` command automatically. On Linux, install `espeak` if needed: `sudo apt install espeak`.

---

## Running the Application

### Recommended: one command, full pipeline

```bash
./analyze.sh path/to/xray.png
```

Clears previous outputs, runs prediction, opens the heatmap and PDF report, then launches the live interactive 3D viewer.

### Manual step-by-step

```bash
python predict_tb.py path/to/xray.png   # classification + heatmap + PDF
python tb_portal_viewer.py              # live interactive 3D viewer
```

### Web interface

```bash
python app.py
```
Then open `http://127.0.0.1:5000` and upload a chest X-ray (PNG/JPG/JPEG).

---

## Output

* TB Prediction + Confidence Score
* R/L-labeled GradCAM++ heatmap
* Lobe-wise relative activation table (not a confidence score)
* Multi-region lesion localization (pixel + locked 3D coordinates, honest mid-depth Z)
* Coordinate verification PNG (2D↔3D cross-check)
* Annotated anatomical lung diagram PNG
* Interactive, labeled 3D lung visualization with audio narration
* Optional 360° rotation MP4
* PDF patient report

---

## Known Limitations

* **3D lung shape is a fixed anatomical reference model**, not reconstructed from the patient's own X-ray silhouette — only the lesion marker's X/Y position is patient-derived.
* **Depth (Z-axis) is fixed at mid-depth by design**, since a single 2D X-ray contains no real depth information — this was previously mislabeled as a measured percentage and has since been corrected to avoid overclaiming.
* **Lobe boundaries are approximated by vertical position**, not traced fissure anatomy.
* Not validated against CT or an Indian-specific patient population — current validation uses the Shenzhen chest X-ray dataset.
* This is a proof-of-concept research and educational pipeline, **not a clinically validated diagnostic system**.

---

## Future Improvements

* Real localization validation against ground-truth lesion bounding boxes (e.g. TBX11K dataset).
* Clinical severity scoring (requires severity-labeled training data).
* Patient-specific anatomical scaling.
* True volumetric depth from CT-based reconstruction (separate, larger scope than current 2D-derived viewer).
* DICOM support.
* Validation on an Indian-specific patient dataset with clinical correlation.

---

## Disclaimer

This project is a proof-of-concept research and educational pipeline. It is **not** a clinically validated diagnostic system and should not be used as a substitute for professional medical diagnosis or treatment decisions.
