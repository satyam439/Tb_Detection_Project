# TB Detection and 3D Lung Visualization System

## Overview

This project is an AI-assisted Tuberculosis (TB) Detection and 3D Visualization System developed using chest X-ray images.

The system combines deep learning-based TB classification, lesion localization, heatmap generation, and an **interactive, anatomically-labeled 3D lung model** with a **locked (fixed) lesion coordinate marker**, **spoken audio alerts**, and **exportable verification diagrams**. It is designed to assist in identifying potential TB-infected regions and visualizing their exact anatomical location within a 3D lung model — without the lesion marker drifting during rotation.

---

## Problem Statement

Chest X-rays are commonly used for Tuberculosis screening. While deep learning models can classify TB-positive and healthy lungs with high accuracy, they often lack interpretability and spatial visualization.

This project addresses that limitation by:

* Detecting TB from chest X-rays.
* Localizing suspicious TB regions.
* Extracting lesion hotspot coordinates.
* Mapping lesion coordinates onto a 3D anatomical lung model.
* **Locking** the lesion marker to a fixed real-world coordinate so it stays anatomically accurate through any rotation.
* Providing interactive, labeled 3D visualization of the suspected lesion location, with audio narration and exportable verification images.

---

## Key Features

### TB Classification

* DenseNet121-based deep learning classifier.
* Binary classification:
  * Healthy
  * Tuberculosis (TB)

### Lung Segmentation

* OpenCV-based lung segmentation (with a U-Net fallback path).
* Isolates lung regions for focused analysis.

### Heatmap Generation

* GradCAM++-based visual attention maps highlighting suspicious TB regions.
* Improves explainability of model predictions.
* Rendered as a true 3D volumetric "heat" scalar on the lung mesh (JET colormap, gamma-adjusted opacity curve) — not just a flat 2D overlay.

### TB Hotspot Localization

* Identifies the most probable lesion region from the 2D heatmap.
* Extracts pixel-space hotspot coordinates `(cx, cy)`.

### Coordinate Mapping — Locked / Fixed 3D Coordinates

This is the core upgrade in the latest version:

* The 2D pixel hotspot is back-projected into millimeter-accurate 3D space `(x, y, z)` using the lung mesh's pre-rotation bounds.
* These coordinates are **snapshotted once at load time** (`INITIAL_LESION_3D`, `INITIAL_LUNG_BOUNDS`) and **never recalculated mid-rotation**.
* A dedicated **fixed lesion marker** (red sphere + white XYZ axis cross + always-visible label showing the exact `(x, y, z) mm` coordinate) is drawn separately from the rotating lung mesh, so the marker's world position stays locked and accurate at every viewing angle — it does not drift, jump, or desync the way an in-mesh scalar marker would.
* Depth is computed once from frozen anatomical bounds:
  * Depth from anterior surface (mm and % into lung)
  * Depth from posterior surface
  * Total anterior-posterior lung depth
  * Distance from carina (tracheal bifurcation / rotation pivot)
* These frozen values are printed to the terminal at 90°, 180°, 270°, and 360° rotation milestones, and appear identically in the on-screen overlay — terminal and viewer numbers always match.

### New 3D Model with Full Anatomical Labels

* Dual STL lung shells (left/right), individually scaled to **average Indian adult lung anatomy**:
  * Right lung: 95 × 220 × 130 mm (3 lobes)
  * Left lung: 85 × 210 × 120 mm (2 lobes, cardiac notch)
* Three-pass translucent shell rendering (deep tissue tone → mid layer → soft specular highlight) for a realistic, semi-transparent look.
* Labeled leader-line callouts for:
  * Left/Right apex (with live coordinates)
  * Left/Right hilum (with live coordinates)
  * Upper / middle / lower lobes
  * Lung bases (diaphragm)
  * Carina
  * A dedicated **TB Infection Site** callout box (lung, lobe, 3D coordinates, pixel coordinates, depth, carina distance) when TB is detected
* Bronchial airway trees are loaded and geometrically validated internally but are hidden from the rendered view (removed white airway clutter from the upper lobes per current design).
* Four-point cinematic light rig plus a dedicated spotlight aimed directly at the lesion.

### Audio Narration

* On launch, if TB is detected, the system **speaks** the finding aloud (e.g. *"TB Infected Area detected. Right lung. Right upper lobe."*) using `pyttsx3` (cross-platform), with `say` (macOS) and `espeak` (Linux) as fallbacks.

### Interactive 3D Viewer Controls

* Mouse: left-drag to rotate, right-drag to pan, **scroll wheel to zoom** (dolly-based, VTK observer bound — smooth zoom in/out without clipping artifacts).
* Keyboard:

| Key | Action |
|---|---|
| `F` | Front view |
| `B` | Back view |
| `L` | Left view |
| `R` | Right view |
| `Y` | Slow 360° Y-axis rotation (with depth milestone logging) |
| `V` | Record 360° rotation to `outputs/lung_rotation.mp4` |
| `Space` | Stop rotation / recording |
| `=` / `+` / numpad `+` | Zoom in |
| `-` / numpad `-` | Zoom out |

* Rotation is around the Y-axis only, pivoting at the combined lung center (approximate carina position) — the fixed lesion marker and all anatomical labels remain correctly anchored throughout.

### Exportable Verification Outputs

Generated automatically every time the viewer script runs, before the interactive window opens:

* **`outputs/coord_verification.png`** — side-by-side 2D GradCAM heatmap (with pixel-space hotspot marker) and a coordinate audit panel showing the back-projected 3D→2D position, pixel error, lung/lobe, depth, and a PASS/CHECK verdict confirming the 2D and 3D coordinates agree.
* **`outputs/lung_annotated_diagram.png`** — a full anatomical 2D lung diagram (apex, hilum, lobes, fissures, trachea, carina, diaphragm) with the TB hotspot overlaid as a multi-ring glow, a coordinate-system legend, and a detailed infection callout.
* **`outputs/lung_rotation.mp4`** — optional 360° rotation video, triggered with the `V` key inside the viewer.

### Automated Report Generation

* Generates patient-specific analysis reports (PDF).
* Includes prediction, confidence score, lesion localization, and visualization outputs.

---

## Novel Contribution

Unlike conventional TB detection systems that only provide a prediction score, this project performs lesion localization by extracting TB hotspot coordinates from chest X-ray heatmaps and projecting them onto a 3D anatomical lung model — then **locks** that 3D coordinate so it is immune to rotation drift, and cross-verifies it against the original 2D pixel location via an automatically generated audit image.

This approach allows precise, reproducible visualization of the approximate spatial location of suspected TB lesions inside the lungs, improving interpretability and clinical understanding of model predictions.

---

## System Workflow

### Step 1: Chest X-ray Input
The user uploads a chest X-ray image via the Flask web interface.

### Step 2: TB Classification
A DenseNet121-based model predicts whether the image is Healthy or TB Positive.

### Step 3: Heatmap Generation
A GradCAM++ lesion attention heatmap is generated highlighting regions contributing to the TB prediction.

### Step 4: TB Hotspot Localization
The system identifies the most probable TB lesion region and extracts pixel-space hotspot coordinates.

### Step 5: Coordinate Mapping & Locking
The extracted coordinates are transformed into 3D millimeter space, mapped onto the standardized anatomical lung model, and **frozen** so they no longer change during rotation.

### Step 6: 3D Lesion Placement
A fixed lesion marker (sphere + axis cross + label) is placed inside the lung model at the locked anatomical location, independent of the rotating mesh.

### Step 7: Interactive Visualization
The user rotates, zooms, and explores the labeled 3D lungs; audio narrates the finding; verification PNGs are auto-generated for cross-checking.

### Step 8: Report Generation
A patient PDF report is generated summarizing prediction, confidence, lesion localization, and visualization outputs.

---

## Project Structure

```text
TB_Detection_Project/
│
├── app.py                          # Flask web app entry point
├── predict_tb.py                   # DenseNet121 TB classification
├── train_tb.py                     # TB classifier training script
├── train_unet.py                   # Lung segmentation model training
├── lung_segmentation.py            # OpenCV-based lung segmentation
├── lung_unet.pth                   # Trained U-Net weights
├── tb_model.pth                    # Trained DenseNet121 weights
│
├── tb_portal_viewer.py             # Main 3D viewer — locked coords, labels, audio, zoom
├── tb_lesion_from_mask.py          # Lesion mask → 3D coordinate extraction
├── anatomical_lung.py              # Anatomical lung geometry helpers
├── save_tb_center.py               # Saves tb_center.npy (pixel hotspot)
├── split_lungs.py                  # Splits combined STL into left/right lungs
├── split_lung_layers.py            # Splits lung shell vs internal tree layers
├── export_3d_viewer.py             # 3D export utilities
├── inspect_lung_stl.py             # STL inspection / debugging tool
├── check_and_fix_lungs.py          # STL bounds/geometry repair
├── fix_gradcam_save.py             # GradCAM output path fixes
├── verify_coords.py                # Standalone coordinate verification script
├── generate_case_series.py         # Batch case-series report generation
├── batch_test.py                   # Batch testing across sample images
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
│   ├── left_lung.stl
│   ├── right_lung.stl
│   ├── left_lung_shell.stl
│   ├── right_lung_shell.stl
│   ├── left_lung_tree.stl
│   ├── right_lung_tree.stl
│   └── *_original_backup.stl       # Pre-edit backups of the STL meshes
│
└── outputs/
    ├── tb_center.npy                    # Saved pixel hotspot (cx, cy)
    ├── gradcam.npy                      # Saved GradCAM++ heatmap array
    ├── coord_verification.png           # 2D/3D coordinate audit image
    ├── lung_annotated_diagram.png       # Annotated 2D anatomical diagram
    ├── lung_rotation.mp4                # 360° rotation video (on request)
    └── tb_report_*.pdf                  # Generated patient reports
```

---

## Technologies Used

### Programming Language
* Python 3.13

### Deep Learning
* PyTorch
* TorchVision

### Image Processing
* OpenCV
* NumPy
* Pillow
* SciPy

### 3D Visualization
* PyVista
* VTK (direct observer bindings for zoom/dolly control)

### Audio
* pyttsx3 (cross-platform TTS), with `say` (macOS) / `espeak` (Linux) fallback

### Reporting & Diagrams
* Matplotlib (coordinate verification + annotated lung diagram)
* ReportLab / FPDF (PDF report generation)

### Web Interface
* Flask
* HTML / CSS

---

## Installation

### Clone Repository

```bash
git clone https://github.com/satyam439/Tb_Detection_Project.git
cd Tb_Detection_Project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

> On macOS, if you want spoken audio narration, `pyttsx3` will use the built-in `say` command automatically — no extra setup needed. On Linux, install `espeak` if `pyttsx3` isn't available: `sudo apt install espeak`.

---

## Running the Application

### 1. Start the Flask web app (classification + upload UI)

```bash
python app.py
```

Open your browser at:

```text
http://127.0.0.1:5000
```

Upload a chest X-ray (PNG/JPG/JPEG). The app will run classification, generate the heatmap and hotspot, and give you the option to launch the 3D viewer for that result.

### 2. Launch the 3D viewer directly (standalone)

If `outputs/tb_center.npy` and `outputs/gradcam.npy` already exist (from a prior prediction), you can open the labeled, locked-coordinate 3D viewer on its own:

```bash
python tb_portal_viewer.py
```

This single command will:
1. Generate `outputs/coord_verification.png`
2. Generate `outputs/lung_annotated_diagram.png`
3. Open the interactive 3D viewer window

Use the mouse/keyboard controls listed above to rotate, zoom, record video, and inspect the locked lesion marker and anatomical labels.

---

## Input

Supported formats:
* PNG
* JPG
* JPEG

Chest X-ray images can be uploaded through the web interface.

---

## Output

The system generates:

* TB Prediction + Confidence Score
* 2D GradCAM++ Heatmap Visualization
* TB Lesion Localization (pixel + locked 3D coordinates)
* Coordinate verification PNG (2D↔3D cross-check)
* Annotated anatomical lung diagram PNG
* Interactive, labeled 3D lung visualization with audio narration
* Optional 360° rotation MP4
* PDF patient report

---

## Models

### TB Detection Model
* Architecture: DenseNet121
* Output Classes: Healthy / TB Positive

### Lung Segmentation
* OpenCV-based segmentation (primary), with a U-Net model available as an alternative path
* Purpose: lung region isolation and lesion localization support

---

## Future Improvements

* Patient-specific anatomical scaling (beyond the current Indian-adult population averages).
* Multi-lesion visualization (currently supports a single locked hotspot per scan).
* Quantitative TB burden estimation.
* Clinical severity scoring.
* DICOM support.
* Volumetric lesion reconstruction from CT scans (MedNeRF / ViT-based reconstruction approaches under evaluation).
* External validation on additional TB datasets.

---

## Disclaimer

This project is a proof-of-concept research and educational pipeline. It is **not** a clinically validated diagnostic system and should not be used as a substitute for professional medical diagnosis or treatment decisions.
