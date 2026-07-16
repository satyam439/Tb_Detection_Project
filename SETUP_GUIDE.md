# AI TB Analysis Platform — Setup & Run Guide

**For:** Project Manager / Non-Technical User  
**Project:** 3D Lung TB Visualisation System  
**Platform:** macOS or Windows  

---

## What This Tool Does

This tool takes a chest X-ray, runs an AI model to detect TB (Tuberculosis), and displays a **3D rotating lung model** showing exactly where the infection is located — including depth, lobe, and distance from key anatomical landmarks.

---

## Step 1 — Install Python

You need **Python 3.10 or higher**.

### macOS
1. Open **Terminal** (press `Cmd + Space`, type `Terminal`, press Enter)
2. Check if Python is already installed:
   ```
   python3 --version
   ```
3. If not installed, download from: **https://www.python.org/downloads/**
4. Run the installer and follow the prompts

### Windows
1. Open **Command Prompt** (press `Win + R`, type `cmd`, press Enter)
2. Check if Python is already installed:
   ```
   python --version
   ```
3. If not installed, download from: **https://www.python.org/downloads/**
4. ✅ During install — check the box **"Add Python to PATH"** before clicking Install

---

## Step 2 — Get the Project Files

Ask the developer to share the project folder called `Tb_Detection_Project`.  
It should contain these folders and files:

```
Tb_Detection_Project/
├── lung_model/
│   ├── left_lung_shell.stl
│   ├── right_lung_shell.stl
│   ├── left_lung_tree.stl
│   └── right_lung_tree.stl
├── outputs/
│   ├── tb_center.npy        ← generated after running prediction
│   └── gradcam.npy          ← generated after running prediction
├── tb_portal_viewer.py      ← the 3D viewer (main file)
├── predict_tb.py            ← runs AI on a new X-ray
└── requirements.txt         ← list of Python packages needed
```

Place the folder somewhere easy to find, e.g.:
- macOS: `/Users/YourName/Desktop/Tb_Detection_Project`
- Windows: `C:\Users\YourName\Desktop\Tb_Detection_Project`

---

## Step 3 — Install Required Packages

Do this **once only**.

### macOS — Terminal
```bash
cd ~/Desktop/Tb_Detection_Project
pip3 install -r requirements.txt
```

### Windows — Command Prompt
```
cd C:\Users\YourName\Desktop\Tb_Detection_Project
pip install -r requirements.txt
```

This installs: `pyvista`, `numpy`, `scipy`, `torch`, `pyttsx3` and other dependencies.  
It may take 3–5 minutes. Wait for it to finish.

> ⚠️ If you see a red error about `vtk` or `pyvista`, run:
> ```
> pip install pyvista[all]
> ```

---

## Step 4 — Run the AI Prediction on a New X-Ray

Before viewing the 3D model, you must run the prediction on an X-ray image.

### macOS
```bash
cd ~/Desktop/Tb_Detection_Project
python3 predict_tb.py --image /path/to/xray.jpg
```

### Windows
```
cd C:\Users\YourName\Desktop\Tb_Detection_Project
python predict_tb.py --image C:\path\to\xray.jpg
```

Replace `/path/to/xray.jpg` with the actual path to the X-ray file.  
Supported formats: `.jpg`, `.png`, `.jpeg`

✅ When done, you will see:
```
[INFO] TB Positive — hotspot saved to outputs/tb_center.npy
[INFO] GradCAM saved to outputs/gradcam.npy
```

---

## Step 5 — Launch the 3D Viewer

### macOS
```bash
python3 tb_portal_viewer.py
```

### Windows
```
python tb_portal_viewer.py
```

A window will open showing the **3D lung model** with the TB lesion highlighted.

---

## Step 6 — Using the 3D Viewer

| Key | Action |
|-----|--------|
| **F** | Front view (resets camera) |
| **B** | Back view |
| **L** | Left side view |
| **R** | Right side view |
| **Y** | Start 360° auto-rotation |
| **Space** | Stop rotation |
| **V** | Record rotation video (saves to `outputs/lung_rotation.mp4`) |
| **Mouse drag** | Rotate manually |
| **Scroll wheel** | Zoom in/out |

### What you will see on screen

- **Teal/blue translucent shape** — the lung
- **Red/yellow hotspot** — the TB infected area
- **Blue ring** — surrounding consolidation zone
- **White structures** — bronchi (airways)
- **Top-right panel** — full diagnostic report including lobe, depth, coordinates

### Terminal output after 360° rotation (press Y)

After the rotation completes, a full **DEPTH SUMMARY** is printed:
```
  +======================================================+
  |                    DEPTH SUMMARY                     |
  +======================================================+
  | Lung                       : LEFT                    |
  | Lobe                       : LEFT LOWER LOBE         |
  | Distance from carina       :   47.8 mm               |
  | Depth from ANTERIOR        :   32.2 mm  (22% into)  |
  | Lung total A-P depth       :  147.0 mm               |
  +======================================================+
```

---

## Running a New Patient X-Ray

For each new patient, simply repeat **Steps 4 and 5**:

1. Run `predict_tb.py` with the new X-ray path
2. Launch `tb_portal_viewer.py`

The viewer automatically loads the latest `outputs/tb_center.npy` and `outputs/gradcam.npy` — no code changes needed.

---

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `python: command not found` | Use `python3` instead of `python` on macOS |
| `No module named pyvista` | Run `pip3 install pyvista[all]` |
| `No lung STL files found` | Make sure `lung_model/` folder has the `.stl` files |
| Window opens but is black | Wait 5 seconds — it loads the 3D mesh on startup |
| Audio voice not working | Install `pyttsx3`: `pip3 install pyttsx3` |
| Video not saving | Make sure the `outputs/` folder exists |

---

## System Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| **OS** | macOS 12 / Windows 10 | macOS 13+ / Windows 11 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Not required | Any GPU speeds up AI prediction |
| **Python** | 3.10 | 3.11 or 3.12 |
| **Disk space** | 2 GB free | 5 GB free |

---
