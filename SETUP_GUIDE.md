# TB Detection and 3D Lung Visualization System

## Requirements

### Software

* Python 3.10 or higher
* Git
* pip (Python Package Manager)

### Hardware

Minimum:

* 8 GB RAM
* Dual-Core Processor

Recommended:

* 16 GB RAM
* Dedicated GPU (Optional)

---

## Project Files Required

The following files must be present in the project directory:

### Trained Models

* tb_model.pth
* lung_unet.pth

### 3D Lung Models

Inside the `lung_model` folder:

* left_lung.stl
* right_lung.stl
* left_lung_tree.stl
* right_lung_tree.stl
* lungs.stl

### Templates

Inside the `templates` folder:

* index.html
* result.html

---

## Installation

### Clone Repository

```bash
git clone <repository-link>
cd Tb_Detection_Project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Input

Supported Formats:

* PNG
* JPG
* JPEG

Upload a chest X-ray image through the web interface.

---

## Output Generated

The system provides:

* TB Prediction
* Confidence Score
* Heatmap Visualization
* TB Hotspot Localization
* Coordinate Mapping to 3D Lung Model
* Interactive 3D Lung Visualization
* Patient Report Generation

---

## Workflow

1. Upload Chest X-ray.
2. Run TB Detection.
3. Generate Heatmap.
4. Extract TB Hotspot Coordinates.
5. Map Coordinates to 3D Lung Model.
6. Visualize Lesion in 3D.
7. Generate Report.

---

## Troubleshooting

If model loading fails:

Verify:

* tb_model.pth exists
* lung_unet.pth exists

If 3D viewer fails:

Verify STL files exist inside the lung_model folder.

If Flask application fails:

Run:

```bash
pip install -r requirements.txt
```

again to install all dependencies.
