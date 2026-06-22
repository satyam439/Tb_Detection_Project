# TB Detection and 3D Lung Visualization System

## Overview

This project is an AI-assisted Tuberculosis (TB) Detection and 3D Visualization System developed using chest X-ray images.

The system combines deep learning-based TB classification, lesion localization, heatmap generation, and interactive 3D lung visualization. It is designed to assist in identifying potential TB-infected regions and visualizing their approximate anatomical location within a 3D lung model.

---

## Problem Statement

Chest X-rays are commonly used for Tuberculosis screening. While deep learning models can classify TB-positive and healthy lungs with high accuracy, they often lack interpretability and spatial visualization.

This project addresses that limitation by:

* Detecting TB from chest X-rays.
* Localizing suspicious TB regions.
* Extracting lesion hotspot coordinates.
* Mapping lesion coordinates onto a 3D anatomical lung model.
* Providing interactive 3D visualization of the suspected lesion location.

---

## Key Features

### TB Classification

* DenseNet121-based deep learning classifier.
* Binary classification:

  * Healthy
  * Tuberculosis (TB)

### Lung Segmentation

* U-Net-based lung segmentation model.
* Isolates lung regions for focused analysis.

### Heatmap Generation

* Generates visual attention maps highlighting suspicious TB regions.
* Improves explainability of model predictions.

### TB Hotspot Localization

* Identifies the most probable lesion region.
* Extracts hotspot coordinates from the detected area.

### Coordinate Mapping

* Converts extracted lesion coordinates into positions within a standardized 3D lung model.
* Enables spatial localization of TB lesions.

### Interactive 3D Visualization

* Displays lesion location inside a 3D lung model.
* Supports:

  * Rotation
  * Zooming
  * Free exploration
  * Anatomical lesion inspection

### Automated Report Generation

* Generates patient-specific analysis reports.
* Includes:

  * Prediction
  * Confidence Score
  * Lesion Localization
  * Visualization Outputs

---

## Novel Contribution

Unlike conventional TB detection systems that only provide a prediction score, this project performs lesion localization by extracting TB hotspot coordinates from chest X-ray heatmaps and projecting them onto a 3D anatomical lung model.

This approach allows visualization of the approximate spatial location of suspected TB lesions inside the lungs, improving interpretability and clinical understanding of model predictions.

---

## System Workflow

### Step 1: Chest X-ray Input

The user uploads a chest X-ray image.

### Step 2: TB Classification

A DenseNet121-based model predicts whether the image is:

* Healthy
* TB Positive

### Step 3: Heatmap Generation

A lesion attention heatmap is generated highlighting regions contributing to the TB prediction.

### Step 4: TB Hotspot Localization

The system identifies the most probable TB lesion region and extracts hotspot coordinates.

### Step 5: Coordinate Mapping

The extracted coordinates are transformed and mapped onto a standardized 3D anatomical lung model.

### Step 6: 3D Lesion Placement

A lesion marker is placed inside the lung model at the mapped anatomical location.

### Step 7: Interactive Visualization

The user can rotate, zoom, and explore the lungs in 3D to inspect lesion placement.

### Step 8: Report Generation

A patient report is generated summarizing the analysis results.

---

## Project Structure

```text
TB_Detection_Project/
│
├── app.py
├── predict_tb.py
├── tb_portal_viewer.py
├── tb_lesion_from_mask.py
├── anatomical_lung.py
├── save_tb_center.py
├── split_lungs.py
├── split_lung_layers.py
│
├── tb_model.pth
├── lung_unet.pth
│
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── lung_model/
│   ├── left_lung.stl
│   ├── right_lung.stl
│   ├── left_lung_tree.stl
│   ├── right_lung_tree.stl
│   └── lungs.stl
│
└── outputs/
```

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* TorchVision

### Image Processing

* OpenCV
* NumPy
* Pillow
* SciPy

### Visualization

* PyVista
* VTK
* Plotly

### Web Interface

* Flask
* HTML
* CSS

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

---

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

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

* TB Prediction
* Confidence Score
* Heatmap Visualization
* TB Lesion Localization
* 3D Lung Visualization
* PDF Report

---

## Models

### TB Detection Model

* Architecture: DenseNet121
* Output Classes:

  * Healthy
  * TB Positive

### Lung Segmentation Model

* Architecture: U-Net
* Purpose:

  * Lung Region Segmentation
  * Lesion Localization Support

---

## Future Improvements

* Patient-specific anatomical scaling.
* Multi-lesion visualization.
* Quantitative TB burden estimation.
* Clinical severity scoring.
* DICOM support.
* Volumetric lesion reconstruction from CT scans.
* External validation on additional TB datasets.

---

## Disclaimer

This project is intended for research, educational, and demonstration purposes only. It should not be used as a substitute for professional medical diagnosis or treatment decisions.
