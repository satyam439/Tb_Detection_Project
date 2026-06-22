import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, HRFlowable
)

from lung_segmentation import (
    load_lung_unet, segment_lungs, split_left_right_lung,
    apply_lung_mask_to_cam
)
from export_3d_viewer import export_3d_viewer

# ============================================================
# PATIENT CLASS
# ============================================================

@dataclass
class Patient:
    name: str
    patient_id: str
    gender: str
    age: int
    referred_by: str
    notes: str


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "tb_model.pth")
UNET_PATH = os.path.join(BASE_DIR, "lung_unet.pth")

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL LOADING (done once at import time, reused across requests)
# ============================================================

def load_model(path: str) -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


print(f"[INFO] Using device: {device}")
model = load_model(MODEL_PATH)
print(f"[INFO] TB model loaded from: {MODEL_PATH}")

lung_unet = load_lung_unet(UNET_PATH, device)
print(f"[INFO] Lung segmentation U-Net loaded from: {UNET_PATH}")

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

CLASSES = ["Healthy", "TB"]

# ============================================================
# GRADCAM++
# ============================================================

def compute_gradcam(model, input_tensor, target_class):
    """
    GradCAM++ on the LAST dense layer before pooling (features.norm5),
    which gives a 7x7 activation map upsampled smoothly to 224x224.

    The CAM is generated for the PREDICTED class (target_class), so the
    heatmap reflects what the model actually based its decision on,
    rather than always highlighting "TB-like" regions on healthy scans.
    """
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[model.features.norm5]
    )

    grayscale = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    grayscale = cv2.resize(
        grayscale, (224, 224), interpolation=cv2.INTER_CUBIC
    )

    grayscale = np.clip(grayscale, 0, None)
    g_min, g_max = grayscale.min(), grayscale.max()
    grayscale = (grayscale - g_min) / (g_max - g_min + 1e-8)

    return grayscale


def build_heatmap_overlay(original_rgb, grayscale_cam, pred_label,
                           pred_confidence, risk_level, risk_color_cv):
    """
    Single overlay: original X-ray blended with a JET heatmap of the
    GradCAM++ activation, plus a corner label with prediction info.
    """
    heatmap_bgr = cv2.applyColorMap(
        np.uint8(grayscale_cam * 255), cv2.COLORMAP_JET
    )
    base_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    blend_bgr = cv2.addWeighted(base_bgr, 0.55, heatmap_bgr, 0.45, 0)

    tag = f"{pred_label} | {pred_confidence}% | Risk: {risk_level}"
    bar_h = 22
    bar = np.zeros((bar_h, blend_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, tag, (6, 16), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                risk_color_cv, 1, cv2.LINE_AA)
    blend_bgr = np.vstack([bar, blend_bgr])

    return blend_bgr

# ============================================================
# PDF HELPERS
# ============================================================

def hex_color(h: str):
    h = h.lstrip("#")
    return colors.Color(*[int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])


def _conf_interp(c: float) -> str:
    if c >= 85:
        return "Strong positive indication"
    if c >= 65:
        return "Moderate positive indication"
    if c >= 45:
        return "Weak positive — review advised"
    return "Below threshold"


def _risk_interp(r: str) -> str:
    return {
        "HIGH": "Immediate clinical follow-up recommended",
        "MODERATE": "Further testing advised",
        "LOW": "Unlikely TB; monitor if symptomatic"
    }.get(r, "—")


C_DARK = hex_color("#0D1B2A")
C_BLUE = hex_color("#1565C0")
C_ACCENT = hex_color("#E53935")
C_ORANGE = hex_color("#FB8C00")
C_GREEN = hex_color("#2E7D32")
C_LIGHT = hex_color("#F5F7FA")
C_WHITE = colors.white
C_GREY = hex_color("#607D8B")

RISK_COLOR_MAP = {"HIGH": C_ACCENT, "MODERATE": C_ORANGE, "LOW": C_GREEN}


def build_pdf_report(patient: Patient, pred_label, pred_confidence,
                      confidence_tb, confidence_healthy, risk_level,
                      heatmap_path):
    report_path = os.path.join(
        OUTPUT_DIR,
        f"tb_report_{patient.patient_id.replace('/', '-')}.pdf"
    )

    risk_rl_color = RISK_COLOR_MAP[risk_level]

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    W, H = A4
    CW = W - 3.6 * cm

    h2_s = ParagraphStyle("h2_s", fontSize=11, textColor=C_BLUE,
                           fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    body_s = ParagraphStyle("body_s", fontSize=9, textColor=C_DARK,
                             fontName="Helvetica", leading=14)
    small_s = ParagraphStyle("small_s", fontSize=8, textColor=C_GREY,
                              fontName="Helvetica", leading=11)
    foot_s = ParagraphStyle("foot_s", fontSize=7.5, textColor=C_GREY,
                             fontName="Helvetica", alignment=TA_CENTER)

    story = []

    # ---- Header bar ----
    header_data = [[
        Paragraph(
            "<font color='#FFFFFF'><b>TB SCREENING REPORT</b></font><br/>"
            "<font color='#90CAF9' size='8'>AI-Assisted Chest X-Ray Analysis</font>",
            ParagraphStyle("hdr", fontName="Helvetica-Bold",
                           fontSize=14, textColor=C_WHITE, leading=18)
        ),
        Paragraph(
            f"<font color='#FFFFFF' size='8'>"
            f"Report Date: {datetime.now().strftime('%d %b %Y  %H:%M')}<br/>"
            f"System: DenseNet121 + GradCAM++<br/>"
            f"Device: {str(device).upper()}</font>",
            ParagraphStyle("hdr2", fontName="Helvetica",
                           fontSize=8, textColor=C_WHITE,
                           alignment=TA_RIGHT, leading=13)
        )
    ]]
    header_tbl = Table(header_data, colWidths=[CW * 0.65, CW * 0.35])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 14),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 10))

    # ---- Risk badge ----
    risk_badge_data = [[
        Paragraph(
            f"<b>RISK LEVEL: {risk_level}</b>",
            ParagraphStyle("rbdg", fontName="Helvetica-Bold",
                           fontSize=11, textColor=C_WHITE, alignment=TA_CENTER)
        )
    ]]
    risk_badge = Table(risk_badge_data, colWidths=[CW])
    risk_badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_rl_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(risk_badge)
    story.append(Spacer(1, 10))

    # ---- Patient Info ----
    story.append(Paragraph("Patient Information", h2_s))
    story.append(HRFlowable(width=CW, thickness=1, color=C_BLUE, spaceAfter=6))

    pi_data = [
        ["Patient Name", patient.name, "Patient ID", patient.patient_id],
        ["Gender", patient.gender, "Age", str(patient.age)],
        ["Referred By", patient.referred_by, "Scan Date", datetime.now().strftime("%d %b %Y")],
        ["Clinical Notes", Paragraph(patient.notes, body_s), "", ""],
    ]
    pi_tbl = Table(pi_data, colWidths=[CW * 0.2, CW * 0.3, CW * 0.2, CW * 0.3])
    pi_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -2), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), C_DARK),
        ("TEXTCOLOR", (2, 0), (2, -2), C_DARK),
        ("TEXTCOLOR", (1, 0), (1, -1), C_GREY),
        ("TEXTCOLOR", (3, 0), (3, -2), C_GREY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_WHITE, C_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("SPAN", (1, 3), (3, 3)),
        ("FONTNAME", (0, 3), (0, 3), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 3), (3, 3), C_GREY),
        ("GRID", (0, 0), (-1, -1), 0.4, hex_color("#CFD8DC")),
    ]))
    story.append(pi_tbl)
    story.append(Spacer(1, 12))

    # ---- Diagnosis Results ----
    story.append(Paragraph("Diagnosis Results", h2_s))
    story.append(HRFlowable(width=CW, thickness=1, color=C_BLUE, spaceAfter=6))

    diag_data = [
        ["Parameter", "Value", "Interpretation"],
        ["Prediction", pred_label, "AI Model Output"],
        ["Confidence (TB)", f"{confidence_tb}%", _conf_interp(confidence_tb)],
        ["Confidence (Healthy)", f"{confidence_healthy}%", "—"],
        ["Risk Level", risk_level, _risk_interp(risk_level)],
    ]
    diag_tbl = Table(diag_data, colWidths=[CW * 0.35, CW * 0.25, CW * 0.4])
    diag_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (0, -1), C_DARK),
        ("TEXTCOLOR", (1, 1), (1, -1), C_BLUE),
        ("TEXTCOLOR", (2, 1), (2, -1), C_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, hex_color("#CFD8DC")),
    ]))
    story.append(diag_tbl)
    story.append(Spacer(1, 12))

    # ---- Heatmap image ----
    story.append(Paragraph("X-Ray Heatmap (GradCAM++)", h2_s))
    story.append(HRFlowable(width=CW, thickness=1, color=C_BLUE, spaceAfter=6))

    img_w = CW * 0.6
    img_h = img_w * (246 / 224)  # account for the 22px label bar

    if os.path.exists(heatmap_path):
        img_data = [
            [RLImage(heatmap_path, width=img_w, height=img_h)],
            [Paragraph("<i>Warmer colors indicate regions the model "
                       "weighted more heavily for the predicted class.</i>", small_s)],
        ]
        img_tbl = Table(img_data, colWidths=[img_w])
        img_tbl.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BOX", (0, 0), (-1, -1), 0.5, hex_color("#B0BEC5")),
            ("BACKGROUND", (0, 0), (-1, -2), hex_color("#F5F7FA")),
        ]))
        story.append(img_tbl)
    else:
        story.append(Paragraph("<i>[heatmap_overlay.png not found]</i>", small_s))

    story.append(Spacer(1, 12))

    # ---- Clinical Disclaimer ----
    disc_data = [[
        Paragraph(
            "<b>CLINICAL DISCLAIMER</b><br/>"
            "This report is generated by an AI-assisted screening tool and is intended "
            "to support, not replace, the judgment of a qualified medical professional. "
            "A confirmed TB diagnosis requires microbiological testing, clinical "
            "examination, and radiologist review. All findings must be validated by a "
            "licensed physician before any clinical decisions are made.",
            ParagraphStyle("disc", fontName="Helvetica", fontSize=8,
                           textColor=C_DARK, leading=12)
        )
    ]]
    disc_tbl = Table(disc_data, colWidths=[CW])
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), hex_color("#FFF8E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 1, hex_color("#FFB300")),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(disc_tbl)
    story.append(Spacer(1, 8))

    # ---- Footer ----
    story.append(HRFlowable(width=CW, thickness=0.5, color=C_GREY, spaceAfter=4))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M:%S')}  |  "
        f"Model: DenseNet121  |  Dataset: Shenzhen + Montgomery  |  "
        f"System Version: 1.0",
        foot_s))

    doc.build(story)
    return report_path

# ============================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================

def run_pipeline(image_path, patient_name, patient_id, gender, age,
                  referred_by="", notes="N/A"):
  
    patient = Patient(
        name=patient_name,
        patient_id=patient_id,
        gender=gender,
        age=int(age),
        referred_by=referred_by or "Self",
        notes=notes or "N/A",
    )

    # ---- Load & preprocess image ----
    pil_img = Image.open(image_path).convert("L")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # ---- Prediction ----
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    confidence_tb = round(probs[0][1].item() * 100, 2)
    confidence_healthy = round(probs[0][0].item() * 100, 2)
    pred_label = CLASSES[pred_idx]
    pred_confidence = round(probs[0][pred_idx].item() * 100, 2)

    if confidence_tb >= 85:
        risk_level = "HIGH"
        risk_color_cv = (0, 0, 220)
    elif confidence_tb >= 55:
        risk_level = "MODERATE"
        risk_color_cv = (0, 140, 255)
    else:
        risk_level = "LOW"
        risk_color_cv = (0, 200, 80)

    # ---- Original image as numpy ----
    original = np.array(pil_img.resize((224, 224)))
    original_rgb = np.stack([original] * 3, axis=-1)

    # ---- Lung segmentation ----
    lung_mask, lung_prob = segment_lungs(
        lung_unet, pil_img, device, target_size=(224, 224)
    )
    left_lung_mask, right_lung_mask = split_left_right_lung(lung_mask)
    lung_coverage_pct = int(np.count_nonzero(lung_mask) / lung_mask.size * 100)

    # ---- GradCAM, masked to lung field ----
    grayscale_cam_raw = compute_gradcam(model, input_tensor, target_class=pred_idx)
    grayscale_cam = apply_lung_mask_to_cam(grayscale_cam_raw, lung_mask)

       # ------------------------------------------------
    # TB hotspot handling
    # ------------------------------------------------

    if pred_label == "TB":

        from save_tb_center import save_tb_center

        save_tb_center(
            grayscale_cam,
            OUTPUT_DIR
        )

    else:

        tb_center_file = os.path.join(
            OUTPUT_DIR,
            "tb_center.npy"
        )

        if os.path.exists(tb_center_file):

            os.remove(tb_center_file)

    heatmap_overlay = build_heatmap_overlay(
        original_rgb, grayscale_cam, pred_label, pred_confidence,
        risk_level, risk_color_cv
    )

    heatmap_path = os.path.join(OUTPUT_DIR, "heatmap_overlay.png")
    lung_mask_path = os.path.join(OUTPUT_DIR, "lung_mask.png")
    cv2.imwrite(heatmap_path, heatmap_overlay)
    cv2.imwrite(lung_mask_path, lung_mask)

    # ---- 3D interactive viewer ----
    sex_param = "female" if str(gender).strip().lower().startswith("f") else "male"
    viewer_path = export_3d_viewer(
        original_rgb=original_rgb,
        grayscale_cam=grayscale_cam,
        heatmap_overlay_bgr=heatmap_overlay,  # export script auto-strips label bar
        pred_label=pred_label,
        pred_confidence=pred_confidence,
        risk_level=risk_level,
        output_path=os.path.join(OUTPUT_DIR, "3d_viewer.html"),
        left_lung_mask=left_lung_mask,
        right_lung_mask=right_lung_mask,
        sex=sex_param,
    )

    # ---- PDF report ----
    report_path = build_pdf_report(
        patient=patient,
        pred_label=pred_label,
        pred_confidence=pred_confidence,
        confidence_tb=confidence_tb,
        confidence_healthy=confidence_healthy,
        risk_level=risk_level,
        heatmap_path=heatmap_path,
    )

    return {
        "patient": patient,
        "pred_label": pred_label,
        "pred_confidence": pred_confidence,
        "confidence_tb": confidence_tb,
        "confidence_healthy": confidence_healthy,
        "risk_level": risk_level,
        "lung_coverage_pct": lung_coverage_pct,
        "heatmap_path": heatmap_path,
        "lung_mask_path": lung_mask_path,
        "viewer_path": viewer_path,
        "report_path": report_path,
    }