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

from lung_mask_xrv import generate_lung_masks
from lung_segmentation import apply_lung_mask_to_cam, get_lobe_masks
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

# NOTE: lung segmentation now uses lung_mask_xrv.py (pretrained
# torchxrayvision PSPNet), which loads its own weights lazily on first
# call — no separate U-Net loading needed here anymore.

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

    NOTE: a two-layer fusion (denseblock3 + norm5) was tried and reverted.
    Averaging in the mid-level denseblock3 layer diluted the sharp peak
    from the final layer, producing a flat, washed-out heatmap with barely
    any visible red — worse for readability than the original single-layer
    version, even though the underlying activation math was "more
    sophisticated". Back to single-layer, which reliably shows a clear,
    strong hotspot.

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

    # For a Healthy prediction, the CAM shows "where the model found support
    # for Healthy" — NOT a danger zone. Because the heatmap is min-max
    # normalized, it will always show red somewhere even when raw activation
    # is low and flat, which looks alarming/misleading on a negative result.
    # So for Healthy, blend it in very lightly instead of using the same
    # high-intensity overlay as a TB-positive result.
    if pred_label == "Healthy":
        blend_bgr = cv2.addWeighted(base_bgr, 0.92, heatmap_bgr, 0.08, 0)
    else:
        blend_bgr = cv2.addWeighted(base_bgr, 0.55, heatmap_bgr, 0.45, 0)

    tag = f"{pred_label} | {pred_confidence}% | Risk: {risk_level}"
    bar_h = 22
    bar = np.zeros((bar_h, blend_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, tag, (6, 16), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                risk_color_cv, 1, cv2.LINE_AA)
    blend_bgr = np.vstack([bar, blend_bgr])

    # ---- Anatomical R/L side markers (standard PA chest X-ray convention) ----
    # On a PA film, the film is viewed as if facing the patient, so the
    # patient's RIGHT lung appears on the LEFT side of the image, and the
    # patient's LEFT lung appears on the RIGHT side of the image. This is
    # a fixed anatomical convention, independent of the model's prediction
    # — it just labels the image the way a doctor expects any chest X-ray
    # to be labeled, so they can orient themselves immediately.
    img_h, img_w = blend_bgr.shape[:2]
    label_y = 40  # just below the top prediction bar
    cv2.putText(blend_bgr, "R", (10, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_bgr, "L", (img_w - 30, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)

    return blend_bgr


# ============================================================
# NEW: LOBE-WISE RISK SCORING
# ============================================================

def compute_lobe_risk(grayscale_cam_masked: np.ndarray,
                       left_lung_mask: np.ndarray,
                       right_lung_mask: np.ndarray) -> dict:
    """
    Breaks the (already lung-field-masked) GradCAM++ activation down
    into a percentage share per lobe, instead of one blurry whole-lung
    heatmap. This directly addresses activation that leaks onto ribs /
    pleura by (a) already being restricted to the lung field via
    apply_lung_mask_to_cam, and (b) here, normalizing so the report can
    say "62% of activation is in the Right Upper Lobe" rather than just
    showing a diffuse color wash.

    Method: mean CAM activation *within* each lobe's own mask, then
    normalized across all 5 lobes so shares sum to ~100%.

    Args:
        grayscale_cam_masked : HxW float32 [0,1], lung-masked CAM
        left_lung_mask        : HxW uint8 (0/255), one lung only
        right_lung_mask       : HxW uint8 (0/255), one lung only

    Returns:
        dict lobe_name -> percentage share (float, sums to ~100 if any
        activation exists; all-zero dict if there's no activation at
        all, e.g. a confidently Healthy scan).
    """
    lobe_masks = {}
    lobe_masks.update(get_lobe_masks(right_lung_mask, side="right"))
    lobe_masks.update(get_lobe_masks(left_lung_mask,  side="left"))

    raw_scores = {}
    for lobe_name, mask in lobe_masks.items():
        mask_bool = mask > 0
        if not mask_bool.any():
            raw_scores[lobe_name] = 0.0
            continue
        raw_scores[lobe_name] = float(grayscale_cam_masked[mask_bool].mean())

    total = sum(raw_scores.values())
    if total <= 1e-9:
        return {name: 0.0 for name in raw_scores}

    return {
        name: round(score / total * 100, 1)
        for name, score in raw_scores.items()
    }


def _lobe_risk_tier(pct: float) -> str:
    if pct >= 50:
        return "HIGH"
    if pct >= 20:
        return "MODERATE"
    if pct > 0:
        return "LOW"
    return "—"


LOBE_ORDER = [
    "RIGHT UPPER LOBE", "RIGHT MIDDLE LOBE", "RIGHT LOWER LOBE",
    "LEFT UPPER LOBE", "LEFT LOWER LOBE",
]

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
                      heatmap_path, lobe_risk_pct=None):
    report_path = os.path.join(
        OUTPUT_DIR,
        f"tb_report_{patient.patient_id.replace('/', '-')}.pdf"
    )

    risk_rl_color = RISK_COLOR_MAP[risk_level]
    lobe_risk_pct = lobe_risk_pct or {}

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

    # ---- NEW: Lobe-Wise Risk Distribution ----
    story.append(Paragraph("Lobe-Wise Risk Distribution", h2_s))
    story.append(HRFlowable(width=CW, thickness=1, color=C_BLUE, spaceAfter=6))

    if pred_label == "TB" and any(v > 0 for v in lobe_risk_pct.values()):
        lobe_data = [["Lobe", "Relative Share (not confidence)", "Tier"]]
        for lobe_name in LOBE_ORDER:
            pct = lobe_risk_pct.get(lobe_name, 0.0)
            lobe_data.append([lobe_name.title(), f"{pct}%", _lobe_risk_tier(pct)])

        lobe_tbl = Table(lobe_data, colWidths=[CW * 0.45, CW * 0.30, CW * 0.25])
        lobe_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 1), (0, -1), C_DARK),
            ("TEXTCOLOR", (1, 1), (1, -1), C_BLUE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, hex_color("#CFD8DC")),
        ]))
        story.append(lobe_tbl)
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<i>NOTE: this is a RELATIVE spatial distribution, not a second "
            "confidence score — the 5 shares always sum to ~100% regardless "
            "of the overall prediction confidence above. It shows WHERE "
            "within the lungs the model's activation is concentrated, not "
            "HOW confident the model is that TB is present (see Confidence "
            "(TB) above for that). Share = mean GradCAM++ activation, after "
            "lung-field masking, inside each lobe, normalized across all 5 "
            "lobes. Tiers: HIGH &ge;50%, MODERATE 20&ndash;49%, LOW &lt;20%. "
            "Lobe boundaries are approximated by vertical position within "
            "each lung, not traced fissure anatomy.</i>", small_s))
    else:
        story.append(Paragraph(
            "<i>No significant lung-field activation to distribute across "
            "lobes (Healthy prediction).</i>", small_s))

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
                       "weighted more heavily for the predicted class, "
                       "restricted to the segmented lung field.</i>", small_s)],
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

    # ---- Lung segmentation (pretrained torchxrayvision PSPNet) ----
    # Single model pass returns the union mask AND the left/right split
    # together, so we don't run the segmenter twice.
    lung_mask, left_lung_mask, right_lung_mask = generate_lung_masks(
        pil_img, target_size=(224, 224)
    )
    lung_coverage_pct = int(np.count_nonzero(lung_mask) / lung_mask.size * 100)

    # ---- GradCAM, masked to lung field ----
    grayscale_cam_raw = compute_gradcam(model, input_tensor, target_class=pred_idx)
    grayscale_cam = apply_lung_mask_to_cam(grayscale_cam_raw, lung_mask)

    # ---- NEW: lobe-wise risk breakdown (masked CAM, split by lobe) ----
    lobe_risk_pct = compute_lobe_risk(grayscale_cam, left_lung_mask, right_lung_mask)
    if pred_label == "TB":
        top_lobe = max(lobe_risk_pct, key=lobe_risk_pct.get) if lobe_risk_pct else "N/A"
        print(f"[INFO] Lobe risk breakdown: {lobe_risk_pct}")
        print(f"[INFO] Highest-risk lobe: {top_lobe} ({lobe_risk_pct.get(top_lobe, 0)}%)")

    # ── SAVE REAL GRADCAM.NPY so viewer_3d.py uses actual heatmap ──
    np.save(os.path.join(OUTPUT_DIR, "gradcam.npy"), grayscale_cam.astype(np.float32))
    print(f"[INFO] gradcam.npy saved — peak at {np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)}")

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
        tb_centers_file = os.path.join(
            OUTPUT_DIR,
            "tb_centers.npy"
        )

        if os.path.exists(tb_center_file):
            os.remove(tb_center_file)

        if os.path.exists(tb_centers_file):
            os.remove(tb_centers_file)

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
        lobe_risk_pct=lobe_risk_pct,
    )

    return {
        "patient": patient,
        "pred_label": pred_label,
        "pred_confidence": pred_confidence,
        "confidence_tb": confidence_tb,
        "confidence_healthy": confidence_healthy,
        "risk_level": risk_level,
        "lung_coverage_pct": lung_coverage_pct,
        "lobe_risk_pct": lobe_risk_pct,
        "heatmap_path": heatmap_path,
        "lung_mask_path": lung_mask_path,
        "viewer_path": viewer_path,
        "report_path": report_path,
    }

# ============================================================
# CLI ENTRY POINT
# ============================================================
# Without this block, running `python3 predict_tb.py <image_path>` only
# loads the model and exits — run_pipeline() is never actually called, so
# the image path is silently ignored and any leftover outputs/tb_center.npy
# from a previous run gets reused by batch_test.py. This is what caused the
# "every image comes back TB" bug both times it's shown up.

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 predict_tb.py <image_path> [patient_name] [patient_id] [gender] [age]")
        sys.exit(1)

    image_path = sys.argv[1]
    patient_name = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    patient_id = sys.argv[3] if len(sys.argv) > 3 else "N/A"
    gender = sys.argv[4] if len(sys.argv) > 4 else "N/A"
    age = sys.argv[5] if len(sys.argv) > 5 else 0

    result = run_pipeline(
        image_path=image_path,
        patient_name=patient_name,
        patient_id=patient_id,
        gender=gender,
        age=age,
    )

    print(f"[RESULT] {os.path.basename(image_path)} → "
          f"{result['pred_label']} "
          f"(TB={result['confidence_tb']}%, Healthy={result['confidence_healthy']}%)")
