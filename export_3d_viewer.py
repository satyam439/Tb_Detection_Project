"""
export_3d_viewer.py
============================================================
Builds a self-contained HTML viewer showing:

  - An anatomically structured reference lung model (3 lobes right,
    2 lobes left, cardiac notch), built to the real-world dimensions
    supplied (see anatomical_lung.py for the full spec).
  - Locked infection coordinates (from this patient's 2D GradCAM)
    projected onto that model's real-world (cm) coordinate space and
    rendered as glowing markers, each labeled with its (x,y,z) in cm,
    lung side, lobe, and confidence.

See anatomical_lung.py module docstring for the explicit, honest
description of what is patient-derived (X/Y of each marker) vs. what
is a fixed reference shape / modeled estimate (the lung mesh itself,
and each marker's Z depth).
============================================================
"""

import os
import json
import cv2
import base64
import numpy as np

from anatomical_lung import (
    build_anatomical_lung_volume, extract_anatomical_mesh, decimate_mesh,
    voxel_to_cm, get_lung_dims, lock_and_project_coordinates,
)

VIEWER_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_template.html")


def _to_b64(img) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def export_3d_viewer(original_rgb, grayscale_cam, heatmap_overlay_bgr,
                      pred_label, pred_confidence, risk_level, output_path,
                      left_lung_mask=None, right_lung_mask=None,
                      sex="male", population="indian",
                      target_faces=6000, mesh_level=0.25):
    """
    Builds the anatomical 3D lung viewer with locked infection
    coordinate markers.

    original_rgb        : HxWx3 uint8 RGB — clean grayscale X-ray
    grayscale_cam       : HxW float32 [0,1] — lung-masked GradCAM
    heatmap_overlay_bgr : HxWx3 uint8 BGR — JET heatmap blend (flat modes)
    left_lung_mask      : HxW uint8 (0/255) — used to restrict coordinate
                           locking to the lung field (passed through to
                           lock_and_project_coordinates via the union mask)
    right_lung_mask     : HxW uint8 (0/255)
    sex                 : "male" or "female" — selects reference dimensions
    population          : "indian" (default) or "general" — see
                           anatomical_lung.get_lung_dims() for the exact
                           scaling method and its stated limitations
    """
    H, W = original_rgb.shape[:2]
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    if left_lung_mask  is None: left_lung_mask  = np.zeros((H, W), dtype=np.uint8)
    if right_lung_mask is None: right_lung_mask = np.zeros((H, W), dtype=np.uint8)
    full_lung_mask = cv2.bitwise_or(left_lung_mask, right_lung_mask)

    # ── Step 2: anatomical reference model ────────────────────
    vol, lobe_masks, dims_cm = build_anatomical_lung_volume(sex, population)
    verts, faces, normals = extract_anatomical_mesh(vol, level=mesh_level)
    verts, faces, normals = decimate_mesh(verts, faces, normals, target_faces=target_faces)
    verts_cm = voxel_to_cm(verts, dims_cm)

    # Normalize verts to a renderer-friendly scale (Three.js units),
    # keeping cm values available separately for on-screen labels
    render_scale = 1.0 / max(dims_cm["total_width_cm"], dims_cm["length_cm"]) * 1.8
    verts_render = verts_cm * render_scale

    mesh_json = json.dumps({
        "verts":   verts_render.flatten().tolist(),
        "faces":   faces.flatten().tolist(),
        "normals": normals.flatten().tolist(),
    })

    # ── Step 1 + 3: lock 2D coords, project to 3D cm space ────
    # Only treat locked regions as TB findings when the model actually
    # predicted TB. When prediction is Healthy, the GradCAM is computed
    # against the Healthy class (see pipeline.py: target_class=pred_idx)
    # and shows "what made the model think Healthy", NOT infection sites
    # — displaying those as red TB markers would be actively misleading.
    if pred_label.strip().lower() == "tb":
        locked_regions = lock_and_project_coordinates(
            grayscale_cam, full_lung_mask, W, H, dims_cm
        )
    else:
        locked_regions = []

    markers = []
    for r in locked_regions:
        markers.append({
            "x": round(r["x_cm"] * render_scale, 4),
            "y": round(r["y_cm"] * render_scale, 4),
            "z": round(r["z_cm"] * render_scale, 4),
            "xCm": r["x_cm"], "yCm": r["y_cm"], "zCm": r["z_cm"],
            "lungSide": r["lung_side"], "lobe": r["lobe"],
            "peak": r["peak_intensity"], "mean": r["mean_intensity"],
            "id": r["region_id"], "zEstimated": r["z_estimated"],
        })
    markers_json = json.dumps(markers)

    # ── flat 2D textures for X-RAY / HEATMAP / LUNGS modes ────
    xray_b64 = _to_b64(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

    heat = heatmap_overlay_bgr.copy()
    for strip in range(min(40, heat.shape[0])):
        if heat[strip].mean() > 20:
            heat = heat[strip:]
            break
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_b64 = _to_b64(heat)

    left_img  = np.zeros_like(gray)
    right_img = np.zeros_like(gray)
    left_img[left_lung_mask   > 0] = gray[left_lung_mask   > 0]
    right_img[right_lung_mask > 0] = gray[right_lung_mask  > 0]
    left_b64  = _to_b64(cv2.cvtColor(np.stack([left_img]  * 3, axis=-1), cv2.COLOR_RGB2BGR))
    right_b64 = _to_b64(cv2.cvtColor(np.stack([right_img] * 3, axis=-1), cv2.COLOR_RGB2BGR))

    coverage_pct = int(np.count_nonzero(full_lung_mask) / (H * W) * 100)

    rw_lo, rw_hi = dims_cm["right_weight_g_range"]
    tw_lo, tw_hi = dims_cm["total_weight_g_range"]
    tlc_lo, tlc_hi = dims_cm["tlc_ml_range"]
    sa_lo, sa_hi = dims_cm["surface_area_m2_range"]

    with open(VIEWER_TEMPLATE_PATH, "r") as f:
        html = f.read()

    html = html.replace("__MESH_JSON__",     mesh_json)
    html = html.replace("__MARKERS_JSON__",  markers_json)
    html = html.replace("__XRAY_B64__",      xray_b64)
    html = html.replace("__HEATMAP_B64__",   heat_b64)
    html = html.replace("__LEFT_LUNG_B64__", left_b64)
    html = html.replace("__RIGHT_LUNG_B64__",right_b64)
    html = html.replace("__PRED_LABEL__",    pred_label)
    html = html.replace("__PRED_CONF__",     f"{pred_confidence}%")
    html = html.replace("__PRED_RISK__",     risk_level)
    html = html.replace("__LUNG_COVERAGE__", f"{coverage_pct}%")
    html = html.replace("__FACE_COUNT__",    str(len(faces)))
    html = html.replace("__VERT_COUNT__",    str(len(verts)))
    html = html.replace("__MARKER_COUNT__",  str(len(markers)))
    html = html.replace("__SEX__",           f"{sex}, {population}")
    html = html.replace("__DIMS_STR__",
                         f"{dims_cm['total_width_cm']:.1f}W x {dims_cm['length_cm']:.1f}L x {dims_cm['depth_cm']:.1f}D cm")
    html = html.replace("__WEIGHT_STR__",
                         f"R lung {rw_lo}-{rw_hi}g · Total {tw_lo}-{tw_hi}g")
    html = html.replace("__TLC_STR__", f"{tlc_lo}-{tlc_hi} mL")
    html = html.replace("__SA_STR__",  f"{sa_lo}-{sa_hi} m\u00b2")

    with open(output_path, "w") as f:
        f.write(html)

    return output_path