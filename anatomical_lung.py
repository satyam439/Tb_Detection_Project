"""
anatomical_lung.py
============================================================
Implements the pipeline requested:

  1. Lock 2D pixel coordinates of infected (high-GradCAM) regions
     in the original X-ray.
  2. Build a proper ANATOMICAL reference lung model — right lung
     with 3 lobes (upper/middle/lower, larger), left lung with
     2 lobes (upper/lower, smaller, with a cardiac notch) — built
     to the real-world dimensions supplied:

       Length (superior-inferior): ~24-30 cm (using 27cm / 24cm avg)
       Width (chest cavity):        31.8 cm (M) / 28.0 cm (F)
       Depth (anterior-posterior):  21.4 cm (M) / 19.4 cm (F)
       Right lung larger (3 lobes), left lung smaller (2 lobes,
       cardiac notch for the heart)

  3. Project each locked 2D coordinate onto this anatomical model's
     real-world (cm) coordinate system.
  4. Hand off labeled 3D coordinates + the anatomical mesh for
     rendering.

HONEST LIMITATION (stated explicitly so it can be repeated to your
sir if asked): the lung SHAPE here is now a fixed anatomical
reference model built to standard population measurements — it is
NOT derived from this patient's own X-ray silhouette anymore. The
X (left-right) and Y (up-down) coordinates of each locked infection
region ARE derived from this specific patient's 2D GradCAM output,
proportionally mapped onto the reference model's real-world cm
dimensions. The Z (front-to-back depth) coordinate is NOT measured —
a 2D X-ray contains no depth information for any patient — so Z is
placed at the anatomical mid-depth of whichever lobe the region falls
into, as a labeled estimate, not a measurement.
============================================================
"""

import numpy as np
import cv2
from skimage import measure
from scipy.ndimage import map_coordinates, gaussian_filter

VOL = 72  # voxel grid resolution for the anatomical model

# ============================================================
# REAL-WORLD ANATOMICAL DIMENSIONS (from supplied spec)
# ============================================================
#
# Length (superior-inferior): ~24cm at normal breathing, up to 27-30cm
#   fully expanded -> using 27cm (M) / 24cm (F) as the displayed model size
# Width (chest cavity): 31.8cm (M) / 28.0cm (F)
# Depth (chest/thickness): 21.4cm (M) / 19.4cm (F)
# Weight: right lung 155-720g (M), 100-590g (F); left slightly lighter;
#   combined total ~1000-1200g
# Total Lung Capacity (TLC): ~5700-6000mL (M), ~4200-4500mL (F)
# Combined gas-exchange surface area: ~80-100 sq. meters (both lungs)
# Structure: right lung = 3 lobes, larger; left lung = 2 lobes, smaller
#   (cardiac notch for the heart)
#
# INDIAN-POPULATION-SCALED VARIANT ("population='indian'"):
# No single Indian-population study directly reports lung length/width/
# depth in the same format as the general spec above, so these figures
# are derived rather than directly measured. Method: a chest CT study of
# Indian adults (cardiothoracic ratio research at a Chennai facility,
# n=102) reported mean transverse thoracic diameter of 29.7cm (male) and
# 27.1cm (female) -- noticeably smaller than the general/Western-population
# 31.8cm / 28.0cm figures above. The ratio between these two width figures
# (29.7/31.8 = 0.934 male, 27.1/28.0 = 0.968 female) is applied UNIFORMLY
# to length, width, and depth, since no comparably-sourced Indian-specific
# figures exist for the other two axes. This is an approximation, not a
# direct measurement of Indian lung length/depth -- stated explicitly so
# it isn't presented as more precise than it is.
# ============================================================

ANATOMICAL_DIMS = {
    "male": {
        "length_cm": 27.0,        # mid-range of 24-30cm (normal..fully expanded)
        "depth_cm": 21.4,         # chest depth/thickness
        "total_width_cm": 31.8,
        "right_weight_g_range": (155, 720),
        "total_weight_g_range": (1000, 1200),
        "tlc_ml_range": (5700, 6000),
        "surface_area_m2_range": (80, 100),
    },
    "female": {
        "length_cm": 24.0,
        "depth_cm": 19.4,
        "total_width_cm": 28.0,
        "right_weight_g_range": (100, 590),
        "total_weight_g_range": (1000, 1200),   # spec gives one combined figure for both sexes
        "tlc_ml_range": (4200, 4500),
        "surface_area_m2_range": (80, 100),
    },
}

# Indian-population scale factors (see note above) — applied uniformly
# to length_cm, depth_cm, and total_width_cm only. Weight/TLC/surface
# area figures are kept as the general reference values since no
# Indian-specific data for those was available.
_INDIAN_SCALE = {
    "male": 29.7 / 31.8,    # ≈ 0.934
    "female": 27.1 / 28.0,  # ≈ 0.968
}


def get_lung_dims(sex: str = "male", population: str = "general") -> dict:
    """
    Real-world lung dimensions and physiological reference figures.

    Args:
        sex        : "male" or "female"
        population : "general" (default; broad/Western-population
                      reference values as originally supplied) or
                      "indian" (proportionally scaled using measured
                      Indian-population chest width data — see the
                      module-level note above for the exact method
                      and its limitations)
    """
    sex_key = sex.lower() if sex.lower() in ANATOMICAL_DIMS else "male"
    spec = dict(ANATOMICAL_DIMS[sex_key])  # copy so we don't mutate the original

    if population.lower() == "indian":
        sf = _INDIAN_SCALE[sex_key]
        spec["length_cm"] = round(spec["length_cm"] * sf, 1)
        spec["depth_cm"] = round(spec["depth_cm"] * sf, 1)
        spec["total_width_cm"] = round(spec["total_width_cm"] * sf, 1)

    total_width = spec["total_width_cm"]
    return {
        "length_cm": spec["length_cm"],
        "depth_cm": spec["depth_cm"],
        "right_width_cm": total_width * 0.53,   # right lung larger
        "left_width_cm": total_width * 0.47,    # left lung smaller (cardiac notch)
        "total_width_cm": total_width,
        "right_weight_g_range": spec["right_weight_g_range"],
        "total_weight_g_range": spec["total_weight_g_range"],
        "tlc_ml_range": spec["tlc_ml_range"],
        "surface_area_m2_range": spec["surface_area_m2_range"],
        "population": population.lower(),
    }



# ============================================================
# STEP 1 — LOCK 2D INFECTION COORDINATES
# ============================================================

def lock_infection_coordinates(grayscale_cam: np.ndarray,
                                lung_mask: np.ndarray,
                                min_area: int = 40) -> list:
    """
    Locks pixel coordinates of infected (high-activation) regions from
    a lung-masked GradCAM heatmap, using an adaptive threshold relative
    to that image's own activation range (robust across varying
    confidence levels rather than a single fixed global percentile).

    Returns a list of region dicts, sorted by peak intensity descending:
        x, y, w, h        — bounding box in pixel space
        area_px            — region area in pixels
        centroid_x/y        — region center in pixel space
        peak_intensity      — max GradCAM value in this region [0,1]
        mean_intensity       — mean GradCAM value in this region [0,1]
    """
    H, W = grayscale_cam.shape
    inside_lung = grayscale_cam[lung_mask > 0]

    if inside_lung.size == 0 or inside_lung.max() <= 0:
        return []

    cam_max = float(inside_lung.max())
    thresh_val = max(cam_max * 0.55, float(np.percentile(inside_lung, 85)))

    binary = ((grayscale_cam >= thresh_val) & (lung_mask > 0)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2

        mask_c = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        region_cam = grayscale_cam[mask_c > 0]
        peak = float(region_cam.max()) if region_cam.size else 0.0
        mean_i = float(region_cam.mean()) if region_cam.size else 0.0

        regions.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area_px": int(area),
            "centroid_x": float(cx), "centroid_y": float(cy),
            "peak_intensity": round(peak, 4),
            "mean_intensity": round(mean_i, 4),
        })

    regions.sort(key=lambda r: r["peak_intensity"], reverse=True)
    return regions


# ============================================================
# STEP 2 — ANATOMICAL REFERENCE LUNG MODEL (with lobes)
# ============================================================

def build_anatomical_lung_volume(sex: str = "male", population: str = "general"):
    """
    Builds a voxel scalar field for an anatomically structured pair of
    lungs: right lung = 3 lobes (upper/middle/lower), larger; left
    lung = 2 lobes (upper/lower), smaller, with a cardiac notch carved
    out of its medial-inferior border for the heart.

    This is a FIXED reference shape built to population-average real
    -world proportions, not derived from any individual patient's scan.

    Args:
        sex        : "male" or "female"
        population : "general" or "indian" — see get_lung_dims() for
                      the exact scaling method and its limitations

    Returns:
        vol        : VOL x VOL x VOL float32 [0,1] scalar field
        lobe_masks : dict of per-lobe voxel masks, used later to assign
                     each locked 2D region to a specific lobe in 3D
        dims_cm    : real-world dimensions dict for this sex/population
    """
    dims_cm = get_lung_dims(sex, population)

    zz, yy, xx = np.meshgrid(
        np.linspace(-0.5, 0.5, VOL),
        np.linspace(-0.5, 0.5, VOL),
        np.linspace(-0.5, 0.5, VOL),
        indexing='ij'
    )
    # xx: medial-lateral (patient left/right), yy: superior-inferior, zz: anterior-posterior

    def lobe_blob(cx, cy, cz, rx, ry, rz, power=2.0):
        d = (np.abs(xx - cx) / rx) ** power + (np.abs(yy - cy) / ry) ** power + (np.abs(zz - cz) / rz) ** power
        return np.clip(1.0 - d, 0, 1)

    # Right lung (negative X = patient's right): 3 lobes, generously
    # overlapping radii so they fuse into one continuous lung after
    # smoothing rather than reading as separate floating balls
    right_upper  = lobe_blob(-0.26,  0.21,  0.00, 0.21, 0.23, 0.17)
    right_middle = lobe_blob(-0.27,  0.00,  0.02, 0.22, 0.19, 0.18)
    right_lower  = lobe_blob(-0.25, -0.22, -0.01, 0.23, 0.25, 0.19)
    right_lung = np.maximum(np.maximum(right_upper, right_middle), right_lower)

    # Left lung (positive X = patient's left): 2 lobes, same fused approach
    left_upper = lobe_blob(0.26,  0.20,  0.00, 0.20, 0.25, 0.17)
    left_lower = lobe_blob(0.25, -0.20, -0.01, 0.21, 0.27, 0.18)
    left_lung_raw = np.maximum(left_upper, left_lower)

    # Cardiac notch — a SHALLOW concave dent on the medial-inferior
    # border (partial-strength subtraction, smoothed afterward) so it
    # reads as a gentle notch rather than disconnecting the lobe into
    # separate fragments
    notch = lobe_blob(0.06, -0.16, 0.08, 0.16, 0.18, 0.22, power=2.0)
    left_lung = np.clip(left_lung_raw - notch * 0.55, 0, 1)

    vol = np.maximum(right_lung, left_lung)
    vol = gaussian_filter(vol, sigma=1.3)
    if vol.max() > 0:
        vol /= vol.max()

    lobe_masks = {
        "right_upper": right_upper, "right_middle": right_middle, "right_lower": right_lower,
        "left_upper": left_lung_raw * (left_upper > left_lower),
        "left_lower": left_lung_raw * (left_lower >= left_upper),
    }

    return vol, lobe_masks, dims_cm


# ============================================================
# STEP 3 — PROJECT LOCKED 2D COORDINATES -> 3D ANATOMICAL SPACE
# ============================================================

def determine_lung_side(cx_norm: float) -> str:
    """PA chest X-ray convention: patient's RIGHT lung appears on the
    LEFT side of the image."""
    return "right" if cx_norm < 0.5 else "left"


def estimate_lobe(cy_norm: float, lung_side: str) -> str:
    """Rough lobe estimate from vertical position in the 2D image."""
    if lung_side == "right":
        if cy_norm < 0.40:
            return "upper"
        elif cy_norm < 0.62:
            return "middle"
        else:
            return "lower"
    else:
        return "upper" if cy_norm < 0.50 else "lower"


def project_region_to_3d(region: dict, img_w: int, img_h: int, dims_cm: dict) -> dict:
    """
    Projects one locked 2D region (pixel coordinates) onto the
    anatomical model's real-world (cm) coordinate system.

    X, Y come directly from this patient's 2D coordinates, scaled
    proportionally into the reference model's real-world width/length.
    Z is NOT measured — see module docstring — and is left to be
    resolved to a specific lobe's mid-depth by the caller.
    """
    cx_norm = region["centroid_x"] / img_w
    cy_norm = region["centroid_y"] / img_h

    lung_side = determine_lung_side(cx_norm)
    lobe = estimate_lobe(cy_norm, lung_side)

    lung_width_cm = dims_cm["right_width_cm"] if lung_side == "right" else dims_cm["left_width_cm"]

    if lung_side == "right":
        frac = cx_norm / 0.5            # 0 = lateral edge, 1 = medial edge
        x_cm = -lung_width_cm * (1 - frac)
    else:
        frac = (cx_norm - 0.5) / 0.5    # 0 = medial edge, 1 = lateral edge
        x_cm = lung_width_cm * frac

    y_cm = (0.5 - cy_norm) * dims_cm["length_cm"]

    return {
        "x_cm": round(float(x_cm), 2),
        "y_cm": round(float(y_cm), 2),
        "lung_side": lung_side,
        "lobe": lobe,
        "peak_intensity": region["peak_intensity"],
        "mean_intensity": region["mean_intensity"],
        "area_px": region["area_px"],
    }


def resolve_z_from_lobe(lung_side: str, lobe: str, dims_cm: dict) -> dict:
    """
    Resolves the Z (anterior-posterior) coordinate for a region by
    placing it at the anatomical mid-depth of its assigned lobe.

    This is an explicit, labeled APPROXIMATION — not a measurement.
    A single 2D X-ray contains no information about how far forward
    or backward within the chest a lesion sits.
    """
    half_depth = dims_cm["depth_cm"] / 2
    # Slightly anterior bias for upper/middle lobes (common TB site),
    # neutral mid-depth otherwise — purely a presentation choice, not
    # a clinical measurement.
    if lobe in ("upper", "middle"):
        z_cm = half_depth * 0.15
    else:
        z_cm = -half_depth * 0.05

    return {
        "z_cm": round(float(z_cm), 2),
        "z_estimated": True,
        "z_method": "anatomical lobe mid-depth (not measured from the X-ray)",
    }


def lock_and_project_coordinates(grayscale_cam: np.ndarray,
                                  lung_mask: np.ndarray,
                                  img_w: int, img_h: int,
                                  dims_cm: dict,
                                  min_area: int = 40) -> list:
    """
    Full Step 1 + Step 3 pipeline: lock 2D infection coordinates, then
    project each onto the anatomical model's real-world coordinate
    system, including the labeled Z-depth estimate.

    Returns a list of fully-resolved coordinate dicts ready for
    rendering and for printing/reporting.
    """
    regions = lock_infection_coordinates(grayscale_cam, lung_mask, min_area=min_area)

    results = []
    for i, region in enumerate(regions, start=1):
        proj = project_region_to_3d(region, img_w, img_h, dims_cm)
        z_info = resolve_z_from_lobe(proj["lung_side"], proj["lobe"], dims_cm)
        proj.update(z_info)
        proj["region_id"] = i
        results.append(proj)

    return results


# ============================================================
# MESH EXTRACTION + DECIMATION (anatomical model + markers)
# ============================================================

def extract_anatomical_mesh(vol: np.ndarray, level: float = 0.3):
    """Marching-cubes surface extraction of the anatomical lung volume."""
    verts, faces, normals, _ = measure.marching_cubes(vol, level=level)
    return verts, faces, normals


def decimate_mesh(verts, faces, normals, target_faces=6000):
    """Vertex-clustering decimation (same method as mesh_3d.py)."""
    if len(faces) <= target_faces:
        return verts, faces, normals

    bounds_min = verts.min(axis=0)
    bounds_max = verts.max(axis=0)
    extent = bounds_max - bounds_min
    extent[extent == 0] = 1

    reduction_factor = (target_faces / len(faces)) ** 0.5
    grid_res = max(8, int(VOL * reduction_factor))

    grid_idx = ((verts - bounds_min) / extent * (grid_res - 1)).astype(np.int32)
    grid_idx = np.clip(grid_idx, 0, grid_res - 1)
    cell_keys = grid_idx[:, 0] * grid_res * grid_res + grid_idx[:, 1] * grid_res + grid_idx[:, 2]

    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)
    n_clusters = len(unique_keys)

    new_verts = np.zeros((n_clusters, 3), dtype=np.float32)
    new_normals = np.zeros((n_clusters, 3), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int32)

    np.add.at(new_verts, inverse, verts)
    np.add.at(new_normals, inverse, normals)
    np.add.at(counts, inverse, 1)

    counts = np.maximum(counts, 1)
    new_verts /= counts[:, None]
    new_normals /= counts[:, None]
    norm_lens = np.linalg.norm(new_normals, axis=1, keepdims=True)
    norm_lens[norm_lens == 0] = 1
    new_normals /= norm_lens

    new_face_idx = inverse[faces]
    valid = ((new_face_idx[:, 0] != new_face_idx[:, 1]) &
             (new_face_idx[:, 1] != new_face_idx[:, 2]) &
             (new_face_idx[:, 0] != new_face_idx[:, 2]))
    new_faces = new_face_idx[valid]

    return new_verts, new_faces, new_normals


def voxel_to_cm(verts: np.ndarray, dims_cm: dict) -> np.ndarray:
    """
    Converts mesh vertex positions from normalized voxel space (-0.5..0.5
    per axis, as built in build_anatomical_lung_volume) into real-world
    centimeters, matching the same coordinate convention used by
    project_region_to_3d / resolve_z_from_lobe (X=medial-lateral,
    Y=superior-inferior, Z=anterior-posterior).
    """
    # verts come out of marching_cubes in [0, VOL] voxel index space
    centered = (verts / VOL) - 0.5   # -> -0.5..0.5
    out = np.zeros_like(centered)
    out[:, 0] = centered[:, 2] * dims_cm["total_width_cm"]   # marching_cubes axis order is (z,y,x)
    out[:, 1] = centered[:, 1] * dims_cm["length_cm"]
    out[:, 2] = centered[:, 0] * dims_cm["depth_cm"]
    return out