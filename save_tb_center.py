import os
import numpy as np
from scipy import ndimage

# Tunable thresholds — adjust these if regions are being missed or
# noise is being picked up as false regions.
MIN_REGION_PEAK = 0.5    # a region must reach at least this activation to count
MIN_REGION_AREA_PX = 15  # minimum connected pixel count (filters single-pixel noise)
MAX_REGIONS = 5          # cap on how many regions to report, strongest first


def find_infection_regions(heatmap_2d: np.ndarray,
                            min_peak: float = MIN_REGION_PEAK,
                            min_area_px: int = MIN_REGION_AREA_PX,
                            max_regions: int = MAX_REGIONS) -> list:

    binary = heatmap_2d > min_peak
    labeled, n = ndimage.label(binary)

    regions = []
    for i in range(1, n + 1):
        mask = labeled == i
        area = int(mask.sum())
        if area < min_area_px:
            continue

        ys, xs = np.where(mask)
        weights = heatmap_2d[mask]
        peak = float(heatmap_2d[mask].max())
        mean = float(heatmap_2d[mask].mean())
        cy = float(np.average(ys, weights=weights))
        cx = float(np.average(xs, weights=weights))

        regions.append({
            "cx": cx, "cy": cy,
            "peak": peak, "mean": mean,
            "area_px": area,
        })

    regions.sort(key=lambda r: r["peak"], reverse=True)
    return regions[:max_regions]


def save_tb_center(grayscale_cam: np.ndarray, output_dir: str):
    """
    Detect infection region(s) in a GradCAM heatmap and save them for
    the 3D viewer. Called from predict_tb.py's run_pipeline() only
    when the model predicts TB.

    Args:
        grayscale_cam : HxW float32 [0,1], lung-masked GradCAM++ output
        output_dir     : directory to save tb_center.npy / tb_centers.npy into
    """
    os.makedirs(output_dir, exist_ok=True)

    regions = find_infection_regions(grayscale_cam)

    if not regions:
        # Fallback: if nothing clears the threshold (e.g. very diffuse,
        # low-confidence activation), fall back to the single global
        # peak so the viewer still has SOMETHING to show rather than
        # silently rendering nothing.
        idx = np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)
        regions = [{
            "cx": float(idx[1]), "cy": float(idx[0]),
            "peak": float(grayscale_cam.max()),
            "mean": float(grayscale_cam.max()),
            "area_px": 1,
        }]
        print("[WARN] No region cleared the activation threshold — "
              "falling back to single global peak.")

    # ---- backward-compatible single-point file (strongest region) ----
    top = regions[0]
    np.save(os.path.join(output_dir, "tb_center.npy"),
            np.array([top["cx"], top["cy"]], dtype=np.float32))

    # ---- new multi-region file ----
    centers_arr = np.array(
        [[r["cx"], r["cy"], r["peak"], r["mean"], r["area_px"]] for r in regions],
        dtype=np.float32,
    )
    np.save(os.path.join(output_dir, "tb_centers.npy"), centers_arr)

    print(f"[INFO] Detected {len(regions)} infection region(s):")
    for i, r in enumerate(regions):
        print(f"  Region {i+1}: px=({r['cx']:.0f},{r['cy']:.0f})  "
              f"peak={r['peak']:.2f}  mean={r['mean']:.2f}  area={r['area_px']}px")