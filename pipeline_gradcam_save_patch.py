
# ══════════════════════════════════════════════════════════════════
# PATTERN 1 — if your pipeline looks like this (pytorch-grad-cam):
# ══════════════════════════════════════════════════════════════════

def example_pattern_1(model, input_tensor, target_layers, output_dir="outputs"):
    import numpy as np
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)   # shape (1, 224, 224)
    grayscale_cam = grayscale_cam[0]                 # shape (224, 224)

    # ← ADD THIS (3 lines) ──────────────────────────────────────────
    import os
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "gradcam.npy"),
            grayscale_cam.astype(np.float32))
    print(f"[INFO] gradcam.npy saved, peak at "
          f"{np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)}")
    # ───────────────────────────────────────────────────────────────

    return grayscale_cam


# ══════════════════════════════════════════════════════════════════
# PATTERN 2 — if your pipeline uses a custom CAM function:
# ══════════════════════════════════════════════════════════════════

def example_pattern_2(heatmap_2d, output_dir="outputs"):
    """heatmap_2d is already a (224,224) numpy array from your model."""
    import numpy as np, os

    # Normalise to [0,1]
    mn, mx = heatmap_2d.min(), heatmap_2d.max()
    cam_norm = (heatmap_2d - mn) / (mx - mn + 1e-9)

    # ← ADD THIS (2 lines) ──────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "gradcam.npy"),
            cam_norm.astype(np.float32))
    # ───────────────────────────────────────────────────────────────

    return cam_norm


# ══════════════════════════════════════════════════════════════════
# PATTERN 3 — also save tb_center.npy at the same time
#             (verifies both files update together per scan)
# ══════════════════════════════════════════════════════════════════

def save_both_outputs(cam_norm, cx: int, cy: int, output_dir="outputs"):
  
    import numpy as np, os

    os.makedirs(output_dir, exist_ok=True)

    # Save GradCAM heatmap
    np.save(os.path.join(output_dir, "gradcam.npy"),
            cam_norm.astype(np.float32))

    # Save TB hotspot centre
    np.save(os.path.join(output_dir, "tb_center.npy"),
            np.array([cx, cy], dtype=np.float64))

    # Verify peak alignment immediately
    peak_r, peak_c = divmod(int(cam_norm.argmax()), 224)
    drift_col = abs(peak_c - cx)
    drift_row = abs(peak_r - cy)

    print(f"\n[SCAN SAVED]")
    print(f"  gradcam.npy  → peak at col={peak_c}  row={peak_r}")
    print(f"  tb_center    → cx={cx}  cy={cy}")
    print(f"  Peak drift   → Δcol={drift_col}px  Δrow={drift_row}px  "
          + ("✅ OK" if drift_col <= 2 and drift_row <= 2 else "⚠️ CHECK ALIGNMENT"))
    print(f"  Files ready for viewer_3d.py\n")


# ══════════════════════════════════════════════════════════════════
# WHAT HAPPENS PER SCAN — full flow summary
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(__doc__)
    print("Search your pipeline code for where GradCAM is computed")
    print("and add the np.save() line shown in the patterns above.")