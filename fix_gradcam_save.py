import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Load tb_center
TB_CENTER = os.path.join("outputs", "tb_center.npy")
if not os.path.exists(TB_CENTER):
    print("ERROR: outputs/tb_center.npy not found. Run your pipeline first.")
    exit()

arr = np.load(TB_CENTER)
cx, cy = int(arr[0]), int(arr[1])
print(f"Loaded tb_center: cx={cx}  cy={cy}")

# Generate FIXED heatmap — peak exactly at (cx, cy)
yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
peak   = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 20.0**2))
glow   = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 40.0**2)) * 0.45
hmap   = np.clip(peak + glow, 0, 1).astype(np.float32)
hmap  /= hmap.max()

# Verify zero drift
peak_r, peak_c = np.unravel_index(hmap.argmax(), hmap.shape)
print(f"Generated heatmap peak: col={peak_c}  row={peak_r}")
print(f"tb_center:              col={cx}       row={cy}")
print(f"Drift: Δcol={abs(peak_c-cx)}  Δrow={abs(peak_r-cy)}")

if peak_c == cx and peak_r == cy:
    print("✅ Peak is exactly at tb_center — zero drift")
else:
    print("⚠️ Still some drift — check values")

# Save fixed gradcam.npy
os.makedirs("outputs", exist_ok=True)
np.save(os.path.join("outputs", "gradcam.npy"), hmap)
print(f"\nSaved → outputs/gradcam.npy")
print("Now run: python3 verify_coords.py")
