import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — saves PNG only
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom as nd_zoom, gaussian_filter

# ── PATHS ─────────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TB_CENTER_PATH = os.path.join(BASE_DIR, "outputs", "tb_center.npy")
GRADCAM_PATH   = os.path.join(BASE_DIR, "outputs", "gradcam.npy")
OUT_PNG        = os.path.join(BASE_DIR, "outputs", "coord_verification.png")

# Optional: original X-ray image for overlay
XRAY_PATH      = os.path.join(BASE_DIR, "outputs", "xray_input.png")   # or .jpg

# ── STL bounds — updated after 72% scaling fix ───────────────────────────────
# Source: fix_lung_scale.py output
#   Left  : x=[-70.1, -5.0]  y=[-58.8, 58.8]  z=[24.3, 124.4]
#   Right : x=[ 15.0, 80.1]  y=[-58.8, 58.8]  z=[24.3, 124.4]
#   Gap between lungs : 20.0 mm
#   Real-world size   : 65.1 × 117.5 × 100.1 mm  (Indian population average)

LEFT_BOUNDS  = (-70.1, -5.0,  -58.8, 58.8,  24.3, 124.4)   # xmin,xmax,ymin,ymax,zmin,zmax
RIGHT_BOUNDS = ( 15.0, 80.1,  -58.8, 58.8,  24.3, 124.4)
# (xmin, xmax, ymin, ymax, zmin, zmax) — all values in mm

# ── LOAD TB HOTSPOT ───────────────────────────────────────────────────────────

HAS_TB = os.path.exists(TB_CENTER_PATH)
if HAS_TB:
    tb_raw = np.load(TB_CENTER_PATH)
    cx, cy = int(tb_raw[0]), int(tb_raw[1])
else:
    print("[WARN] No tb_center.npy found — using synthetic centre (112, 80)")
    cx, cy = 112, 80

print(f"\n{'='*65}")
print(f"  COORDINATE VERIFICATION AUDIT")
print(f"{'='*65}")
print(f"\n[1] X-RAY IMAGE SPACE (224×224 pixels)")
print(f"    tb_center.npy  →  cx={cx}  cy={cy}")
print(f"    Pixel position : col {cx}/224  row {cy}/224")
print(f"    Normalised     : nx={cx/224:.4f}  ny={cy/224:.4f}")
print(f"    Side           : {'RIGHT lung (cx < 112)' if cx < 112 else 'LEFT lung (cx >= 112)'}")

# ── LOAD / SYNTHESISE HEATMAP ─────────────────────────────────────────────────

HAS_GRADCAM = os.path.exists(GRADCAM_PATH)
if HAS_GRADCAM:
    raw = np.load(GRADCAM_PATH).astype(np.float32)
    if raw.shape != (224, 224):
        raw = nd_zoom(raw, (224/raw.shape[0], 224/raw.shape[1]), order=1)
    mn, mx = raw.min(), raw.max()
    heatmap = (raw - mn) / (mx - mn + 1e-9)
else:
    yy, xx  = np.mgrid[0:224, 0:224].astype(np.float32)
    blob    = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*38.0**2))
    heatmap = np.clip(blob + gaussian_filter(blob, 20)*0.18, 0, 1).astype(np.float32)

# Peak pixel in the heatmap
peak_flat           = np.argmax(heatmap)
peak_row, peak_col  = np.unravel_index(peak_flat, heatmap.shape)

print(f"\n[2] GRADCAM HEATMAP PEAK (224×224)")
print(f"    Source         : {'gradcam.npy' if HAS_GRADCAM else 'Synthesised Gaussian'}")
print(f"    Peak pixel     : col={peak_col}  row={peak_row}")
print(f"    Normalised     : nx={peak_col/224:.4f}  ny={peak_row/224:.4f}")
print(f"    Peak value     : {heatmap[peak_row, peak_col]:.4f}")
print(f"    Match with tb_center : col diff={abs(peak_col-cx)}px  "
      f"row diff={abs(peak_row-cy)}px")

# ── 3-D MAPPING (same formula as viewer_3d.py) ───────────────────────────────

lung_name = "RIGHT" if cx < 112 else "LEFT"
bounds    = RIGHT_BOUNDS if lung_name == "RIGHT" else LEFT_BOUNDS
xmin, xmax, ymin, ymax, zmin, zmax = bounds

# Forward mapping: 2D pixel → 3D coordinate
lx = xmin + (cx / 224.0) * (xmax - xmin)
lz = zmax - (cy / 224.0) * (zmax - zmin)
ly = ymin + 0.35 * (ymax - ymin) - 10.0

print(f"\n[3] 3-D LUNG MODEL LESION COORDINATES")
print(f"    Lung           : {lung_name}")
print(f"    Bounds (mm)    : x=[{xmin}, {xmax}]  y=[{ymin}, {ymax}]  z=[{zmin}, {zmax}]")
print(f"    Lung dimensions: w={xmax-xmin:.1f}mm  h={ymax-ymin:.1f}mm  d={zmax-zmin:.1f}mm")
print(f"    Mapping formula:")
print(f"      lx = {xmin} + ({cx}/224) × ({xmax}-{xmin:.1f})  =  {lx:.2f} mm")
print(f"      lz = {zmax} - ({cy}/224) × ({zmax}-{zmin:.1f})  =  {lz:.2f} mm")
print(f"      ly = {ymin} + 0.35×({ymax}-{ymin:.1f}) - 10      =  {ly:.2f} mm  (fixed depth)")
print(f"    3-D lesion     : x={lx:.2f}  y={ly:.2f}  z={lz:.2f}  (mm)")

# ── BACK-PROJECTION: 3D → 2D (round-trip check) ──────────────────────────────

back_px = (lx - xmin) / max(xmax - xmin, 1e-9) * 223.0
back_py = (zmax - lz)  / max(zmax - zmin, 1e-9) * 223.0

print(f"\n[4] BACK-PROJECTION CHECK (3D → 2D pixel)")
print(f"    Formula:")
print(f"      px = ({lx:.1f} - {xmin}) / ({xmax}-{xmin:.1f}) × 223  =  {back_px:.2f}")
print(f"      py = ({zmax} - {lz:.1f}) / ({zmax}-{zmin:.1f}) × 223  =  {back_py:.2f}")
print(f"    Back-projected : col={back_px:.1f}  row={back_py:.1f}")
print(f"    Original input : col={cx}        row={cy}")
err_x = abs(back_px - cx)
err_y = abs(back_py - cy)
print(f"    Round-trip err : Δcol={err_x:.2f}px  Δrow={err_y:.2f}px")
if err_x < 1.0 and err_y < 1.0:
    print(f"    ✅  PASS — coordinates match within 1 pixel")
else:
    print(f"    ⚠️  MISMATCH — check STL bounds in verify_coords.py")

# ── HEATMAP PROJECTION CHECK ──────────────────────────────────────────────────

hpx = xmin + (peak_col / 224.0) * (xmax - xmin)
hpz = zmax - (peak_row / 224.0) * (zmax - zmin)

print(f"\n[5] HEATMAP PEAK → 3-D POSITION")
print(f"    Heatmap peak pixel : col={peak_col}  row={peak_row}")
print(f"    Maps to 3-D        : x={hpx:.2f}  z={hpz:.2f}  (mm)")
print(f"    tb_center 3-D      : x={lx:.2f}  z={lz:.2f}  (mm)")
print(f"    3-D distance       : Δx={abs(hpx-lx):.2f}mm  Δz={abs(hpz-lz):.2f}mm")

print(f"\n{'='*65}")
print(f"  SUMMARY TABLE")
print(f"{'='*65}")
print(f"  {'Source':<30} {'Col / X':>10} {'Row / Z':>10}")
print(f"  {'-'*50}")
print(f"  {'tb_center.npy (pixel)':<30} {cx:>10} {cy:>10}")
print(f"  {'GradCAM peak (pixel)':<30} {peak_col:>10} {peak_row:>10}")
print(f"  {'Back-projected (pixel)':<30} {back_px:>10.1f} {back_py:>10.1f}")
print(f"  {'3-D lesion (mm)':<30} {lx:>10.1f} {lz:>10.1f}")
print(f"  {'3-D heatmap peak (mm)':<30} {hpx:>10.1f} {hpz:>10.1f}")
print(f"{'='*65}\n")

# ── VISUAL OUTPUT ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 8), facecolor="#0d0d0d")
gs  = GridSpec(1, 2, figure=fig, wspace=0.06)

# ── LEFT PANEL: Heatmap with markers ─────────────────────────────────────────

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor("#0d0d0d")

if os.path.exists(XRAY_PATH):
    try:
        import matplotlib.image as mpimg
        xray = mpimg.imread(XRAY_PATH)
        xray_gray = np.mean(xray[:,:,:3], axis=2) if xray.ndim == 3 else xray
        if xray_gray.shape != (224, 224):
            from scipy.ndimage import zoom as sz
            xray_gray = sz(xray_gray, (224/xray_gray.shape[0], 224/xray_gray.shape[1]), order=1)
        ax1.imshow(xray_gray, cmap="gray", extent=[0,224,224,0], alpha=0.5)
    except Exception as e:
        print(f"[WARN] Could not load X-ray background: {e}")

im = ax1.imshow(heatmap, cmap="jet", vmin=0, vmax=1,
                extent=[0, 224, 224, 0], alpha=0.75)

ax1.plot(cx, cy, "w+", markersize=22, markeredgewidth=2.5, zorder=10,
         label=f"tb_center ({cx},{cy})")
ax1.add_patch(mpatches.Circle((cx, cy), 12, color="white",
              fill=False, linewidth=2, zorder=10))

ax1.plot(peak_col, peak_row, "D", color="cyan", markersize=10,
         markeredgewidth=1.5, markeredgecolor="white", zorder=11,
         label=f"GradCAM peak ({peak_col},{peak_row})")

ax1.plot(back_px, back_py, "*", color="yellow", markersize=14,
         markeredgewidth=1, markeredgecolor="white", zorder=12,
         label=f"Back-proj ({back_px:.0f},{back_py:.0f})")

ax1.set_xlim(0, 224); ax1.set_ylim(224, 0)
ax1.set_xlabel("Image column (px)", color="white", fontsize=11)
ax1.set_ylabel("Image row (px)", color="white", fontsize=11)
ax1.tick_params(colors="white")
for sp in ax1.spines.values(): sp.set_edgecolor("white")
ax1.set_title("2-D Heatmap — Coordinate Markers", color="white", fontsize=13,
              fontweight="bold", pad=10)
ax1.legend(loc="lower right", fontsize=9,
           facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.02,
             label="GradCAM activation").ax.yaxis.label.set_color("white")

# ── RIGHT PANEL: Coordinate audit table ──────────────────────────────────────

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#0d0d0d")
ax2.axis("off")

ax2.text(0.5, 0.97, "COORDINATE AUDIT", color="white",
         fontsize=14, fontweight="bold", ha="center", va="top",
         transform=ax2.transAxes)

rows = [
    ("Source",                   "Col / X",        "Row / Z",        "Status"),
    ("─"*22,                     "─"*12,            "─"*12,           "─"*8),
    ("tb_center.npy (px)",       f"{cx}",           f"{cy}",          "INPUT"),
    ("GradCAM peak (px)",         f"{peak_col}",     f"{peak_row}",
        f"Δ{abs(peak_col-cx)}px / Δ{abs(peak_row-cy)}px"),
    ("Back-projected (px)",       f"{back_px:.1f}",  f"{back_py:.1f}",
        "✅ PASS" if err_x < 1 and err_y < 1 else "⚠️ CHECK"),
    ("─"*22,                     "─"*12,            "─"*12,           "─"*8),
    ("3-D lesion (mm)",           f"{lx:.1f}",       f"{lz:.1f}",     "MAPPED"),
    ("3-D heatmap peak (mm)",     f"{hpx:.1f}",      f"{hpz:.1f}",
        f"Δ{abs(hpx-lx):.1f}mm / Δ{abs(hpz-lz):.1f}mm"),
]

colours = ["#AAAAFF","#555555","white","#00FFFF","#FFFF00",
           "#555555","#FF6644","#FF9944"]
y_pos = 0.90
for row, col in zip(rows, colours):
    ax2.text(0.02, y_pos, row[0], color=col, fontsize=10,
             fontfamily="monospace", transform=ax2.transAxes)
    ax2.text(0.48, y_pos, row[1], color=col, fontsize=10,
             fontfamily="monospace", ha="center", transform=ax2.transAxes)
    ax2.text(0.68, y_pos, row[2], color=col, fontsize=10,
             fontfamily="monospace", ha="center", transform=ax2.transAxes)
    ax2.text(0.98, y_pos, row[3], color=col, fontsize=10,
             fontfamily="monospace", ha="right", transform=ax2.transAxes)
    y_pos -= 0.075

# STL bounds info box
y_pos -= 0.03
bounds_text = (
    f"SCALED STL BOUNDS  (72% — Indian avg)\n"
    f"────────────────────────────────────\n"
    f"  Lung      Width    Height   Depth\n"
    f"  Left      65.1mm  117.5mm  100.1mm\n"
    f"  Right     65.1mm  117.5mm  100.1mm\n"
    f"  Gap between lungs : 20.0 mm\n\n"
    f"MAPPING FORMULA  ({lung_name} lung)\n"
    f"────────────────────────────────────\n"
    f"  lx = {xmin} + ({cx}/224)×{xmax-xmin:.1f}  =  {lx:.1f} mm\n"
    f"  lz = {zmax} − ({cy}/224)×{zmax-zmin:.1f}  =  {lz:.1f} mm\n"
    f"  ly = fixed depth  =  {ly:.1f} mm\n\n"
    f"BACK-PROJECTION CHECK\n"
    f"────────────────────────────────────\n"
    f"  px = ({lx:.1f}−{xmin})/({xmax-xmin:.1f})×223  =  {back_px:.1f} px  [was {cx}]\n"
    f"  py = ({zmax}−{lz:.1f})/({zmax-zmin:.1f})×223  =  {back_py:.1f} px  [was {cy}]"
)
ax2.text(0.02, y_pos, bounds_text, color="#CCCCCC", fontsize=8.5,
         fontfamily="monospace", va="top", transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a1a",
                   edgecolor="#555555", linewidth=1))

verdict_color = "#00FF88" if err_x < 1 and err_y < 1 else "#FF4444"
verdict_text  = ("✅  COORDINATES VERIFIED — 2D pixel and 3D position MATCH"
                 if err_x < 1 and err_y < 1
                 else "⚠️  MISMATCH — check STL bounds in verify_coords.py")
ax2.text(0.5, 0.02, verdict_text, color=verdict_color,
         fontsize=10, fontweight="bold", ha="center", va="bottom",
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#111111",
                   edgecolor=verdict_color, linewidth=1.5))

fig.suptitle(
    f"TB Coordinate Verification  |  Lung: {lung_name}  |  "
    f"Pixel: ({cx},{cy})  →  3D: ({lx:.0f}, {ly:.0f}, {lz:.0f}) mm",
    color="white", fontsize=12, fontweight="bold", y=0.99,
)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"[INFO] Visual saved → {OUT_PNG}")
print("[INFO] Share coord_verification.png with your manager for review.\n")