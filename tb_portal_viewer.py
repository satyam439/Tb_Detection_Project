import os, sys, time, threading
import numpy as np
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
from scipy.ndimage import zoom as nd_zoom

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEFT_SHELL  = os.path.join(BASE_DIR, "lung_model", "left_lung_shell.stl")
RIGHT_SHELL = os.path.join(BASE_DIR, "lung_model", "right_lung_shell.stl")
LEFT_TREE   = os.path.join(BASE_DIR, "lung_model", "left_lung_tree.stl")
RIGHT_TREE  = os.path.join(BASE_DIR, "lung_model", "right_lung_tree.stl")
LEFT_RAW    = os.path.join(BASE_DIR, "lung_model", "left_lung.stl")
RIGHT_RAW   = os.path.join(BASE_DIR, "lung_model", "right_lung.stl")

TB_CENTER_PATH = os.path.join(BASE_DIR, "outputs", "tb_center.npy")
GRADCAM_PATH   = os.path.join(BASE_DIR, "outputs", "gradcam.npy")
VIDEO_PATH     = os.path.join(BASE_DIR, "outputs", "lung_rotation.mp4")

DIAGRAM_PATH   = os.path.join(BASE_DIR, "outputs", "lung_annotated_diagram.png")
VERIFY_PATH    = os.path.join(BASE_DIR, "outputs", "coord_verification.png")
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

R_W, R_H, R_D = 95.0, 220.0, 130.0
L_W, L_H, L_D = 85.0, 210.0, 120.0

PUSH_MM  = 14.0
TARGET_D = 125.0

def speak_async(text):
    def _speak():
        try:
            import pyttsx3; e = pyttsx3.init()
            e.setProperty("rate", 155); e.say(text); e.runAndWait(); return
        except Exception: pass
        if sys.platform == "darwin": os.system(f'say "{text}"'); return
        os.system(f'espeak "{text}" 2>/dev/null')
    threading.Thread(target=_speak, daemon=True).start()

USE_SPLIT = os.path.exists(LEFT_SHELL) and os.path.exists(RIGHT_SHELL)
if not USE_SPLIT:
    print("[WARNING] Shell files not found — falling back to raw STL.")
    if not os.path.exists(LEFT_RAW) or not os.path.exists(RIGHT_RAW):
        print("[ERROR] No lung STL files found."); sys.exit(1)

TB_CENTERS_PATH = os.path.join(BASE_DIR, "outputs", "tb_centers.npy")

raw_regions = []  # list of dicts: {cx, cy, peak, mean, area_px}

if os.path.exists(TB_CENTERS_PATH):
    arr = np.load(TB_CENTERS_PATH)
    for row in arr:
        raw_regions.append({
            "cx": float(row[0]), "cy": float(row[1]),
            "peak": float(row[2]) if len(row) > 2 else 1.0,
            "mean": float(row[3]) if len(row) > 3 else 1.0,
            "area_px": int(row[4]) if len(row) > 4 else 0,
        })
    print(f"[INFO] Loaded {len(raw_regions)} infection region(s) from tb_centers.npy")
elif os.path.exists(TB_CENTER_PATH):
    tb_raw = np.load(TB_CENTER_PATH)
    raw_regions.append({
        "cx": float(tb_raw[0]), "cy": float(tb_raw[1]),
        "peak": 1.0, "mean": 1.0, "area_px": 0,
    })
    print("[INFO] Loaded single legacy TB hotspot from tb_center.npy")

raw_regions.sort(key=lambda r: r["peak"], reverse=True)
HAS_TB = len(raw_regions) > 0

if HAS_TB:
    # cx, cy kept for anything downstream that still expects a single
    # "primary" hotspot — this is the strongest region (highest peak).
    cx, cy = int(round(raw_regions[0]["cx"])), int(round(raw_regions[0]["cy"]))
    print(f"[INFO] Primary TB hotspot: cx={cx}  cy={cy}  "
          f"({len(raw_regions)} region(s) total)")
else:
    cx = cy = None
    print("[INFO] Healthy mode.")


HAS_GRADCAM = os.path.exists(GRADCAM_PATH)
if HAS_GRADCAM:
    raw = np.load(GRADCAM_PATH).astype(np.float32)
    if raw.shape != (224, 224):
        raw = nd_zoom(raw, (224/raw.shape[0], 224/raw.shape[1]), order=1)
    mn, mx = raw.min(), raw.max()
    HEATMAP = (raw - mn) / (mx - mn + 1e-9)
    print("[INFO] GradCAM loaded")
elif HAS_TB:
    yy, xx  = np.mgrid[0:224, 0:224].astype(np.float32)
    peak    = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*20.0**2))
    glow    = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*40.0**2)) * 0.45
    HEATMAP = np.clip(peak + glow, 0, 1).astype(np.float32)
    HEATMAP /= HEATMAP.max()
else:
    HEATMAP = np.zeros((224, 224), dtype=np.float32)

def load_shell(path, side):
    mesh = pv.read(path)
    mesh = mesh.connectivity(extraction_mode='largest')
    mesh = mesh.extract_surface(algorithm='dataset_surface')
    mesh = mesh.clean()
    mesh = mesh.smooth(n_iter=300, relaxation_factor=0.04)
    mesh.compute_normals(inplace=True, auto_orient_normals=True)

    tw = R_W if side == 'right' else L_W
    th = R_H if side == 'right' else L_H
    td = R_D if side == 'right' else L_D

    b        = mesh.bounds
    raw_w    = b[1]-b[0]; raw_h = b[3]-b[2]; raw_d = b[5]-b[4]
    sf = min(tw/max(raw_w,1e-6), th/max(raw_h,1e-6), td/max(raw_d,1e-6))
    c  = np.array(mesh.center)
    pts = mesh.points.copy()
    pts[:,0] = c[0] + (pts[:,0]-c[0]) * sf
    pts[:,1] = c[1] + (pts[:,1]-c[1]) * sf
    pts[:,2] = c[2] + (pts[:,2]-c[2]) * sf
    mesh.points = pts

    b2 = mesh.bounds
    mesh.translate([0, -(b2[2]+b2[3])/2, -(b2[4]+b2[5])/2], inplace=True)

    b3 = mesh.bounds
    if side == 'left':
        mesh.translate([-b3[1] + PUSH_MM, 0, 0], inplace=True)
    else:
        mesh.translate([-b3[0] - PUSH_MM, 0, 0], inplace=True)

    mesh.compute_normals(inplace=True, auto_orient_normals=True)
    b4 = mesh.bounds
    print(f"[INFO] {side.upper()} raw=({raw_w:.0f}x{raw_h:.0f}x{raw_d:.0f})  "
          f"sf={sf:.3f}  final W={b4[1]-b4[0]:.1f} H={b4[3]-b4[2]:.1f} "
          f"D={b4[5]-b4[4]:.1f} mm  (target {tw}x{th}x{td})")
    return mesh

src_l = LEFT_SHELL  if USE_SPLIT else LEFT_RAW
src_r = RIGHT_SHELL if USE_SPLIT else RIGHT_RAW

print("[INFO] Loading left lung ...")
left_shell  = load_shell(src_l, 'left')
print("[INFO] Loading right lung ...")
right_shell = load_shell(src_r, 'right')

lb, rb = left_shell.bounds, right_shell.bounds
print(f"[INFO] Gap at midline: {rb[0]-lb[1]:.1f} mm  "
      f"({'overlap=no gap' if rb[0]-lb[1]<0 else 'gap present'})")

def load_tree(path, ref_shell):
    if not os.path.exists(path):
        print(f"[INFO] Tree not found: {path}"); return None
    tree = pv.read(path)
    if tree.n_points < 100:
        print(f"[WARN] Tree too small — skip"); return None

    sb = ref_shell.bounds
    tb = tree.bounds

    if (tb[1]-tb[0] > (sb[1]-sb[0])*1.5 or
        tb[3]-tb[2] > (sb[3]-sb[2])*1.5 or
        tb[5]-tb[4] > (sb[5]-sb[4])*1.5):
        print(f"[WARN] Tree much larger than shell — skipping"); return None

    margin = 5.0
    sf = min(
        ((sb[1]-sb[0])-2*margin) / max(tb[1]-tb[0], 1e-6),
        ((sb[3]-sb[2])-2*margin) / max(tb[3]-tb[2], 1e-6),
        ((sb[5]-sb[4])-2*margin) / max(tb[5]-tb[4], 1e-6),
        0.85
    )

    c  = np.array(tree.center)
    cs = np.array(ref_shell.center)
    pts = tree.points.copy()
    pts[:,0] = c[0] + (pts[:,0]-c[0])*sf
    pts[:,1] = c[1] + (pts[:,1]-c[1])*sf
    pts[:,2] = c[2] + (pts[:,2]-c[2])*sf
    tree.points = pts

    ct = np.array(tree.center)
    tree.translate(list(cs-ct), inplace=True)

    b = ref_shell.bounds
    tree = tree.clip_box(
        [b[0]+4, b[1]-4, b[2]+4, b[3]-4, b[4]+4, b[5]-4], invert=False)

    if tree.n_points < 50:
        print("[WARN] Tree clipped away — skip"); return None

    tb2 = tree.bounds
    if (tb2[0] < sb[0] or tb2[1] > sb[1] or
        tb2[2] < sb[2] or tb2[3] > sb[3] or
        tb2[4] < sb[4] or tb2[5] > sb[5]):
        print("[WARN] Tree still outside shell — skip"); return None

    print(f"[INFO] Tree OK: x=[{tb2[0]:.1f},{tb2[1]:.1f}]  pts={tree.n_points}")
    return tree

print("[INFO] Loading bronchi trees ...")
left_tree  = load_tree(LEFT_TREE,  left_shell)
right_tree = load_tree(RIGHT_TREE, right_shell)
print(f"[INFO] Trees: L={'OK' if left_tree else 'SKIP'}  R={'OK' if right_tree else 'SKIP'}")

PIVOT = np.array(left_shell.merge(right_shell).center)
print(f"[INFO] Pivot: {[round(v,1) for v in PIVOT]}")

lobe = "N/A"; lung_name = "N/A"; lesion_3d = None; target_lung = None
regions_3d = []  # list of dicts: {lung, lung_name, lobe, lesion_3d, peak, mean, area_px, px}

def _lung_medial_lateral_x(bounds):
    """
    Determine which side of a lung's x-bounds is medial (near the
    shared midline at x=0, toward the mediastinum) vs lateral (far
    from the midline, toward the outer chest wall). Done by comparing
    absolute values rather than assuming xmin/xmax map to a fixed
    side — this is geometry-agnostic, so it doesn't depend on which
    way the raw STL happened to be oriented.
    """
    xmin, xmax = bounds[0], bounds[1]
    if abs(xmax) < abs(xmin):
        return xmax, xmin   # medial, lateral
    return xmin, xmax        # medial, lateral


def _pixel_x_to_mm(rcx, lung_name, bounds):
    """
    Map a 2D pixel column to an X position (mm) within the correct
    lung — normalized to the HALF of the 224px image that actually
    belongs to that lung (0-112 for RIGHT, 112-224 for LEFT), not the
    full 0-224 range.

    BUG THIS FIXES: the old formula did
        lx = xmin + (rcx/224.0)*(xmax-xmin)
    using the full image width as the denominator even though rcx was
    already constrained to one half of it. For the LEFT lung, rcx only
    ever ranges 112-224, so rcx/224.0 only ever produced 0.5-1.0 —
    meaning every left-lung lesion was mathematically compressed into
    only the medial (near-mediastinum) half of the lung; the lateral
    half of the lung could never be reached, regardless of where the
    real hotspot was in the 2D heatmap. Same issue for the right lung
    mapping into the wrong half.
    """
    medial_x, lateral_x = _lung_medial_lateral_x(bounds)
    if lung_name == "RIGHT":
        frac_x = np.clip(rcx / 112.0, 0.0, 1.0)             # 0=lateral edge, 1=midline
        return lateral_x + frac_x * (medial_x - lateral_x)
    else:
        frac_x = np.clip((rcx - 112.0) / 112.0, 0.0, 1.0)   # 0=midline, 1=lateral edge
        return medial_x + frac_x * (lateral_x - medial_x)


def _mm_x_to_pixel(lx, lung_name, bounds):
    """Inverse of _pixel_x_to_mm — used by the coordinate-verification audit."""
    medial_x, lateral_x = _lung_medial_lateral_x(bounds)
    if lung_name == "RIGHT":
        denom = medial_x - lateral_x
        denom = denom if abs(denom) > 1e-9 else 1e-9
        frac_x = (lx - lateral_x) / denom
        return frac_x * 112.0
    else:
        denom = lateral_x - medial_x
        denom = denom if abs(denom) > 1e-9 else 1e-9
        frac_x = (lx - medial_x) / denom
        return 112.0 + frac_x * 112.0


def _classify_region(rcx, rcy):
    """Given one region's 2D pixel coords, determine its lung, lobe, and 3D position."""
    if rcx < 112: t_lung, l_name = right_shell, "RIGHT"
    else:         t_lung, l_name = left_shell,  "LEFT"

    xmin,xmax,ymin,ymax,zmin,zmax = t_lung.bounds
    lx = _pixel_x_to_mm(rcx, l_name, t_lung.bounds)

    # FIX: cy (image row from the 2D heatmap) must drive the VERTICAL axis
    # (Y = apex-to-base), since Y is what's actually rendered as up/down on
    # screen (rotate_y, camera up=(0,1,0)). It was previously routed into Z
    # (depth) while Y was hardcoded to the lung midpoint, so the 3D marker
    # never moved vertically no matter where the heatmap hotspot actually was.
    ly = ymax - (rcy/224.0)*(ymax-ymin)   # small cy (top of image) -> high Y (apex)

    # Depth (Z) is fixed at mid-depth — a single 2D X-ray has no real depth
    # data to place a lesion front-to-back, so this doesn't pretend otherwise.
    lz = (zmin + zmax) / 2.0

    l3d = np.array([float(lx), float(ly), float(lz)], dtype=np.float64)

    # Lobe classification now uses the SAME axis (Y) that actually moves the
    # marker, so the label and the dot's position always agree with each other.
    y_pct = (ly-ymin)/max(ymax-ymin,1e-9)
    if l_name == "RIGHT":
        lb_ = ("RIGHT UPPER LOBE"  if y_pct>0.66 else
               "RIGHT MIDDLE LOBE" if y_pct>0.33 else "RIGHT LOWER LOBE")
    else:
        lb_ = "LEFT UPPER LOBE" if y_pct>0.50 else "LEFT LOWER LOBE"

    return t_lung, l_name, lb_, l3d


if HAS_TB:
    for r in raw_regions:
        t_lung, l_name, lb_, l3d = _classify_region(r["cx"], r["cy"])
        regions_3d.append({
            "lung": t_lung, "lung_name": l_name, "lobe": lb_,
            "lesion_3d": l3d, "peak": r["peak"], "mean": r["mean"],
            "area_px": r["area_px"], "px": (r["cx"], r["cy"]),
        })

    # "Primary" region = strongest peak (raw_regions is already sorted,
    # so regions_3d[0] is it). Everything below that historically used
    # a single target_lung/lung_name/lobe/lesion_3d keeps working
    # unchanged, now driven by the primary region.
    target_lung = regions_3d[0]["lung"]
    lung_name   = regions_3d[0]["lung_name"]
    lobe        = regions_3d[0]["lobe"]
    lesion_3d   = regions_3d[0]["lesion_3d"]

    print(f"[INFO] {len(regions_3d)} region(s) classified:")
    for i, rg in enumerate(regions_3d):
        tag = " (PRIMARY)" if i == 0 else ""
        print(f"  [{i+1}] {rg['lung_name']} | {rg['lobe']}{tag} | "
              f"3D=({rg['lesion_3d'][0]:.1f},{rg['lesion_3d'][1]:.1f},{rg['lesion_3d'][2]:.1f})mm "
              f"peak={rg['peak']:.2f} area={rg['area_px']}px")

INITIAL_LUNG_BOUNDS = tuple(float(v) for v in target_lung.bounds) if target_lung is not None else None
INITIAL_LESION_3D   = tuple(float(v) for v in lesion_3d)          if lesion_3d   is not None else None

if HAS_TB and lesion_3d is not None and target_lung is not None:
    _b = INITIAL_LUNG_BOUNDS

    ANT_Z    = _b[4]
    POST_Z   = _b[5]
    LUNG_AP  = POST_Z - ANT_Z

    DEPTH_FROM_ANT_MM  = lesion_3d[2] - ANT_Z
    DEPTH_FROM_POST_MM = POST_Z - lesion_3d[2]
    DEPTH_PCT          = (DEPTH_FROM_ANT_MM / max(LUNG_AP, 1e-6)) * 100.0
    DIST_FROM_CARINA   = float(np.linalg.norm(lesion_3d - PIVOT))

    print()
    print(f"  [DEPTH] Anterior surface Z  : {ANT_Z:8.1f} mm")
    print(f"  [DEPTH] Lesion Z            : {lesion_3d[2]:8.1f} mm")
    print(f"  [DEPTH] Posterior surface Z : {POST_Z:8.1f} mm")
    print(f"  [DEPTH] Depth from anterior : {DEPTH_FROM_ANT_MM:8.1f} mm  ({DEPTH_PCT:.0f}% into lung)")
    print(f"  [DEPTH] Depth from posterior: {DEPTH_FROM_POST_MM:8.1f} mm")
    print(f"  [DEPTH] Total lung A-P      : {LUNG_AP:8.1f} mm")
    print(f"  [DEPTH] Distance from carina: {DIST_FROM_CARINA:8.1f} mm")
    print()
else:
    ANT_Z = POST_Z = LUNG_AP = 0.0
    DEPTH_FROM_ANT_MM = DEPTH_FROM_POST_MM = DEPTH_PCT = DIST_FROM_CARINA = 0.0

depth_estimates = []

def log_depth_report(angle_int):
    if not HAS_TB or lesion_3d is None: return

    lx_o, ly_o, lz_o = INITIAL_LESION_3D
    theta = np.deg2rad(angle_int % 360)
    s, c  = np.sin(theta), np.cos(theta)
    dx0   = lx_o - PIVOT[0];  dz0 = lz_o - PIVOT[2]
    rot_x = PIVOT[0] + dx0*c - dz0*s
    rot_z = PIVOT[2] + dx0*s + dz0*c
    depth_estimates.append(DEPTH_FROM_ANT_MM)

    labels = {90: "RIGHT LAT", 180: "BACK", 270: "LEFT LAT", 360: "FRONT"}
    label  = labels.get(angle_int, "")
    W      = 52

    print()
    print(f"  +{'-'*W}+")
    print(f"  | DEPTH @ {angle_int:3d} deg  [{label:<9s}]{' '*(W-24)}|")
    print(f"  +{'-'*W}+")
    print(f"  | {'Lesion world pos (fixed)':<22s}  x={lx_o:7.1f}  y={ly_o:7.1f}  z={lz_o:7.1f} mm  |")
    print(f"  | {'If it had rotated':<22s}  x={rot_x:7.1f}            z={rot_z:7.1f} mm  |")
    print(f"  | {'Depth from anterior':<22s}  {DEPTH_FROM_ANT_MM:7.1f} mm  ({DEPTH_PCT:4.0f}% into lung){' '*6}|")
    print(f"  | {'Depth from posterior':<22s}  {DEPTH_FROM_POST_MM:7.1f} mm{' '*20}|")
    print(f"  | {'Total A-P lung depth':<22s}  {LUNG_AP:7.1f} mm{' '*20}|")
    print(f"  | {'Distance from carina':<22s}  {DIST_FROM_CARINA:7.1f} mm{' '*20}|")
    print(f"  +{'-'*W}+")

def make_3d_heat_multi(mesh, regions_for_this_lung):
    """
    Sum (max-combine) Gaussian heat blobs for ALL infection regions
    belonging to this lung mesh — replaces the old single-blob version
    so multiple hot regions in the same lung (or across both lungs)
    are each rendered distinctly instead of only the strongest one
    surviving.
    """
    pts = mesh.points
    total = np.zeros(len(pts), dtype=np.float32)
    if not regions_for_this_lung:
        return total
    xmin,xmax = mesh.bounds[0], mesh.bounds[1]
    for rg in regions_for_this_lung:
        # Size THIS region's own blob by its own detected pixel area,
        # not the whole heatmap's active-pixel count as before — so a
        # small secondary region doesn't get sized as if it were the
        # large primary one.
        area_px = max(rg["area_px"], 20)
        sigma_xz = np.clip(np.sqrt(area_px/np.pi)*(xmax-xmin)/224.0, 10.0, 28.0)
        sigma_y  = np.clip(sigma_xz*0.55, 7.0, 18.0)
        lx,ly,lz = rg["lesion_3d"]
        dx=pts[:,0]-lx; dy=pts[:,1]-ly; dz=pts[:,2]-lz
        h = np.exp(-(dx**2/(2*sigma_xz**2) +
                     dy**2/(2*sigma_y**2)  +
                     dz**2/(2*sigma_xz**2))).astype(np.float32)
        h = np.clip(h, 0.0, 1.0) ** 0.5
        h *= rg["peak"]   # weight by this region's own activation strength
        total = np.maximum(total, h)   # max-combine so overlapping blobs don't oversaturate
        print(f"[INFO] Heat blob ({rg['lung_name']} {rg['lobe']}) "
              f"sigma_xz={sigma_xz:.1f}mm  sigma_y={sigma_y:.1f}mm  peak={h.max():.3f}")
    return total

left_regions  = [rg for rg in regions_3d if rg["lung_name"] == "LEFT"]
right_regions = [rg for rg in regions_3d if rg["lung_name"] == "RIGHT"]

left_shell["heat"]  = make_3d_heat_multi(left_shell,  left_regions)
right_shell["heat"] = make_3d_heat_multi(right_shell, right_regions)


def generate_coord_verification():
    print("[INFO] Generating coordinate verification diagram ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 8), facecolor="#0d0d0d")
        gs  = GridSpec(1, 2, figure=fig, wspace=0.06)

        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor("#0d0d0d")
        im = ax1.imshow(HEATMAP, cmap="jet", vmin=0, vmax=1,
                        extent=[0,224,224,0], alpha=0.85)

        # Mark and back-project EVERY detected region, not just the
        # primary one — a distinct color per region so this diagram
        # stays consistent with what the 3D viewer actually shows.
        REGION_COLORS = ["white", "#FF8800", "#00CCFF", "#00FF88", "#FF44FF"]
        region_audit = []  # (region_idx, cx, cy, bpx, bpy, err_x, err_y)

        if HAS_TB and regions_3d:
            xmin_,xmax_,_,_,zmin_,zmax_ = INITIAL_LUNG_BOUNDS
            for i, rg in enumerate(regions_3d):
                col = REGION_COLORS[i % len(REGION_COLORS)]
                rcx, rcy = rg["px"]
                is_primary = (i == 0)
                marker_size = 22 if is_primary else 16

                ax1.plot(rcx, rcy, "+", color=col, markersize=marker_size,
                          markeredgewidth=2.5 if is_primary else 1.8, zorder=10,
                          label=f"Region {i+1} ({rcx:.0f},{rcy:.0f})")
                ax1.add_patch(mpatches.Circle((rcx,rcy), 12 if is_primary else 9,
                              color=col, fill=False,
                              linewidth=2 if is_primary else 1.4, zorder=10))

                # Back-project this region's own 3D position, using its
                # own lung's bounds (regions can be in different lungs)
                # and the CORRECT inverse of the fixed forward mapping.
                r_lung = rg["lung"]
                rxmin_,rxmax_,_,_,rzmin_,rzmax_ = r_lung.bounds
                bpx = _mm_x_to_pixel(rg["lesion_3d"][0], rg["lung_name"], r_lung.bounds)
                bpy = (rzmax_-rg["lesion_3d"][2])/max(rzmax_-rzmin_,1e-9)*223
                ax1.plot(bpx, bpy, "*", color=col,
                          markersize=13 if is_primary else 10,
                          markeredgecolor="white", zorder=12)
                err_x = abs(bpx-rcx); err_y = abs(bpy-rcy)
                region_audit.append((i, rcx, rcy, bpx, bpy, err_x, err_y, rg))

        ax1.set_xlim(0,224); ax1.set_ylim(224,0)
        ax1.set_xlabel("Image column (px)",color="white",fontsize=11)
        ax1.set_ylabel("Image row (px)",color="white",fontsize=11)
        ax1.tick_params(colors="white")
        for sp in ax1.spines.values(): sp.set_edgecolor("white")
        ax1.set_title("2-D GradCAM Heatmap — Coordinate Markers (all regions)",
                      color="white",fontsize=13,fontweight="bold")
        if HAS_TB and regions_3d:
            ax1.legend(loc="lower right",fontsize=8,
                       facecolor="#1a1a1a",edgecolor="white",labelcolor="white")
        plt.colorbar(im,ax=ax1,fraction=0.035,pad=0.02,
                     label="GradCAM activation").ax.yaxis.label.set_color("white")

        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor("#0d0d0d"); ax2.axis("off")
        ax2.text(0.5,0.97,"COORDINATE AUDIT",color="white",
                 fontsize=14,fontweight="bold",ha="center",va="top",
                 transform=ax2.transAxes)

        lines = [
            ("Coordinate System","","",""),
            ("  X","Right <-> Left","horizontal",""),
            ("  Y","Inf <-> Sup","VERTICAL (up)","Rotation axis"),
            ("  Z","Post <-> Ant","DEPTH","Camera axis"),
            ("","","",""),
            ("Source","Col/X","Row/Z","Status"),
        ]

        all_pass = True
        if HAS_TB and region_audit:
            for i, rcx, rcy, bpx, bpy, err_x, err_y, rg in region_audit:
                pass_fail = "PASS" if err_x<1 and err_y<1 else "CHECK"
                if pass_fail != "PASS":
                    all_pass = False
                tag = " (PRI)" if i == 0 else f" ({i+1})"
                lines += [
                    (f"Region{tag} px", f"{rcx:.0f}", f"{rcy:.0f}", "INPUT"),
                    (f"  Back-proj px", f"{bpx:.1f}", f"{bpy:.1f}", pass_fail),
                    (f"  3D lesion (mm)", f"{rg['lesion_3d'][0]:.1f}", f"{rg['lesion_3d'][2]:.1f}", "MAPPED"),
                    (f"  Lung / Lobe", rg["lung_name"], rg["lobe"][:12], ""),
                    ("","","",""),
                ]
            lines += [
                (f"Carina dist (primary)", f"{DIST_FROM_CARINA:.1f}mm", "", ""),
            ]

        colours=["#AAAAFF","#CCCCFF","#88FFAA","#6699FF","white","#AAAAFF"]
        y_pos=0.86
        for i,row in enumerate(lines):
            col = colours[i] if i < len(colours) else "white"
            fs = 9 if i < 6 else 8
            ax2.text(0.02,y_pos,row[0],color=col,fontsize=fs,
                     fontfamily="monospace",transform=ax2.transAxes)
            ax2.text(0.46,y_pos,row[1],color=col,fontsize=fs,
                     fontfamily="monospace",ha="center",transform=ax2.transAxes)
            ax2.text(0.70,y_pos,row[2],color=col,fontsize=fs,
                     fontfamily="monospace",ha="center",transform=ax2.transAxes)
            ax2.text(0.98,y_pos,row[3],color=col,fontsize=fs,
                     fontfamily="monospace",ha="right",transform=ax2.transAxes)
            y_pos -= 0.052

        n_regions = len(regions_3d) if HAS_TB else 0
        verdict = (f"ALL {n_regions} REGION(S) VERIFIED -- 2D pixels match 3D positions"
                   if HAS_TB and all_pass
                   else (f"CHECK: {n_regions} region(s), some mismatched"
                         if HAS_TB else "Healthy scan -- no coordinates to verify"))
        vcol = "#00FF88" if HAS_TB and all_pass else ("#FFAA00" if HAS_TB else "#888888")
        ax2.text(0.5,0.02,verdict,color=vcol,fontsize=10,
                 fontweight="bold",ha="center",va="bottom",transform=ax2.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4",facecolor="#111111",
                           edgecolor=vcol,linewidth=1.5))

        title = (f"TB Coord Verification | {n_regions} region(s) | Primary: {lung_name} "
                 f"Pixel:({cx},{cy}) -> 3D:({lesion_3d[0]:.0f},{lesion_3d[1]:.0f},{lesion_3d[2]:.0f})mm"
                 if HAS_TB and lesion_3d is not None else "Coord Verification — Healthy scan")
        fig.suptitle(title,color="white",fontsize=11,y=0.99)
        plt.savefig(VERIFY_PATH,dpi=150,bbox_inches="tight",facecolor="#0d0d0d")
        plt.close()
        print(f"[INFO] Coord verification saved -> {VERIFY_PATH}")
    except Exception as e:
        print(f"[WARN] Coord verification failed: {e}")


def generate_lung_diagram():
    print("[INFO] Generating annotated lung diagram ...")
    import traceback
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Ellipse, FancyBboxPatch
        import matplotlib.patheffects as pe
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        from matplotlib import cm

        fig, ax = plt.subplots(figsize=(20, 15), facecolor="#04060A")
        ax.set_facecolor("#04060A")
        ax.set_xlim(-175, 175); ax.set_ylim(-135, 148)
        ax.set_aspect('equal'); ax.axis('off')

        def draw_lung_shape(cx_d, base_y, top_y, half_w, side, fill, edge, alpha=0.38):
            h = top_y - base_y
            if side == 'left':
                xs = [cx_d+half_w*0.08, cx_d+half_w*0.05, cx_d-half_w*0.9,
                      cx_d-half_w,       cx_d-half_w*0.95, cx_d-half_w*0.7,
                      cx_d-half_w*0.25,  cx_d+half_w*0.10, cx_d+half_w*0.08]
                ys = [base_y, base_y-h*0.03, base_y+h*0.05,
                      base_y+h*0.25,     base_y+h*0.55,    base_y+h*0.82,
                      top_y,             top_y,             base_y]
            else:
                xs = [cx_d-half_w*0.08, cx_d-half_w*0.05, cx_d+half_w*0.9,
                      cx_d+half_w,       cx_d+half_w*0.95, cx_d+half_w*0.7,
                      cx_d+half_w*0.25,  cx_d-half_w*0.10, cx_d-half_w*0.08]
                ys = [base_y, base_y-h*0.03, base_y+h*0.05,
                      base_y+h*0.25,     base_y+h*0.55,    base_y+h*0.82,
                      top_y,             top_y,             base_y]
            ax.fill(xs, ys, color=fill, alpha=alpha, zorder=2)
            ax.plot(xs, ys, color=edge, linewidth=2.0, zorder=3)

        L_CX=-60; R_CX=60; BASE_Y=-92; TOP_Y=118; HW=50

        draw_lung_shape(R_CX,BASE_Y,TOP_Y,HW,'right',"#1A4A6A","#4AAED8")
        draw_lung_shape(L_CX,BASE_Y,TOP_Y,HW,'left', "#1A4A6A","#4AAED8")

        ax.plot([R_CX+HW*0.35,R_CX-HW*0.05,R_CX-HW*0.30],
                [BASE_Y+12,BASE_Y+68,TOP_Y-8],
                color="#7ACCE0",linewidth=1.5,linestyle='--',alpha=0.8,zorder=4)
        ax.plot([R_CX-HW*0.05,R_CX+HW*0.90],
                [BASE_Y+100,BASE_Y+82],
                color="#7ACCE0",linewidth=1.5,linestyle='--',alpha=0.8,zorder=4)
        ax.plot([L_CX-HW*0.35,L_CX+HW*0.05,L_CX+HW*0.30],
                [BASE_Y+12,BASE_Y+68,TOP_Y-8],
                color="#7ACCE0",linewidth=1.5,linestyle='--',alpha=0.8,zorder=4)

        ax.plot([0,0],[TOP_Y-2,TOP_Y+28],color="#A0C8E0",linewidth=6,zorder=3)
        ax.plot([-20,20],[TOP_Y-2,TOP_Y-2],color="#A0C8E0",linewidth=5,zorder=3)
        ax.plot([-20,L_CX+8],[TOP_Y-2,TOP_Y-38],color="#A0C8E0",linewidth=3,zorder=3)
        ax.plot([20,R_CX-8],[TOP_Y-2,TOP_Y-38],color="#A0C8E0",linewidth=3,zorder=3)

        dx=np.linspace(-135,135,100)
        dy=BASE_Y-10+14*np.sin(np.pi*(dx+135)/270)
        ax.plot(dx,dy,color="#5A8A9A",linewidth=2.5,alpha=0.7,zorder=3)
        ax.fill_between(dx,dy-22,dy,color="#1A3A4A",alpha=0.45,zorder=2)

        for hx,hy in [(L_CX+10,BASE_Y+68),(R_CX-10,BASE_Y+68)]:
            hx, hy = float(hx), float(hy)
            if not (np.isfinite(hx) and np.isfinite(hy)):
                continue
            ax.add_patch(Ellipse((hx,hy),14,24,color="#2A6A8A",alpha=0.7,zorder=4))
            ax.add_patch(Ellipse((hx,hy),14,24,fill=False,
                                 edgecolor="#7ACCE0",linewidth=1.5,zorder=5))

        region_diagram_pts = []  # (region_idx, ldx, ldy, rg) for later annotation
        if HAS_TB and regions_3d and INITIAL_LUNG_BOUNDS is not None:
            for i, rg in enumerate(regions_3d):
                r_l3d = rg["lesion_3d"]
                r_lung_bounds = rg["lung"].bounds
                rxmin_,rxmax_,_,_,rzmin_,rzmax_ = r_lung_bounds
                tw = R_W if rg["lung_name"]=="RIGHT" else L_W
                r_cx_diag = R_CX if rg["lung_name"]=="RIGHT" else L_CX
                r_ldx = r_cx_diag + (r_l3d[0]-(rxmin_+rxmax_)/2)/(tw/2)*HW*0.82
                r_ldy = BASE_Y + (r_l3d[2]-rzmin_)/(rzmax_-rzmin_)*(TOP_Y-BASE_Y)

                is_primary = (i == 0)
                # Secondary regions get a slightly smaller/dimmer stack so
                # the primary region still reads as the main finding.
                rings = ([(30,"#0000FF",0.10),(24,"#00CCFF",0.14),
                          (18,"#00FF88",0.18),(13,"#FFFF00",0.32),
                          (8, "#FF6600",0.55),(5, "#FF0000",0.88)]
                         if is_primary else
                         [(22,"#0000FF",0.08),(17,"#00CCFF",0.11),
                          (13,"#FFCC00",0.24),(6, "#FF8800",0.60)])
                edge_col = "#FF2200" if is_primary else "#FFAA00"

                for r,col,alp in rings:
                    ax.add_patch(plt.Circle((r_ldx,r_ldy),r,color=col,alpha=alp,zorder=8))
                ax.add_patch(plt.Circle((r_ldx,r_ldy), 5 if is_primary else 3.5, fill=False,
                                        edgecolor=edge_col,linewidth=2.2 if is_primary else 1.6,zorder=9))

                region_diagram_pts.append((i, r_ldx, r_ldy, rg))

        # Kept for compatibility with anything below expecting a single point
        ldx, ldy = (region_diagram_pts[0][1], region_diagram_pts[0][2]) if region_diagram_pts else (0, 0)

        def ann(text, xy, xytext, color="#DDDDDD", fs=9, ac="#AAAAAA"):
            ax.annotate(text,xy=xy,xytext=xytext,
                fontsize=fs,color=color,fontfamily='monospace',
                ha='center',va='center',zorder=15,
                bbox=dict(boxstyle='round,pad=0.35',facecolor='#08101A',
                          edgecolor=color,linewidth=0.9,alpha=0.93),
                arrowprops=dict(arrowstyle='->',color=ac,lw=1.3,
                                connectionstyle='arc3,rad=0.18'),
                path_effects=[pe.withStroke(linewidth=2,foreground='#000000')])

        ann(f"RIGHT APEX\n(W={R_W:.0f}mm, H={R_H:.0f}mm)",
            (R_CX,TOP_Y),(R_CX+95,TOP_Y+22),color="#A8D8F8")
        ann(f"R. UPPER LOBE",
            (R_CX+18,BASE_Y+108),(R_CX+100,BASE_Y+118),color="#7ACCE0")
        ann(f"R. MIDDLE LOBE",
            (R_CX+20,BASE_Y+88),(R_CX+100,BASE_Y+88),color="#7ACCE0")
        ann(f"R. LOWER LOBE",
            (R_CX+10,BASE_Y+38),(R_CX+100,BASE_Y+42),color="#7ACCE0")
        ann(f"OBLIQUE FISSURE",
            (R_CX-2,BASE_Y+55),(R_CX+100,BASE_Y+62),color="#5ABCD0",fs=8)
        ann(f"HORIZ. FISSURE",
            (R_CX+22,BASE_Y+90),(R_CX+100,BASE_Y+74),color="#5ABCD0",fs=8)
        ann(f"RIGHT HILUM",
            (R_CX-12,BASE_Y+68),(R_CX-95,BASE_Y+52),color="#9ADCE8")

        ann(f"LEFT APEX\n(W={L_W:.0f}mm, H={L_H:.0f}mm)",
            (L_CX,TOP_Y),(L_CX-95,TOP_Y+22),color="#A8D8F8")
        ann(f"L. UPPER LOBE",
            (L_CX-18,BASE_Y+108),(L_CX-100,BASE_Y+118),color="#7ACCE0")
        ann(f"L. LOWER LOBE",
            (L_CX-10,BASE_Y+38),(L_CX-100,BASE_Y+42),color="#7ACCE0")
        ann(f"OBLIQUE FISSURE",
            (L_CX+2,BASE_Y+55),(L_CX-100,BASE_Y+62),color="#5ABCD0",fs=8)
        ann(f"LEFT HILUM",
            (L_CX+12,BASE_Y+68),(L_CX+95,BASE_Y+52),color="#9ADCE8")

        ann(f"CARINA",(0,TOP_Y-2),(0,TOP_Y+50),color="#C8E8F8")
        ann(f"TRACHEA",(0,TOP_Y+16),(45,TOP_Y+50),color="#C8E8F8")
        ann(f"DIAPHRAGM",(0,BASE_Y-8),(0,BASE_Y-32),color="#5AAAB8")

        if HAS_TB and region_diagram_pts:
            for i, r_ldx, r_ldy, rg in region_diagram_pts:
                is_primary = (i == 0)
                tag = " (PRIMARY)" if is_primary and len(region_diagram_pts) > 1 else ""
                inf_txt=(f"TB INFECTION{tag if is_primary else f' — REGION {i+1}'}\n"
                         f"Lung  : {rg['lung_name']}\n"
                         f"Lobe  : {rg['lobe']}\n"
                         f"3D    : ({rg['lesion_3d'][0]:.0f},{rg['lesion_3d'][1]:.0f},{rg['lesion_3d'][2]:.0f})mm\n"
                         f"Pixel : ({rg['px'][0]:.0f},{rg['px'][1]:.0f}) / 224x224\n"
                         f"Peak  : {rg['peak']:.2f}")
                xt = (L_CX-105 if rg["lung_name"]=="LEFT" else R_CX+105)
                yt = r_ldy - 20 - (0 if is_primary else 55)  # stack secondary labels lower to avoid overlap
                txt_col = "#FFDD00" if is_primary else "#FFB84D"
                edge_col = "#FF4400" if is_primary else "#FF9900"
                ax.annotate(inf_txt,xy=(r_ldx,r_ldy),xytext=(xt,yt),
                    fontsize=9 if is_primary else 8,color=txt_col,fontfamily='monospace',
                    ha='center',va='center',zorder=20,
                    bbox=dict(boxstyle='round,pad=0.5',facecolor='#180808',
                              edgecolor=edge_col,linewidth=2 if is_primary else 1.4,alpha=0.96),
                    arrowprops=dict(arrowstyle='->',color=edge_col,
                                    lw=2.2 if is_primary else 1.6,connectionstyle='arc3,rad=0.22'))

        ox,oy=138,-112
        ax.add_patch(FancyBboxPatch((ox-55,oy-30),110,68,
            boxstyle="round,pad=3",facecolor="#080E18",
            edgecolor="#334455",linewidth=1,zorder=6))
        for dxy,lbl,col in [((22,0),"X (R-L)","#FF6666"),
                              ((0,22),"Y (Inf-Sup)","#66FF66"),
                              ((-14,-14),"Z (Post-Ant)","#6699FF")]:
            ax.annotate('',xy=(ox+dxy[0],oy+dxy[1]),xytext=(ox,oy),
                arrowprops=dict(arrowstyle='->',color=col,lw=2))
            ax.text(ox+dxy[0]*1.35,oy+dxy[1]*1.35,lbl,
                    color=col,fontsize=8,ha='center',va='center')
        ax.text(ox,oy-26,"Rotation: Y axis only",
                color="#AAAAAA",fontsize=7.5,ha='center',fontfamily='monospace')

        cax=fig.add_axes([0.92,0.25,0.016,0.42])
        cb=ColorbarBase(cax,cmap=cm.jet,norm=Normalize(0,1),orientation='vertical')
        cb.set_label('TB Activation\n(GradCAM++)',color='white',fontsize=9)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(),color='white',fontsize=8)
        cb.outline.set_edgecolor('white')

        ax.plot([],[],color="#7ACCE0",lw=1.3,ls='--',label='Fissure')
        ax.add_patch(mpatches.Patch(color="#1A4A6A",alpha=0.5,label='Lung parenchyma'))
        ax.add_patch(mpatches.Patch(color="#A0C8E0",label='Airways'))
        if HAS_TB:
            ax.add_patch(mpatches.Patch(color="#FF0000",label='TB hotspot'))
        ax.legend(loc='lower left',fontsize=9,facecolor='#08101A',
                  edgecolor='#334455',labelcolor='white',framealpha=0.9)

        ax.set_title(
            "AI TB ANALYSIS PLATFORM — Annotated Lung Diagram with 3D Coordinates",
            color='white',fontsize=14,fontweight='bold',pad=14)
        ax.text(0,-128,
            "X=Right-Left | Y=Inferior-Superior (vertical/rotation axis) | "
            "Z=Posterior-Anterior (depth) | Indian-adult anatomy targets",
            color='#777777',fontsize=8.5,ha='center',fontfamily='monospace')

        plt.savefig(DIAGRAM_PATH,dpi=180,bbox_inches='tight',
                    facecolor="#04060A",edgecolor='none')
        plt.close()
        print(f"[INFO] Annotated diagram saved -> {DIAGRAM_PATH}")
    except Exception as e:
        print(f"[WARN] Lung diagram failed: {e}")
        print("[WARN] Full traceback for debugging:")
        traceback.print_exc()


print("\n" + "="*55)
print("  GENERATING OUTPUTS (single command) ...")
print("="*55)
generate_coord_verification()
generate_lung_diagram()
print(f"[INFO] Saved: outputs/{os.path.basename(VERIFY_PATH)}")
print(f"[INFO] Saved: outputs/{os.path.basename(DIAGRAM_PATH)}")
print("[INFO] Opening 3-D viewer ...\n")

plotter = pv.Plotter(window_size=[1600, 900])
plotter.enable_trackball_style()
plotter.set_background("#06080D")
plotter.enable_anti_aliasing("ssaa")

CLIP_NEAR, CLIP_FAR = 0.01, 50000

def _apply_clip_range():
    plotter.camera.SetClippingRange(CLIP_NEAR, CLIP_FAR)
    plotter.renderer.ResetCameraClippingRange()

_apply_clip_range()

ZOOM_STEP    = 1.15
MIN_CAM_DIST = 15.0

def _dolly(factor):
    cam = plotter.camera
    pos = np.array(cam.position, dtype=float)
    foc = np.array(cam.focal_point, dtype=float)
    direction = pos - foc
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return
    new_dist = max(dist / factor, MIN_CAM_DIST)
    cam.position = tuple(foc + direction / dist * new_dist)
    _apply_clip_range()
    plotter.render()

def zoom_in():
    _dolly(ZOOM_STEP)

def zoom_out():
    _dolly(1.0 / ZOOM_STEP)

plotter.iren.add_observer("MouseWheelForwardEvent",  lambda obj, evt: zoom_in())
plotter.iren.add_observer("MouseWheelBackwardEvent", lambda obj, evt: zoom_out())

plotter.add_light(pv.Light(position=( 200, 300, 600),
    color=[1.0,0.97,0.90], intensity=1.20, light_type="scene light"))
plotter.add_light(pv.Light(position=(-400, 150, 400),
    color=[0.70,0.82,1.00], intensity=0.55, light_type="scene light"))
plotter.add_light(pv.Light(position=(  0, 500,-400),
    color=[1.0,0.92,0.85],  intensity=0.35, light_type="scene light"))
plotter.add_light(pv.Light(position=(  0,-500, 200),
    color=[0.55,0.50,0.45], intensity=0.14, light_type="scene light"))

spotlight = None
if lesion_3d is not None:
    spotlight = pv.Light(
        position    = tuple(lesion_3d + np.array([0, 0, 180])),
        focal_point = tuple(lesion_3d),
        color="white", intensity=2.2, cone_angle=10,
        positional=True, light_type="scene light")
    plotter.add_light(spotlight)

def add_shell(mesh):
    plotter.add_mesh(mesh, color="#1A6080", opacity=0.28,
        smooth_shading=True, specular=0.06, specular_power=3,
        ambient=0.22, diffuse=0.78, lighting=True)
    plotter.add_mesh(mesh, color="#5AAED0", opacity=0.12,
        smooth_shading=True, specular=0.25, specular_power=12,
        ambient=0.08, diffuse=0.92, lighting=True)
    plotter.add_mesh(mesh, color="#C8E8F8", opacity=0.07,
        smooth_shading=True, specular=0.98, specular_power=128,
        ambient=0.02, diffuse=0.98, lighting=True)

add_shell(left_shell)
add_shell(right_shell)

def add_tree(tree):
    if tree is None or tree.n_points < 50: return
    plotter.add_mesh(tree, color="#E8F0F8", opacity=0.88,
        smooth_shading=True, specular=0.50, specular_power=22,
        ambient=0.20, diffuse=0.80, lighting=True)

def add_lesion(mesh):
    # Steeper opacity ramp: mid-range activation becomes solidly opaque
    # much sooner, so the red/yellow core is clearly visible instead of
    # staying translucent until scalar values approach 1.0.
    opac = [
        0.00, 0.00, 0.05, 0.25,
        0.55, 0.78, 0.90, 0.96,
        0.99, 1.00, 1.00,
    ]
    plotter.add_mesh(mesh, scalars="heat", cmap="jet",
        clim=[0.0,1.0], opacity=opac,
        smooth_shading=True, show_scalar_bar=False,
        specular=0.55, specular_power=18, ambient=0.18)

def add_lesion_glow(lesion_centre):
    """Soft outer glow behind the hotspot — purely visual, makes the
    infection site pop at a glance instead of relying only on the
    gradient's own falloff."""
    if lesion_centre is None:
        return
    for radius, color, opacity in [
        (26, "#FF3300", 0.05),
        (18, "#FF5500", 0.10),
        (11, "#FF8800", 0.18),
    ]:
        glow = pv.Sphere(radius=radius, center=lesion_centre)
        plotter.add_mesh(glow, color=color, opacity=opacity,
                          lighting=False, smooth_shading=True)

if HAS_TB:
    if right_regions:
        add_lesion(right_shell)
        for rg in right_regions:
            add_lesion_glow(rg["lesion_3d"])
    if left_regions:
        add_lesion(left_shell)
        for rg in left_regions:
            add_lesion_glow(rg["lesion_3d"])

def add_fixed_lesion_markers():
    """
    One fixed coordinate marker per detected infection region — the
    primary (strongest) region gets a larger red marker matching the
    old single-marker look; any additional regions get a smaller
    orange marker so multiple infection sites are all visible and
    distinguishable at a glance.

    Labels are offset OUTWARD from each marker (with a thin leader
    line) rather than placed directly at the point — when two regions
    are close together (e.g. both near the mediastinum), placing
    labels right at the marker causes the text boxes to overlap and
    become unreadable, which is exactly what happened before this fix.
    """
    if not HAS_TB or not regions_3d:
        return
    for i, rg in enumerate(regions_3d):
        pt = np.array(rg["lesion_3d"])
        is_primary = (i == 0)
        color  = "#FF0000" if is_primary else "#FF8800"
        radius = 2.4 if is_primary else 1.8

        plotter.add_mesh(pv.Sphere(radius=radius, center=pt),
                          color=color, opacity=0.95, lighting=False)

        axis_len = 18 if is_primary else 12
        for d in (np.array([axis_len,0,0]), np.array([0,axis_len,0]), np.array([0,0,axis_len])):
            line = pv.Line(pt - d, pt + d)
            plotter.add_mesh(line, color="#FFFFFF",
                              line_width=1.5 if is_primary else 1.0,
                              opacity=0.55 if is_primary else 0.40)

        # Stagger label positions: primary stays near-center-high,
        # each additional region is pushed further out and alternates
        # up/down so labels don't stack on top of each other.
        if is_primary:
            offset = np.array([0.0, 45.0, 0.0])
        else:
            side = 1.0 if (i % 2 == 1) else -1.0
            spread = 40.0 + 25.0 * ((i - 1) // 2)
            vertical = -35.0 - 20.0 * ((i - 1) // 2)
            offset = np.array([side * spread, vertical, 0.0])

        label_pt = pt + offset

        leader = pv.Line(pt, label_pt)
        plotter.add_mesh(leader, color=color, line_width=1.2, opacity=0.5)

        tag = " (PRIMARY)" if is_primary else ""
        label = (f"REGION {i+1}{tag}\n"
                 f"({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}) mm\n"
                 f"{rg['lung_name']} | {rg['lobe']}\n"
                 f"peak={rg['peak']:.2f}")

        plotter.add_point_labels(
            [label_pt], [label],
            font_size=20 if is_primary else 15,
            text_color="#FFFFFF", bold=True,
            shape_color="#3A0000" if is_primary else "#2A1800",
            shape_opacity=0.90 if is_primary else 0.82,
            margin=10, show_points=False, always_visible=True, shadow=True,
        )

add_fixed_lesion_markers()


plotter.add_scalar_bar(
    title="TB Activation (GradCAM++)", n_labels=5, fmt="%.1f",
    position_x=0.90, position_y=0.25, width=0.06, height=0.40,
    label_font_size=10, title_font_size=10, color="white")

def add_anatomical_labels():
    lb = left_shell.bounds
    rb = right_shell.bounds
    mid_z = (min(lb[4],rb[4]) + max(lb[5],rb[5])) / 2

    span_x = max(rb[1], lb[1]) - min(rb[0], lb[0])
    OUTSET = span_x * 0.55

    anchors      = []
    label_ends   = []
    texts        = []
    text_colors  = []
    shape_colors = []

    def add(anchor_xyz, outward_dir, text, tcol="#CFEFFF", scol="#0E1B2B"):
        anchor_xyz = np.array(anchor_xyz, dtype=float)
        outward_dir = np.array(outward_dir, dtype=float)
        n = np.linalg.norm(outward_dir)
        outward_dir = outward_dir / n if n > 1e-6 else outward_dir
        end_xyz = anchor_xyz + outward_dir * OUTSET
        anchors.append(anchor_xyz)
        label_ends.append(end_xyz)
        texts.append(text)
        text_colors.append(tcol)
        shape_colors.append(scol)

    l_apex  = [(lb[0]+lb[1])/2, lb[3]-5,                (lb[4]+lb[5])/2]
    l_base  = [(lb[0]+lb[1])/2, lb[2]+5,                (lb[4]+lb[5])/2]
    l_hilum = [lb[1]-8,         (lb[2]+lb[3])/2,        (lb[4]+lb[5])/2]
    l_upper = [(lb[0]+lb[1])/2, lb[2]+(lb[3]-lb[2])*0.72,(lb[4]+lb[5])/2]
    l_lower = [(lb[0]+lb[1])/2, lb[2]+(lb[3]-lb[2])*0.22,(lb[4]+lb[5])/2]

    add(l_apex,  [-1, 0.6, 0], f"LEFT APEX\n({l_apex[0]:.0f}, {l_apex[1]:.0f}, {l_apex[2]:.0f}) mm")
    add(l_hilum, [ 1, 0.0, 0], f"LEFT HILUM\n({l_hilum[0]:.0f}, {l_hilum[1]:.0f}, {l_hilum[2]:.0f}) mm")
    add(l_upper, [-1, 0.2, 0], "L. UPPER LOBE")
    add(l_lower, [-1,-0.2, 0], "L. LOWER LOBE")
    add(l_base,  [-1,-0.6, 0], "LEFT BASE\n(diaphragm)")

    r_apex   = [(rb[0]+rb[1])/2, rb[3]-5,                 (rb[4]+rb[5])/2]
    r_base   = [(rb[0]+rb[1])/2, rb[2]+5,                 (rb[4]+rb[5])/2]
    r_hilum  = [rb[0]+8,         (rb[2]+rb[3])/2,         (rb[4]+rb[5])/2]
    r_upper  = [(rb[0]+rb[1])/2, rb[2]+(rb[3]-rb[2])*0.78,(rb[4]+rb[5])/2]
    r_middle = [(rb[0]+rb[1])/2, rb[2]+(rb[3]-rb[2])*0.50,(rb[4]+rb[5])/2]
    r_lower  = [(rb[0]+rb[1])/2, rb[2]+(rb[3]-rb[2])*0.18,(rb[4]+rb[5])/2]

    add(r_apex,   [ 1, 0.6, 0], f"RIGHT APEX\n({r_apex[0]:.0f}, {r_apex[1]:.0f}, {r_apex[2]:.0f}) mm")
    add(r_hilum,  [-1, 0.0, 0], f"RIGHT HILUM\n({r_hilum[0]:.0f}, {r_hilum[1]:.0f}, {r_hilum[2]:.0f}) mm")
    add(r_upper,  [ 1, 0.3, 0], "R. UPPER LOBE")
    add(r_middle, [ 1, 0.0, 0], "R. MIDDLE LOBE")
    add(r_lower,  [ 1,-0.3, 0], "R. LOWER LOBE")
    add(r_base,   [ 1,-0.6, 0], "RIGHT BASE\n(diaphragm)")

    carina_pt = [0, max(lb[3], rb[3]) + 4, mid_z]
    add(carina_pt, [0, 1, 0], f"CARINA\n(x=0, z={mid_z:.0f}) mm", tcol="#E8F4FF")

    for a, e in zip(anchors, label_ends):
        line = pv.Line(a, e)
        plotter.add_mesh(line, color="#7ACCE0", line_width=1.3, opacity=0.55)
        plotter.add_mesh(pv.Sphere(radius=1.6, center=a),
                          color="#7ACCE0", opacity=0.9)

    plotter.add_point_labels(
        label_ends, texts,
        font_size=26, text_color="#CFEFFF", bold=True,
        shape_color="#0E1B2B", shape_opacity=0.90,
        margin=12,
        show_points=False,
        always_visible=True, shadow=True,
        italic=False,
    )

    if HAS_TB and lesion_3d is not None:
        region_note = f" (PRIMARY — 1 of {len(regions_3d)} regions)" if len(regions_3d) > 1 else ""
        infection_text = (
            f"TB INFECTION SITE{region_note}\n"
            f"Lung: {lung_name}   Lobe: {lobe}\n"
            f"3D coords: ({lesion_3d[0]:.1f}, {lesion_3d[1]:.1f}, {lesion_3d[2]:.1f}) mm\n"
            f"Pixel: ({cx},{cy})   Depth: {DEPTH_FROM_ANT_MM:.1f} mm ({DEPTH_PCT:.0f}%)\n"
            f"Carina dist: {DIST_FROM_CARINA:.1f} mm"
        )
        outward = np.array([1.0, 1.0, 0.0]) if lung_name == "RIGHT" else np.array([-1.0, 1.0, 0.0])
        tb_end  = np.array(lesion_3d) + outward/np.linalg.norm(outward) * (OUTSET*1.15)

        tb_line = pv.Line(lesion_3d, tb_end)
        plotter.add_mesh(tb_line, color="#FF4400", line_width=2.2, opacity=0.85)
        plotter.add_mesh(pv.Sphere(radius=2.2, center=lesion_3d),
                          color="#FF2200", opacity=0.95)

        plotter.add_point_labels(
            [tb_end], [infection_text],
            font_size=30, text_color="#FFE45C", bold=True,
            shape_color="#220A0A", shape_opacity=0.95,
            margin=14,
            show_points=False,
            always_visible=True, shadow=True,
        )

add_anatomical_labels()

plotter.add_text("3-D ANALYSIS MODEL",
    position="upper_left", font_size=16, color="white")

if HAS_TB and lesion_3d is not None:
    region_line = (f"Regions    : {len(regions_3d)} detected\n\n"
                   if len(regions_3d) > 1 else "")
    primary_tag = " (PRIMARY)" if len(regions_3d) > 1 else ""
    info = (
        f"Prediction : TB Positive\n\n"
        f"Confidence : 100%\n\n"
        f"{region_line}"
        f"Lung{primary_tag}       : {lung_name}\n\n"
        f"Lobe{primary_tag}       : {lobe}\n\n"
        f"Hotspot px : ({cx}, {cy})\n\n"
        f"Lesion 3D  : ({lesion_3d[0]:.0f}, {lesion_3d[1]:.0f}, {lesion_3d[2]:.0f}) mm\n\n"
        f"Ant. surf Z: {ANT_Z:.1f} mm\n\n"
        f"Lesion Z   : {lesion_3d[2]:.1f} mm\n\n"
        f"Depth (ant): {DEPTH_FROM_ANT_MM:.1f} mm  ({DEPTH_PCT:.0f}% into lung)\n\n"
        f"Depth (post): {DEPTH_FROM_POST_MM:.1f} mm\n\n"
        f"Lung A-P   : {LUNG_AP:.1f} mm\n\n"
        f"Carina dist: {DIST_FROM_CARINA:.1f} mm\n\n"
        f"Heatmap    : {'GradCAM .npy' if HAS_GRADCAM else 'Synthesised'}"
    )
else:
    info = "Prediction : Healthy\n\nNo TB hotspot detected."

plotter.add_text(info, position="upper_right", font_size=13, color="white")
plotter.add_text(
    "RED = primary region   ORANGE = additional region(s)   YELLOW/GREEN = moderate activation",
    position="lower_left", font_size=9, color="#666666")
plotter.add_text(
    "F=Front  B=Back  L=Left  R=Right  Y=Rotate  V=Video  "
    "+/- or =/- = Zoom  Space=Stop  (scroll wheel also zooms)",
    position=(20,20), font_size=10, color="#555555")

_cam = {}

def _init_camera():
    plotter.camera_position = [
        (PIVOT[0], PIVOT[1], PIVOT[2]+600),
        tuple(PIVOT), (0,1,0)]
    plotter.reset_camera()
    plotter.camera.zoom(0.80)
    _apply_clip_range()
    plotter.render()
    _cam['pos']   = tuple(plotter.camera.position)
    _cam['focal'] = tuple(plotter.camera.focal_point)
    _cam['up']    = tuple(plotter.camera.up)

def front_view():
    if _cam:
        plotter.camera.position    = _cam['pos']
        plotter.camera.focal_point = _cam['focal']
        plotter.camera.up          = _cam['up']
        _apply_clip_range()
        plotter.render()
    else:
        _init_camera()

def back_view():
    plotter.camera_position=[
        (PIVOT[0],PIVOT[1],PIVOT[2]-600),tuple(PIVOT),(0,1,0)]
    plotter.reset_camera(); plotter.camera.zoom(0.80)
    _apply_clip_range(); plotter.render()

def left_view():
    plotter.camera_position=[
        (PIVOT[0]-600,PIVOT[1],PIVOT[2]),tuple(PIVOT),(0,1,0)]
    plotter.reset_camera(); plotter.camera.zoom(0.80)
    _apply_clip_range(); plotter.render()

def right_view():
    plotter.camera_position=[
        (PIVOT[0]+600,PIVOT[1],PIVOT[2]),tuple(PIVOT),(0,1,0)]
    plotter.reset_camera(); plotter.camera.zoom(0.80)
    _apply_clip_range(); plotter.render()

rotation_running = False
STEP_DEG   = 1
FRAME_WAIT = 0.04

def _actors():
    yield left_shell;  yield right_shell
    if left_tree  is not None: yield left_tree
    if right_tree is not None: yield right_tree

def _step(deg):
    for m in _actors(): m.rotate_y(deg, point=PIVOT, inplace=True)

def rotate_y_slow():
    global rotation_running
    if rotation_running: print("[INFO] Already rotating."); return
    rotation_running=True; front_view()
    print("[INFO] 360 deg rotation ...")
    total=0.0
    while total<360.0 and rotation_running:
        _step(STEP_DEG); total+=STEP_DEG
        plotter.render(); time.sleep(FRAME_WAIT)
        i=int(round(total))
        if i in (90,180,270,360):
            log_depth_report(i)
    rotation_running=False
    if HAS_TB and depth_estimates:
        W = 54
        lname = lung_name if HAS_TB else "N/A"
        lobe_s = lobe      if HAS_TB else "N/A"
        lesion_s = (f"x={lesion_3d[0]:.1f}  y={lesion_3d[1]:.1f}  z={lesion_3d[2]:.1f} mm"
                    if lesion_3d is not None else "N/A")
        print()
        print(f"  +{'='*W}+")
        print(f"  |{'  DEPTH SUMMARY  ':^{W}}|")
        print(f"  +{'='*W}+")
        print(f"  | {'Rotation axis':<28}: Y (trachea pivot){'':<{W-46}}|")
        print(f"  | {'Lung':<28}: {lname:<{W-31}}|")
        print(f"  | {'Lobe':<28}: {lobe_s:<{W-31}}|")
        print(f"  | {'Lesion position (fixed)':<28}: {lesion_s:<{W-31}}|")
        print(f"  | {'Distance from carina':<28}: {DIST_FROM_CARINA:>6.1f} mm{'':<{W-38}}|")
        print(f"  +{'-'*W}+")
        print(f"  | {'Anterior surface Z':<28}: {ANT_Z:>6.1f} mm{'':<{W-38}}|")
        print(f"  | {'Lesion Z':<28}: {lesion_3d[2]:>6.1f} mm{'':<{W-38}}|")
        print(f"  | {'Posterior surface Z':<28}: {POST_Z:>6.1f} mm{'':<{W-38}}|")
        print(f"  +{'-'*W}+")
        print(f"  | {'Depth from ANTERIOR':<28}: {DEPTH_FROM_ANT_MM:>6.1f} mm  ({DEPTH_PCT:.0f}% into lung){'':<{W-50}}|")
        print(f"  | {'Depth from posterior':<28}: {DEPTH_FROM_POST_MM:>6.1f} mm{'':<{W-38}}|")
        print(f"  | {'Lung total A-P depth':<28}: {LUNG_AP:>6.1f} mm{'':<{W-38}}|")
        print(f"  +{'-'*W}+")
        tw_s = f"{R_W if lung_name=='RIGHT' else L_W}"
        th_s = f"{R_H if lung_name=='RIGHT' else L_H}"
        td_s = f"{R_D if lung_name=='RIGHT' else L_D}"
        print(f"  | {'Lung dimensions (target)':<28}: W={tw_s}  H={th_s}  D={td_s} mm{'':<{W-46}}|")
        print(f"  | {'Estimates collected':<28}: {len(depth_estimates):<{W-31}}|")
        print(f"  +{'='*W}+")
    print("[INFO] Rotation complete.")

def stop_rotation():
    global rotation_running; rotation_running=False; print("[INFO] Stopped.")

def record_rotation_video():
    global rotation_running
    if rotation_running: print("[INFO] Stop first."); return
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    rotation_running=True; front_view()
    plotter.open_movie(VIDEO_PATH, framerate=24)
    total=0.0
    while total<360.0 and rotation_running:
        _step(STEP_DEG); total+=STEP_DEG
        plotter.render(); plotter.write_frame()
        i=int(round(total))
        if i in (90,180,270,360):
            print(f"[INFO] {i} deg captured")
            log_depth_report(i)
    plotter.close_movie(); rotation_running=False
    print(f"[INFO] Saved -> {VIDEO_PATH}" if total>=360 else "[INFO] Cancelled.")

plotter.add_key_event("f",     front_view)
plotter.add_key_event("b",     back_view)
plotter.add_key_event("l",     left_view)
plotter.add_key_event("r",     right_view)
plotter.add_key_event("y",     rotate_y_slow)
plotter.add_key_event("v",     record_rotation_video)
plotter.add_key_event("space", stop_rotation)

plotter.add_key_event("equal",        zoom_in)
plotter.add_key_event("KP_Add",       zoom_in)
plotter.add_key_event("plus",         zoom_in)
plotter.add_key_event("minus",        zoom_out)
plotter.add_key_event("KP_Subtract",  zoom_out)
plotter.add_key_event("underscore",   zoom_out)

print('[INFO] Zoom keys bound: "=" / "+" / numpad+  -> zoom IN')
print('[INFO] Zoom keys bound: "-" / numpad-        -> zoom OUT')
print('[INFO] Mouse scroll wheel is now also bound directly to zoom in/out (dolly-based).')
print('[INFO] Click inside the 3-D render window first so it has keyboard focus.')

if HAS_TB:
    speech=(f"T B Infected Area detected. {lung_name.title()} lung. "
            f"{lobe.replace('_',' ').title()}.")
    speak_async(speech); print(f'[INFO] Audio: "{speech}"')

plotter.enable_trackball_style()
_apply_clip_range()

_init_camera()
plotter.show()