import os, sys, time, threading
import numpy as np
import pyvista as pv
from scipy.ndimage import zoom as nd_zoom

# ── PATHS ─────────────────────────────────────────────────────────────────────

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

# ── ANATOMY TARGETS — Average Indian Adult Lungs ──────────────────────────────
# Source: Indian Journal of Radiology, CT volumetry studies on Indian adults
#
#   Right lung: 3 lobes, wider (pushed by heart leftward), no cardiac notch
#     Width (mediolateral) : ~95 mm
#     Height (craniocaudal): ~220 mm
#     Depth (A-P)          : ~130 mm
#
#   Left lung: 2 lobes, cardiac notch indents lower-medial border
#     Width (mediolateral) : ~85 mm
#     Height (craniocaudal): ~210 mm
#     Depth (A-P)          : ~120 mm
#
# Scale each axis independently TO these targets so both lungs match real
# Indian anatomy exactly — shape is preserved by the STL smoothing.

R_W, R_H, R_D = 95.0, 220.0, 130.0   # right lung mm
L_W, L_H, L_D = 85.0, 210.0, 120.0   # left lung mm

PUSH_MM  = 14.0    # inward push at midline to close mediastinal gap
TARGET_D = 125.0   # average A-P for depth calculations

# ── AUDIO ─────────────────────────────────────────────────────────────────────

def speak_async(text):
    def _speak():
        try:
            import pyttsx3; e = pyttsx3.init()
            e.setProperty("rate", 155); e.say(text); e.runAndWait(); return
        except Exception: pass
        if sys.platform == "darwin": os.system(f'say "{text}"'); return
        os.system(f'espeak "{text}" 2>/dev/null')
    threading.Thread(target=_speak, daemon=True).start()

# ── DETECT MESHES ─────────────────────────────────────────────────────────────

USE_SPLIT = os.path.exists(LEFT_SHELL) and os.path.exists(RIGHT_SHELL)
if not USE_SPLIT:
    print("[WARNING] Shell files not found — falling back to raw STL.")
    if not os.path.exists(LEFT_RAW) or not os.path.exists(RIGHT_RAW):
        print("[ERROR] No lung STL files found."); sys.exit(1)

# ── LOAD TB HOTSPOT ───────────────────────────────────────────────────────────

HAS_TB = os.path.exists(TB_CENTER_PATH)
if HAS_TB:
    tb_raw = np.load(TB_CENTER_PATH)
    cx, cy = int(tb_raw[0]), int(tb_raw[1])
    print(f"[INFO] TB hotspot: cx={cx}  cy={cy}")
else:
    cx = cy = None
    print("[INFO] Healthy mode.")

# ── HEATMAP ───────────────────────────────────────────────────────────────────

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

# ── LOAD SHELL ────────────────────────────────────────────────────────────────

def load_shell(path, side):
    mesh = pv.read(path)
    mesh = mesh.connectivity(extraction_mode='largest')
    mesh = mesh.extract_surface(algorithm='dataset_surface')
    mesh = mesh.clean()
    mesh = mesh.smooth(n_iter=300, relaxation_factor=0.04)
    mesh.compute_normals(inplace=True, auto_orient_normals=True)

    # ── Indian adult lung targets ─────────────────────────────────────────────
    tw = R_W if side == 'right' else L_W
    th = R_H if side == 'right' else L_H
    td = R_D if side == 'right' else L_D

    # Step 1 — uniform scale: fit the mesh so its LONGEST axis matches the
    # corresponding target dimension. This avoids distorting the organic shape.
    b        = mesh.bounds
    raw_w    = b[1]-b[0]; raw_h = b[3]-b[2]; raw_d = b[5]-b[4]
    # Pick the axis whose ratio is most constrained (smallest → avoids clipping)
    sf = min(tw/max(raw_w,1e-6), th/max(raw_h,1e-6), td/max(raw_d,1e-6))
    c  = np.array(mesh.center)
    pts = mesh.points.copy()
    pts[:,0] = c[0] + (pts[:,0]-c[0]) * sf
    pts[:,1] = c[1] + (pts[:,1]-c[1]) * sf
    pts[:,2] = c[2] + (pts[:,2]-c[2]) * sf
    mesh.points = pts

    # Centre in Y and Z
    b2 = mesh.bounds
    mesh.translate([0, -(b2[2]+b2[3])/2, -(b2[4]+b2[5])/2], inplace=True)

    # Pin medial face to x=0 then push inward PUSH_MM to close gap
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

# ── BRONCHI ───────────────────────────────────────────────────────────────────

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

# ── PIVOT ─────────────────────────────────────────────────────────────────────

PIVOT = np.array(left_shell.merge(right_shell).center)
print(f"[INFO] Pivot: {[round(v,1) for v in PIVOT]}")

# ── LOBE + LESION ─────────────────────────────────────────────────────────────

lobe = "N/A"; lung_name = "N/A"; lesion_3d = None; target_lung = None

if HAS_TB:
    if cx < 112: target_lung = right_shell; lung_name = "RIGHT"
    else:        target_lung = left_shell;  lung_name = "LEFT"

    xmin,xmax,ymin,ymax,zmin,zmax = target_lung.bounds
    lx = xmin + (cx/224.0)*(xmax-xmin)
    lz = zmax - (cy/224.0)*(zmax-zmin)
    ly = (ymin+ymax)/2.0
    lesion_3d = np.array([lx, ly, lz])

    z_pct = (lz-zmin)/max(zmax-zmin,1e-9)
    if lung_name == "RIGHT":
        lobe = ("RIGHT UPPER LOBE"  if z_pct>0.66 else
                "RIGHT MIDDLE LOBE" if z_pct>0.33 else "RIGHT LOWER LOBE")
    else:
        lobe = "LEFT UPPER LOBE" if z_pct>0.50 else "LEFT LOWER LOBE"

    print(f"[INFO] {lung_name} | {lobe}")
    print(f"[INFO] Lesion 3D: x={lx:.1f}  y={ly:.1f}  z={lz:.1f} mm")

# ── DEPTH — computed ONCE from initial (pre-rotation) bounds ──────────────────
#
# ROOT CAUSE of the mismatch:
#   OLD code used target_lung.bounds[5] (max-Z = POSTERIOR face) and called it
#   "depth from surface". But the camera faces the ANTERIOR side (max-Z in
#   PyVista's default orientation where +Z points toward the viewer), so
#   "depth from anterior surface" should be:
#       depth = lesion_z - lung_anterior_z   (distance lesion is behind the front)
#   where lung_anterior_z = bounds[4]  (min-Z = the face closest to camera).
#
#   Also: bounds change every frame during rotation because the mesh moves,
#   so computing depth mid-rotation gave different numbers each time.
#   FIX: snapshot all depth values at load time (before any rotation) and
#        reuse those frozen values everywhere — overlay AND terminal are
#        identical because they read the same variables.

# Snapshot initial bounds BEFORE any rotation
INITIAL_LUNG_BOUNDS = tuple(target_lung.bounds) if target_lung is not None else None
INITIAL_LESION_3D   = tuple(lesion_3d)          if lesion_3d   is not None else None

if HAS_TB and lesion_3d is not None and target_lung is not None:
    _b = INITIAL_LUNG_BOUNDS

    # Anterior face = smallest Z value (closest to camera in front view)
    ANT_Z    = _b[4]   # min-Z  = anterior surface
    POST_Z   = _b[5]   # max-Z  = posterior surface
    LUNG_AP  = POST_Z - ANT_Z           # total anterior-posterior depth

    # Depth FROM anterior surface = how far the lesion is behind the front face
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

# ── DEPTH TRACKING — terminal milestones ─────────────────────────────────────
# Uses ONLY the frozen values computed above — no re-calculation from bounds.
# The rotation angle is shown for context (it tells you which face is now
# "anterior"), but the depth number is the same frozen value, which is correct
# because depth is a property of the anatomy, not the camera angle.

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
    print(f"  | {'Lesion world pos':<22s}  x={lx_o:7.1f}  y={ly_o:7.1f}  z={lz_o:7.1f} mm  |")
    print(f"  | {'Lesion rotated':<22s}  x={rot_x:7.1f}            z={rot_z:7.1f} mm  |")
    print(f"  | {'Depth from anterior':<22s}  {DEPTH_FROM_ANT_MM:7.1f} mm  ({DEPTH_PCT:4.0f}% into lung){' '*6}|")
    print(f"  | {'Depth from posterior':<22s}  {DEPTH_FROM_POST_MM:7.1f} mm{' '*20}|")
    print(f"  | {'Total A-P lung depth':<22s}  {LUNG_AP:7.1f} mm{' '*20}|")
    print(f"  | {'Distance from carina':<22s}  {DIST_FROM_CARINA:7.1f} mm{' '*20}|")
    print(f"  +{'-'*W}+")

# ── 3-D HEAT BLOB ─────────────────────────────────────────────────────────────

def make_3d_heat(mesh, lesion_centre, heatmap_2d):
    pts = mesh.points
    if lesion_centre is None:
        return np.zeros(len(pts), dtype=np.float32)
    xmin,xmax = mesh.bounds[0], mesh.bounds[1]
    active   = (heatmap_2d > 0.5).sum()
    sigma_xz = np.clip(np.sqrt(max(active,1)/np.pi)*(xmax-xmin)/224.0, 6.0, 22.0)
    sigma_y  = np.clip(sigma_xz*0.55, 4.0, 14.0)
    lx,ly,lz = lesion_centre
    dx=pts[:,0]-lx; dy=pts[:,1]-ly; dz=pts[:,2]-lz
    h = np.exp(-(dx**2/(2*sigma_xz**2) +
                 dy**2/(2*sigma_y**2)  +
                 dz**2/(2*sigma_xz**2))).astype(np.float32)
    print(f"[INFO] Heat blob sigma_xz={sigma_xz:.1f}mm  sigma_y={sigma_y:.1f}mm  peak={h.max():.3f}")
    return h

if HAS_TB and lung_name=="RIGHT":
    right_shell["heat"] = make_3d_heat(right_shell, lesion_3d, HEATMAP)
    left_shell["heat"]  = np.zeros(left_shell.n_points,  dtype=np.float32)
elif HAS_TB and lung_name=="LEFT":
    left_shell["heat"]  = make_3d_heat(left_shell,  lesion_3d, HEATMAP)
    right_shell["heat"] = np.zeros(right_shell.n_points, dtype=np.float32)
else:
    left_shell["heat"]  = np.zeros(left_shell.n_points,  dtype=np.float32)
    right_shell["heat"] = np.zeros(right_shell.n_points, dtype=np.float32)

# ── PLOTTER ───────────────────────────────────────────────────────────────────

plotter = pv.Plotter(window_size=[1600, 900])
plotter.enable_trackball_style()
plotter.set_background("#06080D")
plotter.enable_anti_aliasing("ssaa")

plotter.add_light(pv.Light(position=( 200, 300, 600),
    color=[1.0,0.97,0.90], intensity=1.20, light_type="scene light"))
plotter.add_light(pv.Light(position=(-400, 150, 400),
    color=[0.70,0.82,1.00], intensity=0.55, light_type="scene light"))
plotter.add_light(pv.Light(position=(  0, 500,-400),
    color=[1.0,0.92,0.85],  intensity=0.35, light_type="scene light"))
plotter.add_light(pv.Light(position=(  0,-500, 200),
    color=[0.55,0.50,0.45], intensity=0.14, light_type="scene light"))

# ── SPOTLIGHT ─────────────────────────────────────────────────────────────────

spotlight = None
if lesion_3d is not None:
    spotlight = pv.Light(
        position    = tuple(lesion_3d + np.array([0, 0, 180])),
        focal_point = tuple(lesion_3d),
        color="white", intensity=2.2, cone_angle=10,
        positional=True, light_type="scene light")
    plotter.add_light(spotlight)

# ── LUNG SHELLS ───────────────────────────────────────────────────────────────

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

# ── BRONCHI ───────────────────────────────────────────────────────────────────

def add_tree(tree):
    if tree is None or tree.n_points < 50: return
    plotter.add_mesh(tree, color="#E8F0F8", opacity=0.88,
        smooth_shading=True, specular=0.50, specular_power=22,
        ambient=0.20, diffuse=0.80, lighting=True)

add_tree(left_tree)
add_tree(right_tree)

# ── TB LESION ─────────────────────────────────────────────────────────────────

def add_lesion(mesh):
    opac = [
        0.00, 0.00, 0.00, 0.00,
        0.10, 0.35, 0.62, 0.80,
        0.90, 0.96, 0.99,
    ]
    plotter.add_mesh(mesh, scalars="heat", cmap="jet",
        clim=[0.0,1.0], opacity=opac,
        smooth_shading=True, show_scalar_bar=False,
        specular=0.55, specular_power=18, ambient=0.18)

if HAS_TB and lung_name=="RIGHT": add_lesion(right_shell)
elif HAS_TB and lung_name=="LEFT": add_lesion(left_shell)

plotter.add_scalar_bar(
    title="TB Activation (GradCAM++)", n_labels=5, fmt="%.1f",
    position_x=0.90, position_y=0.25, width=0.06, height=0.40,
    label_font_size=10, title_font_size=10, color="white")

# ── OVERLAYS — use the SAME frozen depth values as terminal ───────────────────

plotter.add_text("3-D ANALYSIS MODEL",
    position="upper_left", font_size=16, color="white")

if HAS_TB and lesion_3d is not None:
    info = (
        f"Prediction : TB Positive\n\n"
        f"Confidence : 100%\n\n"
        f"Lung       : {lung_name}\n\n"
        f"Lobe       : {lobe}\n\n"
        f"Hotspot px : ({cx}, {cy})\n\n"
        f"Lesion 3D  : ({lesion_3d[0]:.0f}, {lesion_3d[1]:.0f}, {lesion_3d[2]:.0f}) mm\n\n"
        # ── These values are IDENTICAL to what terminal prints ──
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

plotter.add_text(info, position="upper_right", font_size=11, color="white")
plotter.add_text(
    "RED = TB hotspot   YELLOW/GREEN = moderate activation   WHITE = bronchi",
    position="lower_left", font_size=9, color="#666666")
plotter.add_text(
    "F=Front  B=Back  L=Left  R=Right  Y=Rotate  V=Video  Space=Stop",
    position=(20,20), font_size=10, color="#555555")

# ── CAMERA ────────────────────────────────────────────────────────────────────

_cam = {}

def _init_camera():
    plotter.camera_position = [
        (PIVOT[0], PIVOT[1], PIVOT[2]+600),
        tuple(PIVOT), (0,1,0)]
    plotter.reset_camera()
    plotter.camera.zoom(0.80)
    plotter.render()
    _cam['pos']   = tuple(plotter.camera.position)
    _cam['focal'] = tuple(plotter.camera.focal_point)
    _cam['up']    = tuple(plotter.camera.up)

def front_view():
    if _cam:
        plotter.camera.position    = _cam['pos']
        plotter.camera.focal_point = _cam['focal']
        plotter.camera.up          = _cam['up']
        plotter.render()
    else:
        _init_camera()

def back_view():
    plotter.camera_position=[
        (PIVOT[0],PIVOT[1],PIVOT[2]-600),tuple(PIVOT),(0,1,0)]
    plotter.reset_camera(); plotter.camera.zoom(0.80); plotter.render()

def left_view():
    plotter.camera_position=[
        (PIVOT[0]-600,PIVOT[1],PIVOT[2]),tuple(PIVOT),(0,1,0)]
    plotter.reset_camera(); plotter.camera.zoom(0.80); plotter.render()

def right_view():
    plotter.camera_position=[
        (PIVOT[0]+600,PIVOT[1],PIVOT[2]),tuple(PIVOT),(0,1,0)]
    plotter.reset_camera(); plotter.camera.zoom(0.80); plotter.render()

# ── ROTATION ──────────────────────────────────────────────────────────────────

rotation_running = False
STEP_DEG   = 1
FRAME_WAIT = 0.04

def _rotate_spotlight(deg):
    if spotlight is None: return
    rad=np.deg2rad(deg); c,s=np.cos(rad),np.sin(rad)
    def rot_y(pt):
        dx=pt[0]-PIVOT[0]; dz=pt[2]-PIVOT[2]
        return np.array([PIVOT[0]+dx*c+dz*s, pt[1], PIVOT[2]-dx*s+dz*c])
    spotlight.position    = tuple(rot_y(np.array(spotlight.position)))
    spotlight.focal_point = tuple(rot_y(np.array(spotlight.focal_point)))

def _actors():
    yield left_shell;  yield right_shell
    if left_tree  is not None: yield left_tree
    if right_tree is not None: yield right_tree

def _step(deg):
    for m in _actors(): m.rotate_y(deg, point=PIVOT, inplace=True)
    _rotate_spotlight(deg)

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
        print(f"  | {'Lesion position':<28}: {lesion_s:<{W-31}}|")
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

# ── KEYS ──────────────────────────────────────────────────────────────────────

plotter.add_key_event("f",     front_view)
plotter.add_key_event("b",     back_view)
plotter.add_key_event("l",     left_view)
plotter.add_key_event("r",     right_view)
plotter.add_key_event("y",     rotate_y_slow)
plotter.add_key_event("v",     record_rotation_video)
plotter.add_key_event("space", stop_rotation)

# ── AUDIO + LAUNCH ────────────────────────────────────────────────────────────

if HAS_TB:
    speech=(f"T B Infected Area detected. {lung_name.title()} lung. "
            f"{lobe.replace('_',' ').title()}.")
    speak_async(speech); print(f'[INFO] Audio: "{speech}"')

_init_camera()
plotter.show()
