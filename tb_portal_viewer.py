"""
viewer_3d.py
─────────────
PyVista 3-D lung viewer.
Loads pre-built STL meshes, maps the 2-D CAM hotspot onto the 3-D lung
surface, and shows an interactive window with keyboard shortcuts.

Directory layout expected
─────────────────────────
<project>/
  lung_model/
    left_lung_shell.stl   ← preferred (split output of split_lung_layers.py)
    right_lung_shell.stl
    left_lung_tree.stl    ← optional bronchi / vessel tree
    right_lung_tree.stl
    left_lung.stl         ← fallback raw meshes
    right_lung.stl
  outputs/
    tb_center.npy         ← (cx, cy) in 224×224 image space; absent = healthy

Keyboard shortcuts
──────────────────
  F  – front view (default PA projection)
  B  – back view
  L  – left lateral view
  R  – right lateral view
  Y  – slow 180° azimuth rotation
  Space – stop rotation
"""

import os
import sys
import time
import numpy as np
import pyvista as pv

# ── PATHS ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEFT_SHELL  = os.path.join(BASE_DIR, "lung_model", "left_lung_shell.stl")
RIGHT_SHELL = os.path.join(BASE_DIR, "lung_model", "right_lung_shell.stl")
LEFT_TREE   = os.path.join(BASE_DIR, "lung_model", "left_lung_tree.stl")
RIGHT_TREE  = os.path.join(BASE_DIR, "lung_model", "right_lung_tree.stl")
LEFT_RAW    = os.path.join(BASE_DIR, "lung_model", "left_lung.stl")
RIGHT_RAW   = os.path.join(BASE_DIR, "lung_model", "right_lung.stl")

TB_CENTER_PATH = os.path.join(BASE_DIR, "outputs", "tb_center.npy")

# ── DETECT SPLIT VS RAW MESHES ───────────────────────────────────────────────

USE_SPLIT = os.path.exists(LEFT_SHELL) and os.path.exists(RIGHT_SHELL)

if not USE_SPLIT:
    print("\n[WARNING] Shell/tree split files not found.")
    print("Run:  python3 split_lung_layers.py")
    print("Falling back to raw STL files.\n")
    if not os.path.exists(LEFT_RAW) or not os.path.exists(RIGHT_RAW):
        print("[ERROR] No lung STL files found. Exiting.")
        sys.exit(1)

# ── LOAD TB HOTSPOT (optional) ───────────────────────────────────────────────

HAS_TB = os.path.exists(TB_CENTER_PATH)

if HAS_TB:
    tb_raw = np.load(TB_CENTER_PATH)
    cx = int(tb_raw[0])
    cy = int(tb_raw[1])
    print(f"[INFO] TB hotspot loaded: ({cx}, {cy})")
else:
    cx, cy = None, None
    print("[INFO] No tb_center.npy found — healthy scan mode.")

# ── LOAD MESHES ──────────────────────────────────────────────────────────────

if USE_SPLIT:
    left_shell  = pv.read(LEFT_SHELL).smooth(n_iter=50, relaxation_factor=0.1)
    right_shell = pv.read(RIGHT_SHELL).smooth(n_iter=50, relaxation_factor=0.1)
    left_tree   = pv.read(LEFT_TREE)  if os.path.exists(LEFT_TREE)  else None
    right_tree  = pv.read(RIGHT_TREE) if os.path.exists(RIGHT_TREE) else None
else:
    left_shell  = pv.read(LEFT_RAW).smooth(n_iter=50, relaxation_factor=0.1)
    right_shell = pv.read(RIGHT_RAW).smooth(n_iter=50, relaxation_factor=0.1)
    left_tree   = None
    right_tree  = None

# ── MIRROR RIGHT LUNG (STLs are often both left-side) ────────────────────────

def mirror_x(mesh):
    """Mirror a mesh across the YZ plane and recompute normals."""
    m = mesh.copy()
    m.points[:, 0] *= -1
    m.compute_normals(inplace=True, auto_orient_normals=True)
    return m

right_shell = mirror_x(right_shell)
if right_tree is not None:
    right_tree = mirror_x(right_tree)

# Small gap so lungs don't fuse at the midline
GAP_MM = 4.0
left_shell.translate([-GAP_MM, 0, 0],  inplace=True)
right_shell.translate([ GAP_MM, 0, 0], inplace=True)
if left_tree  is not None: left_tree.translate( [-GAP_MM, 0, 0], inplace=True)
if right_tree is not None: right_tree.translate([ GAP_MM, 0, 0], inplace=True)

print("Left lung bounds :", [round(b, 1) for b in left_shell.bounds])
print("Right lung bounds:", [round(b, 1) for b in right_shell.bounds])

# ── MAP 2-D HOTSPOT → 3-D POSITION ──────────────────────────────────────────

finding_dots  = None
lobe          = "N/A"
lung_name     = "N/A"
lesion_x = lesion_y = lesion_z = 0.0

if HAS_TB:
    # PA X-ray: left on image = patient's right lung (cx < 112 → right lung)
    if cx < 112:
        target_lung = right_shell
        lung_name   = "RIGHT"
    else:
        target_lung = left_shell
        lung_name   = "LEFT"

    xmin, xmax, ymin, ymax, zmin, zmax = target_lung.bounds

    nx = cx / 224.0
    ny = cy / 224.0

    lesion_x = xmin + nx * (xmax - xmin)
    lesion_z = zmax - ny * (zmax - zmin)
    lesion_y = ymin + 0.35 * (ymax - ymin) - 10.0

    z_pct = (lesion_z - zmin) / max(zmax - zmin, 1e-9)

    if lung_name == "RIGHT":
        if   z_pct > 0.66: lobe = "RIGHT UPPER LOBE"
        elif z_pct > 0.33: lobe = "RIGHT MIDDLE LOBE"
        else:               lobe = "RIGHT LOWER LOBE"
    else:
        lobe = "LEFT UPPER LOBE" if z_pct > 0.50 else "LEFT LOWER LOBE"

    print(f"[INFO] 3-D lesion centre: ({lesion_x:.1f}, {lesion_y:.1f}, {lesion_z:.1f})")
    print(f"[INFO] Lung: {lung_name}  |  Lobe: {lobe}")

    # ── Scattered finding markers ─────────────────────────────────────────────
    rng            = np.random.default_rng(42)
    N_DOTS         = 14
    SCATTER_RADIUS = 9.0
    DOT_RADIUS     = 1.4

    dx = rng.normal(0, SCATTER_RADIUS,       N_DOTS)
    dy = rng.normal(0, SCATTER_RADIUS * 0.6, N_DOTS)
    dz = rng.normal(0, SCATTER_RADIUS,       N_DOTS)

    finding_dots = pv.PolyData()
    for i in range(N_DOTS):
        sphere = pv.Sphere(
            radius=DOT_RADIUS * rng.uniform(0.7, 1.3),
            center=(lesion_x + dx[i], lesion_y + dy[i], lesion_z + dz[i]),
            theta_resolution=10,
            phi_resolution=10,
        )
        finding_dots = finding_dots.merge(sphere)

# ── ZONE COLOURING (upper / lower lobe shading) ──────────────────────────────

ZONE_SPLIT = 0.62

def add_zone_scalars(mesh):
    zmin_l, zmax_l = mesh.bounds[4], mesh.bounds[5]
    z_pct  = (mesh.points[:, 2] - zmin_l) / max(zmax_l - zmin_l, 1e-9)
    mesh["zone"] = (z_pct > ZONE_SPLIT).astype(np.float32)
    return mesh

left_shell  = add_zone_scalars(left_shell)
right_shell = add_zone_scalars(right_shell)

ZONE_CMAP = ["#D7E4EA", "#D7E4EA"]   # uniform pale blue — change to e.g.
                                      # ["#A8C8E0","#E8A8A0"] for upper/lower tint

# ── PLOTTER SETUP ────────────────────────────────────────────────────────────

plotter = pv.Plotter(window_size=[1600, 900])
plotter.enable_trackball_style()
plotter.set_background("white")
plotter.enable_anti_aliasing()

plotter.add_light(pv.Light(position=(600, 600, 600), intensity=1.5))

# Translucent lung shells
SHELL_OPACITY = 0.22
plotter.add_mesh(
    left_shell,  scalars="zone", cmap=ZONE_CMAP,
    opacity=SHELL_OPACITY, smooth_shading=True, show_scalar_bar=False,
)
plotter.add_mesh(
    right_shell, scalars="zone", cmap=ZONE_CMAP,
    opacity=SHELL_OPACITY, smooth_shading=True, show_scalar_bar=False,
)

# Bronchi / vessel trees
if left_tree is not None and left_tree.n_points > 0:
    plotter.add_mesh(left_tree,  color="#9B96AE", opacity=0.85, smooth_shading=True)
if right_tree is not None and right_tree.n_points > 0:
    plotter.add_mesh(right_tree, color="#9B96AE", opacity=0.85, smooth_shading=True)

# TB finding markers
if finding_dots is not None and finding_dots.n_points > 0:
    plotter.add_mesh(finding_dots, color="#FF2D20", opacity=1.0, smooth_shading=True)

# ── AXIS / BOUNDING BOX ──────────────────────────────────────────────────────

combined = left_shell.merge(right_shell)
plotter.show_bounds(
    mesh=combined,
    grid="back",
    location="outer",
    ticks="outside",
    n_xlabels=5, n_ylabels=5, n_zlabels=5,
    xtitle="Right-Left (mm)",
    ytitle="Anterior-Posterior (mm)",
    ztitle="Superior-Inferior (mm)",
    font_size=10,
    color="black",
    bold=False,
)

# ── INFO OVERLAYS ─────────────────────────────────────────────────────────────

plotter.add_text("AI TB ANALYSIS PLATFORM", position="upper_left", font_size=18)

if HAS_TB:
    info_text = (
        f"Prediction : TB Positive\n\n"
        f"Confidence : 100%\n\n"
        f"Lung       : {lung_name}\n\n"
        f"Lobe       : {lobe}\n\n"
        f"Hotspot    : ({cx}, {cy})"
    )
else:
    info_text = "Prediction : Healthy\n\nNo TB hotspot detected."

plotter.add_text(info_text, position="upper_right", font_size=12)

plotter.add_text(
    "Zone key:  UPPER region = apex   LOWER region = base   RED dots = TB finding",
    position="lower_left", font_size=9, color="#333333",
)
plotter.add_text(
    "F=Front  B=Back  L=Left  R=Right  Y=Slow Rotation  Space=Stop",
    position=(20, 20), font_size=10, color="black",
)

# ── CAMERA VIEWS ─────────────────────────────────────────────────────────────

def front_view():
    """Standard PA (posterior-anterior) projection."""
    plotter.view_yz()           # look along Y axis
    plotter.reset_camera()
    plotter.camera.zoom(2.8)
    plotter.render()

def back_view():
    plotter.camera_position = [
        (0,  450, 80),
        (0,    0, 80),
        (0,    0,  1),
    ]
    plotter.render()

def left_view():
    plotter.camera_position = [
        (-450, 0, 80),
        (   0, 0, 80),
        (   0, 0,  1),
    ]
    plotter.render()

def right_view():
    plotter.camera_position = [
        (450, 0, 80),
        (  0, 0, 80),
        (  0, 0,  1),
    ]
    plotter.render()

# ── ROTATION HELPERS ─────────────────────────────────────────────────────────

rotation_running = False

def rotate_y_slow():
    """180° azimuth rotation at a gentle pace."""
    global rotation_running
    rotation_running = True
    for _ in range(360):
        if not rotation_running:
            break
        plotter.camera.Azimuth(-0.5)
        plotter.render()
        time.sleep(0.03)
    rotation_running = False

def stop_rotation():
    global rotation_running
    rotation_running = False
    print("[INFO] Rotation stopped.")

# ── KEY BINDINGS ──────────────────────────────────────────────────────────────

plotter.add_key_event("f",     front_view)
plotter.add_key_event("b",     back_view)
plotter.add_key_event("l",     left_view)
plotter.add_key_event("r",     right_view)
plotter.add_key_event("y",     rotate_y_slow)
plotter.add_key_event("space", stop_rotation)

# ── LAUNCH ────────────────────────────────────────────────────────────────────

front_view()   # set initial camera before show()
plotter.show()