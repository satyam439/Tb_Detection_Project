import os
import sys
import time
import threading
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
VIDEO_PATH     = os.path.join(BASE_DIR, "outputs", "lung_rotation.mp4")

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
    left_shell  = pv.read(LEFT_SHELL).smooth(n_iter=10, relaxation_factor=0.05)
    right_shell = pv.read(RIGHT_SHELL).smooth(n_iter=10, relaxation_factor=0.05)
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

# ── SHARED ROTATION PIVOT  ───────────────────────────────────────────────────
# All rotation is around a single world Y-axis passing through the combined
# centroid so both lungs (and any markers) rotate as one rigid body.

combined_for_pivot = left_shell.merge(right_shell)
PIVOT = np.array(combined_for_pivot.center)   # (cx, cy, cz) world centroid
print(f"[INFO] Rotation pivot (world centroid): {[round(v,1) for v in PIVOT]}")

# ── COLLECT ALL ROTATING ACTORS INTO ONE LIST ─────────────────────────────────

def get_rotating_meshes():
    """Return the list of meshes that participate in Y-axis rotation."""
    meshes = [left_shell, right_shell]
    if left_tree  is not None: meshes.append(left_tree)
    if right_tree is not None: meshes.append(right_tree)
    return meshes


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

ZONE_CMAP = ["#D7E4EA", "#D7E4EA"]   # uniform pale blue

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
    ytitle="Superior-Inferior (mm)",   # Y is vertical per spec
    ztitle="Anterior-Posterior (mm)",  # Z is depth per spec
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
    "F=Front  B=Back  L=Left  R=Right  Y=Rotate  V=Create Video  Space=Stop",
    position=(20, 20), font_size=10, color="black",
)

# ── CAMERA VIEWS ─────────────────────────────────────────────────────────────
# Camera is FIXED; the meshes rotate around the world Y-axis.
# Front view: camera sits along the -Z axis (anterior of patient faces viewer).

def front_view():
    plotter.camera_position = [
        (0, PIVOT[1], PIVOT[2] - 450),   # camera in front along -Z
        tuple(PIVOT),                      # look at pivot
        (0, 1, 0),                         # Y is up
    ]
    plotter.render()

def back_view():
    plotter.camera_position = [
        (0, PIVOT[1], PIVOT[2] + 450),
        tuple(PIVOT),
        (0, 1, 0),
    ]
    plotter.render()

def left_view():
    plotter.camera_position = [
        (PIVOT[0] - 450, PIVOT[1], PIVOT[2]),
        tuple(PIVOT),
        (0, 1, 0),
    ]
    plotter.render()

def right_view():
    plotter.camera_position = [
        (PIVOT[0] + 450, PIVOT[1], PIVOT[2]),
        tuple(PIVOT),
        (0, 1, 0),
    ]
    plotter.render()

# ── ROTATION HELPERS ─────────────────────────────────────────────────────────
#
# Rotation strategy
# ─────────────────
# We rotate every mesh IN PLACE around the shared PIVOT using the world Y-axis
# [0, 1, 0].  The camera stays at the front-view position so the viewer sees
# the lungs spin — front → right-side (90°) → back (180°) → left-side (270°)
# → front again (360°).
#
# PyVista's rotate_y(angle, point=...) applies a RIGHT-HAND rotation around
# the Y-axis.  A positive angle rotates:
#   +X → +Z  (i.e. the right side of the model swings toward the camera)
# which is CLOCKWISE when viewed from above (–Y looking down) and matches
# the "clockwise" requirement when the camera sits on the –Z axis looking in.
#
# We step 1° per frame at ~0.05 s/frame → ~18 seconds for a full revolution.

rotation_running = False

STEP_DEG   = 1          # degrees per frame
FRAME_WAIT = 0.05       # seconds between frames  (~20 fps)


def _all_rotating_actors():
    """Yield every mesh that must rotate together."""
    yield left_shell
    yield right_shell
    if left_tree  is not None: yield left_tree
    if right_tree is not None: yield right_tree
    if finding_dots is not None: yield finding_dots


def _rotate_one_step(deg: float):
    """Rotate all actors by *deg* degrees around the world Y-axis at PIVOT."""
    for mesh in _all_rotating_actors():
        mesh.rotate_y(deg, point=PIVOT, inplace=True)


def rotate_y_slow():
    """
    360° clockwise rotation around the vertical (Y) world axis.
    Camera is FIXED at the front-view position.
    Milestones are printed at 90 / 180 / 270 / 360°.
    """
    global rotation_running

    if rotation_running:
        print("[INFO] Rotation already running — press Space to stop first.")
        return

    rotation_running = True
    front_view()          # ensure camera is at the canonical front position
    print("[INFO] Starting 360° Y-axis rotation (clockwise from front) …")

    total_angle = 0.0

    while total_angle < 360.0 and rotation_running:
        _rotate_one_step(STEP_DEG)
        total_angle += STEP_DEG
        plotter.render()
        time.sleep(FRAME_WAIT)

        # ── milestone logging ──
        angle_int = int(round(total_angle))
        if angle_int == 90:
            print("[INFO]  90° — RIGHT LATERAL VIEW")
        elif angle_int == 180:
            print("[INFO] 180° — BACK VIEW (posterior facing viewer)")
        elif angle_int == 270:
            print("[INFO] 270° — LEFT LATERAL VIEW")
        elif angle_int == 360:
            print("[INFO] 360° — FRONT VIEW RESTORED")

    # If stopped early, bring back to front view
    if not rotation_running:
        print("[INFO] Rotation cancelled by user.")
    else:
        rotation_running = False
        print("[INFO] Full 360° rotation complete.")


def stop_rotation():
    global rotation_running
    rotation_running = False
    print("[INFO] Rotation stopped.")


# ── VIDEO EXPORT ──────────────────────────────────────────────────────────────

def record_rotation_video():
    """
    Capture one full 360° rotation to VIDEO_PATH (MP4).
    The camera is fixed at the front view; the meshes rotate.
    Each degree-step is written as one frame.
    """
    global rotation_running

    if rotation_running:
        print("[INFO] Already rotating — stop first (Space) before recording.")
        return

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)

    print(f"[INFO] Recording 360° rotation to {VIDEO_PATH} …")
    rotation_running = True
    front_view()

    # Open the movie writer (requires ffmpeg on PATH)
    plotter.open_movie(VIDEO_PATH, framerate=20)

    total_angle = 0.0
    while total_angle < 360.0 and rotation_running:
        _rotate_one_step(STEP_DEG)
        total_angle += STEP_DEG
        plotter.render()
        plotter.write_frame()

        angle_int = int(round(total_angle))
        if angle_int in (90, 180, 270, 360):
            print(f"[INFO] {angle_int:3d}° frame captured")

    plotter.close_movie()
    rotation_running = False

    if total_angle >= 360.0:
        print(f"[INFO] Video saved → {VIDEO_PATH}")
    else:
        print("[INFO] Recording cancelled; partial video may exist.")


# ── KEY BINDINGS ──────────────────────────────────────────────────────────────

plotter.add_key_event("f",     front_view)
plotter.add_key_event("b",     back_view)
plotter.add_key_event("l",     left_view)
plotter.add_key_event("r",     right_view)
plotter.add_key_event("y",     rotate_y_slow)
plotter.add_key_event("v",     record_rotation_video)
plotter.add_key_event("space", stop_rotation)

# ── LAUNCH ────────────────────────────────────────────────────────────────────

front_view()   # set initial camera before show()
plotter.show()
