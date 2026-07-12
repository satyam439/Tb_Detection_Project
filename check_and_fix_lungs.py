"""
Run this ONCE to permanently fix the STL files:
  python3 fix_stl_and_viewer.py

Fixes:
1. Re-splits lungs.stl with 6mm overlap so mediastinum faces overlap → NO GAP
2. Saves clean left/right shell STL files
"""
import pyvista as pv
import numpy as np
import os

BASE     = os.path.dirname(os.path.abspath(__file__))
LUNG_DIR = os.path.join(BASE, "lung_model")
SRC      = os.path.join(LUNG_DIR, "lungs.stl")

if not os.path.exists(SRC):
    print(f"[ERROR] {SRC} not found"); exit()

print("Loading lungs.stl ...")
m      = pv.read(SRC)
x_vals = m.points[:, 0]
print(f"  X range: [{x_vals.min():.1f}, {x_vals.max():.1f}]")

# ── SPLIT WITH 6mm OVERLAP ────────────────────────────────────────────────────
# Left  lung = x <= +6  (takes 6mm past centreline)
# Right lung = x >= -6  (takes 6mm past centreline)
# The 6mm overlap means both meshes share the mediastinum region →
# when rendered together there is zero visible gap.

OVERLAP = 6.0
left_ids  = np.where(x_vals <= +OVERLAP)[0]
right_ids = np.where(x_vals >= -OVERLAP)[0]

def extract_clean(mesh, ids):
    sub  = mesh.extract_points(ids, adjacent_cells=True)
    sub  = sub.connectivity(extraction_mode='largest')
    sub  = sub.extract_surface(algorithm='dataset_surface')
    sub  = sub.clean()
    sub  = sub.smooth(n_iter=150, relaxation_factor=0.05)
    sub.compute_normals(inplace=True, auto_orient_normals=True)
    return sub

print("Extracting left  lung (x ≤ +6mm) ...")
left  = extract_clean(m, left_ids)
print("Extracting right lung (x ≥ -6mm) ...")
right = extract_clean(m, right_ids)

lb, rb = left.bounds, right.bounds
print(f"\n  Left  : x=[{lb[0]:.1f},{lb[1]:.1f}]  pts={left.n_points}")
print(f"  Right : x=[{rb[0]:.1f},{rb[1]:.1f}]  pts={right.n_points}")
print(f"  Overlap at midline: {lb[1]-rb[0]:.1f} mm  (positive = overlap = NO GAP)")

# Save
left.save(os.path.join(LUNG_DIR,  "left_lung_shell.stl"))
right.save(os.path.join(LUNG_DIR, "right_lung_shell.stl"))
left.save(os.path.join(LUNG_DIR,  "left_lung.stl"))
right.save(os.path.join(LUNG_DIR, "right_lung.stl"))
print("\n✅ STL files saved — now run: python3 tb_portal_viewer.py")