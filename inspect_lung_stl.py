"""
inspect_lung_stl.py
============================================================
Diagnostic + cleanup tool for lung_model/left_lung.stl and
right_lung.stl.

If your 3D viewer shows a tangled, branching, wireframe-like shape
instead of a smooth solid lung, it almost always means the STL file
contains MULTIPLE disconnected pieces merged together — typically the
actual lung pleural surface PLUS the bronchial airway tree and/or
blood vessels, which are thin branching tubes that look exactly like
what you described.

This script:
  1. Loads each STL and reports how many separate disconnected
     pieces (connected components) it contains, with each piece's
     size (number of points) and bounding box.
  2. Identifies the LARGEST piece by volume/point count — this is
     almost always the actual lung surface, since airway/vessel trees
     are made of many small thin segments.
  3. Saves a cleaned version containing ONLY the largest piece as
     left_lung_clean.stl / right_lung_clean.stl, ready to use in
     tb_portal_viewer.py instead of the originals.
============================================================
"""

import os
import pyvista as pv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEFT_STL  = os.path.join(BASE_DIR, "lung_model", "left_lung.stl")
RIGHT_STL = os.path.join(BASE_DIR, "lung_model", "right_lung.stl")


def inspect_and_clean(stl_path: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}: {stl_path}")
    print(f"{'='*60}")

    mesh = pv.read(stl_path)
    print(f"Total points: {mesh.n_points}")
    print(f"Total cells:  {mesh.n_cells}")
    print(f"Bounds: {mesh.bounds}")

    # Split into connected pieces
    connected = mesh.connectivity(largest=False)
    region_ids = connected["RegionId"]
    n_regions = int(region_ids.max()) + 1
    print(f"\nDisconnected pieces found: {n_regions}")

    region_sizes = []
    for rid in range(n_regions):
        piece = connected.threshold([rid, rid], scalars="RegionId")
        region_sizes.append((rid, piece.n_points, piece.bounds))

    region_sizes.sort(key=lambda r: r[1], reverse=True)

    print("\nPiece sizes (largest first):")
    for rid, npts, bounds in region_sizes[:10]:
        size_x = bounds[1] - bounds[0]
        size_y = bounds[3] - bounds[2]
        size_z = bounds[5] - bounds[4]
        print(f"  Region {rid}: {npts} points, "
              f"bbox size ({size_x:.1f}, {size_y:.1f}, {size_z:.1f})")
    if len(region_sizes) > 10:
        print(f"  ... and {len(region_sizes) - 10} smaller pieces")

    # Extract only the largest piece (almost certainly the lung surface;
    # airway/vessel trees are made of many small thin branches)
    largest = mesh.connectivity(largest=True)
    print(f"\nLargest piece alone: {largest.n_points} points "
          f"({largest.n_points / mesh.n_points * 100:.1f}% of original)")

    clean_path = stl_path.replace(".stl", "_clean.stl")
    largest.save(clean_path)
    print(f"Saved cleaned mesh -> {clean_path}")

    return n_regions, largest.n_points, mesh.n_points


if __name__ == "__main__":
    for path, label in [(LEFT_STL, "LEFT LUNG"), (RIGHT_STL, "RIGHT LUNG")]:
        if not os.path.exists(path):
            print(f"\nSKIPPED — file not found: {path}")
            continue
        inspect_and_clean(path, label)

    print(f"\n{'='*60}")
    print("DONE. If a file had multiple disconnected pieces, a")
    print("'_clean.stl' version was saved containing only the largest")
    print("piece (the lung surface, with thin airway/vessel branches")
    print("removed). Point tb_portal_viewer.py at the _clean.stl files")
    print("if the result looks better.")
    print(f"{'='*60}")