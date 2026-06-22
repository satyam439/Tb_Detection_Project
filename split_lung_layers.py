"""
split_lung_layers.py
============================================================
TB Portals' viewer (tbportals.niaid.nih.gov/tb-in-3d) shows TWO
separate layers per lung:
  1. A faint, translucent outer lung SHELL (low opacity, ~20%)
  2. A clearly visible bronchi/vessel TREE inside it (white branching
     structure, full opacity)

Your STL files contain both merged into one mesh. This script splits
each file into exactly those two layers:

  - lung_shell.stl  : the single largest connected piece (the outer
                       pleural surface)
  - lung_tree.stl   : everything else combined (bronchi, vessels, any
                       other branching/fragment pieces) — kept as ONE
                       mesh so it renders as a connected tree, not
                       scattered dots

This is different from inspect_lung_stl.py, which DISCARDED the
small pieces. This script KEEPS both halves, since for the TB Portals
look the tree is the main visual element, not noise to be removed.
============================================================
"""

import os
import pyvista as pv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEFT_STL  = os.path.join(BASE_DIR, "lung_model", "left_lung.stl")
RIGHT_STL = os.path.join(BASE_DIR, "lung_model", "right_lung.stl")


def split_shell_and_tree(stl_path: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}: {stl_path}")
    print(f"{'='*60}")

    mesh = pv.read(stl_path)
    connected = mesh.connectivity(extraction_mode="all")
    region_ids = connected["RegionId"]
    n_regions = int(region_ids.max()) + 1

    sizes = []
    for rid in range(n_regions):
        piece = connected.threshold([rid, rid], scalars="RegionId")
        sizes.append((rid, piece.n_points))
    sizes.sort(key=lambda r: r[1], reverse=True)

    shell_rid = sizes[0][0]
    tree_rids = [r[0] for r in sizes[1:]]

    print(f"Total pieces: {n_regions}")
    print(f"Shell piece: region {shell_rid} ({sizes[0][1]} points)")
    print(f"Tree pieces: {len(tree_rids)} pieces "
          f"({sum(s[1] for s in sizes[1:])} points combined)")

    shell = connected.threshold([shell_rid, shell_rid], scalars="RegionId").extract_surface()

    if tree_rids:
        tree_pieces = [
            connected.threshold([rid, rid], scalars="RegionId").extract_surface()
            for rid in tree_rids
        ]
        tree = tree_pieces[0]
        for piece in tree_pieces[1:]:
            tree = tree.merge(piece)
    else:
        tree = None
        print("WARNING: no separate bronchi/vessel pieces found in this file — "
              "the tree layer will be empty. Your STL may only contain the "
              "lung shell with no separate airway/vessel mesh.")

    shell_path = stl_path.replace(".stl", "_shell.stl")
    shell.save(shell_path)
    print(f"Saved shell -> {shell_path}")

    if tree is not None:
        tree_path = stl_path.replace(".stl", "_tree.stl")
        tree.save(tree_path)
        print(f"Saved tree  -> {tree_path}")
    else:
        tree_path = None

    return shell_path, tree_path


if __name__ == "__main__":
    for path, label in [(LEFT_STL, "LEFT LUNG"), (RIGHT_STL, "RIGHT LUNG")]:
        if not os.path.exists(path):
            print(f"\nSKIPPED — file not found: {path}")
            continue
        split_shell_and_tree(path, label)

    print(f"\n{'='*60}")
    print("DONE. Each lung is now split into:")
    print("  *_shell.stl  -> the translucent outer lung surface")
    print("  *_tree.stl   -> the bronchi/vessel branching structure")
    print("Use these with tb_portal_viewer.py (updated version) to")
    print("match the TB Portals visual style.")
    print(f"{'='*60}")