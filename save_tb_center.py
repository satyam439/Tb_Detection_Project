import os
import numpy as np


def save_tb_center(grayscale_cam: np.ndarray, output_dir: str) -> tuple:
    """
    Finds the pixel coordinate of peak GradCAM activation and saves
    it to outputs/tb_center.npy as [cx, cy].

    Args:
        grayscale_cam : HxW float32 array (e.g. 224x224), lung-masked
                         GradCAM output
        output_dir    : directory to save tb_center.npy into (usually
                         your OUTPUT_DIR from pipeline.py)

    Returns:
        (cx, cy) — the saved pixel coordinates, for convenience/logging
    """
    peak_idx = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    cy, cx = peak_idx  # numpy argmax returns (row, col) = (y, x)

    out_path = os.path.join(output_dir, "tb_center.npy")
    np.save(out_path, np.array([cx, cy], dtype=np.float32))

    print(f"[INFO] TB hotspot saved: ({cx}, {cy}) -> {out_path}")
    return cx, cy


if __name__ == "__main__":
    # Standalone usage: re-run on an existing heatmap if you already
    # have a saved grayscale CAM array, or adjust this block to load
    # your specific output format.
    print("This module is meant to be imported and called as:")
    print("    from save_tb_center import save_tb_center")
    print("    save_tb_center(grayscale_cam, OUTPUT_DIR)")
    print("right after grayscale_cam is computed in your pipeline.")