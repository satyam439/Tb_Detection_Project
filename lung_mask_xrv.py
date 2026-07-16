import os
import sys
import numpy as np
import cv2
import torch

try:
    import torchxrayvision as xrv
except ImportError as e:
    raise ImportError(
        "torchxrayvision is required for this module.\n"
        "Install it with:  pip install torchxrayvision"
    ) from e


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_seg_model = None  # loaded lazily, once, on first call


def _load_segmentation_model():
    """Load (and cache) the pretrained PSPNet chest anatomy segmenter."""
    global _seg_model
    if _seg_model is None:
        print("[INFO] Loading torchxrayvision PSPNet chest segmentation model ...")
        _seg_model = xrv.baseline_models.chestx_det.PSPNet()
        _seg_model.to(_device)
        _seg_model.eval()
        print(f"[INFO] Segmentation model ready on {_device}")
        print(f"[INFO] Available organ channels: {_seg_model.targets}")
    return _seg_model


def _load_grayscale(image_path_or_array) -> np.ndarray:
    """Accepts a file path, a PIL Image, or a numpy array; returns HxW uint8 grayscale."""
    if isinstance(image_path_or_array, str):
        gray = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path_or_array}")
        return gray
    if hasattr(image_path_or_array, "convert"):  # PIL Image
        return np.array(image_path_or_array.convert("L"))
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _center_crop_square(img: np.ndarray):
    """
    Crop the longer dimension so the image is square, centered — this
    is required by the model (it only accepts square input; see
    torchxrayvision's own XRayCenterCrop, replicated here so we can
    invert the crop afterward to map the mask back to full-image size).

    Returns:
        cropped image, and (y0, x0, side) so the crop can be undone.
    """
    h, w = img.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side], (y0, x0, side)


def generate_lung_masks(image_path_or_array, target_size=(224, 224), threshold=0.5):
    """
    Generate lung field masks from a chest X-ray using a pretrained
    anatomical segmentation model (not a brightness threshold).

    Args:
        image_path_or_array : file path, PIL Image, or numpy array
        target_size          : (W, H) for the returned masks — matches
                               the rest of the TB pipeline (224, 224)
        threshold             : probability threshold for binarizing
                               each lung channel (model outputs sigmoid
                               probabilities per organ, per pixel)

    Returns:
        lung_mask       : HxW uint8 (0/255) — union of both lungs
        left_lung_mask  : HxW uint8 (0/255) — patient's anatomical LEFT lung
        right_lung_mask : HxW uint8 (0/255) — patient's anatomical RIGHT lung

        (Radiographic convention: patient faces the viewer, so the
        patient's RIGHT lung appears on the IMAGE-LEFT half. The model
        outputs "Left Lung"/"Right Lung" channels directly in patient
        anatomical terms, so no manual left/right inference is needed
        here — unlike the classical image-half-split approach.)
    """
    model = _load_segmentation_model()

    gray_full = _load_grayscale(image_path_or_array)
    orig_h, orig_w = gray_full.shape

    # ---- normalize to the model's expected [-1024, 1024] range ----
    normalized = xrv.utils.normalize(gray_full.astype(np.float32), maxval=255)

    # ---- center-crop to square (model requires H == W) ----
    cropped, (y0, x0, side) = _center_crop_square(normalized)

    tensor = torch.from_numpy(cropped).unsqueeze(0).unsqueeze(0).float().to(_device)

    with torch.no_grad():
        output = model(tensor)                     # [1, 14, 512, 512]
        probs = torch.sigmoid(output)[0].cpu().numpy()

    targets = model.targets
    try:
        left_idx  = targets.index("Left Lung")
        right_idx = targets.index("Right Lung")
    except ValueError:
        raise RuntimeError(
            f"Expected 'Left Lung' / 'Right Lung' channels not found in "
            f"model output. Available targets were: {targets}"
        )

    left_prob_512  = probs[left_idx]
    right_prob_512 = probs[right_idx]

    left_bin_512  = (left_prob_512  > threshold).astype(np.uint8) * 255
    right_bin_512 = (right_prob_512 > threshold).astype(np.uint8) * 255

    # ---- map each 512x512 prediction back to the ORIGINAL image ----
    # first: resize prediction from 512x512 back to the cropped square size
    left_bin_crop  = cv2.resize(left_bin_512,  (side, side), interpolation=cv2.INTER_NEAREST)
    right_bin_crop = cv2.resize(right_bin_512, (side, side), interpolation=cv2.INTER_NEAREST)

    # second: paste back into a full-size canvas at the crop's original
    # position (anything outside the cropped square — possible on very
    # non-square films — is correctly left as "no lung" / zero)
    left_full  = np.zeros((orig_h, orig_w), dtype=np.uint8)
    right_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    left_full[y0:y0 + side, x0:x0 + side]  = left_bin_crop
    right_full[y0:y0 + side, x0:x0 + side] = right_bin_crop

    # ---- light cleanup: keep only the largest component per lung ----
    # The model is already anatomically aware, so this is just tidying
    # stray false-positive pixels, not doing the heavy lifting.
    def _clean(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    left_full  = _clean(left_full)
    right_full = _clean(right_full)

    # ---- resize to the pipeline's target size ----
    left_lung_mask  = cv2.resize(left_full,  target_size, interpolation=cv2.INTER_NEAREST)
    right_lung_mask = cv2.resize(right_full, target_size, interpolation=cv2.INTER_NEAREST)
    lung_mask = cv2.bitwise_or(left_lung_mask, right_lung_mask)

    return lung_mask, left_lung_mask, right_lung_mask


# ============================================================
# DROP-IN REPLACEMENT for lung_segmentation.segment_lungs()
# Matches the existing call signature used in predict_tb.py so you
# can swap this in with a one-line import change once you've verified
# the output looks right (see the __main__ block below).
# ============================================================

def segment_lungs(unet_model, pil_or_gray_img, device, target_size=(224, 224), threshold=0.5):
    """
    Same call signature as lung_segmentation.segment_lungs, so it can
    be swapped in directly. `unet_model` and `device` are accepted but
    unused (kept for API compatibility) — this always uses the
    pretrained PSPNet segmenter loaded internally.
    """
    lung_mask, _, _ = generate_lung_masks(pil_or_gray_img, target_size, threshold)
    lung_prob = (lung_mask / 255.0).astype(np.float32)
    return lung_mask, lung_prob


def split_left_right_lung_xrv(image_path_or_array, target_size=(224, 224), threshold=0.5):
    """
    Same call signature/output order as lung_segmentation.split_left_right_lung,
    but derived directly from the model's own Left/Right Lung channels
    instead of an image-half heuristic — so it's correct even on
    rotated films or images where the patient isn't perfectly centered.
    """
    _, left_lung_mask, right_lung_mask = generate_lung_masks(
        image_path_or_array, target_size, threshold
    )
    return left_lung_mask, right_lung_mask


# ============================================================
# STANDALONE TEST / VISUAL SANITY CHECK
# ============================================================

def _save_debug_overlay(gray_full, lung_mask, left_mask, right_mask, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    h, w = lung_mask.shape
    gray_resized = cv2.resize(gray_full, (w, h))
    bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)

    overlay = bgr.copy()
    overlay[left_mask  > 0] = (255, 130, 0)    # blue-ish  = anatomical LEFT lung
    overlay[right_mask > 0] = (0, 130, 255)    # orange-ish = anatomical RIGHT lung
    blend = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)

    mask_path    = os.path.join(out_dir, "lung_mask_xrv.png")
    overlay_path = os.path.join(out_dir, "lung_mask_xrv_overlay.png")
    cv2.imwrite(mask_path, lung_mask)
    cv2.imwrite(overlay_path, blend)
    print(f"[INFO] Saved mask     -> {mask_path}")
    print(f"[INFO] Saved overlay  -> {overlay_path}  (blue=LEFT lung, orange=RIGHT lung)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lung_mask_xrv.py <path_to_xray_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"[INFO] Segmenting: {img_path}")

    lung_mask, left_mask, right_mask = generate_lung_masks(img_path)

    coverage_pct = int(np.count_nonzero(lung_mask) / lung_mask.size * 100)
    print(f"[INFO] Lung coverage: {coverage_pct}% of frame")
    print(f"[INFO] Left lung pixels : {np.count_nonzero(left_mask)}")
    print(f"[INFO] Right lung pixels: {np.count_nonzero(right_mask)}")

    gray_full = _load_grayscale(img_path)
    _save_debug_overlay(gray_full, lung_mask, left_mask, right_mask)