import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# ============================================================
#  LUNG SEGMENTATION
#
#  Uses classical image processing (Otsu threshold + morphology +
#  connected components) instead of the U-Net, because the U-Net
#  was trained on pre-cropped "lung_only" images (Shenzhen dataset)
#  and produces incorrect masks when given full uncropped X-rays.
#
#  The U-Net loader is kept below for future use if you retrain on
#  full-image data, but inference now uses the classical pipeline.
# ============================================================


# ============================================================
#  KEPT FOR FUTURE USE — U-Net loader
#  (currently not called; swap back in segment_lungs() if you
#   retrain lung_unet.pth on full uncropped chest X-rays)
# ============================================================

def load_lung_unet(model_path: str, device):
    """Load the trained lung segmentation U-Net (not used at inference)."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Lung U-Net loaded (not used at inference — classical segmentation active)")
    return model


# ============================================================
#  CLASSICAL LUNG SEGMENTATION
# ============================================================

def _segment_lungs_classical(gray: np.ndarray, target_size=(224, 224)):
    """
    Segments lung fields from a grayscale frontal chest X-ray using
    classical image processing:

      1. CLAHE to normalize brightness
      2. Otsu threshold (lung fields are the bright regions on CXR)
      3. Border removal (body edges often get thresholded in — widened
         at the top to also strip the clavicle/shoulder-heavy strip,
         which is where this segmenter previously leaked)
      4. Morphological close + open to fill holes and remove noise
      5. Erode to break weak bridges to adjacent bright soft tissue
         (shoulder/deltoid), take the largest connected component in
         each half, then dilate ONLY that winning blob back out — this
         stops the segmenter from swallowing the shoulder into the
         "lung" blob when they're touching, without permanently
         shrinking the true lung boundary
      6. Combine into full lung mask

    Returns:
        lung_mask      : HxW uint8 (0/255) at target_size
        left_lung_mask : HxW uint8 (0/255), image-left (patient right lung)
        right_lung_mask: HxW uint8 (0/255), image-right (patient left lung)
    """
    H, W = gray.shape[:2]

    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Otsu threshold
    _, thresh = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Remove image border. Top border widened to ~6% of image
    # height to strip the clavicle/shoulder region, which on portable
    # or slightly rotated films is often bright enough to merge with
    # the lung blob after morphological closing.
    top_border  = max(int(H * 0.06), 8)
    side_border = max(int(W * 0.03), 8)
    thresh[:top_border, :]   = 0
    thresh[-8:, :]           = 0
    thresh[:, :side_border]  = 0
    thresh[:, -side_border:] = 0

    # Step 4: Morphological cleanup
    kernel  = np.ones((9, 9), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    mid = W // 2

    # Step 5: erode to break thin bridges between lung and shoulder
    # tissue BEFORE picking the largest component, so a touching
    # shoulder blob doesn't get pulled in as "the largest component".
    erode_kernel = np.ones((17, 17), np.uint8)
    eroded = cv2.erode(cleaned, erode_kernel, iterations=1)

    left_half_eroded  = eroded.copy(); left_half_eroded[:, mid:]  = 0
    right_half_eroded = eroded.copy(); right_half_eroded[:, :mid] = 0

    def largest_component(binary):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if n <= 1:
            return np.zeros_like(binary)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out  = np.zeros_like(binary)
        out[labels == best] = 255
        return out

    left_seed  = largest_component(left_half_eroded)
    right_seed = largest_component(right_half_eroded)

    # Dilate the ISOLATED winning blob back out to restore true lung
    # extent (apex/base aren't clipped), but AND it against the
    # original (non-eroded) half-mask so it can't grow back into the
    # shoulder tissue we just cut away.
    left_half  = cleaned.copy(); left_half[:, mid:]  = 0
    right_half = cleaned.copy(); right_half[:, :mid] = 0

    left_mask  = cv2.dilate(left_seed,  erode_kernel, iterations=1)
    right_mask = cv2.dilate(right_seed, erode_kernel, iterations=1)
    left_mask  = cv2.bitwise_and(left_mask,  left_half)
    right_mask = cv2.bitwise_and(right_mask, right_half)

    # Step 6: Combine + light feathering
    combined = cv2.bitwise_or(left_mask, right_mask)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    combined = (combined > 127).astype(np.uint8) * 255

    # Resize to target
    lung_mask  = cv2.resize(combined,   target_size, interpolation=cv2.INTER_NEAREST)
    left_mask  = cv2.resize(left_mask,  target_size, interpolation=cv2.INTER_NEAREST)
    right_mask = cv2.resize(right_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return lung_mask, left_mask, right_mask


# ============================================================
#  PUBLIC API  (same signatures as before — pipeline.py unchanged)
# ============================================================

def segment_lungs(unet_model, pil_or_gray_img, device,
                   target_size=(224, 224), threshold=0.5):
    """
    Segment lung fields from a chest X-ray.

    Args:
        unet_model      : loaded U-Net (passed in but not used currently)
        pil_or_gray_img : PIL Image (mode "L") or HxW uint8 numpy array
        device          : torch device (unused, kept for API compatibility)
        target_size     : (W, H) output mask size
        threshold       : unused (kept for API compatibility)

    Returns:
        lung_mask  : HxW uint8 (0/255)
        lung_prob  : HxW float32 [0,1]  (binary 0/1 from classical seg)
    """
    if hasattr(pil_or_gray_img, "convert"):
        gray = np.array(pil_or_gray_img.convert("L"))
    else:
        gray = pil_or_gray_img.copy()

    gray_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
    lung_mask, _, _ = _segment_lungs_classical(gray_resized, target_size)
    lung_prob = (lung_mask / 255.0).astype(np.float32)

    return lung_mask, lung_prob


def split_left_right_lung(lung_mask: np.ndarray):
    """
    Split a binary lung mask into left-lung and right-lung masks,
    using standard PA chest X-ray radiographic convention: the patient
    faces the viewer, so the patient's RIGHT lung appears on the
    IMAGE-LEFT half, and the patient's LEFT lung appears on the
    IMAGE-RIGHT half.

    This matches the convention already used in tb_portal_viewer.py's
    3-D pipeline (`if cx < 112: ... lung_name = "RIGHT"`), so 2-D
    reports, lobe-risk scoring, and the 3-D viewer all agree on which
    side is which.

    NOTE: previously this function returned (image-left-half,
    image-right-half) *labeled* as (left_mask, right_mask), which was
    anatomically backwards — it silently swapped patient left/right.
    Fixed here to return the correct anatomical sides.

    Returns:
        left_lung_mask  : HxW uint8 (0/255) — anatomical LEFT lung
                          (image-RIGHT half of the X-ray)
        right_lung_mask : HxW uint8 (0/255) — anatomical RIGHT lung
                          (image-LEFT half of the X-ray)
    """
    H, W = lung_mask.shape
    mid  = W // 2

    image_left_half  = lung_mask.copy(); image_left_half[:, mid:]  = 0
    image_right_half = lung_mask.copy(); image_right_half[:, :mid] = 0

    anatomical_left_mask  = image_right_half   # patient's left lung
    anatomical_right_mask = image_left_half    # patient's right lung

    return anatomical_left_mask, anatomical_right_mask


def apply_lung_mask_to_cam(grayscale_cam: np.ndarray, lung_mask: np.ndarray,
                            outside_value: float = 0.0,
                            feather_px: int = 9) -> np.ndarray:
    """
    Restrict a GradCAM activation map to the lung field only.

    Args:
        grayscale_cam : HxW float32 in [0,1]
        lung_mask     : HxW uint8 (0/255), same size
        outside_value : value outside lung (default 0 = no activation)
        feather_px    : gaussian blur size for soft mask edges

    Returns:
        masked_cam : HxW float32 in [0,1]
    """
    mask_f = lung_mask.astype(np.float32) / 255.0

    if feather_px > 0:
        k      = feather_px if feather_px % 2 == 1 else feather_px + 1
        mask_f = cv2.GaussianBlur(mask_f, (k, k), 0)

    masked_cam = grayscale_cam * mask_f + outside_value * (1.0 - mask_f)
    return np.clip(masked_cam, 0, 1)


# ============================================================
#  NEW: LOBE-WISE SPLITTING
#
#  Splits a single lung's binary mask into its anatomical lobes,
#  purely by vertical (superior <-> inferior) position within that
#  lung's own bounding box. This mirrors the z_pct thresholds already
#  used for lobe classification in tb_portal_viewer.py's 3-D pipeline
#  (RIGHT: >66% sup = upper, 33-66% = middle, <33% = lower;
#   LEFT:  >50% sup = upper, else lower), so 2-D and 3-D lobe labels
#  stay consistent with each other.
#
#  This does NOT attempt to trace real fissure anatomy (that would
#  need a dedicated fissure-segmentation model) — it's a practical
#  approximation good enough for percentage risk reporting.
# ============================================================

def get_lobe_masks(single_lung_mask: np.ndarray, side: str) -> dict:
    """
    Split one lung's binary mask into lobe sub-masks.

    Args:
        single_lung_mask : HxW uint8 (0/255) — mask for ONE lung only
                            (output of split_left_right_lung, not the
                            combined lung_mask)
        side              : "right" (3 lobes) or "left" (2 lobes)

    Returns:
        dict of lobe_name -> HxW uint8 (0/255) mask, e.g.
        {"RIGHT UPPER LOBE": mask, "RIGHT MIDDLE LOBE": mask, ...}
        Returns {} if the lung mask is empty.
    """
    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got: {side}")

    ys, xs = np.where(single_lung_mask > 0)
    if len(ys) == 0:
        return {}

    y_min, y_max = int(ys.min()), int(ys.max())
    height = max(y_max - y_min, 1)

    H, W = single_lung_mask.shape
    # row_pct: 0.0 = top of lung (superior), 1.0 = bottom of lung (inferior)
    row_idx  = np.arange(H).reshape(-1, 1).astype(np.float32)
    row_pct  = np.clip((row_idx - y_min) / height, 0.0, 1.0)
    row_pct  = np.repeat(row_pct, W, axis=1)   # broadcast to HxW

    mask_bool = single_lung_mask > 0

    lobes = {}
    if side == "right":
        upper_b  = mask_bool & (row_pct < 0.33)
        middle_b = mask_bool & (row_pct >= 0.33) & (row_pct < 0.66)
        lower_b  = mask_bool & (row_pct >= 0.66)
        lobes["RIGHT UPPER LOBE"]  = (upper_b  * 255).astype(np.uint8)
        lobes["RIGHT MIDDLE LOBE"] = (middle_b * 255).astype(np.uint8)
        lobes["RIGHT LOWER LOBE"]  = (lower_b  * 255).astype(np.uint8)
    else:
        upper_b = mask_bool & (row_pct < 0.50)
        lower_b = mask_bool & (row_pct >= 0.50)
        lobes["LEFT UPPER LOBE"] = (upper_b * 255).astype(np.uint8)
        lobes["LEFT LOWER LOBE"] = (lower_b * 255).astype(np.uint8)

    return lobes