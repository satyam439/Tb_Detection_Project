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
      3. Border removal (body edges often get thresholded in)
      4. Morphological close + open to fill holes and remove noise
      5. Largest connected component in each half → left / right lung
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

    # Step 3: Remove image border (8px on each side)
    border = 8
    thresh[:border, :]  = 0
    thresh[-border:, :] = 0
    thresh[:, :border]  = 0
    thresh[:, -border:] = 0

    # Step 4: Morphological cleanup
    kernel  = np.ones((9, 9), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    # Step 5: Largest component in each image half
    mid = W // 2
    left_half  = cleaned.copy(); left_half[:, mid:]  = 0
    right_half = cleaned.copy(); right_half[:, :mid] = 0

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

    left_mask  = largest_component(left_half)
    right_mask = largest_component(right_half)

    # Step 6: Combine + light feathering
    combined = cv2.bitwise_or(left_mask, right_mask)
    combined = cv2.dilate(combined, np.ones((5, 5), np.uint8), iterations=1)
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
    Split a binary lung mask into left-lung and right-lung masks.

    Uses the same classical segmentation internally to guarantee the
    split is consistent with how the full mask was produced.

    Returns:
        left_lung_mask  : HxW uint8 (0/255) — anatomical left lung
                          (image RIGHT side, radiographic convention)
        right_lung_mask : HxW uint8 (0/255) — anatomical right lung
                          (image LEFT side, radiographic convention)
    """
    H, W = lung_mask.shape
    mid  = W // 2

    # Simple halve-and-intersect with the existing mask
    left_mask  = lung_mask.copy(); left_mask[:, mid:]  = 0
    right_mask = lung_mask.copy(); right_mask[:, :mid] = 0

    return left_mask, right_mask


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