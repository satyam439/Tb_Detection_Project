import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using Device:", device)

# ==========================================
# PATHS
# ==========================================

IMAGE_DIR = "Shenzhen/lung_only"
MASK_DIR  = "Shenzhen/masks"

# ==========================================
# COLLECT FILES
# ==========================================

pairs = []

for img_name in os.listdir(IMAGE_DIR):

    if not img_name.endswith(".png"):
        continue

    mask_name = img_name.replace(
        ".png",
        "_mask.png"
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    if os.path.exists(mask_path):

        pairs.append(
            (
                os.path.join(
                    IMAGE_DIR,
                    img_name
                ),
                mask_path
            )
        )

print("Total Pairs:", len(pairs))

# ==========================================
# SPLIT
# ==========================================

train_pairs, val_pairs = train_test_split(
    pairs,
    test_size=0.2,
    random_state=42
)

# ==========================================
# DATASET
# ==========================================

class LungDataset(Dataset):

    def __init__(self, pairs):

        self.pairs = pairs

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (256,256)
        )

        mask = cv2.resize(
            mask,
            (256,256)
        )

        image = image.astype(np.float32) / 255.0
        mask  = mask.astype(np.float32) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        return (
            torch.tensor(image),
            torch.tensor(mask)
        )

# ==========================================
# LOADERS
# ==========================================

train_loader = DataLoader(
    LungDataset(train_pairs),
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    LungDataset(val_pairs),
    batch_size=4
)

# ==========================================
# MODEL
# ==========================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

model = model.to(device)

# ==========================================
# LOSS
# ==========================================

loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# ==========================================
# TRAIN
# ==========================================

best_loss = 999

for epoch in range(20):

    model.train()

    train_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)

            loss = loss_fn(
                outputs,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/20 "
        f"Train={train_loss:.4f} "
        f"Val={val_loss:.4f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            "lung_unet.pth"
        )

        print("Best Model Saved")

print("\nDone")