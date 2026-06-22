import os
import random
import shutil

SOURCE_DIR = "Shenzhen/lung_only"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

random.seed(42)

for folder in [
    f"{TRAIN_DIR}/normal",
    f"{TRAIN_DIR}/tb",
    f"{VAL_DIR}/normal",
    f"{VAL_DIR}/tb"
]:
    os.makedirs(folder, exist_ok=True)

all_images = [
    f for f in os.listdir(SOURCE_DIR)
    if f.endswith(".png")
]

normal = []
tb = []

for img in all_images:

    if img.endswith("_0.png"):
        normal.append(img)

    elif img.endswith("_1.png"):
        tb.append(img)

random.shuffle(normal)
random.shuffle(tb)

split_normal = int(len(normal) * 0.8)
split_tb = int(len(tb) * 0.8)

train_normal = normal[:split_normal]
val_normal = normal[split_normal:]

train_tb = tb[:split_tb]
val_tb = tb[split_tb:]

for img in train_normal:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(TRAIN_DIR, "normal", img)
    )

for img in val_normal:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(VAL_DIR, "normal", img)
    )

for img in train_tb:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(TRAIN_DIR, "tb", img)
    )

for img in val_tb:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(VAL_DIR, "tb", img)
    )

print("Dataset Ready")
print("Train Normal:", len(train_normal))
print("Val Normal:", len(val_normal))
print("Train TB:", len(train_tb))
print("Val TB:", len(val_tb))