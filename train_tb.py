import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using Device:", device)

# --------------------------------
# TRANSFORMS
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# --------------------------------
# DATASETS
# --------------------------------

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=transform
)

print("Classes:", train_dataset.class_to_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)
print("Classes:", train_dataset.class_to_idx)
print("Train Images:", len(train_dataset))
print("Val Images:", len(val_dataset))

# --------------------------------
# MODEL
# --------------------------------

model = models.densenet121(
    weights="DEFAULT"
)

num_ftrs = model.classifier.in_features

model.classifier = nn.Linear(
    num_ftrs,
    2
)

model = model.to(device)

# --------------------------------
# TB CLASS WEIGHTING
# --------------------------------

weights = torch.tensor(
    [1.0, 5.0]
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

best_val_acc = 0

# --------------------------------
# TRAINING
# --------------------------------

epochs = 35

for epoch in range(35):

    model.train()

    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        _, pred = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            pred == labels
        ).sum().item()

    train_acc = (
        100 * train_correct / train_total
    )

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, pred = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                pred == labels
            ).sum().item()

    val_acc = (
        100 * val_correct / val_total
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train={train_acc:.2f}% "
        f"Val={val_acc:.2f}%"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "tb_model.pth"
        )

        print(
            f"Best Model Saved "
            f"(Val={val_acc:.2f}%)"
        )

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_acc:.2f}%"
)