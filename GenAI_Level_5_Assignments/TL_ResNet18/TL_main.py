import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------------
# Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Image Transformations
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Load Dataset (downloaded images from google)
# -------------------------------
# Training dataset should be organized as:
# dataset/train/car
# dataset/train/truck
# dataset/train/bicycle
# dataset/train/pedestrian
# -------------------------------
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=train_transform
)

# Validation dataset should be organized as:
# dataset/validation/car
# dataset/validation/truck
# dataset/validation/bicycle
# dataset/validation/pedestrian
# -------------------------------
val_dataset = datasets.ImageFolder(
    root="dataset/validation",
    transform=val_transform
)

# -------------------------------
# Data Loaders for training and validation
# -------------------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# -------------------------------
# Load Pretrained ResNet
# -------------------------------
# To download the pretrained ResNet model weights
# model will be downloaded to path:
# Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to 
# C:\Users\bijay/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
# -------------------------------

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

# -------------------------------
# Replace Final Layer
# -------------------------------
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 4)

model = model.to(device)

# -------------------------------
# Loss and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

# -------------------------------
# Save the trained Model
# -------------------------------
torch.save(model.state_dict(), "vehicle_classifier.pth")

print("Model saved successfully!")
