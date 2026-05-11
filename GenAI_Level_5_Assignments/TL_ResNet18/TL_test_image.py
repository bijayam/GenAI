from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Class Names
classes = ['bicycle', 'car', 'pedestrian', 'truck']

# Load model
model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load("vehicle_classifier.pth"))

model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
#image = Image.open("dataset/validation/car/test1.jpg").convert("RGB")
#image = Image.open("dataset/validation/truck/test1.jpg").convert("RGB")
#image = Image.open("dataset/validation/bicycle/test1.jpg").convert("RGB")
image = Image.open("dataset/validation/pedestrian/test1.jpg").convert("RGB")

image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

print("Predicted Class:", classes[predicted.item()])
