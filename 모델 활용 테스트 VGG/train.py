import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class ScrapWeightDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        weight = torch.tensor([self.data.iloc[idx, 1]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, weight

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ScrapWeightDataset("data.csv", "images", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ VGG16 기반 회귀 모델
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    running_loss = 0.0
    for images, weights in train_loader:
        images, weights = images.to(device), weights.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

torch.save(model.state_dict(), "model.pth")
