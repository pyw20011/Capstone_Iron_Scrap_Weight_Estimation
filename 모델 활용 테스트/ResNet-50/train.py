import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

# ✅ 사용자 정의 Dataset
class ScrapWeightDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        weight = self.data.iloc[idx, 1]  # assume 2nd column is weight

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(weight, dtype=torch.float32)

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 기준
                         std=[0.229, 0.224, 0.225])
])

# ✅ Dataset & DataLoader
train_dataset = ScrapWeightDataset(csv_file='data.csv', image_dir='images/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ✅ ResNet 불러오기 (ResNet50 기준)
model = models.resnet50(pretrained=True)

# ✅ 회귀용으로 마지막 FC 레이어 수정
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # weight는 1개 값이니까

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ✅ 손실함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, weights in train_loader:
        images = images.to(device)
        weights = weights.to(device).unsqueeze(1)  # (B, 1) 형태로

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "model.pth")
