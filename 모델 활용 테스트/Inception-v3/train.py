import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

# 하이퍼파라미터 설정
config = {
    "hidden_layers": [256, 64],
    "dropout": 0.3,
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "model_full.pth"
}

# 데이터셋 클래스
class ScrapWeightDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        weight = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(weight, dtype=torch.float32)

# FC Head 생성
def get_custom_fc(in_features, hidden_layers, dropout):
    layers = []
    for hidden in hidden_layers:
        layers.append(nn.Linear(in_features, hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_features = hidden
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)

# 모델 생성
def build_model(config):
    model = models.inception_v3(weights=None, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = get_custom_fc(in_features, config["hidden_layers"], config["dropout"])
    return model.to(config["device"])

# 학습 함수
def train_model(model, train_loader, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0

        for images, weights in train_loader:
            images = images.to(config["device"])
            weights = weights.to(config["device"]).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model, config["model_save_path"])
    print(f"전체 모델 저장 완료: {config['model_save_path']}")

# 실행
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ScrapWeightDataset("../data.csv", "../[images]", transform)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = build_model(config)
    train_model(model, loader, config)


