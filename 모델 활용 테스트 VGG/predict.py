import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

model_path = sys.argv[1]
image_path = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ VGG16 회귀 모델
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_weight(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return output.item()

predicted_weight = predict_weight(model, image_path)
print(f"예측된 스크랩 무게: {predicted_weight:.2f} kg")
