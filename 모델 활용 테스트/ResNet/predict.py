import torch
from torchvision import models, transforms
from PIL import Image
import sys
import os

# 🔧 이미지 전처리: 학습 때와 동일해야 함!
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ 모델 정의: ResNet50 + FC(1)
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ✅ 이미지 예측 함수
def predict_weight(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        output = model(image)
    return output.item()

# ✅ 메인 실행 부분
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python predict.py model.pth image.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("모델 파일 또는 이미지 파일 경로가 잘못되었습니다.")
        sys.exit(1)

    model = load_model(model_path)
    weight = predict_weight(model, image_path)
    print(f"📦 예측된 스크랩 무게: {weight:.2f} kg")