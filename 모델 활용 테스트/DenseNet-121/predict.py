import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DenseNet 입력 사이즈
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 모델 로드 함수
def load_model(model_path):
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    return model

# 예측 함수
def predict_weight(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    return output.item()

# 실행
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python predict.py model_full.pth test.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    model = load_model(model_path)
    predicted_weight = predict_weight(model, image_path)
    print(f"예측된 무게: {predicted_weight:.1f}g")
