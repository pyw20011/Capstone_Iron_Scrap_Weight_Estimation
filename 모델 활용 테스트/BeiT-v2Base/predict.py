import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import timm  # 반드시 필요

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# FC Head 정의 (학습 시와 동일하게!)
def get_custom_fc(in_features, hidden_layers, dropout):
    layers = []
    for hidden in hidden_layers:
        layers.append(torch.nn.Linear(in_features, hidden))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        in_features = hidden
    layers.append(torch.nn.Linear(in_features, 1))  # 회귀 출력
    return torch.nn.Sequential(*layers)

# 모델 로드 (state_dict 기반)
def load_model(model_path):
    model = timm.create_model("beitv2_base_patch16_224", pretrained=False, num_classes=0)
    model.head = get_custom_fc(model.num_features, [256, 64], 0.3)  # 학습 구조와 동일해야 함
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model

# 예측 함수
def predict_weight(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        output = model(image)
    return output.item()

# 실행
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python predict.py model_beitv2.pth ../test.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"모델 파일 없음: {model_path}")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"이미지 파일 없음: {image_path}")
        sys.exit(1)

    model = load_model(model_path)
    predicted_weight = predict_weight(model, image_path)
    print(f"예측된 무게: {predicted_weight:.1f}g")

