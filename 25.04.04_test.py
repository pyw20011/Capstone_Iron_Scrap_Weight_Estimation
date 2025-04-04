import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from config import cfg
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner

cfg.IMG_SIZE = 224
cfg.MEAN = [0.5, 0.5, 0.5]
cfg.STD = [0.5, 0.5, 0.5]

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
    ])
    image = transform(image)
    return image.unsqueeze(0)  # [1, 3, H, W]

# DataParallel 저장된 state_dict 처리 함수
def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

# 모델 초기화
encoder = Encoder(cfg).to(device)
decoder = Decoder(cfg).to(device)
merger = Merger(cfg).to(device)
refiner = Refiner(cfg).to(device)

# 체크포인트 경로
checkpoint_path = './checkpoints/Pix2Vox-A-ShapeNet.pth'

# 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location=device)

# "module." prefix 제거하고 로드
encoder.load_state_dict(remove_module_prefix(checkpoint['encoder_state_dict']))
decoder.load_state_dict(remove_module_prefix(checkpoint['decoder_state_dict']))
merger.load_state_dict(remove_module_prefix(checkpoint['merger_state_dict']))
refiner.load_state_dict(remove_module_prefix(checkpoint['refiner_state_dict']))

# 평가 모드 설정
encoder.eval()
decoder.eval()
merger.eval()
refiner.eval()

# ====== 이미지 경로 입력 ======
image_path = './chair.jpg'  # 이 부분을 실제 경로로 수정

# ====== 이미지 불러오기 및 예측 ======
images = preprocess_image(image_path).to(device)  # [1, 3, 224, 224]
images = images.unsqueeze(1)  # [1, 1, 3, 224, 224] (encoder 요구사항 맞춤)

image_features = encoder(images)  # [1, n, 512, 28, 28]
raw_features, gen_volumes = decoder(image_features)  # 🔹 수정된 부분

# 🔹 크기 출력 (디버깅 용도)
print("Raw Feature Shape:", raw_features.shape)  # [1, n_views, 9, 32, 32, 32]
print("Generated Volume Shape:", gen_volumes.shape)  # [1, n_views, 32, 32, 32]

# 🔹 `gen_volumes`만 사용해서 `coarse_volumes` 생성
coarse_volumes = gen_volumes  # [B, n_views, 32, 32, 32]

# 🔹 Merger 적용 (raw_features 추가)
merged_volume = merger(raw_features, coarse_volumes)  # [1, 32, 32, 32]

# 🔹 Refinement 적용
refined_volume = refiner(merged_volume)

# 🔹 이진화 및 출력
binary_voxel = (refined_volume > 0.3).float()

# 결과 확인 (간단히 형태 출력)
print("Predicted Voxel Shape:", binary_voxel.shape)  # [1, 32, 32, 32]
print("Non-zero voxel count:", binary_voxel.sum().item())




# --------------------------- 시각화 ------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NumPy 변환
voxel_data = binary_voxel.squeeze().cpu().numpy()  # [32, 32, 32]

# 3D 시각화
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# 활성화된 Voxel 좌표 추출
x, y, z = voxel_data.nonzero()

ax.scatter(x, y, z, zdir='z', c='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
