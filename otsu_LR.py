import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 2. 경로 설정
from pathlib import Path
base_dir = Path.cwd() / "Alphonso Mangoes Image Dataset"
image_dir = base_dir
csv_path = base_dir / "Physical_properties_Alphonso_Images.xlsx"


# 3. 데이터 불러오기
df = pd.read_excel(csv_path, header=1)

# 4. 이미지 전처리 (OTSU 이진화 → 픽셀 개수 추출)
pixel_areas = []
weights = []
pixel_areas_dict = {}

for idx, row in df.iterrows():
    mango_id = row['Sample No']
    pixel_counts = []

    for i in [1, 2]:  # 각 망고당 2장 이미지
        img_filename = f"{mango_id}_{i}.jpg"
        img_path = base_dir / img_filename
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found. Skipping...")
            continue
        
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: {img_path} could not be loaded. Skipping...")
            continue
        
        # OTSU 이진화
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 픽셀 개수 세기 (하얀색 픽셀)
        pixel_count = np.sum(binary_img == 0)
        # print(pixel_count)
        pixel_counts.append(pixel_count)

    if pixel_counts:
        avg_pixel_count = np.mean(pixel_counts)
        pixel_areas_dict[mango_id] = avg_pixel_count
        # print("평균",avg_pixel_count)
        pixel_areas.append(avg_pixel_count)
        weights.append(mango_id)
        

# 5. 특징(X), 타겟(y) 만들기
X = np.array(pixel_areas).reshape(-1, 1)  # (N, 1) 형태
df_weight = df[df['Sample No'].isin(weights)].reset_index(drop=True)
y = df_weight['Actual Weight (gms)'].values  # 무게 값

# 6. 데이터 분할 (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 모델 학습
reg = LinearRegression()
reg.fit(X_train, y_train)


# 8. 모델 저장
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(reg, f)
print("✅ 모델 저장 완료: linear_regression_model.pkl")

# 9. 모델 테스트 (불러오기)
with open('linear_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

y_pred = loaded_model.predict(X_test)

# 10. 결과 평가
r2 = r2_score(y_test, y_pred)
print(f"✅ 테스트 R2 Score: {r2:.4f}")

# 11. 실제 vs 예측 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Line")
plt.xlabel("Actual Weight (gms)")
plt.ylabel("Predicted Weight (gms)")
plt.title("Actual vs Predicted Mango Weights")
plt.legend()
plt.grid()
plt.show()

# 시각화: 5개 샘플
sample_ids = df['Sample No'].tolist()[:5]

for idx, mango_id in enumerate(sample_ids):
    mango_id = str(mango_id)
    actual_weight = df.loc[df['Sample No'] == mango_id, 'Actual Weight (gms)'].values[0]
    predicted_weight = reg.predict(pixel_areas[idx].reshape(1, -1))[0]

    img_name = f"{mango_id}_1.jpg"
    img_path = os.path.join(image_dir, img_name)
    
    with open(img_path, 'rb') as f:
        img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"이미지 로드 실패: {img_path}")
        continue

    # Otsu 이진화
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Original Image: {mango_id}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title(f"Actual: {actual_weight:.1f}gms\nPredicted: {predicted_weight:.1f}gms")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 회귀선
a = reg.coef_[0]
b = reg.intercept_
equation = f"y = {a:.5e} * x + {b:.2f}"

# 2. 픽셀 수 vs 실제 무게 + 회귀선 시각화
plt.figure(figsize=(8, 6))

# X축은 픽셀 수, Y축은 실제 무게
plt.scatter(X_test, y_test, c='green', label="Pixel Area vs Actual Weight")

# 회귀선 그리기
x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)  # 픽셀 수 범위
y_line = reg.predict(x_line)

plt.plot(x_line, y_line, color='red', linestyle='-', label=f"Regression Line\n{equation}")

plt.xlabel("Pixel Area (number of black pixels)")
plt.ylabel("Actual Weight (gms)")
plt.title("Pixel Area vs Actual Mango Weight with Regression Line")
plt.legend()
plt.grid()
plt.show()
