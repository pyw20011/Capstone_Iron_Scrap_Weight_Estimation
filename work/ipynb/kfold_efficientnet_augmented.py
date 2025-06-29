import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

start_time = time.time()

# ================================
# 데이터셋 정의
# ================================
class ScrapDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, label_encoder=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = label_encoder or LabelEncoder()
        self.data['class_idx'] = self.label_encoder.fit_transform(self.data['weight_class'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.data.iloc[idx]['class_idx'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

class ScrapDatasetWithFilenames(ScrapDataset):
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.data.iloc[idx]['class_idx'], dtype=torch.long)
        filename = self.data.iloc[idx]['filename']
        if self.transform:
            image = self.transform(image)
        return image, label, filename

# ================================
# 모델 정의
# ================================
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.base_model = models.efficientnet_v2_l(models.EfficientNet_V2_L_Weights.DEFAULT)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, 3)

        # grad-cam  
        self.feature_map_for_gradcam = None
        self._register_gradcam_hook()

    def _register_gradcam_hook(self):
        def forward_hook(module, input, output):
            self.feature_map_for_gradcam = output
            if output.requires_grad:  # 🔒 조건 추가 evaluate에서 
                output.retain_grad() # backward용 gradient 확보

        # 마지막 Conv layer (efficientnet에서는 block7[-1]이 conv)
        self.base_model.features[-1].register_forward_hook(forward_hook)

    def forward(self, x):
        return self.base_model(x)

# ================================
# 평가 함수
# ================================
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            output = model(imgs)
            _, pred = torch.max(output, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    acc = (np.array(preds) == np.array(labels)).mean() * 100
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return acc, precision, recall, f1

def get_predictions(model, dataloader, device, label_encoder):
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            pred_classes = label_encoder.inverse_transform(preds.cpu().numpy())
            true_classes = label_encoder.inverse_transform(labels.cpu().numpy())
            for fname, pred_cls, true_cls in zip(filenames, pred_classes, true_classes):
                results.append({
                    "filename": fname,
                    "predicted_label": pred_cls,
                    "true_label": true_cls
                })
    return results

# ================================
# 실행
# ================================
csv_path = "data.csv"
img_dir = "images"
df = pd.read_csv(csv_path)

df['filename'] = df['filename'].apply(lambda x: f"{int(x):03}.jpg")

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

augmentation_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_predictions = []
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
    print(f"\n===== Fold {fold} =====")
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    label_encoder = LabelEncoder()
    train_df['class_idx'] = label_encoder.fit_transform(train_df['weight_class'])

    # 증강 포함 학습셋 생성
    aug_images, aug_labels = [], []
    for _, row in train_df.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['class_idx'], dtype=torch.long)

        aug_images.append(base_transform(image))
        aug_labels.append(label)

        for _ in range(9):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                random.choice(augmentation_transforms),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            aug_images.append(transform(image))
            aug_labels.append(label)

    train_dataset = TensorDataset(torch.stack(aug_images), torch.stack(aug_labels))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = ScrapDataset(test_df, img_dir, transform=base_transform, label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Fold {fold} - Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

    acc, precision, recall, f1 = evaluate(model, test_loader, device)
    print(f"✅ Accuracy: {acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    fold_scores.append({'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})

    test_dataset_with_fname = ScrapDatasetWithFilenames(test_df, img_dir, transform=base_transform, label_encoder=label_encoder)
    test_loader_with_fname = DataLoader(test_dataset_with_fname, batch_size=8)
    fold_preds = get_predictions(model, test_loader_with_fname, device, label_encoder)
    all_predictions.extend(fold_preds)

    from gradcam_utils import show_and_save_gradcam_examples  # 파일로 만든 거 불러오기

    # Grad-CAM 실행 (fold별 폴더에 저장)
    class_labels = list(label_encoder.classes_)
    output_dir = f"gradcam_fold{fold}"
    single_loader = DataLoader(test_dataset_with_fname, batch_size=1, shuffle=False)

    show_and_save_gradcam_examples(model, single_loader, class_labels, output_dir=output_dir, max_per_class=2)

# 평균 성능 출력
avg_scores = pd.DataFrame(fold_scores).mean()
print("\n🔚 K-Fold 평균 성능:")
print(f"✅ Accuracy: {avg_scores['acc']:.2f}%")
print(f"📍 Precision: {avg_scores['precision']:.4f} | Recall: {avg_scores['recall']:.4f} | F1: {avg_scores['f1']:.4f}")

# 결과 저장 및 분석
result_df = pd.DataFrame(all_predictions)
result_df.to_csv("kfold_predictions_with_truth.csv", index=False)
print("📁 예측 결과가 'kfold_predictions_with_truth.csv'에 저장되었습니다.")

print("\n📊 Classification Report:")
print(classification_report(result_df['true_label'], result_df['predicted_label'], digits=4))

cm = confusion_matrix(result_df['true_label'], result_df['predicted_label'], labels=label_encoder.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


end_time = time.time()
elapsed = end_time - start_time
print(f"\n전체 실행 시간: {elapsed:.2f}초")