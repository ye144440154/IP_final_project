import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import StepLR

# ========= 參數設定 ==========
USE_GPU = torch.cuda.is_available()  # 若沒有GPU改成false
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.05
# ============================

# 載入與合併 CSV
csv1 = pd.read_csv('Grading_features_1.csv')
csv2 = pd.read_csv('Grading_features_modified.csv')
csv2 = csv2.iloc[:, 3:] # 去掉前3欄
df = pd.concat([csv1, csv2], axis=1)

# Shuffle 整份資料
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 擷取特徵與標籤
feature_columns = ['Hue_Mean', 'Hue_Std', 'Saturation_Mean', 'Aspect_Ratio', 'H_hist_bin_0', 'H_hist_bin_1', 'H_hist_bin_2', 'H_hist_bin_3', 'H_hist_bin_4', 'H_hist_bin_5', 'H_hist_bin_6', 'H_hist_bin_7', 'S_hist_bin_0', 'S_hist_bin_1', 'S_hist_bin_2', 'S_hist_bin_3', 'S_hist_bin_4', 'S_hist_bin_5', 'S_hist_bin_6', 'S_hist_bin_7', 'V_hist_bin_0', 'V_hist_bin_1', 'V_hist_bin_2', 'V_hist_bin_3', 'V_hist_bin_4', 'V_hist_bin_5', 'V_hist_bin_6', 'V_hist_bin_7', 'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'major_axis_length', 'minor_axis_length', 'edge_density', 'dark_pixels_ratio', 'dark_area_ratio'] 
# feature_columns = ['Hue_Mean', 'Hue_Std', 'Saturation_Mean', 'Aspect_Ratio', 'H_hist_bin_0', 'H_hist_bin_1', 'H_hist_bin_2', 'H_hist_bin_3', 'H_hist_bin_4', 'H_hist_bin_5', 'H_hist_bin_6', 'H_hist_bin_7', 'S_hist_bin_0', 'S_hist_bin_1', 'S_hist_bin_2', 'S_hist_bin_3', 'S_hist_bin_4', 'S_hist_bin_5', 'S_hist_bin_6', 'S_hist_bin_7', 'V_hist_bin_0', 'V_hist_bin_1', 'V_hist_bin_2', 'V_hist_bin_3', 'V_hist_bin_4', 'V_hist_bin_5', 'V_hist_bin_6', 'V_hist_bin_7', 'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'dark_pixels_ratio', 'dark_area_ratio']  
X = df[feature_columns].values
y = df['Label'].values
df.to_csv('combined_output.csv', index=False)

# 特徵標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# # PCA降維
# pca = PCA(n_components=0.95)
# X = pca.fit_transform(X)

# 拆分訓練/測試集
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
) # stratify=y保持各類別在訓練集與測試集的比例一致

print("Any NaN in X?", np.isnan(X).any())
print("Any Inf in X?", np.isinf(X).any())
print("Feature range (min, max):", X.min(), X.max())
print("Label dtype and classes:", y.dtype, np.unique(y))

# 自訂 Dataset
class MangoDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = MangoDataset(X_train, y_train)
test_dataset = MangoDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 定義 MLP 架構
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1] #即column數
num_classes = len(np.unique(y))
model = MLP(input_dim, num_classes).to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss() # 用cross entropy loss處理多類別分類
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 訓練迴圈
def train():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0 # 每一epoch算一次loss
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad() # train loader的每筆資料算loss前先把梯度歸零
            outputs = model(inputs)
            loss = criterion(outputs, labels) # 1. 把模型對於每個class的輸出套softmax 2.計算cross entropy loss
            loss.backward() # backward propagation根據loss算梯度
            optimizer.step() # 由梯度更新模型參數

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# 算準確率
def evaluate():
    model.eval()
    correct = total = 0
    all_results = []
    
    with torch.no_grad(): # 告訴pytorch現在不需要計算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 儲存每筆結果
            for i in range(len(labels)):
                idx = total - len(labels) + i  # test dataset 中對應的原始 index
                result = {
                    "Folder": test_dataset.x[idx].cpu().numpy(),  # placeholder
                    "Filename": "N/A",  # placeholder
                    "True_Label": labels[i].item(),
                    "Predicted_Label": predicted[i].item(),
                    "Correct": labels[i].item() == predicted[i].item()
                }
                all_results.append(result)
                
    print(f"Accuracy: {100 * correct / total:.2f}%")
    
    # 把對應原始 df 的資料夾與檔名抓出來
    df_test = df.loc[idx_test].reset_index(drop=True)
    df_test_result = pd.DataFrame(all_results)
    df_test_result["Folder"] = df_test["Folder"]
    df_test_result["Filename"] = df_test["Filename"]
    df_test_result.to_csv("test_predictions_mlp.csv", index=False)
    print("Saved prediction results to test_predictions.csv")

# 主程式
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()
    evaluate()
