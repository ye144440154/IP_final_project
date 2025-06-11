import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ========= 參數設定 ==========
N_ESTIMATORS = 300
MAX_DEPTH = 10
RANDOM_STATE = 42
# ============================

# 載入與合併 CSV
csv1 = pd.read_csv('Grading_features_1.csv')
csv2 = pd.read_csv('Grading_features_modified.csv')
csv2 = csv2.iloc[:, 3:]  # 去掉前3欄
df = pd.concat([csv1, csv2], axis=1)

# Shuffle 整份資料
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 擷取特徵與標籤
feature_columns = ['Hue_Mean', 'Hue_Std', 'Saturation_Mean', 'Aspect_Ratio', 'H_hist_bin_0', 'H_hist_bin_1', 'H_hist_bin_2', 'H_hist_bin_3', 'H_hist_bin_4', 'H_hist_bin_5', 'H_hist_bin_6', 'H_hist_bin_7', 'S_hist_bin_0', 'S_hist_bin_1', 'S_hist_bin_2', 'S_hist_bin_3', 'S_hist_bin_4', 'S_hist_bin_5', 'S_hist_bin_6', 'S_hist_bin_7', 'V_hist_bin_0', 'V_hist_bin_1', 'V_hist_bin_2', 'V_hist_bin_3', 'V_hist_bin_4', 'V_hist_bin_5', 'V_hist_bin_6', 'V_hist_bin_7', 'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'major_axis_length', 'minor_axis_length', 'edge_density', 'dark_pixels_ratio', 'dark_area_ratio'] 
# feature_columns = ['Hue_Mean', 'Hue_Std', 'Saturation_Mean', 'Aspect_Ratio', 'H_hist_bin_0', 'H_hist_bin_1', 'H_hist_bin_2', 'H_hist_bin_3', 'H_hist_bin_4', 'H_hist_bin_5', 'H_hist_bin_6', 'H_hist_bin_7', 'S_hist_bin_0', 'S_hist_bin_1', 'S_hist_bin_2', 'S_hist_bin_3', 'S_hist_bin_4', 'S_hist_bin_5', 'S_hist_bin_6', 'S_hist_bin_7', 'V_hist_bin_0', 'V_hist_bin_1', 'V_hist_bin_2', 'V_hist_bin_3', 'V_hist_bin_4', 'V_hist_bin_5', 'V_hist_bin_6', 'V_hist_bin_7', 'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'dark_pixels_ratio', 'dark_area_ratio']  
X = df[feature_columns].values
y = df['Label'].values

# 特徵標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA降維 (保留95%變異量)
pca = PCA(n_components=0.95)
X = pca.fit_transform(X)

# 拆分訓練/測試集，保持類別比例一致
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 建立 RandomForest 分類器
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# 預測
y_pred = clf.predict(X_test)

# 計算準確率
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc * 100:.2f}%")

# 儲存測試結果到 CSV
results = pd.DataFrame({
    "Folder": df.loc[idx_test, "Folder"].values,
    "Filename": df.loc[idx_test, "Filename"].values,
    "True_Label": y_test,
    "Predicted_Label": y_pred,
    "Correct": y_test == y_pred
}).reset_index(drop=True)

results.to_csv("test_predictions_rf.csv", index=False)
print("Saved prediction results to test_predictions_rf.csv")
