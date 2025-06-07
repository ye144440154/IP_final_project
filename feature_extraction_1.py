import os
import cv2
import numpy as np
import pandas as pd

# 分類與分級的標籤對應表
classification_mapping = {
    'Anwar Ratool': 0,
    'Chaunsa (Black)': 1,
    'Chaunsa (Summer Bahisht)': 2,
    'Chaunsa (White)': 3,
    'Dosehri': 4,
    'Fajri': 5,
    'Langra': 6,
    'Sindhri': 7
}
grading_mapping = {
    'Class_I': 0,
    'Class_II': 1,
    'Extra_Class': 2
}

# 特徵擷取函數
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 色相分析
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === 特徵一：Hue 色相平均值 ===
    # 用來表示整體偏什麼顏色）
    h_mean = np.mean(hsv[:, :, 0])

    # === 特徵二：Hue 色相標準差 ===
    # 反映顏色是否一致或變化多（色相分布的變化程度）
    h_std = np.std(hsv[:, :, 0])

    # === 特徵三：Saturation 飽和度平均值 ===
    # 可用來判斷熟度或新鮮程度
    s_mean = np.mean(hsv[:, :, 1])

    # === 特徵四：Aspect Ratio 長寬比 ===
    # 可用來區分是較長型或偏圓型芒果
    height, width = img.shape[:2]
    aspect_ratio = width / height if height != 0 else 0

    return [h_mean, h_std, s_mean, aspect_ratio]

# 產出特徵 CSV
def process_dataset(input_dir, label_map, output_csv):
    rows = []
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        label = label_map.get(folder, None)
        if label is None:
            print(f"跳過未知類別: {folder}")
            continue
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(folder_path, filename)
            features = extract_features(img_path)
            if features is None:
                print(f"讀取失敗: {img_path}")
                continue
            row = [folder, label, filename] + features
            rows.append(row)

    columns = ['Folder', 'Label', 'Filename', 'Hue_Mean', 'Hue_Std', 'Saturation_Mean', 'Aspect_Ratio']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"特徵已儲存至 {output_csv}")


classification_dir = "Output/Classification_output"
grading_dir = "Output/Grading_output"

process_dataset(classification_dir, classification_mapping, "Classification_features_1.csv")
process_dataset(grading_dir, grading_mapping, "Grading_features_1.csv")
