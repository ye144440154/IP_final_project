import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage import measure

label_map = {
    'Anwar Ratool': 0,
    'Chaunsa (Black)': 1,
    'Chaunsa (Summer Bahisht)': 2,
    'Chaunsa (White)': 3,
    'Dosehri': 4,
    'Fajri': 5,
    'Langra': 6,
    'Sindhri': 7
}
label_map_grades = {
    'Class_I': 0,
    'Class_II': 1,
    'Extra_Class': 2
   
}
def extract_features(image, mask):
    # 確保只有mask內的區域被分析
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # 特徵清單
    features = []

    # --- 顏色特徵（HSV） ---
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    for i in range(3):  # H, S, V
        channel = hsv[:, :, i]
        channel_masked = channel[mask > 0]
        hist = cv2.calcHist([channel_masked], [0], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # --- 紋理特徵（GLCM） ---
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0
    glcm = graycomatrix(gray_masked, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    features.extend([contrast, homogeneity, energy, correlation])

    # --- 形狀特徵 ---
    props = regionprops(mask)[0]  # 只取一個區域
    features.extend([
        props.area,
        props.perimeter,
        props.eccentricity,
        props.solidity,
        props.extent,
        props.major_axis_length,
        props.minor_axis_length
    ])

    # --- 邊緣密度 ---
    edges = cv2.Canny(gray_masked, 50, 150)
    edge_density = np.sum(edges) / np.sum(mask)
    features.append(edge_density)

    return np.array(features)

def process_all_folders(base_dir, output_csv):
    data = []
    columns_set = False

    for folder_name, label in label_map.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning! Can't find folder：{folder_path}")
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".png") or "_mask" in fname:
                continue

            base_name = fname.split(".")[0]
            mask_name = base_name + ".png"
            img_path = os.path.join(folder_path, fname)
            mask_path = os.path.join(folder_path, mask_name)

            if not os.path.exists(mask_path):
                print(f"Warning! Can't find mask: {mask_name}")
                continue

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error! Can't read: {fname} or mask")
                continue

            features = extract_features(image, mask)
            row = [folder_name, label, fname] + features.tolist()
            data.append(row)

    # 欄位名稱
    feature_len = len(data[0]) - 3
    hsv_labels = [f'{ch}_hist_bin_{i}' for ch in ['H', 'S', 'V'] for i in range(8)]
    glcm_labels = ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation']
    shape_labels = [
        'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
        'major_axis_length', 'minor_axis_length'
    ]
    edge_label = ['edge_density']
    columns = ['class_name', 'label', 'filename'] + hsv_labels + glcm_labels + shape_labels + edge_label

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Output> {output_csv}")
"""
def process_folder(image_dir, output_csv):
    data = []
    filenames = []

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".png") or "_mask" in fname:
            continue
        
        base_name = fname.split(".")[0]
        mask_name = base_name + ".png"
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(image_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"[警告] 找不到對應 mask: {mask_name}")
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"[錯誤] 無法讀取: {fname} 或其 mask")
            continue

        features = extract_features(image, mask)
        data.append(features)
        filenames.append(fname)

    df = pd.DataFrame(data)
    df.insert(0, 'filename', filenames)
    df.to_csv(output_csv, index=False)
    print(f"[完成] 所有特徵已儲存至: {output_csv}")
"""

def process_grading_folders(base_dir, output_csv):
    data = []

    for folder_name, label in label_map_grades.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning! Can't find folder：{folder_path}")
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".png") or "_mask" in fname:
                continue

            base_name = fname.split(".")[0]
            mask_name = base_name + ".png"
            img_path = os.path.join(folder_path, fname)
            mask_path = os.path.join(folder_path, mask_name)

            if not os.path.exists(mask_path):
                print(f"Warning! Can't read mask: {mask_name}")
                continue

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error! Can't read: {fname} or mask")
                continue

            features = extract_features(image, mask)
            row = [folder_name, label, fname] + features.tolist()
            data.append(row)

    # 欄位名稱
    feature_len = len(data[0]) - 3
    hsv_labels = [f'{ch}_hist_bin_{i}' for ch in ['H', 'S', 'V'] for i in range(8)]
    glcm_labels = ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation']
    shape_labels = [
        'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
        'major_axis_length', 'minor_axis_length'
    ]
    edge_label = ['edge_density']
    columns = ['class_name', 'label', 'filename'] + hsv_labels + glcm_labels + shape_labels + edge_label

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Output> {output_csv}")
if __name__ == "__main__":
    base_class_dir = "Output/Classification_output"
    output_class_csv = "Classification_features.csv"
    base_grades_dir = "Output/Grading_output"
    output_grades_csv = "Grading_features.csv"
    process_grading_folders(base_grades_dir, output_grades_csv)
    process_all_folders(base_class_dir, output_class_csv)