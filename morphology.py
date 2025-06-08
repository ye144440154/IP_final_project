import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import heapq
import cv2


# 手動實現形態學操作
def morphology_op(image, kernel, is_erode=True):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + k_h, j : j + k_w]
            if is_erode:
                result[i, j] = np.min(region[kernel == 1])
            else:
                result[i, j] = np.max(region[kernel == 1])
    return result


def cropper_threshold(img_path):
    # 讀取圖片並轉換為RGB陣列
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # 手動實現灰階轉換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 手動實現Otsu's thresholding
    pixel_count = gray.size
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))

    max_variance = -1
    optimal_thresh = 0

    for threshold in range(256):
        w0 = np.sum(hist[:threshold]) / pixel_count
        w1 = np.sum(hist[threshold:]) / pixel_count

        if w0 == 0 or w1 == 0:
            continue

        mu0 = np.sum(np.arange(threshold) * hist[:threshold]) / (w0 * pixel_count)
        mu1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (w1 * pixel_count)

        between_variance = w0 * w1 * (mu0 - mu1) ** 2
        if between_variance > max_variance:
            max_variance = between_variance
            optimal_thresh = threshold

    # 二值化（反轉）
    binary = np.where(gray > optimal_thresh, 0, 255).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)

    # 開運算（先腐蝕後膨脹）
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 三次膨脹
    mask = cv2.dilate(opening, kernel, iterations=3)

    # 應用遮罩
    result = img_np.copy()
    result[mask == 0] = 0

    return result, mask


def cropper_threshold_cv(img_path):

    img = cv2.imread(img_path)
    img_np = np.array(Image.open(img_path).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 形態學開運算去雜訊
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. 膨脹得到「確定背景」
    mask = cv2.dilate(opening, kernel, iterations=3)

    result = img_np
    result[mask == 0] = 0
    return result, mask
