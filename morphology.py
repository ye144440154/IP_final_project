import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import heapq
import cv2


def compute_gradient(gray):
    # 計算簡單的梯度（邊緣強度）
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.hypot(gx, gy)
    grad = (grad / grad.max() * 255).astype(np.uint8)
    return grad


def get_markers(gray, thresh_fg=0.7, thresh_bg=0.2):
    # 前景標記：亮區，背景標記：暗區
    markers = np.zeros_like(gray, dtype=np.int32)
    fg = gray > (gray.max() * thresh_fg)
    bg = gray < (gray.max() * thresh_bg)
    markers[bg] = 1  # 背景
    markers[fg] = 2  # 前景
    return markers


def watershed(img_path):
    # 1. 讀取圖像，轉灰階
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 形態學開運算去雜訊
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. 膨脹得到「確定背景」
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. 距離轉換得到「確定前景」
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 6. 找到未知區域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. 連通元件標記
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 保證背景不是0
    markers[unknown == 255] = 0  # 未知區域設為0

    # 8. 用區域生長模擬分水嶺分割
    # 這裡以鄰近像素的標記進行擴展，直到所有未知區域被標記
    h, w = markers.shape
    changed = True
    while changed:
        changed = False
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if markers[y, x] == 0:
                    neighbor_labels = set()
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            label = markers[y + dy, x + dx]
                            if label > 1:
                                neighbor_labels.add(label)
                    if len(neighbor_labels) == 1:
                        markers[y, x] = neighbor_labels.pop()
                        changed = True
                    elif len(neighbor_labels) > 1:
                        markers[y, x] = -1  # 分水嶺邊界

    # 9. 將分水嶺邊界標記在原圖
    img[markers == -1] = [0, 0, 255]  # 紅色
    return img


def cropper_threshold(img_path):

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
