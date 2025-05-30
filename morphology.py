import numpy as np


def manual_color_segmentation(image_np, hue_range=(10, 30), sat_thresh=50):
    """基於HSV色彩空間的手動閾值分割"""
    # 轉換RGB到HSV（自主實現，避免使用cv2.cvtColor）
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    maxc = np.max(image_np, axis=2)
    minc = np.min(image_np, axis=2)
    diff = maxc - minc + 1e-10

    # 計算色調(Hue)
    h = np.zeros_like(maxc)
    mask = maxc == r
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
    mask = maxc == g
    h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120) % 360
    mask = maxc == b
    h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240) % 360

    # 計算飽和度(Saturation)
    s = (diff / (maxc + 1e-10)) * 255

    # 手動閾值處理
    mask = (h >= hue_range[0]) & (h <= hue_range[1]) & (s > sat_thresh)
    return mask.astype(np.uint8)


def morphological_cleanup(mask):
    """自主實現形態學操作"""
    # 自定義卷積核
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    # 膨脹操作
    dilated = np.zeros_like(mask)
    h, w = mask.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if np.any(mask[i - 1 : i + 2, j - 1 : j + 2] * kernel):
                dilated[i, j] = 1

    # 腐蝕操作
    eroded = np.zeros_like(dilated)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if np.all(dilated[i - 1 : i + 2, j - 1 : j + 2] >= kernel):
                eroded[i, j] = 1
    return eroded


def find_mango_contour(mask):
    """基於邊緣追蹤的輪廓檢測"""
    contours = []
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1 and not visited[i, j]:
                # 區域生長法找輪廓
                queue = [(i, j)]
                current_contour = []
                while queue:
                    x, y = queue.pop(0)
                    if (
                        0 <= x < h
                        and 0 <= y < w
                        and not visited[x, y]
                        and mask[x, y] == 1
                    ):
                        visited[x, y] = True
                        current_contour.append((y, x))  # (x,y)轉換為圖片座標
                        # 8鄰域搜索
                        queue.extend(
                            [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                        )
                contours.append(np.array(current_contour))

    # 選取最大面積輪廓
    if contours:
        largest = max(contours, key=lambda x: x.shape[0])
        return largest
    return None
