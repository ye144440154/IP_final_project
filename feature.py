import numpy as np


def mango_feature_extractor(image, bins=16):
    # 強度特徵計算
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    intensity_features = [
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.quantile(gray, 0.25),
        np.quantile(gray, 0.75),
    ]

    # 自定義邊緣檢測
    dx = np.abs(np.diff(gray, axis=1, prepend=0))
    dy = np.abs(np.diff(gray, axis=0, prepend=0))
    edge_map = np.sqrt(dx**2 + dy**2)
    edge_features = [
        np.mean(edge_map),
        np.std(edge_map),
        np.max(edge_map),
        np.count_nonzero(edge_map > 50) / edge_map.size,
    ]

    # 顏色直方圖特徵
    hist_features = []
    for ch in range(3):  # 處理RGB三通道
        hist, _ = np.histogram(image[:, :, ch], bins=bins, range=(0, 255))
        hist = hist / hist.sum()  # 正規化
        hist_features.extend(hist)

    # 形態特徵
    binary_mask = (gray > np.quantile(gray, 0.7)).astype(float)
    morph_features = [
        binary_mask.mean(),
        binary_mask.std(),
        np.corrcoef(binary_mask.sum(axis=0), binary_mask.sum(axis=1))[0, 1],
    ]

    return np.concatenate(
        [intensity_features, edge_features, hist_features, morph_features]
    )


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.features = None
        self.labels = None

    def fit(self, features, labels):
        self.features = np.array(features)
        self.labels = np.array(labels)

    def predict(self, feature):
        distances = np.sqrt(np.sum((self.features - feature) ** 2, axis=1))
        k_indices = distances.argsort()[: self.k]
        k_labels = self.labels[k_indices]
        counts = {}
        for label in k_labels:
            counts[label] = counts.get(label, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[0][0]
