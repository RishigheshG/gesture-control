import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gesture_detector import normalize_landmarks

def row_to_points(row):
    return [(row[i], row[i+1], row[i+2]) for i in range(0, len(row), 3)]

df = pd.read_csv("gesture_data.csv", header=None)
X = np.array([np.array(normalize_landmarks(row_to_points(r))).flatten() for r in df.iloc[:, :-1].values])
y = df.iloc[:, -1].values

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print("variance captured by these 2 components:", round(pca.explained_variance_ratio_.sum(), 3))

plt.figure(figsize=(8, 6))
for label in sorted(set(y)):
    mask = y == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.5, s=10)
plt.legend()
plt.title("Gesture feature clusters (PCA of normalized landmarks)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("pca_clusters.png", dpi=150)
plt.show()