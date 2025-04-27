import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === Load dataset === #
X = np.load("./stream_saved/X_stream.npy")
y = np.load("./stream_saved/Y_stream.npy")

# === Feature Scaling (중요!) === #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === t-SNE 변환 === #
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_scaled)

# === 시각화 === #
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=f"Class {label}", alpha=0.6)

plt.legend()
plt.title("t-SNE visualization of pose features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()
