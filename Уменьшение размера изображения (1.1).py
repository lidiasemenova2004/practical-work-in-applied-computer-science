from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = fetch_olivetti_faces()
X, y = data.images, data.target 

idx = np.random.randint(0, len(X))
image = X[idx]  # (64, 64)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(image)

X_reconstructed = pca.inverse_transform(X_pca)

cov_matrix = np.cov(image.T) 

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Оригинальное изображение")

axes[0, 1].imshow(X_reconstructed, cmap='gray')
axes[0, 1].set_title("После 1D PCA")

sns.heatmap(cov_matrix, ax=axes[0, 2], cmap="coolwarm", square=True)
axes[0, 2].set_title("Ковариационная матрица")

axes[1, 0].plot(eigenvalues, marker='o', linestyle='--')
axes[1, 0].set_title("Собственные числа")
axes[1, 0].set_xlabel("Компонента")
axes[1, 0].set_ylabel("Значение")

for i in range(5):
    axes[1, 1].plot(eigenvectors[:, i], label=f'Компонента {i+1}')
axes[1, 1].legend()
axes[1, 1].set_title("Первые 5 собственных векторов")

plt.tight_layout()
plt.show()
