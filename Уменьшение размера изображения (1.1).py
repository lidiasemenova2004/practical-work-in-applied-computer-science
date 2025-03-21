from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = fetch_olivetti_faces()
X, y = data.images, data.target  # Оставляем двумерные изображения (400, 64, 64)

# Выбираем случайное изображение
idx = np.random.randint(0, len(X))
image = X[idx]  # (64, 64)

# Применяем одномерный PCA к строкам изображения
pca = PCA(n_components=1)
X_pca = pca.fit_transform(image)  # (64, 1)

# Восстановление изображения из 1D PCA
X_reconstructed = pca.inverse_transform(X_pca)

# Вычисление ковариационной матрицы вручную
cov_matrix = np.cov(image.T)  # (64, 64) — ковариация между пикселями по столбцам

# Получение собственных чисел и собственных векторов
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Оригинальное изображение
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Оригинальное изображение")

# После 1D PCA
axes[0, 1].imshow(X_reconstructed, cmap='gray')
axes[0, 1].set_title("После 1D PCA")

# Ковариационная матрица
sns.heatmap(cov_matrix, ax=axes[0, 2], cmap="coolwarm", square=True)
axes[0, 2].set_title("Ковариационная матрица")

# График собственных чисел
axes[1, 0].plot(eigenvalues, marker='o', linestyle='--')
axes[1, 0].set_title("Собственные числа")
axes[1, 0].set_xlabel("Компонента")
axes[1, 0].set_ylabel("Значение")

# Первые 5 главных компонент (векторов) в виде изображений
for i in range(5):
    axes[1, 1].plot(eigenvectors[:, i], label=f'Компонента {i+1}')
axes[1, 1].legend()
axes[1, 1].set_title("Первые 5 собственных векторов")

plt.tight_layout()
plt.show()
