from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(0)
X = np.random.randn(100, 2)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

# Visualize the reduced data
plt.scatter(X_reduced, np.zeros_like(X_reduced))
plt.title('Principal Component Analysis (PCA)')
plt.xlabel('Principal Component 1')
plt.show()
