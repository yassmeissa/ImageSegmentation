from sklearn.cluster import KMeans
import numpy as np
from .base_model import BaseClusteringModel
import gc

class KMeansClusteringModel(BaseClusteringModel):
    def __init__(self, n_clusters: int = 3):
        super().__init__(f"K-Means (k={n_clusters})", use_vibrant_colors=True)
        self.n_clusters = n_clusters
        # Ultra-optimized K-Means: aggressive convergence for sharp results
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=30,          # Increased from 10 to 30 (exhaustive search)
            max_iter=500,       # Increased from 300 to 500 (very thorough)
            algorithm='lloyd',  # Most precise algorithm
            verbose=0,
            tol=1e-4            # Tighter convergence tolerance
        )
        self.cluster_centers = None

    def fit(self, pixels_data: np.ndarray) -> None:
        self.kmeans.fit(pixels_data)
        self.cluster_centers = self.kmeans.cluster_centers_.astype(np.uint8)
        self.is_fitted = True
        gc.collect()

    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.kmeans.predict(pixels_data)

    def get_cluster_centers(self) -> np.ndarray:
        if self.cluster_centers is None:
            raise RuntimeError("Model must be fitted first")
        return self.cluster_centers.copy()

    def set_n_clusters(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=30,
            max_iter=500,
            algorithm='lloyd',
            verbose=0,
            tol=1e-4
        )
        self.is_fitted = False
        self.name = f"K-Means (k={n_clusters})"
        self.cluster_centers = None
        gc.collect()
