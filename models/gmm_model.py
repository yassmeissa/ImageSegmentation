from sklearn.mixture import GaussianMixture
import numpy as np
from .base_model import BaseClusteringModel
import gc

class GMMClusteringModel(BaseClusteringModel):
    def __init__(self, n_components: int = 3):
        super().__init__(f"GMM (k={n_components})", use_vibrant_colors=True)
        self.n_components = n_components
        # Balanced GMM: good quality, FAST, and DISTINCTIVE
        # Using 'full' covariance for more interesting results
        self.gmm = GaussianMixture(
            n_components=n_components, 
            random_state=42, 
            n_init=5,           # Optimized: 8->5 for speed
            max_iter=100,       # Optimized: 150->100 for faster convergence
            covariance_type='diag',  # Optimized: 'full'->'diag' for ~3x speed boost
            warm_start=False,
            tol=1e-3
        )
        self.cluster_centers = None

    def fit(self, pixels_data: np.ndarray) -> None:
        self.gmm.fit(pixels_data)
        self.cluster_centers = self.gmm.means_.astype(np.uint8)
        self.is_fitted = True
        gc.collect()

    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.gmm.predict(pixels_data)

    def get_cluster_centers(self) -> np.ndarray:
        if self.cluster_centers is None:
            raise RuntimeError("Model must be fitted first")
        return self.cluster_centers.copy()

    def set_n_components(self, n_components: int) -> None:
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components, 
            random_state=42, 
            n_init=5,
            max_iter=100,
            covariance_type='diag',
            warm_start=False,
            tol=1e-3
        )
        self.is_fitted = False
        self.name = f"GMM (k={n_components})"
        self.cluster_centers = None
        gc.collect()
