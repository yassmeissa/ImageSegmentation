from sklearn.cluster import SpectralClustering
import numpy as np
from .base_model import BaseClusteringModel
import gc
try:
    from utils.logger import get_logger
    logger = get_logger()
except:
    logger = None

class SpectralClusteringModel(BaseClusteringModel):
    
    def __init__(self, n_clusters: int = 3, affinity: str = 'nearest_neighbors'):
        super().__init__(f"Spectral (k={n_clusters})", use_vibrant_colors=True)
        if logger:
            logger.info(f"Initializing Spectral Clustering with n_clusters={n_clusters}")
        self.n_clusters = n_clusters
        self.affinity = 'nearest_neighbors'  # Always use nearest_neighbors
        self.spectral = None
        self.cluster_centers = None
        self.labels = None
        self.original_pixels = None

    def fit(self, pixels_data: np.ndarray) -> None:
        if logger:
            logger.info(f"[Spectral] Starting fit with {len(pixels_data)} pixels")
            logger.debug(f"[Spectral] pixels_data shape: {pixels_data.shape}, dtype: {pixels_data.dtype}")
        
        self.original_pixels = pixels_data.copy()
        
        # Sous-échantillonnage ULTRA-AGRESSIF pour speed
        n_pixels = len(pixels_data)
        sample_rate = 1.0
        sample_indices = None
        
        # Aggressive downsampling: max 2000 pixels for RBF affinity matrix
        if n_pixels > 5000:
            sample_rate = max(0.05, 2000 / n_pixels)
            if logger:
                logger.info(f"[Spectral] Large dataset ({n_pixels} pixels). Using {sample_rate:.1%} sampling")
            n_samples = max(1500, int(n_pixels * sample_rate))
            sample_indices = np.random.choice(n_pixels, size=n_samples, replace=False)
            pixels_for_fit = pixels_data[sample_indices]
            if logger:
                logger.debug(f"[Spectral] Downsampled to {len(pixels_for_fit)} pixels")
        else:
            pixels_for_fit = pixels_data
        
        # Créer et entraîner le modèle Spectral Clustering
        if logger:
            logger.info(f"[Spectral] Running Spectral Clustering with {self.n_clusters} clusters, affinity={self.affinity}...")
        
        # Upgraded Spectral parameters - ULTRA-OPTIMIZED FOR SPEED
        self.spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,  # Use the configured affinity type
            assign_labels='kmeans',
            n_init=5,          # Reduced from 8 for speed
            random_state=42,
            n_neighbors=5       # Reduced from 10 for faster affinity matrix
        )
        sample_labels = self.spectral.fit_predict(pixels_for_fit)
        
        if logger:
            logger.info(f"[Spectral] fit_predict completed. Labels shape: {sample_labels.shape}")
        
        # Si sous-échantillonné, prédire sur l'ensemble complet
        if sample_rate < 1.0:
            if logger:
                logger.info("[Spectral] Predicting labels for full dataset...")
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(pixels_for_fit)
            distances, indices = nbrs.kneighbors(pixels_data)
            self.labels = sample_labels[indices.flatten()]
            if logger:
                logger.info("[Spectral] Full dataset labeling completed")
        else:
            self.labels = sample_labels
        
        # Analyser les clusters trouvés
        unique_labels = np.unique(self.labels)
        if logger:
            logger.info(f"[Spectral] Unique labels found: {unique_labels}")
        
        n_clusters = len(unique_labels)
        if logger:
            logger.info(f"[Spectral] Number of clusters: {n_clusters}")
        
        # Calculer les centres de clusters
        if logger:
            logger.info("[Spectral] Computing cluster centers...")
        
        centers = []
        for label_id in sorted(unique_labels):
            mask = self.labels == label_id
            count = np.sum(mask)
            
            # Calculer la moyenne et convertir proprement en uint8
            center_float = pixels_data[mask].mean(axis=0)
            center = np.clip(np.round(center_float), 0, 255).astype(np.uint8)
            
            centers.append(center)
            
            if logger:
                logger.debug(f"[Spectral] Cluster {label_id}: {count} points, RGB: {center}")
        
        self.cluster_centers = np.array(centers, dtype=np.uint8)
        
        if logger:
            logger.info(f"[Spectral] Total clusters: {len(self.cluster_centers)}")
            logger.info(f"[Spectral] Label range: {self.labels.min()} to {self.labels.max()}")
        
        self.is_fitted = True
        
        if logger:
            logger.info("[Spectral] Model fitted successfully")
        
        # Nettoyage mémoire
        gc.collect()

    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        if logger:
            logger.debug(f"[Spectral] predict() called with {len(pixels_data)} pixels")
        
        if not self.is_fitted:
            if logger:
                logger.error("[Spectral] Model not fitted before prediction!")
            raise RuntimeError("Model must be fitted before prediction")
        
        # Vérification de cohérence
        if len(self.labels) != len(pixels_data):
            if logger:
                logger.error(f"[Spectral] Size mismatch! labels: {len(self.labels)}, pixels: {len(pixels_data)}")
            raise RuntimeError(f"Label count ({len(self.labels)}) doesn't match pixel count ({len(pixels_data)})")
        
        result = self.labels.copy()
        
        if logger:
            logger.debug(f"[Spectral] Returning {len(result)} labels, range: [{result.min()}, {result.max()}]")
        
        return result

    def get_cluster_centers(self) -> np.ndarray:
        if logger:
            logger.debug(f"[Spectral] get_cluster_centers() called")
        
        if self.cluster_centers is None:
            if logger:
                logger.error("[Spectral] Cluster centers not available!")
            raise RuntimeError("Model must be fitted first")
        
        if logger:
            logger.debug(f"[Spectral] Returning centers, shape: {self.cluster_centers.shape}")
        
        return self.cluster_centers.copy()

    def set_parameters(self, n_clusters: int = 3, affinity: str = 'nearest_neighbors') -> None:
        if logger:
            logger.info(f"[Spectral] Changing parameters to n_clusters={n_clusters}")
        
        self.n_clusters = n_clusters
        self.affinity = 'nearest_neighbors'  # Always nearest_neighbors
        self.spectral = None
        self.is_fitted = False
        self.name = f"Spectral (k={n_clusters})"
        self.cluster_centers = None
        self.labels = None
        self.original_pixels = None
        
        gc.collect()
        
        if logger:
            logger.info("[Spectral] Parameters changed, model reset")
