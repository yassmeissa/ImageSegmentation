from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from .base_model import BaseClusteringModel
import gc
try:
    from utils.logger import get_logger
    logger = get_logger()
except:
    logger = None

class MeanShiftClusteringModel(BaseClusteringModel):
    
    def __init__(self, bandwidth: float = 25.0):
        self.bandwidth_param = bandwidth
        super().__init__("MeanShift", use_vibrant_colors=True)
        if logger:
            logger.info(f"Initializing MeanShift with bandwidth={bandwidth}")
        self.meanshift = None
        self.cluster_centers = None
        self.labels = None
        self.original_pixels = None

    def fit(self, pixels_data: np.ndarray) -> None:
        if logger:
            logger.info(f"[MeanShift] Starting fit with {len(pixels_data)} pixels")
            logger.debug(f"[MeanShift] pixels_data shape: {pixels_data.shape}, dtype: {pixels_data.dtype}")
        
        self.original_pixels = pixels_data.copy()
        
        n_pixels = len(pixels_data)
        sample_rate = 1.0
        sample_indices = None
        
        # Check if using PCA data (typically has much smaller range)
        data_range = np.max(pixels_data) - np.min(pixels_data)
        is_pca_data = data_range < 50  # PCA data typically has range < 20-30, RGB has 0-255
        
        if logger:
            logger.debug(f"[MeanShift] Data range: {data_range:.2f}, detected PCA: {is_pca_data}")
        
        # Less aggressive downsampling for PCA to preserve cluster structure
        if is_pca_data:
            # For PCA data, use more conservative downsampling
            max_samples = 5000
        else:
            # For RGB data, use aggressive downsampling
            max_samples = 3000
        
        if n_pixels > 8000:
            sample_rate = max(0.05, max_samples / n_pixels)
            if logger:
                logger.info(f"[MeanShift] Large dataset ({n_pixels} pixels, PCA={is_pca_data}). Using {sample_rate:.1%} sampling")
            n_samples = max(2000, int(n_pixels * sample_rate))
            sample_indices = np.random.choice(n_pixels, size=n_samples, replace=False)
            pixels_for_fit = pixels_data[sample_indices]
            if logger:
                logger.debug(f"[MeanShift] Downsampled to {len(pixels_for_fit)} pixels")
        else:
            pixels_for_fit = pixels_data
        
        # Utiliser le paramètre bandwidth spécifié ou auto-estimer pour PCA
        bandwidth = self.bandwidth_param
        
        # For PCA data, auto-estimate bandwidth if default is used
        if is_pca_data and bandwidth <= 25.0:
            try:
                bandwidth = estimate_bandwidth(pixels_for_fit, quantile=0.2, n_samples=min(500, len(pixels_for_fit)))
                if logger:
                    logger.info(f"[MeanShift] Auto-estimated bandwidth for PCA data: {bandwidth:.4f}")
            except Exception as e:
                # Fallback: estimate manually for PCA space
                data_std = np.std(pixels_for_fit)
                bandwidth = data_std * 0.3
                if logger:
                    logger.info(f"[MeanShift] Estimated bandwidth for PCA (std-based): {bandwidth:.4f}, data_std={data_std:.4f}")
        elif is_pca_data:
            if logger:
                logger.info(f"[MeanShift] Using user-specified bandwidth for PCA: {bandwidth}")
        
        if logger:
            logger.info(f"[MeanShift] Using bandwidth: {bandwidth}")
        
        # Créer et entraîner le modèle MeanShift
        if logger:
            logger.info("[MeanShift] Running MeanShift...")
        
        # ULTRA-OPTIMIZED MeanShift parameters
        self.meanshift = MeanShift(
            bandwidth=bandwidth, 
            n_jobs=-1,
            cluster_all=False  # Only include points in neighborhood (sharper clusters)
        )
        sample_labels = self.meanshift.fit_predict(pixels_for_fit)
        
        if logger:
            logger.info(f"[MeanShift] fit_predict completed. Labels shape: {sample_labels.shape}")
        
        # Si sous-échantillonné, prédire sur l'ensemble complet
        if sample_rate < 1.0:
            if logger:
                logger.info("[MeanShift] Predicting labels for full dataset...")
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(pixels_for_fit)
            distances, indices = nbrs.kneighbors(pixels_data)
            self.labels = sample_labels[indices.flatten()]
            if logger:
                logger.info("[MeanShift] Full dataset labeling completed")
        else:
            self.labels = sample_labels
        
        # Analyser les clusters trouvés
        unique_labels = np.unique(self.labels)
        if logger:
            logger.info(f"[MeanShift] Unique labels found: {unique_labels}")
        
        n_clusters = len(unique_labels)
        if logger:
            logger.info(f"[MeanShift] Number of clusters: {n_clusters}")
        
        # Calculer les centres de clusters
        if logger:
            logger.info("[MeanShift] Computing cluster centers...")
        
        centers = []
        for label_id in sorted(unique_labels):
            mask = self.labels == label_id
            count = np.sum(mask)
            
            # Calculer la moyenne
            center_float = pixels_data[mask].mean(axis=0)
            
            # For PCA data, keep values as float (can be small)
            # For RGB data, clip to 0-255
            if is_pca_data:
                center = center_float.astype(np.float32)
            else:
                center = np.clip(np.round(center_float), 0, 255).astype(np.uint8)
            
            centers.append(center)
            
            if logger:
                logger.debug(f"[MeanShift] Cluster {label_id}: {count} points, center: {center}")
        
        self.cluster_centers = np.array(centers, dtype=np.uint8 if not is_pca_data else np.float32)
        
        if logger:
            logger.info(f"[MeanShift] Total clusters: {len(self.cluster_centers)}")
            logger.info(f"[MeanShift] Label range: {self.labels.min()} to {self.labels.max()}")
        
        self.is_fitted = True
        
        if logger:
            logger.info("[MeanShift] Model fitted successfully")
        
        # Nettoyage mémoire
        gc.collect()

    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        if logger:
            logger.debug(f"[MeanShift] predict() called with {len(pixels_data)} pixels")
        
        if not self.is_fitted:
            if logger:
                logger.error("[MeanShift] Model not fitted before prediction!")
            raise RuntimeError("Model must be fitted before prediction")
        
        # Vérification de cohérence
        if len(self.labels) != len(pixels_data):
            if logger:
                logger.error(f"[MeanShift] Size mismatch! labels: {len(self.labels)}, pixels: {len(pixels_data)}")
            raise RuntimeError(f"Label count ({len(self.labels)}) doesn't match pixel count ({len(pixels_data)})")
        
        result = self.labels.copy()
        
        if logger:
            logger.debug(f"[MeanShift] Returning {len(result)} labels, range: [{result.min()}, {result.max()}]")
        
        return result

    def get_cluster_centers(self) -> np.ndarray:
        if logger:
            logger.debug(f"[MeanShift] get_cluster_centers() called")
        
        if self.cluster_centers is None:
            if logger:
                logger.error("[MeanShift] Cluster centers not available!")
            raise RuntimeError("Model must be fitted first")
        
        if logger:
            logger.debug(f"[MeanShift] Returning centers, shape: {self.cluster_centers.shape}")
        
        return self.cluster_centers.copy()

    def set_parameters(self, bandwidth: float = None) -> None:
        if logger:
            logger.info(f"[MeanShift] Changing parameters to bandwidth={bandwidth}")
        
        self.bandwidth_param = bandwidth
        self.meanshift = None
        self.is_fitted = False
        self.name = "MeanShift"
        self.cluster_centers = None
        self.labels = None
        self.original_pixels = None
        
        gc.collect()
        
        if logger:
            logger.info("[MeanShift] Parameters changed, model reset")
