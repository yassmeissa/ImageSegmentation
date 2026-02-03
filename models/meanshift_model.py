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
    """
    Modèle de clustering utilisant l'algorithme MeanShift.
    
    MeanShift trouve les clusters en cherchant les modes de la densité.
    """
    
    def __init__(self, bandwidth: float = 25.0):
        """
        Initialise le modèle MeanShift.
        
        Args:
            bandwidth: Bande passante du kernel. Défaut: 25.0
        """
        self.bandwidth_param = bandwidth
        super().__init__("MeanShift")
        if logger:
            logger.info(f"Initializing MeanShift with bandwidth={bandwidth}")
        self.meanshift = None
        self.cluster_centers = None
        self.labels = None
        self.original_pixels = None

    def fit(self, pixels_data: np.ndarray) -> None:
        """
        Entraîne le modèle MeanShift sur les données de pixels.
        
        Args:
            pixels_data: Données des pixels (n_pixels, 3) en float32
        """
        if logger:
            logger.info(f"[MeanShift] Starting fit with {len(pixels_data)} pixels")
            logger.debug(f"[MeanShift] pixels_data shape: {pixels_data.shape}, dtype: {pixels_data.dtype}")
        
        self.original_pixels = pixels_data.copy()
        
        # Sous-échantillonnage ULTRA-AGRESSIF pour speed
        n_pixels = len(pixels_data)
        sample_rate = 1.0
        sample_indices = None
        
        # Very aggressive downsampling: max 3000 pixels for MeanShift
        if n_pixels > 8000:
            sample_rate = max(0.05, 3000 / n_pixels)
            if logger:
                logger.info(f"[MeanShift] Large dataset ({n_pixels} pixels). Using {sample_rate:.1%} sampling")
            n_samples = max(2000, int(n_pixels * sample_rate))
            sample_indices = np.random.choice(n_pixels, size=n_samples, replace=False)
            pixels_for_fit = pixels_data[sample_indices]
            if logger:
                logger.debug(f"[MeanShift] Downsampled to {len(pixels_for_fit)} pixels")
        else:
            pixels_for_fit = pixels_data
        
        # Utiliser le paramètre bandwidth spécifié
        bandwidth = self.bandwidth_param
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
            
            # Calculer la moyenne et convertir proprement en uint8
            center_float = pixels_data[mask].mean(axis=0)
            center = np.clip(np.round(center_float), 0, 255).astype(np.uint8)
            
            centers.append(center)
            
            if logger:
                logger.debug(f"[MeanShift] Cluster {label_id}: {count} points, RGB: {center}")
        
        self.cluster_centers = np.array(centers, dtype=np.uint8)
        
        if logger:
            logger.info(f"[MeanShift] Total clusters: {len(self.cluster_centers)}")
            logger.info(f"[MeanShift] Label range: {self.labels.min()} to {self.labels.max()}")
        
        self.is_fitted = True
        
        if logger:
            logger.info("[MeanShift] Model fitted successfully")
        
        # Nettoyage mémoire
        gc.collect()

    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        """
        Retourne les labels de cluster pour les données.
        
        Args:
            pixels_data: Données des pixels
            
        Returns:
            Labels de cluster pour chaque pixel
            
        Raises:
            RuntimeError: Si le modèle n'est pas entraîné
        """
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
        """
        Retourne les centres des clusters.
        
        Returns:
            Array des centres de clusters (n_clusters, 3) en uint8
            
        Raises:
            RuntimeError: Si le modèle n'est pas entraîné
        """
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
        """
        Change les paramètres du modèle.
        
        Args:
            bandwidth: Nouvelle bande passante
        """
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
