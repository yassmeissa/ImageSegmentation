
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAPreprocessing:

    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.original_shape = None
    
    def fit_transform(self, pixels):
        self.original_shape = pixels.shape
        
        pixels_scaled = self.scaler.fit_transform(pixels)
        
        pixels_transformed = self.pca.fit_transform(pixels_scaled)
        
        self.is_fitted = True
        
        return pixels_transformed
    
    def get_explained_variance_ratio(self):

        if not self.is_fitted:
            return None
        
        return self.pca.explained_variance_ratio_
    
    def get_total_variance_explained(self):

        if not self.is_fitted:
            return None
        
        return np.sum(self.pca.explained_variance_ratio_)
    
    def get_components(self):

        if not self.is_fitted:
            return None
        
        return self.pca.components_
    
    def get_mean(self):

        if not self.is_fitted:
            return None
        
        return self.scaler.mean_
    
    def get_std(self):

        if not self.is_fitted:
            return None
        
        return self.scaler.scale_
    
    def print_summary(self):

        if not self.is_fitted:
            print("[ERREUR] PCA n'a pas été appliquée")
            return
        
        print("\n" + "="*70)
        print("RÉSUMÉ DE L'ACP")
        print("="*70)
        print(f"Nombre de composantes: {self.n_components}")
        print(f"\nVariance expliquée par composante:")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_):
            print(f"  Composante {i+1}: {ratio*100:.2f}%")
        
        total = np.sum(self.pca.explained_variance_ratio_)
        print(f"\nVariance totale expliquée: {total*100:.2f}%")
        print("="*70 + "\n")

class PCAComparison:

    
    @staticmethod
    def compare_clustering(pixels, labels_original, labels_pca=None):
        stats = {
            'original_clusters': len(np.unique(labels_original)),
            'pca_clusters': len(np.unique(labels_pca)) if labels_pca is not None else None,
            'original_silhouette': None,
            'pca_silhouette': None
        }
        
        try:
            from sklearn.metrics import silhouette_score
            
            stats['original_silhouette'] = silhouette_score(pixels, labels_original)
            
            if labels_pca is not None:
                pass
        except ImportError:
            pass
        
        return stats
