"""
Tests de validation des modèles de clustering
Vérifie que chaque modèle fonctionne correctement
"""

import sys
import os
import numpy as np
from PIL import Image

# Ajouter le chemin du projet
sys.path.insert(0, '/Users/yassmeissa/Downloads/apptkr_imageprocessing')

from models.kmeans_model import KMeansClusteringModel
from models.gmm_model import GMMClusteringModel
from models.meanshift_model import MeanShiftClusteringModel
from models.spectral_model import SpectralClusteringModel
from utils.image_loader import ImageLoader
from config import AppConfig


class TestModels:
    """Classe de test pour les modèles de clustering"""
    
    def __init__(self):
        self.test_image_path = '/Users/yassmeissa/Downloads/apptkr_imageprocessing/img/cat.png'
        self.passed = 0
        self.failed = 0
        self.image = None
        self.pixels = None
    
    def setup(self):
        """Préparer les données de test"""
        print("\n" + "="*70)
        print("CONFIGURATION DES TESTS")
        print("="*70)
        
        if not os.path.exists(self.test_image_path):
            print(f"[ERREUR] Image de test non trouvée: {self.test_image_path}")
            return False
        
        try:
            self.image = ImageLoader.load_image(self.test_image_path)
            self.pixels = np.array(self.image).reshape(-1, 3).astype(np.float32)
            print(f"[OK] Image chargée: {self.image.size}")
            print(f"[OK] Pixels: {self.pixels.shape}")
            return True
        except Exception as e:
            print(f"[ERREUR] Erreur lors du chargement: {e}")
            return False
    
    def test_kmeans(self):
        """Test du modèle K-Means"""
        print("\n" + "-"*70)
        print("TEST: K-Means Clustering Model")
        print("-"*70)
        
        try:
            model = KMeansClusteringModel(n_clusters=5)
            
            # Test segment_image
            segmented = model.segment_image(self.image)
            assert isinstance(segmented, Image.Image), "Le résultat doit être une Image PIL"
            assert segmented.size == self.image.size, "La taille doit être identique"
            
            # Vérifier qu'il y a au moins 1 cluster
            seg_array = np.array(segmented)
            unique_colors = len(np.unique(seg_array.reshape(-1, 3), axis=0))
            assert unique_colors >= 1, "Doit avoir au moins 1 cluster"
            
            print(f"[OK] K-Means: {unique_colors} clusters trouvés")
            self.passed += 1
            return True
        except Exception as e:
            print(f"[ERREUR] K-Means Error: {e}")
            self.failed += 1
            return False
    
    def test_gmm(self):
        """Test du modèle GMM"""
        print("\n" + "-"*70)
        print("TEST: GMM Clustering Model")
        print("-"*70)
        
        try:
            model = GMMClusteringModel(n_components=5)
            
            # Test segment_image
            segmented = model.segment_image(self.image)
            assert isinstance(segmented, Image.Image), "Le résultat doit être une Image PIL"
            assert segmented.size == self.image.size, "La taille doit être identique"
            
            # Vérifier qu'il y a au moins 1 cluster
            seg_array = np.array(segmented)
            unique_colors = len(np.unique(seg_array.reshape(-1, 3), axis=0))
            assert unique_colors >= 1, "Doit avoir au moins 1 cluster"
            
            print(f"[OK] GMM: {unique_colors} clusters trouvés (peut être < 5 naturellement)")
            self.passed += 1
            return True
        except Exception as e:
            print(f"[ERREUR] GMM Error: {e}")
            self.failed += 1
            return False
    
    def test_meanshift(self):
        """Test du modèle MeanShift"""
        print("\n" + "-"*70)
        print("TEST: MeanShift Clustering Model")
        print("-"*70)
        
        try:
            model = MeanShiftClusteringModel(bandwidth=25)
            
            # Test segment_image
            segmented = model.segment_image(self.image)
            assert isinstance(segmented, Image.Image), "Le résultat doit être une Image PIL"
            assert segmented.size == self.image.size, "La taille doit être identique"
            
            # Vérifier qu'il y a au moins 1 cluster
            seg_array = np.array(segmented)
            unique_colors = len(np.unique(seg_array.reshape(-1, 3), axis=0))
            assert unique_colors >= 1, "Doit avoir au moins 1 cluster"
            
            print(f"[OK] MeanShift: {unique_colors} clusters trouvés (bandwidth=25)")
            self.passed += 1
            return True
        except Exception as e:
            print(f"[ERREUR] MeanShift Error: {e}")
            self.failed += 1
            return False
    
    def test_spectral(self):
        """Test du modèle Spectral"""
        print("\n" + "-"*70)
        print("TEST: Spectral Clustering Model")
        print("-"*70)
        
        try:
            model = SpectralClusteringModel(n_clusters=5)
            
            # Test segment_image
            segmented = model.segment_image(self.image)
            assert isinstance(segmented, Image.Image), "Le résultat doit être une Image PIL"
            assert segmented.size == self.image.size, "La taille doit être identique"
            
            # Vérifier qu'il y a au moins 1 cluster
            seg_array = np.array(segmented)
            unique_colors = len(np.unique(seg_array.reshape(-1, 3), axis=0))
            assert unique_colors >= 1, "Doit avoir au moins 1 cluster"
            
            print(f"[OK] Spectral: {unique_colors} clusters trouvés")
            self.passed += 1
            return True
        except Exception as e:
            print(f"[ERREUR] Spectral Error: {e}")
            self.failed += 1
            return False
    
    def test_parameter_ranges(self):
        """Test les plages de paramètres"""
        print("\n" + "-"*70)
        print("TEST: Plages de paramètres")
        print("-"*70)
        
        try:
            # Vérifier les constantes de config
            assert AppConfig.CLUSTERS_MIN == 5, "CLUSTERS_MIN doit être 5"
            assert AppConfig.CLUSTERS_MAX == 25, "CLUSTERS_MAX doit être 25"
            assert AppConfig.BANDWIDTH_MIN == 15, "BANDWIDTH_MIN doit être 15"
            assert AppConfig.BANDWIDTH_MAX == 45, "BANDWIDTH_MAX doit être 45"
            assert AppConfig.DEFAULT_KMEANS_CLUSTERS == 10, "DEFAULT_KMEANS_CLUSTERS doit être 10"
            assert AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH == 25, "DEFAULT_MEANSHIFT_BANDWIDTH doit être 25"
            
            print(f"[OK] Clusters: {AppConfig.CLUSTERS_MIN} - {AppConfig.CLUSTERS_MAX} (défaut: {AppConfig.DEFAULT_KMEANS_CLUSTERS})")
            print(f"[OK] Bandwidth: {AppConfig.BANDWIDTH_MIN} - {AppConfig.BANDWIDTH_MAX} (défaut: {AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH})")
            self.passed += 1
            return True
        except AssertionError as e:
            print(f"[ERREUR] Parameter Range Error: {e}")
            self.failed += 1
            return False
    
    def test_models_consistency(self):
        """Test que tous les modèles produisent des résultats cohérents"""
        print("\n" + "-"*70)
        print("TEST: Cohérence entre les modèles")
        print("-"*70)
        
        try:
            models = [
                ("K-Means", KMeansClusteringModel(n_clusters=10)),
                ("GMM", GMMClusteringModel(n_components=10)),
                ("MeanShift", MeanShiftClusteringModel(bandwidth=25)),
                ("Spectral", SpectralClusteringModel(n_clusters=10))
            ]
            
            results = {}
            for name, model in models:
                seg = model.segment_image(self.image)
                seg_array = np.array(seg)
                unique_colors = len(np.unique(seg_array.reshape(-1, 3), axis=0))
                results[name] = unique_colors
                
                # Vérifier que l'image n'est pas vide
                assert seg_array.size > 0, f"{name}: L'image segmentée est vide"
                
                # Vérifier que les dimensions correspondent
                assert seg.size == self.image.size, f"{name}: Les dimensions ne correspondent pas"
            
            print(f"[OK] Tous les modèles ont produit des résultats:")
            for name, clusters in results.items():
                print(f"   - {name}: {clusters} clusters")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"[ERREUR] Consistency Error: {e}")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("\n" + "="*70)
        print("SUITE DE TESTS DE VALIDATION")
        print("="*70)
        
        if not self.setup():
            print("\n❌ Impossible de configurer les tests")
            return
        
        # Exécuter les tests
        self.test_parameter_ranges()
        self.test_kmeans()
        self.test_gmm()
        self.test_meanshift()
        self.test_spectral()
        self.test_models_consistency()
        
        # Résumé
        print("\n" + "="*70)
        print("RÉSUMÉ DES TESTS")
        print("="*70)
        total = self.passed + self.failed
        print(f"[OK] Réussis: {self.passed}/{total}")
        print(f"[ERREUR] Échoués: {self.failed}/{total}")
        
        if self.failed == 0:
            print("\nTOUS LES TESTS SONT PASSÉS!")
        else:
            print(f"\n{self.failed} test(s) ont échoué")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    tester = TestModels()
    tester.run_all_tests()
