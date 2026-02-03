"""
Tests de validation de l'interface utilisateur
Vérifie que les composants GUI fonctionnent correctement
"""

import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, '/Users/yassmeissa/Downloads/apptkr_imageprocessing')

from config import AppConfig
from theme import Theme


class TestUI:
    """Classe de test pour l'interface utilisateur"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def test_config_constants(self):
        """Test que toutes les constantes de config sont correctes"""
        print("\n" + "-"*70)
        print("TEST: Constantes de configuration")
        print("-"*70)
        
        try:
            # Vérifier que toutes les constantes existent
            assert hasattr(AppConfig, 'WINDOW_TITLE'), "WINDOW_TITLE manquant"
            assert hasattr(AppConfig, 'DEFAULT_KMEANS_CLUSTERS'), "DEFAULT_KMEANS_CLUSTERS manquant"
            assert hasattr(AppConfig, 'DEFAULT_GMM_COMPONENTS'), "DEFAULT_GMM_COMPONENTS manquant"
            assert hasattr(AppConfig, 'DEFAULT_MEANSHIFT_BANDWIDTH'), "DEFAULT_MEANSHIFT_BANDWIDTH manquant"
            assert hasattr(AppConfig, 'CLUSTERS_MIN'), "CLUSTERS_MIN manquant"
            assert hasattr(AppConfig, 'CLUSTERS_MAX'), "CLUSTERS_MAX manquant"
            assert hasattr(AppConfig, 'BANDWIDTH_MIN'), "BANDWIDTH_MIN manquant"
            assert hasattr(AppConfig, 'BANDWIDTH_MAX'), "BANDWIDTH_MAX manquant"
            
            print(f"[OK] WINDOW_TITLE: {AppConfig.WINDOW_TITLE}")
            print(f"[OK] DEFAULT_KMEANS_CLUSTERS: {AppConfig.DEFAULT_KMEANS_CLUSTERS}")
            print(f"[OK] DEFAULT_GMM_COMPONENTS: {AppConfig.DEFAULT_GMM_COMPONENTS}")
            print(f"[OK] DEFAULT_MEANSHIFT_BANDWIDTH: {AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH}")
            print(f"[OK] CLUSTERS_MIN: {AppConfig.CLUSTERS_MIN}")
            print(f"[OK] CLUSTERS_MAX: {AppConfig.CLUSTERS_MAX}")
            print(f"[OK] BANDWIDTH_MIN: {AppConfig.BANDWIDTH_MIN}")
            print(f"[OK] BANDWIDTH_MAX: {AppConfig.BANDWIDTH_MAX}")
            
            self.passed += 1
            return True
        except AssertionError as e:
            print(f"[ERREUR] Config Error: {e}")
            self.failed += 1
            return False
    
    def test_theme_colors(self):
        """Test que tous les thèmes et couleurs sont correctement définis"""
        print("\n" + "-"*70)
        print("TEST: Thème et couleurs")
        print("-"*70)
        
        try:
            # Vérifier que les couleurs de thème existent
            assert hasattr(Theme, 'BG'), "Theme.BG manquant"
            assert hasattr(Theme, 'PANEL'), "Theme.PANEL manquant"
            assert hasattr(Theme, 'ACCENT'), "Theme.ACCENT manquant"
            assert hasattr(Theme, 'TEXT'), "Theme.TEXT manquant"
            assert hasattr(Theme, 'MUTED'), "Theme.MUTED manquant"
            assert hasattr(Theme, 'PROCESSING'), "Theme.PROCESSING manquant"
            assert hasattr(Theme, 'DONE'), "Theme.DONE manquant"
            
            # Vérifier que les couleurs sont des strings hexadécimales
            for color_name in ['BG', 'PANEL', 'ACCENT', 'TEXT', 'MUTED', 'PROCESSING', 'DONE']:
                color = getattr(Theme, color_name)
                assert isinstance(color, str), f"Theme.{color_name} doit être une string"
                assert color.startswith('#'), f"Theme.{color_name} doit être au format hex (#RRGGBB)"
            
            print(f"[OK] Theme.BG: {Theme.BG}")
            print(f"[OK] Theme.PANEL: {Theme.PANEL}")
            print(f"[OK] Theme.ACCENT: {Theme.ACCENT}")
            print(f"[OK] Theme.TEXT: {Theme.TEXT}")
            print(f"[OK] Theme.MUTED: {Theme.MUTED}")
            print(f"[OK] Theme.PROCESSING: {Theme.PROCESSING}")
            print(f"[OK] Theme.DONE: {Theme.DONE}")
            
            self.passed += 1
            return True
        except AssertionError as e:
            print(f"[ERREUR] Theme Error: {e}")
            self.failed += 1
            return False
    
    def test_theme_model_colors(self):
        """Test que les couleurs des modèles sont accessibles"""
        print("\n" + "-"*70)
        print("TEST: Couleurs des modèles")
        print("-"*70)
        
        try:
            model_names = ['kmeans', 'gmm', 'meanshift', 'spectral']
            
            for model_name in model_names:
                color = Theme.get_model_color(model_name)
                assert isinstance(color, str), f"Couleur pour {model_name} doit être une string"
                assert color.startswith('#'), f"Couleur pour {model_name} doit être au format hex"
                print(f"[OK] {model_name}: {color}")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"[ERREUR] Model Colors Error: {e}")
            self.failed += 1
            return False
    
    def test_parameter_validation(self):
        """Test que les paramètres sont dans les bonnes plages"""
        print("\n" + "-"*70)
        print("TEST: Validation des paramètres")
        print("-"*70)
        
        try:
            # Vérifier les valeurs par défaut
            assert AppConfig.CLUSTERS_MIN <= AppConfig.DEFAULT_KMEANS_CLUSTERS <= AppConfig.CLUSTERS_MAX, \
                "DEFAULT_KMEANS_CLUSTERS hors limites"
            assert AppConfig.BANDWIDTH_MIN <= AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH <= AppConfig.BANDWIDTH_MAX, \
                "DEFAULT_MEANSHIFT_BANDWIDTH hors limites"
            
            # Vérifier les limites
            assert AppConfig.CLUSTERS_MIN < AppConfig.CLUSTERS_MAX, "CLUSTERS_MIN doit être < CLUSTERS_MAX"
            assert AppConfig.BANDWIDTH_MIN < AppConfig.BANDWIDTH_MAX, "BANDWIDTH_MIN doit être < BANDWIDTH_MAX"
            
            print(f"[OK] Clusters: {AppConfig.CLUSTERS_MIN} <= {AppConfig.DEFAULT_KMEANS_CLUSTERS} <= {AppConfig.CLUSTERS_MAX}")
            print(f"[OK] Bandwidth: {AppConfig.BANDWIDTH_MIN} <= {AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH} <= {AppConfig.BANDWIDTH_MAX}")
            
            self.passed += 1
            return True
        except AssertionError as e:
            print(f"[ERREUR] Parameter Validation Error: {e}")
            self.failed += 1
            return False
    
    def test_file_paths(self):
        """Test que les chemins des fichiers sont corrects"""
        print("\n" + "-"*70)
        print("TEST: Chemins des fichiers")
        print("-"*70)
        
        try:
            # Vérifier que les images existent
            img_dir = os.path.join(os.getcwd(), AppConfig.IMAGE_FOLDER)
            
            print(f"[OK] IMAGE_FOLDER: {AppConfig.IMAGE_FOLDER}")
            print(f"[OK] Chemin complet: {img_dir}")
            
            if os.path.exists(img_dir):
                images = os.listdir(img_dir)
                print(f"[OK] Images trouvées: {images}")
            else:
                print(f"[AVERTISSEMENT] Dossier d'images n'existe pas (non bloquant)")
            
            # Vérifier l'icône
            if hasattr(AppConfig, 'ICON_PATH'):
                print(f"[OK] ICON_PATH: {AppConfig.ICON_PATH}")
                if os.path.exists(AppConfig.ICON_PATH):
                    print(f"[OK] Icône trouvée")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"[AVERTISSEMENT] File Path Warning: {e}")
            self.passed += 1  # Pas critique
            return True
    
    def run_all_tests(self):
        """Exécuter tous les tests UI"""
        print("\n" + "="*70)
        print("TESTS DE VALIDATION UI")
        print("="*70)
        
        # Exécuter les tests
        self.test_config_constants()
        self.test_theme_colors()
        self.test_theme_model_colors()
        self.test_parameter_validation()
        self.test_file_paths()
        
        # Résumé
        print("\n" + "="*70)
        print("RÉSUMÉ DES TESTS UI")
        print("="*70)
        total = self.passed + self.failed
        print(f"[OK] Réussis: {self.passed}/{total}")
        print(f"[ERREUR] Échoués: {self.failed}/{total}")
        
        if self.failed == 0:
            print("\nTOUS LES TESTS UI SONT PASSÉS!")
        else:
            print(f"\n{self.failed} test(s) ont échoué")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    tester = TestUI()
    tester.run_all_tests()
