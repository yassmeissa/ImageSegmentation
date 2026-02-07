
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, '/Users/yassmeissa/Downloads/apptkr_imageprocessing')

from utils.image_loader import ImageLoader
from models.kmeans_model import KMeansClusteringModel
from models.gmm_model import GMMClusteringModel
from models.meanshift_model import MeanShiftClusteringModel
from models.spectral_model import SpectralClusteringModel
from config import AppConfig
import matplotlib.cm as cm

class ValidationReport:

    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def add_result(self, model_name, image_name, params, execution_time, success=True, error=None):

        if model_name not in self.results:
            self.results[model_name] = []
        
        self.results[model_name].append({
            'image': image_name,
            'params': params,
            'time': execution_time,
            'success': success,
            'error': error
        })
    
    def print_report(self):

        print("\n" + "="*80)
        print("RAPPORT DE VALIDATION")
        print("="*80)
        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Durée totale: {self.end_time - self.start_time:.2f} secondes")
        
        total_tests = 0
        total_success = 0
        
        for model_name, tests in self.results.items():
            print(f"\n{'─'*80}")
            print(f"Modèle: {model_name}")
            print(f"{'─'*80}")
            
            for test in tests:
                status = "OK" if test['success'] else "ERREUR"
                total_tests += 1
                if test['success']:
                    total_success += 1
                
                print(f"\n  Image: {test['image']}")
                print(f"  Paramètres: {test['params']}")
                print(f"  Temps d'exécution: {test['time']:.2f}s")
                print(f"  Status: {status}")
                if test['error']:
                    print(f"  Erreur: {test['error']}")
        
        print(f"\n{'='*80}")
        print(f"RÉSUMÉ: {total_success}/{total_tests} tests réussis ({100*total_success/total_tests:.1f}%)")
        print(f"{'='*80}\n")

def generate_palette(palette_name: str, n_colors: int):

    try:
        colormap = cm.get_cmap(palette_name)
        colors = []
        for i in range(n_colors):
            rgba = colormap(i / max(1, n_colors - 1))
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            colors.append([r, g, b])
        return np.array(colors, dtype=np.uint8)
    except Exception:
        colormap = cm.get_cmap('viridis')
        colors = []
        for i in range(n_colors):
            rgba = colormap(i / max(1, n_colors - 1))
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            colors.append([r, g, b])
        return np.array(colors, dtype=np.uint8)

def test_kmeans(image, image_name, report):

    print("\n" + "-"*80)
    print("TEST K-MEANS")
    print("-"*80)
    
    try:
        config = {'clusters': 5, 'n_init': 10, 'max_iter': 300}
        print(f"\n  Configuration: {config}")
        start = time.time()
        
        model = KMeansClusteringModel(n_clusters=config['clusters'])
        model.kmeans.n_init = config['n_init']
        model.kmeans.max_iter = config['max_iter']
        
        palette = generate_palette('viridis', config['clusters'])
        segmented = model.segment_image(image, shared_palette=palette)
        
        elapsed = time.time() - start
        print(f"  Succès ({elapsed:.2f}s)")
        
        report.add_result('K-Means', image_name, config, elapsed, success=True)
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Erreur: {str(e)}")
        report.add_result('K-Means', image_name, config, elapsed, success=False, error=str(e))

def test_gmm(image, image_name, report):

    print("\n" + "-"*80)
    print("TEST GMM")
    print("-"*80)
    
    try:
        config = {'components': 5, 'max_iter': 100, 'cov_type': 'diag'}
        print(f"\n  Configuration: {config}")
        start = time.time()
        
        model = GMMClusteringModel(n_components=config['components'])
        model.gmm.max_iter = config['max_iter']
        model.gmm.covariance_type = config['cov_type']
        
        palette = generate_palette('plasma', config['components'])
        segmented = model.segment_image(image, shared_palette=palette)
        
        elapsed = time.time() - start
        print(f"  Succès ({elapsed:.2f}s)")
        
        report.add_result('GMM', image_name, config, elapsed, success=True)
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Erreur: {str(e)}")
        report.add_result('GMM', image_name, config, elapsed, success=False, error=str(e))

def test_meanshift(image, image_name, report):

    print("\n" + "-"*80)
    print("TEST MEANSHIFT")
    print("-"*80)
    
    try:
        config = {'bandwidth': 25}
        print(f"\n  Configuration: {config}")
        start = time.time()
        
        model = MeanShiftClusteringModel(bandwidth=config['bandwidth'])
        
        palette = generate_palette('inferno', 10)
        segmented = model.segment_image(image, shared_palette=palette)
        
        elapsed = time.time() - start
        print(f"  Succès ({elapsed:.2f}s)")
        
        report.add_result('MeanShift', image_name, config, elapsed, success=True)
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Erreur: {str(e)}")
        report.add_result('MeanShift', image_name, config, elapsed, success=False, error=str(e))

def test_spectral(image, image_name, report):

    print("\n" + "-"*80)
    print("TEST SPECTRAL CLUSTERING")
    print("-"*80)
    
    try:
        config = {'clusters': 5}
        print(f"\n  Configuration: {config}")
        start = time.time()
        
        model = SpectralClusteringModel(n_clusters=config['clusters'])
        
        palette = generate_palette('cool', config['clusters'])
        segmented = model.segment_image(image, shared_palette=palette)
        
        elapsed = time.time() - start
        print(f"  Succès ({elapsed:.2f}s)")
        
        report.add_result('Spectral', image_name, config, elapsed, success=True)
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Erreur: {str(e)}")
        report.add_result('Spectral', image_name, config, elapsed, success=False, error=str(e))

def display_comparison(image, segmented_results):

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparaison des modèles de clustering', fontsize=16, fontweight='bold')
    
    # Image originale
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image Originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Résultats
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    for (model_name, segmented_img), (row, col) in zip(segmented_results.items(), positions):
        axes[row, col].imshow(np.array(segmented_img))
        axes[row, col].set_title(model_name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():

    print("\n" + "="*80)
    print("TESTS DE VALIDATION - TOUS LES MODÈLES")
    print("="*80)
    
    # Initialiser le rapport
    report = ValidationReport()
    report.start_time = time.time()
    
    # Trouver les images de test
    image_dir = '/Users/yassmeissa/Downloads/apptkr_imageprocessing/img'
    images = ['cat.png', 'rose.png']
    
    available_images = []
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            available_images.append((img_name, img_path))
    
    if not available_images:
        print("Erreur: Aucune image trouvée!")
        return
    
    print(f"\nImages trouvées: {len(available_images)}")
    for name, _ in available_images:
        print(f"  - {name}")
    
    # Tester chaque image
    for image_name, image_path in available_images:
        print("\n" + "="*80)
        print(f"TESTS POUR L'IMAGE: {image_name}")
        print("="*80)
        
        # Charger l'image
        try:
            image = ImageLoader.load_image(image_path)
            img_array = np.array(image)
            print(f"Image chargée: {img_array.shape[0]}x{img_array.shape[1]} pixels")
        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")
            continue
        
        # Tester tous les modèles
        test_kmeans(image, image_name, report)
        test_gmm(image, image_name, report)
        test_meanshift(image, image_name, report)
        test_spectral(image, image_name, report)
    
    # Afficher le rapport
    report.end_time = time.time()
    report.print_report()
    
    print("\nTests de validation terminés!")

if __name__ == "__main__":
    main()
