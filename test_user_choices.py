import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

sys.path.insert(0, '/Users/yassmeissa/Downloads/apptkr_imageprocessing')

from utils.image_loader import ImageLoader
from models.kmeans_model import KMeansClusteringModel
from models.gmm_model import GMMClusteringModel
from models.meanshift_model import MeanShiftClusteringModel
from models.spectral_model import SpectralClusteringModel
from config import AppConfig
import matplotlib.cm as cm

def get_image_info(image_path):

    print(f"\nChargement de l'image: {os.path.basename(image_path)}")
    image = ImageLoader.load_image(image_path)
    img_array = np.array(image)
    
    print(f"Dimensions: {img_array.shape[0]} x {img_array.shape[1]} pixels")
    print(f"Canaux RGB: 3")
    print(f"Total pixels: {img_array.shape[0] * img_array.shape[1]:,}")
    
    return image, img_array

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
    except Exception as e:
        print(f"Palette '{palette_name}' non trouvée, utilisation de 'viridis'")
        colormap = cm.get_cmap('viridis')
        colors = []
        for i in range(n_colors):
            rgba = colormap(i / max(1, n_colors - 1))
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            colors.append([r, g, b])
        return np.array(colors, dtype=np.uint8)

def select_palette():

    palettes = ['viridis', 'plasma', 'inferno', 'cool', 'hot', 'spring', 'summer', 'autumn', 'winter', 'twilight']
    
    print("\nPalettes disponibles:")
    for i, p in enumerate(palettes, 1):
        print(f"  {i}. {p}")
    
    palette_choice = input("\nChoisir une palette (numéro ou nom) [viridis]: ").strip()
    
    if not palette_choice:
        return 'viridis'
    
    try:
        idx = int(palette_choice) - 1
        if 0 <= idx < len(palettes):
            return palettes[idx]
    except ValueError:
        pass
    
    if palette_choice in palettes:
        return palette_choice
    
    print(f"Palette inconnue, utilisation de 'viridis'")
    return 'viridis'

def select_and_configure_kmeans(image):

    print("\n" + "="*70)
    print("K-MEANS CLUSTERING - CONFIGURATION")
    print("="*70)
    
    # Nombre de clusters
    clusters_input = input(f"Nombre de clusters [{AppConfig.DEFAULT_KMEANS_CLUSTERS}] (min: {AppConfig.CLUSTERS_MIN}, max: {AppConfig.CLUSTERS_MAX}): ").strip()
    try:
        n_clusters = int(clusters_input) if clusters_input else AppConfig.DEFAULT_KMEANS_CLUSTERS
        if not (AppConfig.CLUSTERS_MIN <= n_clusters <= AppConfig.CLUSTERS_MAX):
            print(f"Valeur hors limites, utilisation du défaut: {AppConfig.DEFAULT_KMEANS_CLUSTERS}")
            n_clusters = AppConfig.DEFAULT_KMEANS_CLUSTERS
    except ValueError:
        print(f"Valeur invalide, utilisation du défaut: {AppConfig.DEFAULT_KMEANS_CLUSTERS}")
        n_clusters = AppConfig.DEFAULT_KMEANS_CLUSTERS
    
    # n_init
    n_init_input = input(f"n_init (nombre d'exécutions) [{30}] (min: 1, max: 100): ").strip()
    try:
        n_init = int(n_init_input) if n_init_input else 30
        n_init = max(1, min(100, n_init))
    except ValueError:
        n_init = 30
    
    # max_iter
    max_iter_input = input(f"max_iter (itérations) [{500}] (min: 100, max: 1000): ").strip()
    try:
        max_iter = int(max_iter_input) if max_iter_input else 500
        max_iter = max(100, min(1000, max_iter))
    except ValueError:
        max_iter = 500
    
    # Palette
    palette = select_palette()
    
    print(f"\n Configuration K-Means:")
    print(f"  - Clusters: {n_clusters}")
    print(f"  - n_init: {n_init}")
    print(f"  - max_iter: {max_iter}")
    print(f"  - Palette: {palette}")
    
    # Appliquer le modèle
    model = KMeansClusteringModel(n_clusters=n_clusters)
    model.kmeans.n_init = n_init
    model.kmeans.max_iter = max_iter
    
    color_palette = generate_palette(palette, n_clusters)
    segmented = model.segment_image(image, shared_palette=color_palette)
    
    return segmented, f"K-Means (clusters={n_clusters}, n_init={n_init}, max_iter={max_iter}, palette={palette})"

def select_and_configure_gmm(image):

    print("\n" + "="*70)
    print("GMM CLUSTERING - CONFIGURATION")
    print("="*70)
    
    # Nombre de composantes
    components_input = input(f"Nombre de composantes [{AppConfig.DEFAULT_GMM_COMPONENTS}] (min: {AppConfig.CLUSTERS_MIN}, max: {AppConfig.CLUSTERS_MAX}): ").strip()
    try:
        n_components = int(components_input) if components_input else AppConfig.DEFAULT_GMM_COMPONENTS
        if not (AppConfig.CLUSTERS_MIN <= n_components <= AppConfig.CLUSTERS_MAX):
            n_components = AppConfig.DEFAULT_GMM_COMPONENTS
    except ValueError:
        n_components = AppConfig.DEFAULT_GMM_COMPONENTS
    
    # max_iter
    max_iter_input = input(f"max_iter (itérations) [{100}] (min: 50, max: 500): ").strip()
    try:
        max_iter = int(max_iter_input) if max_iter_input else 100
        max_iter = max(50, min(500, max_iter))
    except ValueError:
        max_iter = 100
    
    # covariance_type
    print(f"Types de covariance: full, tied, diag, spherical")
    cov_type = input(f"Covariance type [diag]: ").strip() or "diag"
    if cov_type not in ['full', 'tied', 'diag', 'spherical']:
        cov_type = 'diag'
    
    # Palette
    palette = select_palette()
    
    print(f"\n Configuration GMM:")
    print(f"  - Composantes: {n_components}")
    print(f"  - max_iter: {max_iter}")
    print(f"  - Covariance: {cov_type}")
    print(f"  - Palette: {palette}")
    
    # Appliquer le modèle
    model = GMMClusteringModel(n_components=n_components)
    model.gmm.max_iter = max_iter
    model.gmm.covariance_type = cov_type
    
    color_palette = generate_palette(palette, n_components)
    segmented = model.segment_image(image, shared_palette=color_palette)
    
    return segmented, f"GMM (components={n_components}, max_iter={max_iter}, cov={cov_type}, palette={palette})"

def select_and_configure_meanshift(image):

    print("\n" + "="*70)
    print("MEANSHIFT CLUSTERING - CONFIGURATION")
    print("="*70)
    
    # Bandwidth
    bandwidth_input = input(f"Bandwidth [{AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH}] (min: {AppConfig.BANDWIDTH_MIN}, max: {AppConfig.BANDWIDTH_MAX}): ").strip()
    try:
        bandwidth = float(bandwidth_input) if bandwidth_input else AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
        if not (AppConfig.BANDWIDTH_MIN <= bandwidth <= AppConfig.BANDWIDTH_MAX):
            bandwidth = AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
    except ValueError:
        bandwidth = AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
    
    # Palette
    palette = select_palette()
    
    print(f"\n Configuration MeanShift:")
    print(f"  - Bandwidth: {bandwidth}")
    print(f"  - Palette: {palette}")
    
    # Appliquer le modèle
    model = MeanShiftClusteringModel(bandwidth=bandwidth)
    
    # Pour MeanShift, on utilise une palette avec 10 couleurs par défaut
    color_palette = generate_palette(palette, 10)
    segmented = model.segment_image(image, shared_palette=color_palette)
    
    return segmented, f"MeanShift (bandwidth={bandwidth}, palette={palette})"

def select_and_configure_spectral(image):

    print("\n" + "="*70)
    print("SPECTRAL CLUSTERING - CONFIGURATION")
    print("="*70)
    
    # Nombre de clusters
    clusters_input = input(f"Nombre de clusters [{AppConfig.DEFAULT_KMEANS_CLUSTERS}] (min: {AppConfig.CLUSTERS_MIN}, max: {AppConfig.CLUSTERS_MAX}): ").strip()
    try:
        n_clusters = int(clusters_input) if clusters_input else AppConfig.DEFAULT_KMEANS_CLUSTERS
        if not (AppConfig.CLUSTERS_MIN <= n_clusters <= AppConfig.CLUSTERS_MAX):
            n_clusters = AppConfig.DEFAULT_KMEANS_CLUSTERS
    except ValueError:
        n_clusters = AppConfig.DEFAULT_KMEANS_CLUSTERS
    
    # Palette
    palette = select_palette()
    
    print(f"\n Configuration Spectral:")
    print(f"  - Clusters: {n_clusters}")
    print(f"  - Affinity: nearest_neighbors (fixé)")
    print(f"  - Palette: {palette}")
    
    # Appliquer le modèle
    model = SpectralClusteringModel(n_clusters=n_clusters)
    
    color_palette = generate_palette(palette, n_clusters)
    segmented = model.segment_image(image, shared_palette=color_palette)
    
    return segmented, f"Spectral (clusters={n_clusters}, palette={palette})"

def display_result(original_array, segmented_image, title):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Résultat du clustering - {title}', fontsize=14, fontweight='bold')
    
    # Image originale
    axes[0].imshow(original_array)
    axes[0].set_title('Image Originale', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Image segmentée
    segmented_array = np.array(segmented_image)
    axes[1].imshow(segmented_array)
    axes[1].set_title('Résultat du Clustering', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():

    print("\n" + "="*70)
    print("TEST DE CLUSTERING - WORKFLOW IDENTIQUE AU GUI")
    print("="*70)
    
    # ÉTAPE 1: Sélectionner l'image
    print("\n" + "-"*70)
    print("ÉTAPE 1: SÉLECTIONNER UNE IMAGE")
    print("-"*70)
    
    image_dir = '/Users/yassmeissa/Downloads/apptkr_imageprocessing/img'
    images = ['cat.png', 'rose.png']
    
    available_images = []
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            available_images.append(img_path)
    
    if not available_images:
        print("Erreur: Aucune image trouvée!")
        return
    
    print("\nImages disponibles:")
    for i, img_path in enumerate(available_images, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    
    choice = input("\nChoisir une image (numéro): ").strip()
    try:
        idx = int(choice) - 1
        selected_image = available_images[idx]
    except (ValueError, IndexError):
        print("Erreur: choix invalide")
        return
    
    image, img_array = get_image_info(selected_image)
    
    # ÉTAPE 2: Sélectionner le modèle
    print("\n" + "-"*70)
    print("ÉTAPE 2: SÉLECTIONNER UN MODÈLE")
    print("-"*70)
    
    models = [
        ("K-Means", "kmeans"),
        ("GMM (Gaussian Mixture Model)", "gmm"),
        ("MeanShift", "meanshift"),
        ("Spectral Clustering", "spectral")
    ]
    
    print("\nModèles disponibles:")
    for i, (name, _) in enumerate(models, 1):
        print(f"  {i}. {name}")
    
    model_choice = input("\nChoisir un modèle (numéro): ").strip()
    try:
        model_idx = int(model_choice) - 1
        model_name, model_key = models[model_idx]
    except (ValueError, IndexError):
        print("Erreur: choix invalide")
        return
    
    # ÉTAPE 3: Configurer le modèle et obtenir le résultat
    print("\n" + "-"*70)
    print(f"ÉTAPE 3: CONFIGURER LES PARAMÈTRES DE {model_name.upper()}")
    print("-"*70)
    
    if model_key == "kmeans":
        segmented, title = select_and_configure_kmeans(image)
    elif model_key == "gmm":
        segmented, title = select_and_configure_gmm(image)
    elif model_key == "meanshift":
        segmented, title = select_and_configure_meanshift(image)
    elif model_key == "spectral":
        segmented, title = select_and_configure_spectral(image)
    else:
        print("Erreur: modèle non reconnu")
        return
    
    # ÉTAPE 4: Afficher le résultat
    print("\n" + "-"*70)
    print("ÉTAPE 4: AFFICHAGE DU RÉSULTAT")
    print("-"*70)
    
    display_result(img_array, segmented, title)
    
    print("\nTest terminé!")

if __name__ == "__main__":
    main()
