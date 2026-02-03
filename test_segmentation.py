"""
Script de test - Clustering d'images
Charge une image, applique tous les modèles avec les mêmes paramètres
Affiche une grosse image avec tous les résultats côte à côte
"""

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
from models.base_model import BaseClusteringModel
from config import AppConfig


def get_image_info(image_path):
    """Charge et affiche les infos de l'image"""
    print(f"\nChargement de l'image: {os.path.basename(image_path)}")
    image = ImageLoader.load_image(image_path)
    img_array = np.array(image)
    
    print(f"Dimensions: {img_array.shape[0]} x {img_array.shape[1]} pixels")
    print(f"Canaux RGB: 3")
    print(f"Total pixels: {img_array.shape[0] * img_array.shape[1]:,}")
    
    pixels = img_array.reshape(-1, 3)
    return image, img_array, pixels


def apply_all_models(image, img_array, pixels, n_clusters=10, bandwidth=25):
    """Applique tous les modèles exactement comme dans le GUI (via segment_image())"""
    print(f"\nTraitement en cours...")
    
    results = {}
    models_config = [
        ("K-Means", KMeansClusteringModel(n_clusters=n_clusters)),
        ("GMM", GMMClusteringModel(n_components=n_clusters)),
        ("MeanShift", MeanShiftClusteringModel(bandwidth=bandwidth)),
        ("Spectral", SpectralClusteringModel(n_clusters=n_clusters))
    ]
    
    for model_name, model in models_config:
        print(f"  - {model_name}...", end=" ", flush=True)
        
        # Utiliser segment_image() exactement comme le GUI
        segmented_image = model.segment_image(image)
        segmented_array = np.array(segmented_image)
        
        # Compter les clusters réels trouvés par le modèle
        # En convertissant en labels pour compter les couleurs uniques
        pixels_flat = segmented_array.reshape(-1, 3)
        unique_colors = np.unique(pixels_flat, axis=0)
        actual_clusters = len(unique_colors)
        
        results[model_name] = {
            'image': segmented_array,
            'clusters': actual_clusters
        }
        print(f"OK ({actual_clusters} clusters trouvés)")
    
    return results, img_array


def display_all_results(img_array, results, bandwidth=None):
    """Affiche tous les résultats dans une grosse image"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparaison des modèles de clustering', fontsize=16, fontweight='bold')
    
    # Image originale
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Image Originale', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Résultats des modèles
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    for (model_name, data), (row, col) in zip(results.items(), positions):
        axes[row, col].imshow(data['image'])
        
        # Affichage différent pour MeanShift (bandwidth au lieu de clusters)
        if model_name == "MeanShift" and bandwidth is not None:
            title = f"{model_name}\n(bandwidth={bandwidth})"
        else:
            title = f"{model_name}\n({data['clusters']} clusters)"
        
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Masquer le dernier subplot inutilisé
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("TEST DE CLUSTERING - COMPARAISON DES MODÈLES")
    print("="*70)
    
    # Chemins des images
    image_dir = '/Users/yassmeissa/Downloads/apptkr_imageprocessing/img'
    images = ['cat.png', 'rose.png']
    
    # Trouver les images disponibles
    available_images = []
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            available_images.append(img_path)
    
    if not available_images:
        print("Erreur: Aucune image trouvée!")
        return
    
    # Sélectionner une image
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
    
    # Charger l'image
    image, img_array, pixels = get_image_info(selected_image)
    
    # Demander les paramètres à l'utilisateur
    print(f"\nParamètres (appuyer Entrée pour utiliser les valeurs par défaut):")
    
    # Clusters
    clusters_input = input(f"  Nombre de clusters [{AppConfig.DEFAULT_KMEANS_CLUSTERS}] (min: {AppConfig.CLUSTERS_MIN}, max: {AppConfig.CLUSTERS_MAX}): ").strip()
    try:
        n_clusters = int(clusters_input) if clusters_input else AppConfig.DEFAULT_KMEANS_CLUSTERS
        if not (AppConfig.CLUSTERS_MIN <= n_clusters <= AppConfig.CLUSTERS_MAX):
            print(f"⚠️ Valeur hors limites, utilisation du défaut: {AppConfig.DEFAULT_KMEANS_CLUSTERS}")
            n_clusters = AppConfig.DEFAULT_KMEANS_CLUSTERS
    except ValueError:
        print(f"⚠️ Valeur invalide, utilisation du défaut: {AppConfig.DEFAULT_KMEANS_CLUSTERS}")
        n_clusters = AppConfig.DEFAULT_KMEANS_CLUSTERS
    
    # Bandwidth
    bandwidth_input = input(f"  Bandwidth MeanShift [{AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH}] (min: {AppConfig.BANDWIDTH_MIN}, max: {AppConfig.BANDWIDTH_MAX}): ").strip()
    try:
        bandwidth = float(bandwidth_input) if bandwidth_input else AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
        if not (AppConfig.BANDWIDTH_MIN <= bandwidth <= AppConfig.BANDWIDTH_MAX):
            print(f"⚠️ Valeur hors limites, utilisation du défaut: {AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH}")
            bandwidth = AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
    except ValueError:
        print(f"⚠️ Valeur invalide, utilisation du défaut: {AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH}")
        bandwidth = AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
    
    print(f"\nParamètres utilisés:")
    print(f"  - Clusters (K-Means, GMM, Spectral): {n_clusters}")
    print(f"  - Bandwidth (MeanShift): {bandwidth}")
    
    # Appliquer tous les modèles
    results, img_array = apply_all_models(image, img_array, pixels, n_clusters, bandwidth)
    
    # Afficher les résultats
    print("\nAffichage des résultats...")
    display_all_results(img_array, results, bandwidth)
    
    print("\n✅ Test terminé!")


if __name__ == "__main__":
    main()
