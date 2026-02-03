"""
Visualisation 3D des clusters avec matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Cluster3DVisualization:
    """Classe pour visualiser les clusters en 3D"""
    
    @staticmethod
    def plot_clusters_3d(pixels, labels, title="Clusters 3D (RGB)"):
        """
        Afficher les clusters en 3D dans l'espace RGB
        
        Args:
            pixels: Array de pixels (N, 3) en RGB
            labels: Array de labels des clusters (N,)
            title: Titre du graphique
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Récupérer les couleurs uniques
        unique_labels = np.unique(labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plotter chaque cluster
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            cluster_pixels = pixels[mask]
            
            ax.scatter(
                cluster_pixels[:, 0],
                cluster_pixels[:, 1],
                cluster_pixels[:, 2],
                c=[colors[idx]],
                label=f'Cluster {label}',
                s=10,
                alpha=0.6
            )
        
        ax.set_xlabel('Red (R)')
        ax.set_ylabel('Green (G)')
        ax.set_zlabel('Blue (B)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_clusters_3d_with_centers(pixels, labels, centers, title="Clusters 3D avec centres"):
        """
        Afficher les clusters en 3D avec les centres de clusters
        
        Args:
            pixels: Array de pixels (N, 3) en RGB
            labels: Array de labels des clusters (N,)
            centers: Array des centres de clusters (K, 3)
            title: Titre du graphique
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Récupérer les couleurs uniques
        unique_labels = np.unique(labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plotter chaque cluster
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            cluster_pixels = pixels[mask]
            
            ax.scatter(
                cluster_pixels[:, 0],
                cluster_pixels[:, 1],
                cluster_pixels[:, 2],
                c=[colors[idx]],
                label=f'Cluster {label}',
                s=10,
                alpha=0.6
            )
        
        # Plotter les centres
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c='black',
            marker='*',
            s=500,
            label='Centres',
            edgecolors='white',
            linewidth=2
        )
        
        ax.set_xlabel('Red (R)')
        ax.set_ylabel('Green (G)')
        ax.set_zlabel('Blue (B)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def save_clusters_3d(pixels, labels, output_path, title="Clusters 3D (RGB)"):
        """
        Sauvegarder la visualisation 3D en image
        
        Args:
            pixels: Array de pixels (N, 3) en RGB
            labels: Array de labels des clusters (N,)
            output_path: Chemin du fichier de sortie
            title: Titre du graphique
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Récupérer les couleurs uniques
        unique_labels = np.unique(labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plotter chaque cluster
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            cluster_pixels = pixels[mask]
            
            ax.scatter(
                cluster_pixels[:, 0],
                cluster_pixels[:, 1],
                cluster_pixels[:, 2],
                c=[colors[idx]],
                label=f'Cluster {label}',
                s=10,
                alpha=0.6
            )
        
        ax.set_xlabel('Red (R)')
        ax.set_ylabel('Green (G)')
        ax.set_zlabel('Blue (B)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
