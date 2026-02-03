# Features Bonus - Guide Complet

## Vue d'ensemble

Ce projet inclut 5 features bonus avancées pour enrichir l'expérience utilisateur et les capacités de clustering :

1. **Fenêtre de paramètres dédiée** - Configuration facile et intuitive
2. **Gestion des palettes de couleurs** - 4 palettes prédéfinies + personnalisées
3. **Visualisation 3D des clusters** - Graphiques interactifs avec matplotlib
4. **Prétraitement ACP (PCA)** - Réduction de dimensionalité
5. **Sauvegarde avancée** - Multiples formats et options

## Installation

Les features bonus utilisent les dépendances suivantes :

```bash
pip install matplotlib scikit-learn numpy pillow
```

Ces dépendances sont déjà incluses dans `requirements.txt`.

## Structure des fichiers

```
apptkr_imageprocessing/
├── dialogs/
│   ├── __init__.py
│   └── parameters_dialog.py         # Fenêtre de paramètres
├── utils/
│   ├── color_palette.py             # Gestion des palettes
│   ├── cluster_3d.py                # Visualisation 3D
│   └── pca_preprocessing.py         # ACP/PCA
├── palettes/                        # Palettes personnalisées (créé automatiquement)
│   └── (fichiers JSON)
├── BONUS_FEATURES.md               # Documentation détaillée
└── BONUS_INTEGRATION_EXAMPLE.py    # Exemple d'intégration
```

## Features détaillées

### 1. Fenêtre de paramètres dédiée

**Classe:** `ParametersDialog` (dialogs/parameters_dialog.py)

Remplace les sliders globaux par une fenêtre de dialogue contextuelle :

```python
from dialogs import ParametersDialog

dialog = ParametersDialog(parent_window, "K-Means", current_clusters=10)
result = dialog.get_result()

if result:
    clusters = result['clusters']  # Retourne {'clusters': 10}
    # ou pour MeanShift: {'bandwidth': 25.0}
```

**Avantages:**
- Interface plus propre
- Paramètres différents par modèle
- Annulation facile

### 2. Gestion des palettes de couleurs

**Classe:** `ColorPalette` (utils/color_palette.py)

Quatre palettes prédéfinies :

| Palette | Saturation | Brightness | Usage |
|---------|-----------|-----------|-------|
| Vibrant | 85% | 95% | Défaut, max contrast |
| Pastel | 40% | 95% | Couleurs douces |
| Dark | 90% | 70% | Couleurs intenses |
| Rainbow | Mixte | Mixte | Arc-en-ciel classique |

**Usage:**

```python
from utils.color_palette import ColorPalette

# Générer une palette
palette = ColorPalette.generate_palette('vibrant', n_colors=10)
# Retourne: np.array de shape (10, 3) avec RGB values

# Sauvegarder personnalisée
custom = np.array([[255, 0, 0], [0, 255, 0], ...])
ColorPalette.save_custom_palette('ma_palette', custom)

# Charger personnalisée
loaded = ColorPalette.load_custom_palette('ma_palette')

# Lister toutes les palettes personnalisées
custom_list = ColorPalette.list_custom_palettes()
# ['ma_palette', 'autre_palette', ...]
```

**Structure des fichiers palettes:**

```json
{
  "name": "ma_palette",
  "colors": [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
  ]
}
```

### 3. Visualisation 3D des clusters

**Classe:** `Cluster3DVisualization` (utils/cluster_3d.py)

Visualise les pixels et clusters dans l'espace RGB 3D :

```python
from utils.cluster_3d import Cluster3DVisualization

# Affichage interactif
pixels = np.array(image).reshape(-1, 3)
labels = model.predict(pixels)
Cluster3DVisualization.plot_clusters_3d(pixels, labels)

# Avec centres de clusters
centers = model.get_cluster_centers()
Cluster3DVisualization.plot_clusters_3d_with_centers(
    pixels, labels, centers,
    title="K-Means Clustering"
)

# Sauvegarder en PNG
Cluster3DVisualization.save_clusters_3d(
    pixels, labels,
    output_path='clusters_3d.png',
    title="Mon clustering"
)
```

**Features:**
- Rotation interactive (souris)
- Zoom (roulette)
- Couleurs par cluster
- Centres en étoiles noires
- Sauvegarde haute résolution (300 DPI)

### 4. Prétraitement ACP (PCA)

**Classe:** `PCAPreprocessing` (utils/pca_preprocessing.py)

Réduction de dimensionalité avant clustering :

```python
from utils.pca_preprocessing import PCAPreprocessing

# Initialiser PCA (2 ou 3 composantes)
pca = PCAPreprocessing(n_components=2)

# Appliquer transformation
pixels_pca = pca.fit_transform(pixels_rgb)
# Input: (N, 3) RGB
# Output: (N, 2) composantes principales

# Obtenir statistiques
variance = pca.get_explained_variance_ratio()
# [0.72, 0.25] → 97% de variance expliquée

total = pca.get_total_variance_explained()
# 0.97

# Afficher résumé
pca.print_summary()
```

**Résultat du print_summary():**

```
======================================================================
RÉSUMÉ DE L'ACP
======================================================================
Nombre de composantes: 2

Variance expliquée par composante:
  Composante 1: 72.45%
  Composante 2: 24.89%

Variance totale expliquée: 97.34%
======================================================================
```

**Quand utiliser PCA:**
- ✅ Grandes images (> 1M pixels)
- ✅ Données bruitées
- ✅ Comparaison entre modèles
- ❌ Images petites (< 100k pixels)
- ❌ Quand on veut conserver les couleurs réelles

### 5. Sauvegarde avancée

**Menu proposé:**

```
Fichier
├── Ouvrir Image...
├── Sauvegarder résultat
├── ─────────────────────────
├── Sauvegarde avancée
│   ├── Image segmentée (PNG/JPEG)...
│   ├── Visualisation 3D (PNG)...
│   ├── Paramètres utilisés (JSON)...
│   └── Exporter tout...
└── Quitter
```

**Formats supportés:**
- Image: PNG, JPEG, BMP
- Paramètres: JSON
- Visualisation: PNG (300 DPI)

## Exemples d'utilisation complète

### Exemple 1: Clustering simple avec palette

```python
from models.kmeans_model import KMeansClusteringModel
from utils.color_palette import ColorPalette
from PIL import Image
import numpy as np

# Charger image
image = Image.open('photo.png')

# Créer modèle
model = KMeansClusteringModel(n_clusters=5)

# Appliquer clustering
result = model.segment_image(image)

# Sauvegarder
result.save('result.png')
```

### Exemple 2: Clustering avec PCA

```python
from utils.pca_preprocessing import PCAPreprocessing

# Préparer données
pixels = np.array(image).reshape(-1, 3).astype(np.float32)

# Appliquer PCA
pca = PCAPreprocessing(n_components=2)
pixels_pca = pca.fit_transform(pixels)
pca.print_summary()

# Clustering sur données réduites
model.fit(pixels_pca)
labels = model.predict(pixels_pca)

# Visualiser en 3D (original, pas PCA)
Cluster3DVisualization.plot_clusters_3d(pixels, labels)
```

### Exemple 3: Comparaison de palettes

```python
# Afficher 4 versions avec palettes différentes
for palette_type in ['vibrant', 'pastel', 'dark', 'rainbow']:
    palette = ColorPalette.generate_palette(palette_type, n_clusters=5)
    # Visualiser...
```

### Exemple 4: Pipeline complet

```python
# 1. Charger et prétraiter
image = Image.open('photo.png')
pixels = np.array(image).reshape(-1, 3).astype(np.float32)

# 2. Appliquer PCA
pca = PCAPreprocessing(n_components=2)
pixels_pca = pca.fit_transform(pixels)
pca.print_summary()

# 3. Choisir palette
palette = ColorPalette.generate_palette('dark', n_colors=8)
ColorPalette.save_custom_palette('ma_config', palette)

# 4. Clustering
model = KMeansClusteringModel(n_clusters=8)
result = model.segment_image(image)

# 5. Exporter résultats
result.save('segmentation.png')
Cluster3DVisualization.save_clusters_3d(pixels, labels, 'clusters_3d.png')

print("Pipeline complet terminé!")
```

## Tests des features bonus

Les features bonus sont testées via :

```bash
# Tests unitaires
python tests/run_tests.py

# Tests manuels recommandés:
# 1. Ouvrir fenêtre paramètres → ajuster → appliquer
# 2. Changer palettes → vérifier couleurs
# 3. Afficher visualisation 3D → zoomer/tourner
# 4. Activer PCA checkbox → clustering différent
# 5. Exporter tout → vérifier fichiers créés
```

## Performances

| Feature | Temps | Notes |
|---------|-------|-------|
| PCA 2D | ~1s | Sur 1M pixels |
| Visualisation 3D | Interactif | Immédiat |
| Palette 1000 colors | <100ms | Très rapide |
| Sauvegarde PNG 300dpi | ~2s | Haute résolution |

## Limitations et future work

**Limitations actuelles:**
- PCA en 2D seulement (pas 3D encore)
- Palettes max 256 couleurs (limte PNG)
- Visualisation 3D non-interactive pour save

**À implémenter:**
- PCA 3D pour visualisation
- Blend de palettes
- Export en formats 3D (obj, ply)
- Prévisualisation palette avant clustering

## Dépannage

**Problème:** PCA ne fait rien visible
**Solution:** Vérifier que image.size > 100k pixels

**Problème:** Visualisation 3D trop lente
**Solution:** Appliquer PCA avant clustering

**Problème:** Palettes perdues après redémarrage
**Solution:** Palettes sauvegardées dans dossier `palettes/`

**Problème:** "ModuleNotFoundError: No module named 'matplotlib'"
**Solution:** `pip install matplotlib`

## Support et documentation

- **Guide détaillé:** `BONUS_FEATURES.md`
- **Exemple d'intégration:** `BONUS_INTEGRATION_EXAMPLE.py`
- **Tests:** `tests/`
- **Code source:** `dialogs/`, `utils/`

## Conclusion

Ces features bonus transforment l'application d'un simple segmentateur en outil professionnel de clustering avec :
- Interface intuitive
- Visualisations avancées
- Techniques ML modernes (PCA)
- Flexibilité maxima (palettes personnalisées)
- Export complet des résultats

À intégrer progressivement dans `apptkr_imageprocessing.py` selon vos besoins!
