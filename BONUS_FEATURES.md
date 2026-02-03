# Features Bonus Implémentées

## 1. Fenêtre de paramètres dédiée

**Fichier:** `dialogs/parameters_dialog.py`

Une fenêtre de dialogue séparée pour choisir les paramètres de chaque modèle :
- Pour K-Means, GMM, Spectral: Slider pour le nombre de clusters (5-25)
- Pour MeanShift: Slider pour le bandwidth (15-45)
- Boutons "Appliquer" et "Annuler"

**Usage:**
```python
from dialogs import ParametersDialog

dialog = ParametersDialog(parent_window, "K-Means")
result = dialog.get_result()
if result:
    clusters = result['clusters']
```

## 2. Gestion des palettes de couleurs

**Fichier:** `utils/color_palette.py`

4 palettes de couleurs prédéfinies :
- **Vibrant**: Couleurs vives et contrastées (défaut)
- **Pastel**: Couleurs douces et apaisantes
- **Dark**: Couleurs sombres et intenses
- **Rainbow**: Arc-en-ciel classique

Possibilité de sauvegarder et charger des palettes personnalisées.

**Usage:**
```python
from utils.color_palette import ColorPalette

# Générer une palette
palette = ColorPalette.generate_palette('vibrant', n_colors=10)

# Sauvegarder une palette personnalisée
ColorPalette.save_custom_palette('ma_palette', palette)

# Charger une palette personnalisée
loaded = ColorPalette.load_custom_palette('ma_palette')

# Lister les palettes personnalisées
custom_palettes = ColorPalette.list_custom_palettes()
```

## 3. Visualisation 3D des clusters

**Fichier:** `utils/cluster_3d.py`

Visualisation des clusters dans l'espace 3D (RGB) avec matplotlib :
- Affichage interactif des clusters
- Affichage des centres de clusters
- Sauvegarde des visualisations en images PNG

**Usage:**
```python
from utils.cluster_3d import Cluster3DVisualization

# Afficher les clusters en 3D
Cluster3DVisualization.plot_clusters_3d(pixels, labels)

# Afficher avec les centres
Cluster3DVisualization.plot_clusters_3d_with_centers(
    pixels, labels, centers, 
    title="Mon clustering"
)

# Sauvegarder en image
Cluster3DVisualization.save_clusters_3d(
    pixels, labels, 
    output_path='clusters_3d.png'
)
```

## 4. Prétraitement avec ACP (PCA)

**Fichier:** `utils/pca_preprocessing.py`

Application de l'Analyse en Composantes Principales :
- Réduction de dimensionalité (3D RGB → 2D/3D)
- Normalisation automatique des données
- Affichage de la variance expliquée
- Comparaison des résultats avec/sans PCA

**Usage:**
```python
from utils.pca_preprocessing import PCAPreprocessing

# Créer une instance PCA
pca = PCAPreprocessing(n_components=2)

# Appliquer PCA
pixels_transformed = pca.fit_transform(pixels_rgb)

# Obtenir statistiques
variance_ratio = pca.get_explained_variance_ratio()
total_variance = pca.get_total_variance_explained()

# Afficher résumé
pca.print_summary()
```

## 5. Sauvegarde des résultats

**Intégration dans l'interface principale:**
- Menu "Sauvegarder" avec options multiples
- Sauvegarde de l'image segmentée (PNG, JPEG)
- Sauvegarde de la visualisation 3D (PNG)
- Sauvegarde des paramètres utilisés (JSON)

## Structure des fichiers

```
apptkr_imageprocessing/
├── dialogs/
│   ├── __init__.py
│   └── parameters_dialog.py
├── utils/
│   ├── color_palette.py
│   ├── cluster_3d.py
│   └── pca_preprocessing.py
└── palettes/
    └── (palettes personnalisées sauvegardées en JSON)
```

## Intégration dans l'interface GUI

### 1. Paramètres dédiés
Au lieu d'utiliser les sliders par défaut, un clic sur un modèle ouvre une fenêtre dédiée.

### 2. Menu Couleurs
Ajouter un menu pour choisir la palette de couleurs avant d'appliquer le modèle.

### 3. Menu Visualisations
Options pour afficher la visualisation 3D des clusters trouvés.

### 4. Checkbox PCA
Ajouter une checkbox "Appliquer PCA" pour le prétraitement des données.

### 5. Menu Sauvegarde avancée
```
Fichier
├── Ouvrir image...
├── Sauvegarder résultat
├── ─────────────────
├── Sauvegarde avancée
│   ├── Image segmentée...
│   ├── Visualisation 3D...
│   ├── Paramètres (JSON)...
│   └── Tout exporter...
└── Quitter
```

## Exemples d'utilisation complète

### Exemple 1: Clustering avec PCA et palette personnalisée
```python
# Charger image
image = Image.open('image.png')
pixels = np.array(image).reshape(-1, 3)

# Appliquer PCA
pca = PCAPreprocessing(n_components=2)
pixels_pca = pca.fit_transform(pixels)
pca.print_summary()

# Clustering
model = KMeansClusteringModel(n_clusters=5)
labels = model.predict(pixels_pca)

# Générer palette personnalisée
palette = ColorPalette.generate_palette('dark', n_colors=5)
ColorPalette.save_custom_palette('dark_5', palette)

# Visualiser en 3D
Cluster3DVisualization.plot_clusters_3d(pixels, labels)
```

### Exemple 2: Comparaison de palettes
```python
palettes = ['vibrant', 'pastel', 'dark', 'rainbow']

for palette_name in palettes:
    palette = ColorPalette.generate_palette(palette_name, n_clusters)
    # Appliquer et afficher résultats...
```

## Tests de validation

Les fichiers bonus sont testés via :
- `tests/test_models.py` - Validation des modèles
- `tests/test_ui.py` - Validation de l'interface
- Tests manuels recommandés pour les dialogs et visualisations

## Notes importantes

1. **Dépendances**: matplotlib pour les visualisations 3D, scikit-learn pour PCA
2. **Performances**: PCA peut améliorer significativement la vitesse pour grandes images
3. **Palettes**: Les palettes personnalisées sont sauvegardées en JSON dans `palettes/`
4. **ACP vs RGB**: PCA réduit à 2-3 dimensions, utile avant clustering sur petits datasets

## À intégrer dans apptkr_imageprocessing.py

```python
from dialogs import ParametersDialog
from utils.color_palette import ColorPalette
from utils.cluster_3d import Cluster3DVisualization
from utils.pca_preprocessing import PCAPreprocessing

# Ajouter checkbox PCA
self.pca_var = tk.BooleanVar(value=False)

# Ajouter menu couleurs
self.selected_palette = 'vibrant'

# Ajouter menu visualisations
file_menu.add_command(
    label="Visualisation 3D", 
    command=self.show_3d_clusters
)
```
