# 🎨 Image Segmentation by Clustering

Application tkinter moderne pour segmenter des images avec 4 algorithmes de clustering distincts.

## ✨ Fonctionnalités

### 4 Algorithmes de Clustering
- **K-Means**: Partitions nettes (5-25 clusters) - Vitesse: ⚡⚡⚡ Qualité: ⭐⭐⭐⭐
- **GMM**: Gaussian Mixture Model (5-25 components) - Vitesse: ⚡⚡ Qualité: ⭐⭐⭐⭐
- **MeanShift**: Clustering par densité (bandwidth: 15-45) - Vitesse: ⚡⚡ Qualité: ⭐⭐⭐
- **Spectral**: Clustering topologique (5-25 clusters) - Vitesse: ⚡⚡⚡ Qualité: ⭐⭐⭐⭐⭐

### Prétraitement Avancé ⭐ NOUVEAU
- **PCA Preprocessing**: Réduction dimensionnelle RGB → 3D
- Checkbox "Use PCA Preprocessing" pour chaque segmentation
- Affichage variance expliquée (ex: "PCA: 95.1% variance explained")
- Compatible avec tous les modèles de clustering

### Visualisation 3D Avancée
- Graphique 3D RGB interactif avec matplotlib
- Affichage des centroïdes (étoiles rouges)
- Export en haute résolution (150 DPI)
- Fenêtre Toplevel avec boutons Save/Close stylisés

### Sauvegarde Multi-Formats
- **PNG Standard**: Meilleure qualité, taille intermédiaire
- **PNG Haute Résolution**: 300 DPI pour impression
- **JPEG**: Compression, contrôle qualité 1-100
- **BMP**: Sans compression, qualité maximale
- **Auto-naming**: `segmented_<image>_<model>_<params>_palette-<name>.ext`

### Interface Moderne
- Thème sombre professionnel (#1e1e1e, #0d47a1)
- Affichage côte-à-côte avant/après
- Zoom interactif (molette) + pan (drag)
- Status bar avec indicateurs ⏳/✅
- Paramètres adaptatifs par algorithme
- Couleurs distinctes par modèle (rouge/teal/jaune/violet)

### Performance Optimisée
- Downsampling adaptatif (2-3k pixels max)
- Multi-threading (UI non-bloquante)
- Exécution < 2s par image
- Memory-efficient (float32, garbage collection)

## 🚀 Démarrage Rapide

```bash
python3 apptkr_imageprocessing.py
```

**Prérequis**: Python 3.13+

## 📦 Installation Complète

```bash
# Créer environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Lancer
python3 apptkr_imageprocessing.py
```

## 📁 Structure du Projet

```
apptkr_imageprocessing/
├── apptkr_imageprocessing.py   # Main app (UI + contrôleur)
├── config.py                    # Defaults et intervalles
├── theme.py                     # Système de thème (couleurs/styles)
├── ui_components.py             # Widgets tkinter (ModelButton, ComparisonCanvas)
├── image_processor.py           # Gestion des images et PCA
├── cleanup.py                   # Gestion de la mémoire
│
├── models/                      # Algorithmes de clustering
│   ├── base_model.py           # Classe abstraite
│   ├── kmeans_model.py         # K-Means optimisé
│   ├── gmm_model.py            # GMM (diag covariance = rapide)
│   ├── meanshift_model.py      # MeanShift adaptatif
│   └── spectral_model.py       # Spectral NN (rapide)
│
├── dialogs/                     # Fenêtres de dialogue
│   ├── config_dialog.py        # Configuration des modèles + palettes
│   └── parameters_dialog.py    # Paramètres avancés
│
├── utils/                       # Utilitaires
│   ├── logger.py               # Logging console + fichier
│   ├── image_loader.py         # Chargement d'images
│   ├── color_palette.py        # Gestion des palettes de couleurs
│   ├── cluster_3d.py           # Visualisation 3D matplotlib
│   ├── pca_preprocessing.py    # PCA avec StandardScaler
│   ├── visualization.py        # Visualisations supplémentaires
│   ├── memory_manager.py       # Monitoring mémoire
│   ├── batch_processor.py      # Traitement batch d'images
│   └── background_worker.py    # Workers threading
│
├── tests/                       # Tests unitaires
│   └── test_models.py          # Tests des modèles
│
├── img/                         # Images d'exemple
│   ├── cat.png
│   └── rose.png
│
├── res/                         # Ressources
│   └── icon.png
│
├── requirements.txt             # Dépendances Python
└── README.md                    # Cette documentation
```

## 🎮 Utilisation

### Workflow Complet
1. **Charger image**: Menu File → Open Image (ou bouton "Open Image")
2. **Activer PCA (optionnel)**: Cocher "Use PCA Preprocessing" pour réduction dimensionnelle
3. **Choisir algorithme**: Clic sur K-Means / GMM / MeanShift / Spectral
4. **Configurer paramètres**: 
   - Dialogue de configuration s'ouvre automatiquement
   - Ajuster clusters/bandwidth selon algorithme
   - Choisir palette de couleurs
5. **Visualiser résultat**: Voir segmentation avant/après côte-à-côte
6. **Exporter résultats**:
   - **Save Result**: Menu File → Save Result (formats: PNG/JPEG/BMP)
   - **Export 3D**: Menu File → Export 3D ou bouton "Export 3D"
7. **Analyser PCA**: Menu Tools → PCA Analysis (si enabled)

### Menu Principal
```
File
├── Open Image        → Charger image PNG/JPG/BMP
├── Save Result       → Sauvegarde avancée (formats + auto-naming)
└── Exit              → Quitter app

Visualization
├── View 3D Clusters  → Afficher graphique 3D interactif
└── Export 3D (PNG)   → Exporter 3D en haute résolution

Tools
├── Color Palettes    → Choisir palette (viridis, plasma, etc.)
└── PCA Analysis      → Rapport variance PCA
```

### Panneau Latéral Gauche
```
Clustering Models
├── K-Means       → Fast, sharp partitions
├── GMM           → Smooth, probabilistic
├── MeanShift     → Density-based, auto k
└── Spectral      → Topological structure

Preprocessing
└── ☑ Use PCA Preprocessing  → Affiche variance expliquée

Operations
├── Open Image    → Charger
├── Save Result   → Exporter multi-formats
└── Export 3D     → Visualisation 3D
```

### Comprendre les Paramètres

#### K-Means & GMM (Sliders 5-25)
```
Contrôle: Nombre exact de groupes/composantes
Effet: Augmenter = plus de couleurs/nuances
Usage: Meilleur pour images avec k clusters distincts
Paramètres K-Means:
  - n_init: 30 (initialisations)
  - max_iter: 500 (itérations max)
Paramètres GMM:
  - covariance_type: 'diag' (rapide)
  - max_iter: 100
```

#### MeanShift (Slider 15-45)
```
Contrôle: Bandwidth (rayon de kernel)
Effet: Augmenter = moins de clusters (plus lissé)
Note: Nombre final auto-détecté (≠ slider)
Usage: Clustering naturel basé sur la densité
```

#### Spectral (Slider 5-25)
```
Contrôle: Nombre exact de clusters
Effet: Augmenter = séparation fine topologique
Paramètres:
  - affinity: 'nearest_neighbors' (rapide)
  - assign_labels: 'kmeans'
Usage: Structure topologique, formes complexes
```

### PCA Preprocessing (Nouveau ⭐)
```
Checkbox: "Use PCA Preprocessing" dans panneau Preprocessing
Composantes: 3 (RGB → 3D)
Effet: 
  - Réduit dimensionnalité avant clustering
  - Affiche variance expliquée (ex: "95.1% variance explained")
  - Compatible tous les algorithmes
Usage:
  - Images haute-résolution
  - Clustering difficile
  - Analyse de structure principale
```

### Visualisation 3D & Export
```
Menu: Visualization → Export 3D ou bouton "Export 3D"
Affiche:
  - Scatter plot 3D des pixels RGB
  - Centroïdes en étoiles rouges
  - Axes X=Red, Y=Green, Z=Blue
  - Légende des clusters
  - Titre avec modèle et nombre clusters

Boutons:
  - 💾 Save Plot: Export PNG 150 DPI
  - ❌ Close: Fermer fenêtre
```

### Sauvegarde Avancée
```
Menu: File → Save Result
Options:
  ✓ PNG (Standard)          → Qualité max, taille moyenne
  ✓ JPEG                    → Compression, slider qualité 1-100
  ✓ PNG (High Resolution)   → 300 DPI pour impression
  ✓ BMP                     → Sans compression

Auto-naming:
  Format: segmented_<image>_<model>_<clusters-X>_palette-<name>.ext
  Exemple: segmented_cat_kmeans_clusters-5_palette-viridis.png

Avantages:
  - Retrouve facilement vos segmentations
  - Inclut tous les paramètres dans le nom
```

## 🔧 Configuration

Éditer `config.py` pour personnaliser:

```python
# Intervalles des sliders
CLUSTERS_MIN = 5           # Min clusters (K-Means, GMM, Spectral)
CLUSTERS_MAX = 25          # Max clusters
BANDWIDTH_MIN = 15         # Min bandwidth (MeanShift)
BANDWIDTH_MAX = 45         # Max bandwidth

# Defaults
DEFAULT_KMEANS_CLUSTERS = 10
DEFAULT_GMM_COMPONENTS = 10
DEFAULT_MEANSHIFT_BANDWIDTH = 25
```

## 🎨 Personnaliser le Thème

Éditer `theme.py`:

```python
class Theme:
    BG = "#1e1e1e"           # Fond principal
    PANEL = "#263238"        # Panneaux latéraux
    ACCENT = "#0d47a1"       # Couleur accent (bleu)
    TEXT = "#ffffff"         # Texte
    CANVAS_BG = "#111111"    # Fond canvas
    HOVER = "#37474f"        # Hover buttons
```

## 📊 Performance Comparée

| Algorithme | Temps | Qualité | Cas d'Usage |
|-----------|-------|---------|-----------|
| **K-Means** | 0.5-1s | ⭐⭐⭐⭐ | Images avec clusters distincts |
| **GMM** | 1.5-2s | ⭐⭐⭐⭐ | Transitions douces, probabilistes |
| **MeanShift** | 0.8-1.5s | ⭐⭐⭐ | Clustering naturel, densité |
| **Spectral** | 0.5-1s | ⭐⭐⭐⭐⭐ | Structure topologique, formes |

## 🔍 Logging & Debug

L'app génère `app.log`:

```bash
tail -f app.log              # Logs en temps réel
grep "ERROR" app.log         # Erreurs uniquement
```

Logs incluent:
- Initialisation des modèles
- Progression du clustering
- Temps d'exécution
- Taille des downsamples

## 💡 Conseils d'Usage

### Pour les meilleures résultats:
1. **Tester tous les algorithmes** sur la même image
2. **Tester PCA** pour images complexes (variance affichée)
3. **Varier clusters/bandwidth** pour voir l'effet
4. **Zoomer/pan** pour inspecter les détails
5. **Exporter 3D** pour analyser la structure spatiale
6. **Comparer visuellement** avant/après

### Quand utiliser quoi:
- **K-Means**: Images simples, clusters distincts
- **GMM**: Transitions fluides, clustering soft probabiliste
- **MeanShift**: Clustering naturel, pas de k fixe
- **Spectral**: Images complexes, structures topologiques
- **PCA**: Images haute-res, clustering difficile

### Cas d'usage recommandés:
```
Paysages nature       → Spectral ou GMM
Portraits/objets      → K-Means
Textures complexes    → Spectral + PCA
Objets géométriques   → K-Means + Spectral
```

## ✨ Nouvelles Fonctionnalités (v2.0)

### PCA Preprocessing ⭐
- Réduction dimensionnelle RGB → 3D avant clustering
- Améliore performances images complexes
- Affiche variance expliquée en temps réel
- Compatible tous les modèles

### Visualisation 3D Avancée ⭐
- Graphique 3D RGB interactif matplotlib
- Affichage centroïdes (étoiles rouges)
- Export haute résolution (150 DPI)
- Tooltip info clusters

### Sauvegarde Multi-Formats ⭐
- PNG standard (qualité max)
- PNG Haute Résolution 300 DPI
- JPEG avec slider qualité 1-100
- BMP sans compression
- Auto-naming avec paramètres
- Interface avancée intuitive

### Améliorations UI
- Boutons stylisés par modèle (couleurs distinctes)
- Affichage temps traitement
- Indicateurs statut ⏳/✅
- Palettes de couleurs intégrées
- Menu Visualization complet

## 🐛 Dépannage

### L'app se fige?
→ Traitement en cours, attend 2-3s max (threading actif)

### Sliders ne font rien?
→ Sélectionne un algorithme d'abord (K-Means, GMM, MeanShift ou Spectral)

### "Export 3D" grisé?
→ Segmente une image d'abord avec un modèle

### PCA affiche variance 0%?
→ Modèles avec peu de variation. Normal pour images simples.

### Image floue au zoom?
→ Dézoome avec molette ou redéplace-toi avec drag

### Erreur "Image too large"?
→ Normal, redimensionnée auto à 1024x1024px max pour perf

### Quelle palette pour quelle image?
→ Tester! Menu Tools → Color Palettes (viridis, plasma, etc.)

## 📋 Dépendances

```
numpy==2.4.2              # Opérations matricielles
scikit-learn==1.8.0       # Clustering algorithms
Pillow==12.1.0            # Image processing
scipy==1.17.0             # Scientific functions
matplotlib==3.10.1        # Visualization
psutil==7.2.2             # Memory monitoring
```

## 📝 Architecture Design Patterns

### Utilisés:
- **Abstract Base Class**: `BaseClusteringModel` pour tous les algorithmes
- **Strategy Pattern**: Différentes stratégies de clustering swappables
- **Observer Pattern**: UI mise à jour automatiquement après clustering
- **Singleton-like**: `AppLogger`, `Theme` classes

### Principes:
- Séparation UI / Logique métier
- Responsabilité unique (SRP)
- DRY (Don't Repeat Yourself)
- Noms expressifs et documentation

## 🚀 Optimisations Appliquées

### Algorithmes
- ✅ Downsampling adaptatif par algorithme
- ✅ K-Means: n_init=30, max_iter=500 (ultra-optimisé)
- ✅ GMM: covariance='diag' (3x+ rapide)
- ✅ Spectral: affinity='nearest_neighbors' (5-10x rapide vs RBF)
- ✅ MeanShift: bandwidth estimation rapide (500 samples)
- ✅ PCA: StandardScaler + réduction 3D

### Mémoire & Performance
- ✅ Float32 arrays (moitié moins mémoire que float64)
- ✅ Multi-threading (UI responsive)
- ✅ Garbage collection après chaque fit()
- ✅ Image downsampling (1024x1024 max)
- ✅ Lazy loading d'images

### Interface
- ✅ Tkinter natif (pas de deps lourd)
- ✅ Redraw optimisé (cahced images)
- ✅ Status bar indicateurs temps réel
- ✅ Threading worker pour clustering

### Code Quality
- ✅ Imports nettoyés (0 unused)
- ✅ Modules bien organisés (SRP)
- ✅ Logging structuré
- ✅ Design patterns (Strategy, Observer)
- ✅ Documentation complète

## 📄 Licence

Libre d'utilisation et modification.

---

**Enjoy segmenting!** 🎨✨
