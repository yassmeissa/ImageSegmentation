# 🎨 Image Segmentation by Clustering

Application tkinter moderne pour segmenter des images avec 4 algorithmes de clustering distincts.

## ✨ Fonctionnalités

### 4 Algorithmes de Clustering
- **K-Means**: Partitions nettes (5-25 clusters) - Vitesse: ⚡⚡⚡ Qualité: ⭐⭐⭐⭐
- **GMM**: Gaussian Mixture Model (5-25 components) - Vitesse: ⚡⚡ Qualité: ⭐⭐⭐⭐
- **MeanShift**: Clustering par densité (bandwidth: 15-45) - Vitesse: ⚡⚡ Qualité: ⭐⭐⭐
- **Spectral**: Clustering topologique (5-25 clusters) - Vitesse: ⚡⚡⚡ Qualité: ⭐⭐⭐⭐⭐

### Interface Moderne
- Thème sombre professionnel (#1e1e1e, #0d47a1)
- Affichage côte-à-côte avant/après
- Zoom interactif (molette) + pan (drag)
- Status bar avec indicateurs ⏳/✅
- Paramètres adaptatifs par algorithme

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
├── theme.py                     # Système de thème
├── ui_components.py             # Widgets tkinter
├── image_processor.py           # Gestion des images
│
├── models/                      # Algorithmes de clustering
│   ├── base_model.py           # Classe abstraite
│   ├── kmeans_model.py         # K-Means optimisé
│   ├── gmm_model.py            # GMM (diag covariance = rapide)
│   ├── meanshift_model.py      # MeanShift adaptatif
│   └── spectral_model.py       # Spectral NN (rapide)
│
├── utils/                       # Utilitaires
│   ├── logger.py               # Logging console + fichier
│   └── image_loader.py         # Chargement d'images
│
├── img/                         # Images d'exemple
├── requirements.txt
└── README.md
```

## 🎮 Utilisation

### Workflow
1. **Charger image**: Menu Open → sélectionner PNG/JPG
2. **Choisir algorithme**: Clic sur K-Means / GMM / MeanShift / Spectral
3. **Ajuster paramètres**: Sliders pour clusters ou bandwidth
4. **Visualiser**: Voir avant/après côte-à-côte
5. **Exporter**: Menu Save → enregistrer la segmentation

### Comprendre les Paramètres

#### K-Means & GMM (Sliders 5-25)
```
Contrôle: Nombre exact de groupes/composantes
Effet: Augmenter = plus de couleurs/nuances
Usage: Meilleur pour images avec k clusterss distincts
```

#### MeanShift (Slider 15-45)
```
Contrôle: Bandwidth (rayon de kernel)
Effet: Augmenter = moins de clusters (plus lissé)
Note: Nombre final de clusters auto-détecté (≠ slider)
Usage: Clustering naturel basé sur la densité
```

#### Spectral (Slider 5-25)
```
Contrôle: Nombre exact de clusters
Effet: Augmenter = séparation fine topologique
Usage: Capture structure/formes plutôt que juste couleur
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
2. **Varia les clusters/bandwidth** pour voir l'effet
3. **Zoomer/pan** pour inspecter les détails
4. **Comparer visuellement** avant/après

### Quand utiliser quoi:
- **K-Means**: Images simple, pas beaucoup de nuances
- **GMM**: Transitions fluides, clustering soft
- **MeanShift**: Clustering naturel, pas de k fixe
- **Spectral**: Images complexes, structures fines

## 🐛 Dépannage

### L'app se fige?
→ Traitement en cours, attend 2-3s max.

### Sliders ne font rien?
→ Sélectionne un algorithme d'abord (K-Means, GMM, etc.)

### Image floue au zoom?
→ Dézoome avec molette ou redéplace-toi avec drag.

### Erreur "Image too large"?
→ Normal, redimensionnée auto à 1024x1024px max.

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

- ✅ Downsampling adaptatif par algorithme
- ✅ K-Means: n_init=30, max_iter=500 (ultra-agressif)
- ✅ GMM: covariance='diag' (3x+ rapide)
- ✅ Spectral: affinity='nearest_neighbors' (5-10x rapide vs RBF)
- ✅ MeanShift: bandwidth estimation rapide (500 samples)
- ✅ Float32 arrays (moitié moins mémoire que float64)
- ✅ Multi-threading (UI responsive)
- ✅ Garbage collection après chaque fit()

## 📄 Licence

Libre d'utilisation et modification.

---

**Enjoy segmenting!** 🎨✨
