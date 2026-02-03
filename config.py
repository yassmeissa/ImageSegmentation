class AppConfig:
    # Fenêtre principale
    WINDOW_TITLE = "Image Segmentation by Clustering"
    WINDOW_RESIZABLE = False
    
    # Dimensions du canvas d'affichage
    CANVAS_WIDTH = 600
    CANVAS_HEIGHT = 600
    
    # Valeurs par défaut des paramètres (INCREASED FOR BETTER RESULTS)
    DEFAULT_KMEANS_CLUSTERS = 10     # Changed from 6 to 10 (to match slider min)
    DEFAULT_GMM_COMPONENTS = 10      # Changed from 6 to 10 (to match slider min)
    DEFAULT_DBSCAN_EPS = 25.0
    DEFAULT_DBSCAN_MIN_SAMPLES = 3
    DEFAULT_MEANSHIFT_BANDWIDTH = 25
    
    # Plages de sliders
    CLUSTERS_MIN = 10       # Changed from 2 to 10
    CLUSTERS_MAX = 20
    BANDWIDTH_MIN = 10
    BANDWIDTH_MAX = 50
    EPS_MIN = 5
    EPS_MAX = 100
    
    # Chemins
    ICON_PATH = 'res/icon.png'
    IMAGE_FOLDER = 'img'
    
    # Formats d'image supportés
    SUPPORTED_FORMATS = ((".png", "PNG images"), (".jpg", "JPEG images"), (".*", "All files"))
    
    # Couleurs UI
    BG_COLOR = "#f0f0f0"
    BUTTON_COLOR = "#4CAF50"
    BUTTON_ACTIVE_COLOR = "#45a049"
    
    # Polices
    FONT_TITLE = ("Arial", 12, "bold")
    FONT_LABEL = ("Arial", 10)
    FONT_BUTTON = ("Arial", 11)
