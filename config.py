class AppConfig:
    WINDOW_TITLE = "Image Segmentation by Clustering"
    WINDOW_RESIZABLE = False
    
    CANVAS_WIDTH = 600
    CANVAS_HEIGHT = 600
    
    DEFAULT_KMEANS_CLUSTERS = 10     
    DEFAULT_GMM_COMPONENTS = 10      
    DEFAULT_DBSCAN_EPS = 25.0
    DEFAULT_DBSCAN_MIN_SAMPLES = 3
    DEFAULT_MEANSHIFT_BANDWIDTH = 25

    CLUSTERS_MIN = 5
    CLUSTERS_MAX = 25
    BANDWIDTH_MIN = 15
    BANDWIDTH_MAX = 45
    EPS_MIN = 5
    EPS_MAX = 100
    
    ICON_PATH = 'res/icon.png'
    IMAGE_FOLDER = 'img'
    
    SUPPORTED_FORMATS = ((".png", "PNG images"), (".jpg", "JPEG images"), (".*", "All files"))
    
    BG_COLOR = "#f0f0f0"
    BUTTON_COLOR = "#4CAF50"
    BUTTON_ACTIVE_COLOR = "#45a049"
    
    FONT_TITLE = ("Arial", 12, "bold")
    FONT_LABEL = ("Arial", 10)
    FONT_BUTTON = ("Arial", 11)
