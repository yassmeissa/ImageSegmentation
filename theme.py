

class Theme:

    BG = "#1e1e1e"
    PANEL = "#263238"
    ACCENT = "#0d47a1"
    TEXT = "#ffffff"
    MUTED = "#9e9e9e"


    SUCCESS = "#4CAF50"
    HOVER = "#37474f"
    DISABLED = "#424242"

    CANVAS_BG = "#111111"
    CANVAS_BORDER = "#0d47a1"

    KMEANS_COLOR = "#FF6B6B"
    GMM_COLOR = "#4ECDC4"
    MEANSHIFT_COLOR = "#FFD93D"
    SPECTRAL_COLOR = "#A569BD"

    PROCESSING = "#FFA500"
    DONE = "#4CAF50"             
    
    @staticmethod
    def get_button_style():

        return {
            "font": ("Segoe UI", 11, "bold"),
            "relief": "flat",
            "bg": Theme.PANEL,
            "fg": Theme.TEXT,
            "activebackground": Theme.HOVER,
            "padx": 12,
            "pady": 8,
            "cursor": "hand2",
            "highlightthickness": 0,
            "borderwidth": 0
        }
    
    @staticmethod
    def get_label_style():

        return {
            "bg": Theme.PANEL,
            "fg": Theme.TEXT
        }
    
    @staticmethod
    def get_slider_style():

        return {
            "bg": Theme.PANEL,
            "fg": Theme.TEXT,
            "troughcolor": Theme.BG,
            "highlightthickness": 0
        }
    
    @staticmethod
    def get_model_color(model_name: str) -> str:

        color_map = {
            'kmeans': Theme.KMEANS_COLOR,
            'gmm': Theme.GMM_COLOR,
            'meanshift': Theme.MEANSHIFT_COLOR,
            'spectral': Theme.SPECTRAL_COLOR
        }
        return color_map.get(model_name, Theme.PANEL)

