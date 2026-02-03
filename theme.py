"""
Centralized theme configuration for the application
Ensures visual consistency across all components
"""


class Theme:
    """Color palette and styling constants"""
    
    # Core colors
    BG = "#1e1e1e"              # Main background (dark)
    PANEL = "#263238"           # Panel/button background
    ACCENT = "#0d47a1"          # Primary accent (blue)
    TEXT = "#ffffff"            # Main text color
    MUTED = "#9e9e9e"           # Muted/secondary text
    
    # States
    SUCCESS = "#4CAF50"          # Success state
    HOVER = "#37474f"            # Hover state
    DISABLED = "#424242"         # Disabled state
    
    # Canvas
    CANVAS_BG = "#111111"        # Canvas background (darker)
    CANVAS_BORDER = "#0d47a1"    # Canvas border accent
    
    # Model colors - distinct per algorithm
    KMEANS_COLOR = "#FF6B6B"     # Red/coral for K-Means
    GMM_COLOR = "#4ECDC4"        # Teal for GMM
    MEANSHIFT_COLOR = "#FFD93D"  # Yellow for MeanShift
    SPECTRAL_COLOR = "#A569BD"   # Purple for Spectral
    
    # Processing indicators
    PROCESSING = "#FFA500"       # Orange for processing
    DONE = "#4CAF50"             # Green for done
    
    @staticmethod
    def get_button_style():
        """Returns a dictionary of button styling"""
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
        """Returns a dictionary of label styling"""
        return {
            "bg": Theme.PANEL,
            "fg": Theme.TEXT
        }
    
    @staticmethod
    def get_slider_style():
        """Returns a dictionary of slider styling"""
        return {
            "bg": Theme.PANEL,
            "fg": Theme.TEXT,
            "troughcolor": Theme.BG,
            "highlightthickness": 0
        }
    
    @staticmethod
    def get_model_color(model_name: str) -> str:
        """Returns color for specific model"""
        color_map = {
            'kmeans': Theme.KMEANS_COLOR,
            'gmm': Theme.GMM_COLOR,
            'meanshift': Theme.MEANSHIFT_COLOR,
            'spectral': Theme.SPECTRAL_COLOR
        }
        return color_map.get(model_name, Theme.PANEL)

