
import json
import os
import numpy as np
from pathlib import Path

class ColorPalette:

    
    PALETTES = {
        'vibrant': {
            'name': 'Vibrant',
            'description': 'Couleurs vives et contrastées',
            'type': 'vibrant'
        },
        'pastel': {
            'name': 'Pastel',
            'description': 'Couleurs douces et apaisantes',
            'type': 'pastel'
        },
        'dark': {
            'name': 'Dark',
            'description': 'Couleurs sombres et intenses',
            'type': 'dark'
        },
        'rainbow': {
            'name': 'Rainbow',
            'description': 'Arc-en-ciel classique',
            'type': 'rainbow'
        }
    }
    
    @staticmethod
    def generate_palette(palette_type, n_colors):

        if palette_type == 'vibrant':
            return ColorPalette._vibrant_palette(n_colors)
        elif palette_type == 'pastel':
            return ColorPalette._pastel_palette(n_colors)
        elif palette_type == 'dark':
            return ColorPalette._dark_palette(n_colors)
        elif palette_type == 'rainbow':
            return ColorPalette._rainbow_palette(n_colors)
        else:
            return ColorPalette._vibrant_palette(n_colors)
    
    @staticmethod
    def _vibrant_palette(n_colors):

        colors = []
        for i in range(n_colors):
            hue = (i / n_colors) % 1.0
            saturation = 0.85
            value = 0.95
            
            h = hue * 6.0
            c = value * saturation
            x = c * (1 - abs((h % 2) - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            colors.append([r, g, b])
        
        return np.array(colors, dtype=np.uint8)
    
    @staticmethod
    def _pastel_palette(n_colors):

        colors = []
        for i in range(n_colors):
            hue = (i / n_colors) % 1.0
            saturation = 0.4 
            value = 0.95
            
            h = hue * 6.0
            c = value * saturation
            x = c * (1 - abs((h % 2) - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            colors.append([r, g, b])
        
        return np.array(colors, dtype=np.uint8)
    
    @staticmethod
    def _dark_palette(n_colors):

        colors = []
        for i in range(n_colors):
            hue = (i / n_colors) % 1.0
            saturation = 0.9
            value = 0.7
            
            h = hue * 6.0
            c = value * saturation
            x = c * (1 - abs((h % 2) - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            colors.append([r, g, b])
        
        return np.array(colors, dtype=np.uint8)
    
    @staticmethod
    def _rainbow_palette(n_colors):

        colors = []
        rainbow_hues = [0.0, 0.17, 0.33, 0.5, 0.67, 0.83] 
        
        for i in range(n_colors):
            hue = rainbow_hues[i % len(rainbow_hues)]
            saturation = 0.8
            value = 0.9
            
            h = hue * 6.0
            c = value * saturation
            x = c * (1 - abs((h % 2) - 1))
            m = value - c
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            colors.append([r, g, b])
        
        return np.array(colors, dtype=np.uint8)
    
    @staticmethod
    def save_custom_palette(palette_name, colors):

        palettes_dir = Path('palettes')
        palettes_dir.mkdir(exist_ok=True)
        
        palette_data = {
            'name': palette_name,
            'colors': colors.tolist() if isinstance(colors, np.ndarray) else colors
        }
        
        with open(palettes_dir / f"{palette_name}.json", 'w') as f:
            json.dump(palette_data, f, indent=2)
    
    @staticmethod
    def load_custom_palette(palette_name):

        palette_path = Path('palettes') / f"{palette_name}.json"
        
        if not palette_path.exists():
            return None
        
        with open(palette_path, 'r') as f:
            palette_data = json.load(f)
        
        return np.array(palette_data['colors'], dtype=np.uint8)
    
    @staticmethod
    def list_custom_palettes():

        palettes_dir = Path('palettes')
        
        if not palettes_dir.exists():
            return []
        
        return [f.stem for f in palettes_dir.glob('*.json')]
