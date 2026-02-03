"""
Fenêtre de configuration unifiée : Paramètres + Palette
"""

from tkinter import Toplevel, Frame, Label, Scale, Button, Radiobutton, StringVar
from config import AppConfig
from theme import Theme


class ConfigurationDialog(Toplevel):
    """Fenêtre unifiée pour sélectionner paramètres et palette"""
    
    def __init__(self, parent, model_name):
        super().__init__(parent)
        self.title(f"Configuration - {model_name}")
        self.geometry("600x750")
        self.resizable(False, False)
        self.configure(bg=Theme.BG)
        
        self.model_name = model_name
        self.result = None
        self.is_meanshift = model_name == "MeanShift"
        self.is_gmm = model_name == "GMM"
        self.is_spectral = model_name == "Spectral"
        self.is_kmeans = model_name == "K-Means"
        
        # Valeurs initiales
        self.clusters_value = AppConfig.DEFAULT_KMEANS_CLUSTERS
        self.bandwidth_value = AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
        self.palette_name = 'viridis'
        
        # Paramètres additionnels
        self.n_init_value = 30  # Pour K-Means
        self.max_iter_value = 500  # Pour K-Means et GMM
        self.cov_type = 'diag'  # Pour GMM
        self.affinity_type = 'nearest_neighbors'  # Pour Spectral (fixé à nearest_neighbors)
        
        self.setup_ui()
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        """Configurer l'interface"""
        # Frame principal avec scroll
        main_frame = Frame(self, bg=Theme.BG)
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # ==================== SECTION PARAMÈTRES ====================
        params_label = Label(
            main_frame,
            text="Model Parameters",
            font=("Segoe UI", 12, "bold"),
            bg=Theme.BG,
            fg=Theme.ACCENT
        )
        params_label.pack(anchor='w', pady=(0, 10))
        
        params_frame = Frame(main_frame, bg=Theme.PANEL, relief='groove', borderwidth=1)
        params_frame.pack(fill='x', padx=0, pady=(0, 15))
        
        if self.is_kmeans:
            # K-Means: Clusters, n_init, max_iter
            # Slider Clusters
            clusters_label = Label(
                params_frame,
                text="Number of Clusters",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            clusters_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.clusters_slider = Scale(
                params_frame,
                from_=AppConfig.CLUSTERS_MIN,
                to=AppConfig.CLUSTERS_MAX,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_clusters_changed
            )
            self.clusters_slider.set(self.clusters_value)
            self.clusters_slider.pack(padx=15, pady=5, fill='x')
            
            self.clusters_display = Label(
                params_frame,
                text=f"Value: {self.clusters_value}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.clusters_display.pack(padx=15, pady=(0, 10))
            
            # n_init slider
            ninit_label = Label(
                params_frame,
                text="Number of Initializations",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            ninit_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.ninit_slider = Scale(
                params_frame,
                from_=5,
                to=50,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_ninit_changed
            )
            self.ninit_slider.set(self.n_init_value)
            self.ninit_slider.pack(padx=15, pady=5, fill='x')
            
            self.ninit_display = Label(
                params_frame,
                text=f"Value: {self.n_init_value}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.ninit_display.pack(padx=15, pady=(0, 10))
            
            # max_iter slider
            maxiter_label = Label(
                params_frame,
                text="Max Iterations",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            maxiter_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.maxiter_slider = Scale(
                params_frame,
                from_=100,
                to=1000,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_maxiter_changed
            )
            self.maxiter_slider.set(self.max_iter_value)
            self.maxiter_slider.pack(padx=15, pady=5, fill='x')
            
            self.maxiter_display = Label(
                params_frame,
                text=f"Value: {self.max_iter_value}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.maxiter_display.pack(padx=15, pady=(0, 10))
            
        elif self.is_gmm:
            # GMM: Components, max_iter, covariance_type
            components_label = Label(
                params_frame,
                text="Number of Components",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            components_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.components_slider = Scale(
                params_frame,
                from_=AppConfig.CLUSTERS_MIN,
                to=AppConfig.CLUSTERS_MAX,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_components_changed
            )
            self.components_slider.set(self.clusters_value)
            self.components_slider.pack(padx=15, pady=5, fill='x')
            
            self.components_display = Label(
                params_frame,
                text=f"Value: {self.clusters_value}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.components_display.pack(padx=15, pady=(0, 10))
            
            # max_iter slider
            maxiter_label = Label(
                params_frame,
                text="Max Iterations",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            maxiter_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.maxiter_slider = Scale(
                params_frame,
                from_=50,
                to=500,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_maxiter_changed
            )
            self.maxiter_slider.set(self.max_iter_value)
            self.maxiter_slider.pack(padx=15, pady=5, fill='x')
            
            self.maxiter_display = Label(
                params_frame,
                text=f"Value: {self.max_iter_value}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.maxiter_display.pack(padx=15, pady=(0, 10))
            
            # Covariance type
            cov_label = Label(
                params_frame,
                text="Covariance Type",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            cov_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.cov_var = StringVar(value=self.cov_type)
            for cov in ['full', 'tied', 'diag', 'spherical']:
                rb = Radiobutton(
                    params_frame,
                    text=f"{cov.capitalize()} - " + {
                        'full': 'Most flexible (slower)',
                        'tied': 'Tied (medium)',
                        'diag': 'Diagonal (fast)',
                        'spherical': 'Spherical (fastest)'
                    }.get(cov, ''),
                    variable=self.cov_var,
                    value=cov,
                    font=("Segoe UI", 8),
                    bg=Theme.PANEL,
                    fg=Theme.TEXT,
                    selectcolor=Theme.ACCENT,
                    command=self.on_cov_changed
                )
                rb.pack(anchor='w', padx=15, pady=2)
            
        elif self.is_spectral:
            # Spectral: Clusters only (affinity fixed to nearest_neighbors)
            clusters_label = Label(
                params_frame,
                text="Number of Clusters",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            clusters_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.clusters_slider = Scale(
                params_frame,
                from_=AppConfig.CLUSTERS_MIN,
                to=AppConfig.CLUSTERS_MAX,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_clusters_changed
            )
            self.clusters_slider.set(self.clusters_value)
            self.clusters_slider.pack(padx=15, pady=5, fill='x')
            
            self.clusters_display = Label(
                params_frame,
                text=f"Value: {self.clusters_value}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.clusters_display.pack(padx=15, pady=(0, 10))
            
            # Info label about affinity
            affinity_info = Label(
                params_frame,
                text="Affinity: Nearest Neighbors (Fixed)",
                font=("Segoe UI", 8),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            affinity_info.pack(anchor='w', padx=15, pady=(5, 10))
                
        else:
            # MeanShift: Bandwidth only
            bandwidth_label = Label(
                params_frame,
                text="Bandwidth (MeanShift)",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            bandwidth_label.pack(anchor='w', padx=15, pady=(10, 5))
            
            self.bandwidth_slider = Scale(
                params_frame,
                from_=AppConfig.BANDWIDTH_MIN,
                to=AppConfig.BANDWIDTH_MAX,
                orient='horizontal',
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                troughcolor=Theme.MUTED,
                length=300,
                command=self.on_bandwidth_changed
            )
            self.bandwidth_slider.set(self.bandwidth_value)
            self.bandwidth_slider.pack(padx=15, pady=5, fill='x')
            
            self.bandwidth_display = Label(
                params_frame,
                text=f"Value: {self.bandwidth_value:.1f}",
                font=("Segoe UI", 9),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.bandwidth_display.pack(padx=15, pady=(0, 10))
        
        # ==================== SECTION PALETTE ====================
        palette_label = Label(
            main_frame,
            text="Color Palette",
            font=("Segoe UI", 12, "bold"),
            bg=Theme.BG,
            fg=Theme.ACCENT
        )
        palette_label.pack(anchor='w', pady=(0, 10))
        
        palette_frame = Frame(main_frame, bg=Theme.PANEL, relief='groove', borderwidth=1)
        palette_frame.pack(fill='x', padx=0, pady=(0, 15))
        
        self.palette_var = StringVar(value='viridis')
        
        # Matplotlib colormaps
        palettes = [
            ('viridis', 'Viridis - Perceptually uniform'),
            ('plasma', 'Plasma - High contrast'),
            ('inferno', 'Inferno - Dark to bright'),
            ('cool', 'Cool - Cyan to Magenta'),
            ('hot', 'Hot - Black to Red to White'),
            ('spring', 'Spring - Magenta to Yellow'),
            ('summer', 'Summer - Green to Yellow'),
            ('autumn', 'Autumn - Red to Yellow'),
            ('winter', 'Winter - Blue to Green'),
            ('twilight', 'Twilight - Cyclic palette'),
        ]
        
        for palette_name, palette_desc in palettes:
            rb = Radiobutton(
                palette_frame,
                text=palette_desc,
                variable=self.palette_var,
                value=palette_name,
                font=("Segoe UI", 8),
                bg=Theme.PANEL,
                fg=Theme.TEXT,
                selectcolor=Theme.ACCENT,
                command=self.on_palette_changed
            )
            rb.pack(anchor='w', padx=15, pady=3)
        
        # ==================== BOUTONS ====================
        button_frame = Frame(main_frame, bg=Theme.BG)
        button_frame.pack(fill='x', padx=0, pady=(10, 0))
        
        ok_btn = Button(
            button_frame,
            text="Apply",
            font=("Segoe UI", 10, "bold"),
            bg=Theme.ACCENT,
            fg="#ffffff",
            command=self.on_ok,
            width=12
        )
        ok_btn.pack(side='left', padx=5)
        
        cancel_btn = Button(
            button_frame,
            text="Cancel",
            font=("Segoe UI", 10),
            bg=Theme.MUTED,
            fg="#000000",
            command=self.on_cancel,
            width=12
        )
        cancel_btn.pack(side='left', padx=5)
    
    def on_clusters_changed(self, value):
        """Callback pour le changement du slider clusters"""
        self.clusters_value = int(float(value))
        self.clusters_display.config(text=f"Value: {self.clusters_value}")
    
    def on_components_changed(self, value):
        """Callback pour le changement du slider components (GMM)"""
        self.clusters_value = int(float(value))
        self.components_display.config(text=f"Value: {self.clusters_value}")
    
    def on_bandwidth_changed(self, value):
        """Callback pour le changement du slider bandwidth"""
        self.bandwidth_value = float(value)
        self.bandwidth_display.config(text=f"Value: {self.bandwidth_value:.1f}")
    
    def on_ninit_changed(self, value):
        """Callback pour le changement du slider n_init"""
        self.n_init_value = int(float(value))
        self.ninit_display.config(text=f"Value: {self.n_init_value}")
    
    def on_maxiter_changed(self, value):
        """Callback pour le changement du slider max_iter"""
        self.max_iter_value = int(float(value))
        self.maxiter_display.config(text=f"Value: {self.max_iter_value}")
    
    def on_cov_changed(self):
        """Callback pour le changement de covariance type"""
        self.cov_type = self.cov_var.get()
    
    def on_palette_changed(self):
        """Callback pour le changement de palette"""
        self.palette_name = self.palette_var.get()
    
    def on_ok(self):
        """Valider et fermer"""
        if self.is_kmeans:
            self.result = {
                'clusters': self.clusters_value,
                'n_init': self.n_init_value,
                'max_iter': self.max_iter_value,
                'palette': self.palette_name
            }
        elif self.is_gmm:
            self.result = {
                'components': self.clusters_value,
                'max_iter': self.max_iter_value,
                'covariance_type': self.cov_type,
                'palette': self.palette_name
            }
        elif self.is_spectral:
            self.result = {
                'clusters': self.clusters_value,
                'affinity': self.affinity_type,
                'palette': self.palette_name
            }
        else:  # MeanShift
            self.result = {
                'bandwidth': self.bandwidth_value,
                'palette': self.palette_name
            }
        self.destroy()
    
    def on_cancel(self):
        """Annuler et fermer"""
        self.result = None
        self.destroy()
    
    def get_result(self):
        """Obtenir le résultat"""
        self.wait_window()
        return self.result
