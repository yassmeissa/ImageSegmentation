"""
Fenêtre de dialogue pour les paramètres des modèles
"""

from tkinter import Toplevel, Frame, Label, Scale, Button, messagebox
from config import AppConfig
from theme import Theme


class ParametersDialog(Toplevel):
    """Fenêtre de sélection des paramètres"""
    
    def __init__(self, parent, model_name, current_clusters=None, current_bandwidth=None):
        super().__init__(parent)
        self.title(f"Paramètres - {model_name}")
        self.geometry("400x300")
        self.resizable(False, False)
        self.configure(bg=Theme.BG)
        
        self.model_name = model_name
        self.result = None
        self.is_meanshift = model_name == "MeanShift"
        
        # Valeurs initiales
        self.clusters_value = current_clusters or AppConfig.DEFAULT_KMEANS_CLUSTERS
        self.bandwidth_value = current_bandwidth or AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH
        
        self.setup_ui()
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        """Configurer l'interface"""
        # Frame principal
        main_frame = Frame(self, bg=Theme.PANEL, relief='groove', borderwidth=2)
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Titre
        title_label = Label(
            main_frame,
            text=f"Configuration: {self.model_name}",
            font=("Segoe UI", 14, "bold"),
            bg=Theme.PANEL,
            fg=Theme.ACCENT
        )
        title_label.pack(pady=(0, 20))
        
        if not self.is_meanshift:
            # Slider Clusters
            clusters_label = Label(
                main_frame,
                text=f"Nombre de clusters",
                font=("Segoe UI", 11),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            clusters_label.pack(anchor='w', padx=10)
            
            self.clusters_slider = Scale(
                main_frame,
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
            self.clusters_slider.pack(padx=10, pady=5, fill='x')
            
            self.clusters_display = Label(
                main_frame,
                text=f"Valeur: {self.clusters_value}",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.clusters_display.pack(pady=(0, 20))
        else:
            # Slider Bandwidth
            bandwidth_label = Label(
                main_frame,
                text=f"Bandwidth (MeanShift)",
                font=("Segoe UI", 11),
                bg=Theme.PANEL,
                fg=Theme.TEXT
            )
            bandwidth_label.pack(anchor='w', padx=10)
            
            self.bandwidth_slider = Scale(
                main_frame,
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
            self.bandwidth_slider.pack(padx=10, pady=5, fill='x')
            
            self.bandwidth_display = Label(
                main_frame,
                text=f"Valeur: {self.bandwidth_value}",
                font=("Segoe UI", 10),
                bg=Theme.PANEL,
                fg=Theme.MUTED
            )
            self.bandwidth_display.pack(pady=(0, 20))
        
        # Boutons
        button_frame = Frame(main_frame, bg=Theme.PANEL)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ok_btn = Button(
            button_frame,
            text="Appliquer",
            font=("Segoe UI", 10, "bold"),
            bg=Theme.ACCENT,
            fg=Theme.TEXT,
            command=self.on_ok
        )
        ok_btn.pack(side='left', padx=5)
        
        cancel_btn = Button(
            button_frame,
            text="Annuler",
            font=("Segoe UI", 10),
            bg=Theme.MUTED,
            fg=Theme.TEXT,
            command=self.on_cancel
        )
        cancel_btn.pack(side='left', padx=5)
    
    def on_clusters_changed(self, value):
        """Callback pour le changement du slider clusters"""
        self.clusters_value = int(float(value))
        self.clusters_display.config(text=f"Valeur: {self.clusters_value}")
    
    def on_bandwidth_changed(self, value):
        """Callback pour le changement du slider bandwidth"""
        self.bandwidth_value = float(value)
        self.bandwidth_display.config(text=f"Valeur: {self.bandwidth_value:.1f}")
    
    def on_ok(self):
        """Valider et fermer"""
        if self.is_meanshift:
            self.result = {'bandwidth': self.bandwidth_value}
        else:
            self.result = {'clusters': self.clusters_value}
        self.destroy()
    
    def on_cancel(self):
        """Annuler et fermer"""
        self.result = None
        self.destroy()
    
    def get_result(self):
        """Obtenir le résultat"""
        self.wait_window()
        return self.result
