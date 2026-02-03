"""
Exemple d'intégration des features bonus dans l'application principale

À adapter et intégrer dans apptkr_imageprocessing.py
"""

# Imports à ajouter
from dialogs import ParametersDialog
from utils.color_palette import ColorPalette
from utils.cluster_3d import Cluster3DVisualization
from utils.pca_preprocessing import PCAPreprocessing

# =============================================================================
# 1. AJOUTER CHECKBOX PCA DANS setup_ui()
# =============================================================================

def add_pca_checkbox_to_ui(self, left_panel):
    """Ajouter checkbox PCA dans le panel gauche"""
    
    # Separator
    sep = Frame(left_panel, bg=Theme.MUTED, height=1)
    sep.pack(fill='x', padx=10, pady=10)
    
    # Checkbox PCA
    pca_label = Label(
        left_panel,
        text="Options de prétraitement",
        font=("Segoe UI", 13, "bold"),
        bg=Theme.PANEL,
        fg=Theme.ACCENT
    )
    pca_label.pack(pady=(10, 10), padx=15)
    
    import tkinter as tk
    self.pca_enabled = tk.BooleanVar(value=False)
    
    pca_check = tk.Checkbutton(
        left_panel,
        text="Appliquer ACP (PCA)",
        variable=self.pca_enabled,
        font=("Segoe UI", 10),
        bg=Theme.PANEL,
        fg=Theme.TEXT,
        selectcolor=Theme.ACCENT,
        command=self.on_pca_toggled
    )
    pca_check.pack(anchor='w', padx=20, pady=5)


# =============================================================================
# 2. REMPLACER SLIDERS PAR FENÊTRE DE PARAMÈTRES
# =============================================================================

def apply_model_with_parameters_dialog(self, model_name: str):
    """Appliquer un modèle via fenêtre de paramètres dédiée"""
    
    if self.is_processing:
        messagebox.showwarning("Processing", "A process is already running...")
        return
    
    # Récupérer les paramètres courants
    current_clusters = self.clustering_models['kmeans'].n_clusters
    current_bandwidth = self.clustering_models['meanshift'].bandwidth_param
    
    # Ouvrir la fenêtre de paramètres
    dialog = ParametersDialog(
        self.window,
        model_name,
        current_clusters=current_clusters,
        current_bandwidth=current_bandwidth
    )
    result = dialog.get_result()
    
    if result is None:
        return  # Annulé
    
    # Mettre à jour les paramètres
    if 'clusters' in result:
        self.clustering_models['kmeans'].set_n_clusters(result['clusters'])
        self.clustering_models['gmm'].set_n_components(result['clusters'])
        self.clustering_models['spectral'].set_parameters(result['clusters'])
    
    if 'bandwidth' in result:
        self.clustering_models['meanshift'].set_parameters(result['bandwidth'])
    
    # Appliquer le modèle
    self.apply_model(model_name)


# =============================================================================
# 3. AJOUTER MENU PALETTE DE COULEURS
# =============================================================================

def add_palette_menu(self):
    """Ajouter menu pour choisir la palette de couleurs"""
    
    import tkinter as tk
    
    palette_menu = tk.Menu(self.window.menubar, tearoff=0)
    self.window.menubar.add_cascade(label="Palettes", menu=palette_menu)
    
    # Palettes prédéfinies
    for palette_name in ['vibrant', 'pastel', 'dark', 'rainbow']:
        palette_menu.add_command(
            label=palette_name.capitalize(),
            command=lambda p=palette_name: self.select_palette(p)
        )
    
    palette_menu.add_separator()
    
    # Palettes personnalisées
    custom_palettes = ColorPalette.list_custom_palettes()
    for custom in custom_palettes:
        palette_menu.add_command(
            label=f"Custom: {custom}",
            command=lambda p=custom: self.select_palette(p)
        )
    
    palette_menu.add_separator()
    palette_menu.add_command(
        label="Gérer les palettes...",
        command=self.manage_palettes
    )

def select_palette(self, palette_name: str):
    """Sélectionner une palette de couleurs"""
    self.current_palette = palette_name
    print(f"Palette sélectionnée: {palette_name}")


# =============================================================================
# 4. AJOUTER VISUALISATION 3D
# =============================================================================

def show_3d_clusters(self):
    """Afficher la visualisation 3D des clusters"""
    
    if self.image_processor.current_image is None:
        messagebox.showwarning(
            "Attention",
            "Veuillez d'abord appliquer un modèle de clustering"
        )
        return
    
    # Récupérer les pixels et labels
    original = np.array(self.image_processor.original_image)
    segmented = np.array(self.image_processor.current_image)
    
    pixels = original.reshape(-1, 3).astype(np.float32)
    
    # Récréer les labels à partir de l'image segmentée
    seg_flat = segmented.reshape(-1, 3)
    labels, _ = np.unique(seg_flat, axis=0, return_inverse=True)
    
    # Afficher en 3D
    model_name = self.active_model_name or "Unknown"
    Cluster3DVisualization.plot_clusters_3d(
        pixels,
        _,
        title=f"Clusters 3D - {model_name}"
    )


def save_3d_visualization(self):
    """Sauvegarder la visualisation 3D"""
    
    if self.image_processor.current_image is None:
        messagebox.showwarning("Attention", "Aucun clustering actif")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=(("PNG images", "*.png"), ("All files", "*.*"))
    )
    
    if not file_path:
        return
    
    # Sauvegarder
    original = np.array(self.image_processor.original_image)
    pixels = original.reshape(-1, 3).astype(np.float32)
    
    # Récréer les labels
    segmented = np.array(self.image_processor.current_image)
    seg_flat = segmented.reshape(-1, 3)
    _, labels = np.unique(seg_flat, axis=0, return_inverse=True)
    
    Cluster3DVisualization.save_clusters_3d(
        pixels,
        labels,
        file_path,
        title=f"Clusters 3D - {self.active_model_name}"
    )
    
    messagebox.showinfo("Succès", f"Visualisation 3D sauvegardée:\n{file_path}")


# =============================================================================
# 5. APPLIQUER PCA AVANT CLUSTERING
# =============================================================================

def apply_model_with_pca(self, model_name: str):
    """Appliquer le modèle avec PCA si activé"""
    
    if self.is_processing:
        messagebox.showwarning("Processing", "A process is already running...")
        return
    
    self.active_model_name = model_name
    self.is_processing = True
    
    model = self.clustering_models[model_name]
    
    def process():
        try:
            original = self.image_processor.original_image
            pixels = np.array(original).reshape(-1, 3).astype(np.float32)
            
            # Appliquer PCA si activé
            if hasattr(self, 'pca_enabled') and self.pca_enabled.get():
                pca = PCAPreprocessing(n_components=2)
                pixels_transformed = pca.fit_transform(pixels)
                pca.print_summary()
                
                # Clustering sur données PCA
                model.fit(pixels_transformed)
                labels = model.predict(pixels_transformed)
            else:
                # Clustering normal
                model.fit(pixels)
                labels = model.predict(pixels)
            
            # Créer l'image segmentée
            segmented_image = model.segment_image(original)
            self.image_processor.current_image = segmented_image
            
        except Exception as e:
            print(f"Erreur: {e}")
        finally:
            self.is_processing = False
    
    thread = Thread(target=process, daemon=True)
    thread.start()


def on_pca_toggled(self):
    """Callback quand PCA est activé/désactivé"""
    state = "activé" if self.pca_enabled.get() else "désactivé"
    print(f"PCA {state}")


# =============================================================================
# 6. MENU SAUVEGARDE AVANCÉE
# =============================================================================

def add_advanced_save_menu(self, file_menu):
    """Ajouter menu sauvegarde avancée"""
    
    import tkinter as tk
    
    file_menu.add_separator()
    
    save_menu = tk.Menu(file_menu, tearoff=0)
    file_menu.add_cascade(label="Sauvegarde avancée", menu=save_menu)
    
    save_menu.add_command(
        label="Image segmentée...",
        command=self.save_result_image
    )
    save_menu.add_command(
        label="Visualisation 3D...",
        command=self.save_3d_visualization
    )
    save_menu.add_command(
        label="Paramètres utilisés...",
        command=self.save_parameters
    )
    save_menu.add_separator()
    save_menu.add_command(
        label="Exporter tout...",
        command=self.export_all_results
    )


def save_parameters(self):
    """Sauvegarder les paramètres en JSON"""
    
    import json
    from datetime import datetime
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
    )
    
    if not file_path:
        return
    
    params = {
        'timestamp': datetime.now().isoformat(),
        'model': self.active_model_name,
        'clusters': self.clustering_models['kmeans'].n_clusters,
        'bandwidth': self.clustering_models['meanshift'].bandwidth_param,
        'pca_enabled': self.pca_enabled.get() if hasattr(self, 'pca_enabled') else False,
        'palette': self.current_palette if hasattr(self, 'current_palette') else 'vibrant'
    }
    
    with open(file_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    messagebox.showinfo("Succès", f"Paramètres sauvegardés:\n{file_path}")


def export_all_results(self):
    """Exporter tous les résultats"""
    
    folder = filedialog.askdirectory(title="Sélectionner le dossier d'export")
    
    if not folder:
        return
    
    # Sauvegarder image segmentée
    segmented_path = os.path.join(folder, 'result.png')
    self.image_processor.save_current_image(segmented_path)
    
    # Sauvegarder visualisation 3D
    viz_path = os.path.join(folder, 'clusters_3d.png')
    self.save_3d_visualization_to_path(viz_path)
    
    # Sauvegarder paramètres
    params_path = os.path.join(folder, 'parameters.json')
    self.save_parameters_to_path(params_path)
    
    messagebox.showinfo(
        "Succès",
        f"Tous les résultats ont été exportés dans:\n{folder}"
    )


# =============================================================================
# RÉSUMÉ DES MODIFICATIONS DANS apptkr_imageprocessing.py
# =============================================================================

"""
1. Imports à ajouter en haut:
   from dialogs import ParametersDialog
   from utils.color_palette import ColorPalette
   from utils.cluster_3d import Cluster3DVisualization
   from utils.pca_preprocessing import PCAPreprocessing
   import tkinter as tk
   import json
   from datetime import datetime

2. Dans __init__():
   self.pca_enabled = tk.BooleanVar(value=False)
   self.current_palette = 'vibrant'

3. Dans setup_ui():
   - Remplacer le code des sliders par add_pca_checkbox_to_ui()
   - Ou garder les sliders et ajouter la checkbox PCA

4. Dans setup_menu():
   add_advanced_save_menu(self, file_menu)
   add_palette_menu(self)

5. Modifier apply_model() pour utiliser ParametersDialog
   ou créer une nouvelle méthode apply_model_with_parameters_dialog()

6. Ajouter les nouvelles méthodes du fichier ci-dessus

7. Tester que tout fonctionne!
"""
