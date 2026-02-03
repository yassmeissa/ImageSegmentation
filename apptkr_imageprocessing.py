# Fix matplotlib compatibility with macOS BEFORE any other imports
import os
import sys

# Set matplotlib backend before importing anything else
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.cm as cm
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from tkinter import Tk, Menu, Frame, Label, filedialog, messagebox, Canvas, Checkbutton, IntVar, Toplevel, Listbox, Scrollbar, Radiobutton, StringVar, Text, SINGLE, END, Y, LEFT, RIGHT, BOTH, DISABLED
from PIL import Image, ImageTk
import threading
import numpy as np
from image_processor import ImageProcessor
from ui_components import ImageDisplayCanvas, ModelButton, ParameterSlider, ComparisonCanvas
from models.kmeans_model import KMeansClusteringModel
from models.gmm_model import GMMClusteringModel
from models.meanshift_model import MeanShiftClusteringModel
from models.spectral_model import SpectralClusteringModel
from config import AppConfig
from theme import Theme
from utils.logger import AppLogger
from dialogs.parameters_dialog import ParametersDialog
from dialogs.config_dialog import ConfigurationDialog
from utils.color_palette import ColorPalette
from utils.cluster_3d import Cluster3DVisualization
from utils.pca_preprocessing import PCAPreprocessing

AppLogger.setup()
from utils.logger import get_logger
logger = get_logger()


class ImageSegmentationApplication:
    def __init__(self):
        logger.info("Starting ImageSegmentationApplication")
        self.window = Tk()
        self.window.title(AppConfig.WINDOW_TITLE)
        self.window.geometry("1200x800")
        self.window.resizable(True, True)
        
        self.window.configure(bg=Theme.BG)
        self.setup_window_icon()
        
        self.image_processor = ImageProcessor()
        self.clustering_models = {
            'kmeans': KMeansClusteringModel(n_clusters=AppConfig.DEFAULT_KMEANS_CLUSTERS),
            'gmm': GMMClusteringModel(n_components=AppConfig.DEFAULT_GMM_COMPONENTS),
            'meanshift': MeanShiftClusteringModel(bandwidth=AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH),
            'spectral': SpectralClusteringModel(n_clusters=AppConfig.DEFAULT_KMEANS_CLUSTERS)
        }
        self.active_model_name = None
        self.is_processing = False
        self.processing_thread = None
        
        # Bonus features
        self.color_palette = ColorPalette()
        self.use_pca = IntVar(value=0)
        self.current_palette_name = 'viridis'
        self.pca_components = 3
        
        self.setup_menu()
        self.setup_ui()
        
        # Display image after window is rendered
        self.window.after(100, self.refresh_display)
        self.check_processing()
        self.window.mainloop()

    def setup_window_icon(self):
        try:
            if os.path.exists(AppConfig.ICON_PATH):
                icon = Image.open(AppConfig.ICON_PATH)
                icon_tk = ImageTk.PhotoImage(icon)
                self.window.wm_iconphoto(False, icon_tk)
        except Exception:
            pass

    def setup_menu(self):
        menu_bar = Menu(self.window)
        self.window.config(menu=menu_bar)
        
        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image_file)
        file_menu.add_command(label="Save Result", command=self.save_result_advanced)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.destroy)
        
        # Visualization menu
        viz_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Visualization", menu=viz_menu)
        viz_menu.add_command(label="View 3D Clusters", command=self.show_3d_visualization)
        viz_menu.add_command(label="Export 3D (PNG)", command=self.export_3d_clusters)
        
        # Tools menu
        tools_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Color Palettes", command=self.manage_palettes)
        tools_menu.add_command(label="PCA Analysis", command=self.show_pca_analysis)

    def setup_ui(self):
        # Main container
        main_frame = Frame(self.window, bg=Theme.BG)
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Header
        header_frame = Frame(main_frame, bg=Theme.ACCENT, height=60)
        header_frame.pack(fill='x', padx=0, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        header_label = Label(
            header_frame, 
            text="Image Segmentation Studio",
            font=("Segoe UI", 20, "bold"),
            bg=Theme.ACCENT,
            fg=Theme.TEXT
        )
        header_label.pack(pady=15)
        
        # Content area (left: controls, right: canvas)
        content_frame = Frame(main_frame, bg=Theme.BG)
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Controls
        left_panel = Frame(content_frame, bg=Theme.PANEL, relief='groove', borderwidth=2)
        left_panel.pack(side='left', fill='y', padx=(0, 15))
        left_panel.pack_propagate(False)
        left_panel.config(width=280)
        
        # Models section
        models_label = Label(
            left_panel,
            text="Clustering Models",
            font=("Segoe UI", 13, "bold"),
            bg=Theme.PANEL,
            fg=Theme.ACCENT
        )
        models_label.pack(pady=(15, 10), padx=15)
        
        buttons_frame = Frame(left_panel, bg=Theme.PANEL)
        buttons_frame.pack(fill='x', padx=10, pady=10)
        
        self.kmeans_button = ModelButton(buttons_frame, "K-Means", 'kmeans', lambda: self.apply_model('kmeans'))
        self.kmeans_button.grid_layout(row=0, column=0, padx=5, pady=5)
        
        self.gmm_button = ModelButton(buttons_frame, "GMM", 'gmm', lambda: self.apply_model('gmm'))
        self.gmm_button.grid_layout(row=1, column=0, padx=5, pady=5)
        
        self.meanshift_button = ModelButton(buttons_frame, "MeanShift", 'meanshift', lambda: self.apply_model('meanshift'))
        self.meanshift_button.grid_layout(row=2, column=0, padx=5, pady=5)
        
        self.spectral_button = ModelButton(buttons_frame, "Spectral", 'spectral', lambda: self.apply_model('spectral'))
        self.spectral_button.grid_layout(row=3, column=0, padx=5, pady=5)
        
        # Disable model buttons initially (until image is loaded)
        self.kmeans_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        self.gmm_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        self.meanshift_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        self.spectral_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        
        # Separator
        sep1 = Frame(left_panel, bg=Theme.MUTED, height=1)
        sep1.pack(fill='x', padx=10, pady=10)
        
        # Note: Parameters will be configured in a dialog after model selection
        
        # Separator
        sep2 = Frame(left_panel, bg=Theme.MUTED, height=1)
        sep2.pack(fill='x', padx=10, pady=10)
        
        # PCA Preprocessing section
        pca_label = Label(
            left_panel,
            text="Preprocessing",
            font=("Segoe UI", 13, "bold"),
            bg=Theme.PANEL,
            fg=Theme.ACCENT
        )
        pca_label.pack(pady=(10, 5), padx=15)
        
        pca_checkbox = Checkbutton(
            left_panel,
            text="Use PCA Preprocessing",
            variable=self.use_pca,
            bg=Theme.PANEL,
            fg=Theme.TEXT,
            selectcolor=Theme.ACCENT,
            font=("Segoe UI", 10),
            activebackground=Theme.PANEL,
            activeforeground=Theme.TEXT
        )
        pca_checkbox.pack(anchor='w', padx=15, pady=5)
        
        self.pca_info_label = Label(
            left_panel,
            text="",
            font=("Segoe UI", 8),
            bg=Theme.PANEL,
            fg=Theme.MUTED
        )
        self.pca_info_label.pack(anchor='w', padx=20, pady=(0, 10))
        
        # Separator
        sep3 = Frame(left_panel, bg=Theme.MUTED, height=1)
        sep3.pack(fill='x', padx=10, pady=10)
        
        # File operations
        file_label = Label(
            left_panel,
            text="Operations",
            font=("Segoe UI", 13, "bold"),
            bg=Theme.PANEL,
            fg=Theme.ACCENT
        )
        file_label.pack(pady=(10, 10), padx=15)
        
        operations_frame = Frame(left_panel, bg=Theme.PANEL)
        operations_frame.pack(fill='x', padx=10, pady=10)
        
        open_btn = ModelButton(operations_frame, "Open Image", on_click=self.open_image_file)
        open_btn.grid_layout(row=0, column=0, padx=5, pady=5)
        
        save_btn = ModelButton(operations_frame, "Save Result", on_click=self.save_result_advanced)
        save_btn.grid_layout(row=1, column=0, padx=5, pady=5)
        
        export_3d_btn = ModelButton(operations_frame, "Export 3D", on_click=self.export_3d_clusters)
        export_3d_btn.grid_layout(row=2, column=0, padx=5, pady=5)
        
        # Right panel - Canvas with comparison
        right_panel = Frame(content_frame, bg=Theme.BG)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Model info + canvas label area
        info_frame = Frame(right_panel, bg=Theme.PANEL, height=60)
        info_frame.pack(fill='x', padx=10, pady=(0, 10))
        info_frame.pack_propagate(False)
        
        self.model_info_label = Label(
            info_frame,
            text="Select an image to begin",
            font=("Segoe UI", 11, "bold"),
            bg=Theme.PANEL,
            fg=Theme.TEXT
        )
        self.model_info_label.pack(pady=5, padx=15, anchor='w')
        
        self.params_info_label = Label(
            info_frame,
            text="",
            font=("Segoe UI", 9),
            bg=Theme.PANEL,
            fg=Theme.MUTED
        )
        self.params_info_label.pack(pady=(0, 5), padx=15, anchor='w')
        
        # Use comparison canvas for before/after view
        self.comparison_canvas = ComparisonCanvas(
            right_panel, 
            AppConfig.CANVAS_WIDTH, 
            AppConfig.CANVAS_HEIGHT
        )
        self.comparison_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar - Enhanced with progress indicator
        status_frame = Frame(self.window, bg=Theme.PANEL, height=35)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_indicator = Label(
            status_frame,
            text="●",
            font=("Segoe UI", 16),
            bg=Theme.PANEL,
            fg=Theme.MUTED
        )
        self.status_indicator.pack(side='left', padx=15, pady=5)
        
        self.status_label = Label(
            status_frame,
            text="Ready",
            font=("Segoe UI", 10),
            bg=Theme.PANEL,
            fg=Theme.TEXT
        )
        self.status_label.pack(side='left', padx=5, pady=5)

    def open_image_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), AppConfig.IMAGE_FOLDER),
            title="Select Image",
            filetypes=AppConfig.SUPPORTED_FORMATS
        )
        
        if file_path:
            try:
                self.image_processor.load_from_file(file_path)
                # Store the original image path for auto-naming
                self.image_processor.original_image_path = file_path

                # ✅ Reset segmentation state
                self.image_processor.current_image = None
                self.comparison_canvas.has_segmentation = False

                # ✅ Enable model buttons now that image is loaded
                self.kmeans_button.button.config(state='normal', fg='#000000')
                self.gmm_button.button.config(state='normal', fg='#000000')
                self.meanshift_button.button.config(state='normal', fg='#000000')
                self.spectral_button.button.config(state='normal', fg='#000000')
                
                # Update info label
                self.model_info_label.config(
                    text="Image loaded! Select a clustering model",
                    fg=Theme.ACCENT
                )
                
                self.refresh_display()
            except Exception as e:
                print(f"Error loading image: {e}")

    def apply_model(self, model_name: str):
        if self.is_processing:
            logger.warning(f"Already processing, ignoring {model_name} request")
            messagebox.showwarning("Processing", "A process is already running...")
            return
        
        if self.image_processor.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        # Ouvrir la fenêtre de configuration
        model_names = {'kmeans': 'K-Means', 'gmm': 'GMM', 'meanshift': 'MeanShift', 'spectral': 'Spectral'}
        dialog = ConfigurationDialog(self.window, model_names.get(model_name, model_name))
        result = dialog.get_result()
        
        if not result:
            return  # User cancelled
        
        # Disable model buttons during processing
        self._disable_model_buttons()
        
        # Apply configuration
        logger.info(f"[UI] Applying model: {model_name}")
        self.active_model_name = model_name
        self.current_palette_name = result.get('palette', 'viridis')
        
        # Update parameters based on model type
        if model_name == 'kmeans':
            n_clusters = result.get('clusters', AppConfig.DEFAULT_KMEANS_CLUSTERS)
            n_init = result.get('n_init', 30)
            max_iter = result.get('max_iter', 500)
            self.clustering_models['kmeans'].set_n_clusters(n_clusters)
            self.clustering_models['kmeans'].kmeans.n_init = n_init
            self.clustering_models['kmeans'].kmeans.max_iter = max_iter
            logger.info(f"K-Means: clusters={n_clusters}, n_init={n_init}, max_iter={max_iter}")
        elif model_name == 'gmm':
            n_components = result.get('components', AppConfig.DEFAULT_GMM_COMPONENTS)
            max_iter = result.get('max_iter', 100)
            cov_type = result.get('covariance_type', 'diag')
            self.clustering_models['gmm'].set_n_components(n_components)
            self.clustering_models['gmm'].gmm.max_iter = max_iter
            self.clustering_models['gmm'].gmm.covariance_type = cov_type
            logger.info(f"GMM: components={n_components}, max_iter={max_iter}, cov_type={cov_type}")
        elif model_name == 'spectral':
            n_clusters = result.get('clusters', AppConfig.DEFAULT_KMEANS_CLUSTERS)
            self.clustering_models['spectral'].set_parameters(n_clusters)
            logger.info(f"Spectral: clusters={n_clusters}")
        elif model_name == 'meanshift':
            bandwidth = result.get('bandwidth', AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH)
            self.clustering_models['meanshift'].set_parameters(bandwidth)
            logger.info(f"MeanShift: bandwidth={bandwidth}")
        
        self.update_button_states()
        
        # Update model info label
        self.model_info_label.config(
            text=f"Processing: {model_names.get(model_name, model_name)}",
            fg=Theme.get_model_color(model_name)
        )
        
        # Show parameters
        self._update_params_display(model_name, result)
        
        self.is_processing = True
        self.status_indicator.config(text="●", fg=Theme.PROCESSING)
        self.status_label.config(text=f"Processing {model_names.get(model_name, model_name).upper()}...")
        
        model = self.clustering_models[model_name]
        logger.debug(f"[UI] Starting background thread for {model_name}")
        self.processing_thread = threading.Thread(
            target=self._process_model_background,
            args=(model,),
            daemon=True
        )
        self.processing_thread.start()
        logger.info(f"[UI] Background thread started for {model_name}")
    
    def _update_params_display(self, model_name: str, config_result=None):
        """Update the parameters display based on selected model"""
        params_text = ""
        if model_name == 'kmeans':
            clusters = config_result.get('clusters') if config_result else self.clustering_models['kmeans'].n_clusters
            n_init = config_result.get('n_init', 30) if config_result else 30
            max_iter = config_result.get('max_iter', 500) if config_result else 500
            params_text = f"Clusters: {clusters} | n_init: {n_init} | max_iter: {max_iter}"
        elif model_name == 'gmm':
            components = config_result.get('components') if config_result else self.clustering_models['gmm'].n_components
            max_iter = config_result.get('max_iter', 100) if config_result else 100
            cov_type = config_result.get('covariance_type', 'diag') if config_result else 'diag'
            params_text = f"Components: {components} | max_iter: {max_iter} | cov: {cov_type}"
        elif model_name == 'meanshift':
            bandwidth = config_result.get('bandwidth') if config_result else self.clustering_models['meanshift'].bandwidth_param
            params_text = f"Bandwidth: {bandwidth if bandwidth else 25.0}"
        elif model_name == 'spectral':
            clusters = config_result.get('clusters') if config_result else self.clustering_models['spectral'].n_clusters
            params_text = f"Clusters: {clusters}"
        
        if config_result and 'palette' in config_result:
            params_text += f" | Palette: {config_result['palette']}"
        
        self.params_info_label.config(text=params_text)

    def _process_model_background(self, model):
        logger.info(f"[Thread] Processing started for {model.get_name()}")
        try:
            logger.debug("[Thread] Getting original image...")
            original = self.image_processor.original_image
            logger.debug(f"[Thread] Image obtained: {original.size}")
            
            # Apply PCA preprocessing if enabled
            pca_info = ""
            pca_data = None
            if self.use_pca.get():
                logger.info("[Thread] PCA preprocessing enabled, applying...")
                try:
                    # Get pixel data
                    pixels = original.getdata()
                    pixel_list = list(pixels)
                    rgb_data = []
                    for pixel in pixel_list:
                        if isinstance(pixel, tuple):
                            rgb_data.append(pixel[:3])
                    rgb_array = np.array(rgb_data, dtype=np.float32)
                    
                    # Apply PCA
                    pca = PCAPreprocessing(n_components=self.pca_components)
                    pca_data = pca.fit_transform(rgb_array)
                    
                    # Get variance explained
                    variance_ratio = pca.get_explained_variance_ratio()
                    total_variance = pca.get_total_variance_explained()
                    
                    pca_info = f"PCA: {total_variance:.1%} variance explained"
                    logger.info(f"[Thread] PCA applied: {pca_info}")
                    
                    # Store PCA transformer for potential future use
                    self.image_processor.pca_transformer = pca
                    self.image_processor.pca_data = pca_data
                    self.image_processor.use_pca = True
                except Exception as e:
                    logger.warning(f"[Thread] PCA preprocessing failed: {e}, using original data")
                    self.image_processor.use_pca = False
                    pca_data = None
            else:
                self.image_processor.use_pca = False
            
            # Generate palette based on selected palette type
            palette = self._generate_palette(self.current_palette_name, 10)
            
            logger.info(f"[Thread] Calling segment_image() with palette: {self.current_palette_name}...")
            # Pass PCA data to the model if available
            segmented_image = model.segment_image(original, shared_palette=palette, pca_data=pca_data)
            logger.info(f"[Thread] segment_image() completed")
            
            logger.debug("[Thread] Storing segmented image...")
            self.image_processor.current_image = segmented_image
            
            # Update PCA info label
            if pca_info:
                self.pca_info_label.config(text=pca_info, fg=Theme.ACCENT)
            else:
                self.pca_info_label.config(text="", fg=Theme.MUTED)
            
            logger.info(f"[Thread] Processing completed successfully")
        except Exception as e:
            logger.error(f"[Thread] Error processing model: {e}", exc_info=True)
        finally:
            logger.info("[Thread] Processing finished")
            self.is_processing = False

    def check_processing(self):
        if not self.is_processing and self.processing_thread and self.processing_thread.is_alive():
            self.is_processing = True
        
        if not self.is_processing and self.processing_thread and not self.processing_thread.is_alive():
            # Update status to done
            self.status_indicator.config(text="●", fg=Theme.DONE)
            self.status_label.config(text=f"Processing complete", fg=Theme.TEXT)
            
            # Update model info label
            if self.active_model_name:
                model_names = {'kmeans': 'K-Means', 'gmm': 'GMM', 'meanshift': 'MeanShift', 'spectral': 'Spectral'}
                self.model_info_label.config(
                    text=f"Result: {model_names.get(self.active_model_name, self.active_model_name)}",
                    fg=Theme.get_model_color(self.active_model_name)
                )
            
            # Re-enable model buttons
            self._enable_model_buttons()
            self.update_button_states()
            
            self.refresh_display()
            self.processing_thread = None
        
        self.window.after(100, self.check_processing)

    def update_button_states(self):
        self.kmeans_button.set_active(self.active_model_name == 'kmeans')
        self.gmm_button.set_active(self.active_model_name == 'gmm')
        self.meanshift_button.set_active(self.active_model_name == 'meanshift')
        self.spectral_button.set_active(self.active_model_name == 'spectral')
    
    def _disable_model_buttons(self):
        """Disable model buttons during processing"""
        self.kmeans_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        self.gmm_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        self.meanshift_button.button.config(state=DISABLED, fg=Theme.DISABLED)
        self.spectral_button.button.config(state=DISABLED, fg=Theme.DISABLED)
    
    def _enable_model_buttons(self):
        """Enable model buttons after processing"""
        self.kmeans_button.button.config(state='normal', fg='#000000')
        self.gmm_button.button.config(state='normal', fg='#000000')
        self.meanshift_button.button.config(state='normal', fg='#000000')
        self.spectral_button.button.config(state='normal', fg='#000000')
    
    def _generate_palette(self, palette_name: str, n_colors: int):
        """Generate a color palette using Matplotlib colormaps"""
        try:
            # Get colormap from matplotlib
            colormap = cm.get_cmap(palette_name)
            
            # Generate n_colors evenly spaced from the colormap
            colors = []
            for i in range(n_colors):
                # Get color at position i/n_colors (0 to 1)
                rgba = colormap(i / max(1, n_colors - 1))
                # Convert RGBA to RGB and scale to 0-255
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                colors.append([r, g, b])
            
            return np.array(colors, dtype=np.uint8)
        except Exception as e:
            logger.warning(f"Could not load colormap '{palette_name}', using viridis: {e}")
            # Fallback to viridis
            colormap = cm.get_cmap('viridis')
            colors = []
            for i in range(n_colors):
                rgba = colormap(i / max(1, n_colors - 1))
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                colors.append([r, g, b])
            return np.array(colors, dtype=np.uint8)

    def refresh_display(self):
        current_image = self.image_processor.get_current_image()
        original_image = self.image_processor.original_image
        
        # Display images based on segmentation state
        if current_image is None:
            # Before segmentation: show only original image on left
            self.comparison_canvas.display_images(original_image, None)
        else:
            # After segmentation: show both original and segmented
            self.comparison_canvas.display_images(original_image, current_image)
        
        if not self.is_processing:
            self.status_label.config(text="Ready")

    def compare_all_models(self):
        """Lance le script de comparaison de tous les modèles"""
        import subprocess
        try:
            script_path = os.path.join(os.getcwd(), 'test_clustering.py')
            subprocess.Popen(['.venv/bin/python', script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch comparison script: {e}")
    
    
    # ==================== BONUS FEATURES ====================
    
    def show_3d_visualization(self):
        """Afficher la visualisation 3D des clusters"""
        if self.image_processor.current_image is None:
            messagebox.showwarning("Warning", "No segmented image available. Process an image first.")
            return
        
        try:
            # Get pixel data from current segmented image
            current = self.image_processor.current_image
            if current is None:
                messagebox.showwarning("Warning", "No current image available")
                return
            
            pixels = current.getdata()
            pixel_list = list(pixels)
            
            # Extract RGB values
            rgb_data = []
            for pixel in pixel_list:
                if isinstance(pixel, tuple):
                    rgb_data.append(pixel[:3])
            
            import numpy as np
            rgb_array = np.array(rgb_data)
            
            # Get cluster labels from the segmented image (use color quantization)
            from sklearn.cluster import KMeans as SKKMeans
            kmeans = SKKMeans(n_clusters=10, random_state=42, n_init=10)
            labels = kmeans.fit_predict(rgb_array)
            
            # Visualize
            viz = Cluster3DVisualization()
            viz.plot_clusters_3d_with_centers(rgb_array, labels, kmeans.cluster_centers_)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create 3D visualization: {e}")
            logger.error(f"3D visualization error: {e}", exc_info=True)
    
    def export_3d_clusters(self):
        """Exporter la visualisation 3D en PNG haute résolution pour le modèle sélectionné"""
        if self.image_processor.current_image is None:
            messagebox.showwarning("Warning", "No segmented image available.")
            return
        
        if not self.active_model_name:
            messagebox.showwarning("Warning", "No model selected. Process an image first.")
            return
        
        try:
            # Get pixel data from original image
            original = self.image_processor.original_image
            pixels = original.getdata()
            pixel_list = list(pixels)
            
            rgb_data = []
            for pixel in pixel_list:
                if isinstance(pixel, tuple):
                    rgb_data.append(pixel[:3])
            
            import numpy as np
            rgb_array = np.array(rgb_data)
            
            # Get labels from the active model
            model = self.clustering_models[self.active_model_name]
            
            if self.active_model_name == 'kmeans':
                labels = model.kmeans.labels_
                centers = model.kmeans.cluster_centers_
                n_clusters = model.n_clusters
            elif self.active_model_name == 'gmm':
                labels = model.gmm.predict(rgb_array)
                centers = model.gmm.means_
                n_clusters = model.n_components
            elif self.active_model_name == 'spectral':
                labels = model.spectral.labels_
                n_clusters = model.n_clusters
                # For spectral clustering, compute centroids manually
                centers = []
                for i in range(n_clusters):
                    cluster_points = rgb_array[labels == i]
                    if len(cluster_points) > 0:
                        centers.append(cluster_points.mean(axis=0))
                centers = np.array(centers)
            elif self.active_model_name == 'meanshift':
                labels = model.meanshift.labels_
                centers = model.meanshift.cluster_centers_
                n_clusters = len(np.unique(labels))
            else:
                messagebox.showerror("Error", f"Unknown model: {self.active_model_name}")
                return
            
            # Create 3D plot with matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            fig = plt.figure(figsize=(12, 9), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot each cluster with different colors
            colors = plt.cm.viridis(np.linspace(0, 1, max(n_clusters, 2)))
            
            for i in range(n_clusters):
                cluster_points = rgb_array[labels == i]
                if len(cluster_points) > 0:
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                             c=[colors[i]], label=f'Cluster {i}', s=10, alpha=0.6)
            
            # Plot cluster centers
            if centers is not None and len(centers) > 0:
                ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                          c='red', marker='*', s=500, edgecolors='black', linewidths=2,
                          label='Centroids', zorder=10)
            
            ax.set_xlabel('Red', fontsize=12, fontweight='bold')
            ax.set_ylabel('Green', fontsize=12, fontweight='bold')
            ax.set_zlabel('Blue', fontsize=12, fontweight='bold')
            ax.set_title('3D RGB Cluster Visualization\n' + 
                       f'Model: {self.active_model_name.upper()} | Clusters: {n_clusters}',
                       fontsize=14, fontweight='bold')
            
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Set viewing angle
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            
            # Create a new window to display the plot
            plot_window = Toplevel(self.window)
            plot_window.title(f"3D Cluster Visualization - {self.active_model_name.upper()}")
            plot_window.geometry("1000x800")
            
            # Add buttons at the top
            button_frame = Frame(plot_window, bg=Theme.PANEL, height=50)
            button_frame.pack(fill='x', padx=10, pady=10)
            button_frame.pack_propagate(False)
            
            # Embed matplotlib figure in Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=(0, 10))
            
            def save_plot():
                file_path = filedialog.asksaveasfilename(
                    parent=plot_window,
                    defaultextension=".png",
                    filetypes=(("PNG images", "*.png"), ("All files", "*.*")),
                    initialfile=f"clusters_3d_{self.active_model_name}.png"
                )
                if file_path:
                    fig.savefig(file_path, dpi=150, bbox_inches='tight')
                    messagebox.showinfo("Success", f"3D cluster plot saved to:\n{file_path}")
            
            from tkinter import Button
            save_btn = Button(
                button_frame,
                text="💾 Save Plot",
                command=save_plot,
                font=("Segoe UI", 11, "bold"),
                bg="#4CAF50",
                fg="#0056B3",
                activebackground="#45a049",
                activeforeground="#FFD700",
                relief="raised",
                padx=20,
                pady=10,
                cursor="hand2",
                borderwidth=2
            )
            save_btn.pack(side='left', padx=5)
            
            close_btn = Button(
                button_frame,
                text="❌ Close",
                command=plot_window.destroy,
                font=("Segoe UI", 11, "bold"),
                bg="#FF6B6B",
                fg="#000000",
                activebackground="#E63946",
                activeforeground="#FFD700",
                relief="raised",
                padx=20,
                pady=10,
                cursor="hand2",
                borderwidth=2
            )
            close_btn.pack(side='left', padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not export 3D visualization:\n{str(e)}")
            logger.error(f"3D export error: {e}", exc_info=True)
    
    def manage_palettes(self):
        """Afficher les palettes de couleurs disponibles"""
        palette_window = Toplevel(self.window)
        palette_window.title("Available Color Palettes")
        palette_window.geometry("500x400")
        
        # Matplotlib colormaps
        palettes_list = [
            'viridis', 'plasma', 'inferno', 'cool', 'hot',
            'spring', 'summer', 'autumn', 'winter', 'twilight'
        ]
        
        Label(palette_window, text="Available Matplotlib Colormaps:", font=("Segoe UI", 11, "bold")).pack(pady=10)
        Label(palette_window, text="Click to select a palette for your next segmentation", font=("Segoe UI", 9)).pack(pady=(0, 10))
        
        # Listbox with scrollbar
        frame = Frame(palette_window)
        frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = Scrollbar(frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        listbox = Listbox(frame, yscrollcommand=scrollbar.set, selectmode=SINGLE, font=("Segoe UI", 10))
        listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for palette in palettes_list:
            listbox.insert(END, palette)
        
        # Select current palette
        try:
            idx = palettes_list.index(self.current_palette_name)
            listbox.selection_set(idx)
            listbox.see(idx)
        except ValueError:
            listbox.selection_set(0)
        
        def select_palette():
            selection = listbox.curselection()
            if selection:
                self.current_palette_name = listbox.get(selection[0])
                messagebox.showinfo("Selected", f"Palette '{self.current_palette_name}' will be used for next processing")
                palette_window.destroy()
        
        button_frame = Frame(palette_window)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ModelButton(button_frame, "Select", select_palette).grid_layout(row=0, column=0, padx=5)
        ModelButton(button_frame, "Close", palette_window.destroy).grid_layout(row=0, column=1, padx=5)
    
    def show_pca_analysis(self):
        """Afficher l'analyse PCA de l'image"""
        if self.image_processor.original_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
        
        try:
            # Get pixel data
            original = self.image_processor.original_image
            pixels = original.getdata()
            pixel_list = list(pixels)
            
            rgb_data = []
            for pixel in pixel_list:
                if isinstance(pixel, tuple):
                    rgb_data.append(pixel[:3])
            
            import numpy as np
            rgb_array = np.array(rgb_data)
            
            # Apply PCA
            pca = PCAPreprocessing(n_components=3)
            pca_data = pca.fit_transform(rgb_array)
            
            # Show analysis
            analysis_window = Toplevel(self.window)
            analysis_window.title("PCA Analysis Report")
            analysis_window.geometry("500x400")
            
            text_widget = Text(analysis_window, wrap='word', padx=10, pady=10)
            text_widget.pack(fill=BOTH, expand=True)
            
            # Get summary
            variance_ratio = pca.get_explained_variance_ratio()
            total_variance = pca.get_total_variance_explained()
            
            analysis_text = f"""PCA ANALYSIS REPORT
=====================================

Original Image: {original.size}
Total Pixels: {len(pixel_list)}

Variance Explained by Component:
"""
            for i, var in enumerate(variance_ratio):
                analysis_text += f"  Component {i+1}: {var:.2%}\n"
            
            analysis_text += f"\nTotal Variance Explained: {total_variance:.2%}\n"
            analysis_text += f"\nPCA successfully reduced {len(rgb_array[0])}D data to 3D space."
            
            text_widget.insert(1.0, analysis_text)
            text_widget.config(state=DISABLED)
            
            messagebox.showinfo("PCA Analysis", "Analysis complete. See window for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not perform PCA analysis: {e}")
            logger.error(f"PCA analysis error: {e}", exc_info=True)
    
    def save_result_advanced(self):
        """Sauvegarde avancée avec options supplémentaires"""
        # Check if we have an image to save
        image_to_save = self.image_processor.current_image or self.image_processor.original_image
        
        if image_to_save is None:
            messagebox.showwarning("⚠️  Warning", "No image to save. Load an image first!")
            return
        
        from datetime import datetime
        from tkinter import Scale, HORIZONTAL
        
        save_window = Toplevel(self.window)
        save_window.title("Advanced Save Options")
        save_window.geometry("520x650")
        save_window.resizable(False, False)
        save_window.configure(bg=Theme.PANEL)
        
        # Make window modal
        save_window.transient(self.window)
        save_window.grab_set()
        
        # Title
        title_label = Label(
            save_window, 
            text="📁 Advanced Save Options",
            font=("Segoe UI", 14, "bold"),
            bg=Theme.PANEL,
            fg=Theme.ACCENT
        )
        title_label.pack(pady=15, padx=15, fill='x')
        
        # Separator
        sep1 = Frame(save_window, bg=Theme.MUTED, height=1)
        sep1.pack(fill='x', padx=10)
        
        # Content frame
        content_frame = Frame(save_window, bg=Theme.PANEL)
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Format selection
        Label(content_frame, text="📷 Image Format:", font=("Segoe UI", 11, "bold"), bg=Theme.PANEL, fg=Theme.TEXT).pack(anchor='w', pady=(0, 8))
        
        format_var = StringVar(value="png")
        
        formats = [
            ("PNG (Standard) - Best for quality", "png"),
            ("JPEG (Compressed) - Smaller file size", "jpg"),
            ("PNG (High Resolution - 300 DPI)", "png_hires"),
            ("BMP (Uncompressed) - No quality loss", "bmp"),
        ]
        
        for text, value in formats:
            Radiobutton(
                content_frame, 
                text=text, 
                variable=format_var, 
                value=value, 
                bg=Theme.PANEL, 
                fg=Theme.TEXT, 
                font=("Segoe UI", 10),
                selectcolor=Theme.ACCENT,
                activebackground=Theme.PANEL,
                activeforeground=Theme.TEXT
            ).pack(anchor='w', padx=20, pady=5)
        
        # JPEG Quality section
        quality_frame = Frame(content_frame, bg=Theme.PANEL)
        quality_frame.pack(fill='x', pady=15)
        
        Label(quality_frame, text="JPEG Quality:", font=("Segoe UI", 10, "bold"), bg=Theme.PANEL, fg=Theme.TEXT).pack(anchor='w')
        
        # Create quality slider manually (fix the IntVar issue)
        quality_var = IntVar(value=95)
        quality_slider_container = Frame(quality_frame, bg=Theme.PANEL)
        quality_slider_container.pack(fill='x', pady=5)
        
        quality_scale = Scale(
            quality_slider_container,
            from_=1,
            to=100,
            orient=HORIZONTAL,
            variable=quality_var,
            length=300,
            bg=Theme.PANEL,
            fg=Theme.TEXT,
            troughcolor=Theme.MUTED,
            activebackground=Theme.ACCENT,
            highlightthickness=0,
            font=("Segoe UI", 9)
        )
        quality_scale.pack(side=LEFT, padx=5)
        
        quality_value_label = Label(
            quality_slider_container, 
            textvariable=quality_var, 
            font=("Segoe UI", 10, "bold"), 
            bg=Theme.PANEL, 
            fg=Theme.ACCENT, 
            width=4
        )
        quality_value_label.pack(side=LEFT, padx=10)
        
        # Auto-naming option
        auto_name_var = IntVar(value=1)
        Checkbutton(
            content_frame,
            text="Auto-generate filename with image, model & params",
            variable=auto_name_var,
            bg=Theme.PANEL,
            fg=Theme.TEXT,
            selectcolor=Theme.ACCENT,
            font=("Segoe UI", 10),
            activebackground=Theme.PANEL,
            activeforeground=Theme.TEXT
        ).pack(anchor='w', pady=10)
        
        # Info text
        info_label = Label(
            content_frame,
            text="ℹ️  PNG: Best quality, larger file. JPEG: Smaller file, slight quality loss.",
            font=("Segoe UI", 9),
            bg=Theme.PANEL,
            fg=Theme.MUTED,
            wraplength=450,
            justify='left'
        )
        info_label.pack(pady=10, anchor='w')
        
        def save_with_format():
            """Handle save operation"""
            format_choice = format_var.get()
            quality = quality_var.get()
            
            # Determine file extension
            if format_choice == "jpg":
                ext = "jpg"
                file_types = (("JPEG images", "*.jpg"), ("All files", "*.*"))
            elif format_choice == "bmp":
                ext = "bmp"
                file_types = (("BMP images", "*.bmp"), ("All files", "*.*"))
            else:  # png or png_hires
                ext = "png"
                file_types = (("PNG images", "*.png"), ("All files", "*.*"))
            
            # Generate filename if auto-naming is enabled
            if auto_name_var.get():
                # Get original image filename without extension
                original_filename = "image"
                if hasattr(self.image_processor, 'original_image_path') and self.image_processor.original_image_path:
                    original_filename = os.path.splitext(os.path.basename(self.image_processor.original_image_path))[0]
                
                # Get model name
                model_name = self.active_model_name or "model"
                
                # Get palette name
                palette_name = self.current_palette_name or "viridis"
                
                # Extract parameters from params_info_label and format them
                params_text = self.params_info_label.cget("text")
                params_str = ""
                
                if params_text:
                    # Parse parameters: "Clusters: 5 | n_init: 30 | max_iter: 500 | Palette: viridis"
                    # Convert to: "clusters-5_n_init-30_max_iter-500"
                    parts = params_text.split(" | ")
                    param_list = []
                    
                    for part in parts:
                        if "Palette:" not in part:  # Skip palette in this loop (we add it separately)
                            # Convert "Clusters: 5" to "clusters-5"
                            if ": " in part:
                                key, value = part.split(": ", 1)
                                key_clean = key.lower().replace(" ", "_")
                                value_clean = value.strip().replace(" ", "-")
                                param_list.append(f"{key_clean}-{value_clean}")
                    
                    if param_list:
                        params_str = "_" + "_".join(param_list)
                
                # Create filename with all components:
                # segmented_<original>_<model>_<params>_palette-<palette>.<ext>
                default_filename = f"segmented_{original_filename}_{model_name}{params_str}_palette-{palette_name}.{ext}"
                
                # Clean up filename (remove any problematic characters)
                default_filename = default_filename.replace(" ", "-").replace(":", "-")
            else:
                default_filename = f"segmented_image.{ext}"
            
            # Open file dialog
            file_path = filedialog.asksaveasfilename(
                parent=save_window,
                defaultextension=f".{ext}",
                initialfile=default_filename,
                filetypes=file_types,
                title="Save Image As"
            )
            
            if file_path:
                try:
                    if format_choice == "png_hires":
                        # Save PNG with higher DPI metadata
                        image_to_save.save(file_path, "PNG", dpi=(300, 300))
                        messagebox.showinfo("✅ Success", f"High-resolution PNG saved:\n{file_path}")
                    elif format_choice == "jpg":
                        # Convert to RGB if necessary (JPEG doesn't support RGBA)
                        if image_to_save.mode == 'RGBA':
                            rgb_image = image_to_save.convert('RGB')
                            rgb_image.save(file_path, "JPEG", quality=quality, optimize=True)
                        else:
                            image_to_save.save(file_path, "JPEG", quality=quality, optimize=True)
                        messagebox.showinfo("✅ Success", f"JPEG saved (Quality: {quality}):\n{file_path}")
                    elif format_choice == "bmp":
                        image_to_save.save(file_path, "BMP")
                        messagebox.showinfo("✅ Success", f"BMP saved:\n{file_path}")
                    else:  # png
                        image_to_save.save(file_path, "PNG")
                        messagebox.showinfo("✅ Success", f"PNG saved:\n{file_path}")
                    
                    save_window.destroy()
                except Exception as e:
                    messagebox.showerror("❌ Error", f"Could not save image:\n{str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Separator before buttons
        sep2 = Frame(save_window, bg=Theme.MUTED, height=1)
        sep2.pack(fill='x', padx=10, pady=10)
        
        # Buttons frame
        button_frame = Frame(save_window, bg=Theme.PANEL)
        button_frame.pack(fill='x', padx=15, pady=15)
        
        # Use standard Button widgets instead of ModelButton
        from tkinter import Button
        
        save_btn = Button(
            button_frame,
            text="💾 Save Image",
            command=save_with_format,
            font=("Segoe UI", 11, "bold"),
            bg="#4CAF50",
            fg="#0056B3",
            activebackground="#45a049",
            activeforeground="#FFD700",
            relief="raised",
            padx=20,
            pady=10,
            cursor="hand2",
            borderwidth=2
        )
        save_btn.grid(row=0, column=0, padx=5, sticky='ew')
        
        cancel_btn = Button(
            button_frame,
            text="❌ Cancel",
            command=save_window.destroy,
            font=("Segoe UI", 11, "bold"),
            bg="#FF6B6B",
            fg="#000000",
            activebackground="#E63946",
            activeforeground="#FFD700",
            relief="raised",
            padx=20,
            pady=10,
            cursor="hand2",
            borderwidth=2
        )
        cancel_btn.grid(row=0, column=1, padx=5, sticky='ew')
        
        # Make columns equal width
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

if __name__ == '__main__':
    ImageSegmentationApplication()






