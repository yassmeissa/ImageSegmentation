from tkinter import Tk, Menu, Frame, Label, filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import os
import threading
from image_processor import ImageProcessor
from ui_components import ImageDisplayCanvas, ModelButton, ParameterSlider, ComparisonCanvas
from models.kmeans_model import KMeansClusteringModel
from models.gmm_model import GMMClusteringModel
from models.meanshift_model import MeanShiftClusteringModel
from models.spectral_model import SpectralClusteringModel
from config import AppConfig
from theme import Theme
from utils.logger import AppLogger

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
        file_menu.add_command(label="Save Result", command=self.save_result_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.destroy)

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
        
        # Separator
        sep1 = Frame(left_panel, bg=Theme.MUTED, height=1)
        sep1.pack(fill='x', padx=10, pady=10)
        
        # Parameters section
        params_label = Label(
            left_panel,
            text="Parameters",
            font=("Segoe UI", 13, "bold"),
            bg=Theme.PANEL,
            fg=Theme.ACCENT
        )
        params_label.pack(pady=(10, 10), padx=15)
        
        sliders_frame = Frame(left_panel, bg=Theme.PANEL)
        sliders_frame.pack(fill='x', padx=10, pady=10)
        
        # Configure grid for sliders
        sliders_frame.columnconfigure(0, weight=1)
        
        self.clusters_slider = ParameterSlider(
            sliders_frame, 
            "Clusters (K-Means, GMM, Spectral)", 
            AppConfig.CLUSTERS_MIN, 
            AppConfig.CLUSTERS_MAX, 
            AppConfig.DEFAULT_KMEANS_CLUSTERS
        )
        self.clusters_slider.grid_layout(row=0, column=0, padx=5, pady=5)
        self.clusters_slider.set_command(self.update_cluster_parameter)
        
        self.bandwidth_slider = ParameterSlider(sliders_frame, "Bandwidth (MeanShift)", 15, 45, AppConfig.DEFAULT_MEANSHIFT_BANDWIDTH)
        self.bandwidth_slider.grid_layout(row=1, column=0, padx=5, pady=5)
        self.bandwidth_slider.set_command(self.update_bandwidth_parameter)
        
        # Separator
        sep2 = Frame(left_panel, bg=Theme.MUTED, height=1)
        sep2.pack(fill='x', padx=10, pady=10)
        
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
        
        open_btn = ModelButton(operations_frame, "Open Image", self.open_image_file)
        open_btn.grid_layout(row=0, column=0, padx=5, pady=5)
        
        save_btn = ModelButton(operations_frame, "Save Result", self.save_result_image)
        save_btn.grid_layout(row=1, column=0, padx=5, pady=5)
        
        # Right panel - Canvas with comparison
        right_panel = Frame(content_frame, bg=Theme.BG)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Model info + canvas label area
        info_frame = Frame(right_panel, bg=Theme.PANEL, height=60)
        info_frame.pack(fill='x', padx=10, pady=(0, 10))
        info_frame.pack_propagate(False)
        
        self.model_info_label = Label(
            info_frame,
            text="Select a clustering model to begin",
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
                self.refresh_display()
            except Exception as e:
                print(f"Error loading image: {e}")

    def apply_model(self, model_name: str):
        if self.is_processing:
            logger.warning(f"Already processing, ignoring {model_name} request")
            messagebox.showwarning("Processing", "A process is already running...")
            return
        
        logger.info(f"[UI] Applying model: {model_name}")
        self.active_model_name = model_name
        self.update_button_states()
        
        # Enable/disable sliders based on model
        self.clusters_slider.set_enabled(model_name != "meanshift")
        self.bandwidth_slider.set_enabled(model_name == "meanshift")
        
        # Update model info label
        model_names = {'kmeans': 'K-Means', 'gmm': 'GMM', 'meanshift': 'MeanShift', 'spectral': 'Spectral'}
        self.model_info_label.config(
            text=f"Processing: {model_names.get(model_name, model_name)}",
            fg=Theme.get_model_color(model_name)
        )
        
        # Show parameters
        self._update_params_display(model_name)
        
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
    
    def _update_params_display(self, model_name: str):
        """Update the parameters display based on selected model"""
        params_text = ""
        if model_name == 'kmeans':
            params_text = f"Clusters: {self.clustering_models['kmeans'].n_clusters}"
        elif model_name == 'gmm':
            params_text = f"Components: {self.clustering_models['gmm'].n_components}"
        elif model_name == 'meanshift':
            bandwidth = self.clustering_models['meanshift'].bandwidth_param
            params_text = f"Bandwidth: {bandwidth if bandwidth else 25.0}"
        elif model_name == 'spectral':
            params_text = f"Clusters: {self.clustering_models['spectral'].n_clusters}"
        
        self.params_info_label.config(text=params_text)

    def _process_model_background(self, model):
        logger.info(f"[Thread] Processing started for {model.get_name()}")
        try:
            logger.debug("[Thread] Getting original image...")
            original = self.image_processor.original_image
            logger.debug(f"[Thread] Image obtained: {original.size}")
            
            logger.info(f"[Thread] Calling segment_image()...")
            segmented_image = model.segment_image(original)
            logger.info(f"[Thread] segment_image() completed")
            
            logger.debug("[Thread] Storing segmented image...")
            self.image_processor.current_image = segmented_image
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
            
            self.refresh_display()
            self.update_button_states()
            self.processing_thread = None
        
        self.window.after(100, self.check_processing)

    def update_cluster_parameter(self, value):
        n_clusters = int(float(value))
        self.clustering_models['kmeans'].set_n_clusters(n_clusters)
        self.clustering_models['gmm'].set_n_components(n_clusters)
        self.clustering_models['spectral'].set_parameters(n_clusters)

    def update_bandwidth_parameter(self, value):
        bandwidth = float(value)
        self.clustering_models['meanshift'].set_parameters(bandwidth)

    def update_button_states(self):
        self.kmeans_button.set_active(self.active_model_name == 'kmeans')
        self.gmm_button.set_active(self.active_model_name == 'gmm')
        self.meanshift_button.set_active(self.active_model_name == 'meanshift')
        self.spectral_button.set_active(self.active_model_name == 'spectral')

    def refresh_display(self):
        current_image = self.image_processor.get_current_image()
        original_image = self.image_processor.original_image
        
        # Display both original and segmented images
        self.comparison_canvas.display_images(original_image, current_image)
        
        if not self.is_processing:
            self.status_label.config(text="Ready")

    def save_result_image(self):
        if self.image_processor.current_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG images", "*.png"), ("JPEG images", "*.jpg"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                self.image_processor.save_current_image(file_path)
                print(f"Image saved: {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")


if __name__ == '__main__':
    ImageSegmentationApplication()






