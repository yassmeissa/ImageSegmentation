from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import gc

try:
    from utils.logger import get_logger
    logger = get_logger()
except:
    logger = None


class BaseClusteringModel(ABC):
    def __init__(self, name: str, use_vibrant_colors: bool = True):
        self.name = name
        self.is_fitted = False
        self.use_vibrant_colors = use_vibrant_colors
        if logger:
            logger.info(f"Initializing {name}")
    
    @staticmethod
    def generate_vibrant_palette(n_colors: int) -> np.ndarray:
        """
        Generate a vibrant color palette using HSV color space
        
        Args:
            n_colors: Number of colors to generate
            
        Returns:
            Array of RGB colors (n_colors, 3)
        """
        if n_colors == 0:
            return np.array([[128, 128, 128]], dtype=np.uint8)
        
        colors = []
        for i in range(n_colors):
            # Distribute hues evenly across the spectrum
            hue = (i / n_colors) % 1.0  # 0 to 1
            saturation = 0.85  # High saturation for vibrant colors
            value = 0.95  # High brightness
            
            # Convert HSV to RGB
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
            
            # Convert to 0-255 range
            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)
            
            colors.append([r, g, b])
        
        return np.array(colors, dtype=np.uint8)

    @abstractmethod
    def fit(self, pixels_data: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        pass

    def segment_image(self, image: Image.Image, shared_palette: np.ndarray = None) -> Image.Image:
        if logger:
            logger.info(f"[{self.name}] segment_image() started")
        
        if not isinstance(image, Image.Image):
            if logger:
                logger.error("Input is not a PIL Image")
            raise ValueError("Input must be a PIL Image")
        
        if logger:
            logger.debug(f"[{self.name}] Image size: {image.size}, mode: {image.mode}")
        
        pixel_array = np.array(image, dtype=np.uint8)
        original_shape = pixel_array.shape
        
        if logger:
            logger.debug(f"[{self.name}] Pixel array shape: {original_shape}")
        
        if len(original_shape) == 3 and original_shape[2] == 4:
            if logger:
                logger.info("[BaseModel] Converting RGBA to RGB")
            pixel_array = pixel_array[:, :, :3]
            original_shape = pixel_array.shape
        
        if logger:
            logger.info(f"[{self.name}] Flattening pixels...")
        
        pixels_flat = pixel_array.reshape(-1, original_shape[2]).astype(np.float32)
        del pixel_array
        gc.collect()
        
        if logger:
            logger.info(f"[{self.name}] Flattened shape: {pixels_flat.shape}")
        
        try:
            if logger:
                logger.info(f"[{self.name}] Calling fit()...")
            self.fit(pixels_flat)
            
            if logger:
                logger.info(f"[{self.name}] Calling predict()...")
            cluster_labels = self.predict(pixels_flat)
            del pixels_flat
            gc.collect()
            
            if logger:
                logger.debug(f"[{self.name}] Predictions done. Labels shape: {cluster_labels.shape}")
            
            if logger:
                logger.info(f"[{self.name}] Getting cluster centers...")
            cluster_centers = self.get_cluster_centers()
            
            if logger:
                logger.debug(f"[{self.name}] Cluster centers shape: {cluster_centers.shape}")
            
            # Use vibrant color palette instead of original cluster centers
            if self.use_vibrant_colors:
                if logger:
                    logger.info(f"[{self.name}] Using vibrant color palette...")
                n_clusters = len(cluster_centers)
                # Use shared palette if provided, otherwise generate new one
                if shared_palette is not None and len(shared_palette) >= n_clusters:
                    vibrant_palette = shared_palette[:n_clusters]
                else:
                    vibrant_palette = self.generate_vibrant_palette(n_clusters)
                if logger:
                    logger.debug(f"[{self.name}] Using palette with {n_clusters} colors")
                segmented_pixels = vibrant_palette[cluster_labels]
            else:
                if logger:
                    logger.info(f"[{self.name}] Using original cluster centers...")
                segmented_pixels = cluster_centers[cluster_labels]
            
            del cluster_labels, cluster_centers
            
            segmented_array = segmented_pixels.reshape(original_shape).astype(np.uint8)
            del segmented_pixels
            
            if logger:
                logger.debug(f"[{self.name}] Segmented array shape: {segmented_array.shape}")
            
            if logger:
                logger.info(f"[{self.name}] Creating image from array...")
            result = Image.fromarray(segmented_array)
            del segmented_array
            gc.collect()
            
            if logger:
                logger.info(f"[{self.name}] segment_image() completed successfully")
            return result
        except Exception as e:
            if logger:
                logger.error(f"[{self.name}] Error in segment_image: {e}", exc_info=True)
            raise

    @abstractmethod
    def get_cluster_centers(self) -> np.ndarray:
        pass

    def get_name(self) -> str:
        return self.name
