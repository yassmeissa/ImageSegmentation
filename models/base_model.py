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
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        if logger:
            logger.info(f"Initializing {name}")

    @abstractmethod
    def fit(self, pixels_data: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, pixels_data: np.ndarray) -> np.ndarray:
        pass

    def segment_image(self, image: Image.Image) -> Image.Image:
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
            
            if logger:
                logger.info(f"[{self.name}] Mapping labels to colors...")
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
