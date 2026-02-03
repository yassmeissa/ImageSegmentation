from PIL import Image
from typing import Callable
from utils.image_loader import ImageLoader
import gc
import weakref


class ImageProcessor:
    def __init__(self):
        self.current_image = Image.new("RGB", (600, 600), 'lightgrey')
        self.original_image = self.current_image
        # PCA preprocessing attributes
        self.pca_data = None
        self.use_pca = False
        self.pca_transformer = None
        self.original_image_path = None

    def load_from_file(self, file_path: str) -> None:
        self._cleanup_images()
        gc.collect()
        
        self.original_image = ImageLoader.load_image(file_path)
        self.current_image = self.original_image

    def apply_transformation(self, transformation: Callable[[Image.Image], Image.Image]) -> Image.Image:
        segmented = transformation(self.original_image)
        
        if self.current_image is not self.original_image and self.current_image is not None:
            try:
                self.current_image.close()
            except:
                pass
            del self.current_image
        
        self.current_image = segmented
        gc.collect()
        return self.current_image

    def reset_to_original(self) -> None:
        if self.current_image is not self.original_image and self.current_image is not None:
            try:
                self.current_image.close()
            except:
                pass
            del self.current_image
        
        self.current_image = self.original_image

    def get_current_image(self) -> Image.Image:
        return self.current_image

    def get_dimensions(self) -> tuple[int, int]:
        if self.current_image is None:
            return (0, 0)
        return self.current_image.size

    def save_current_image(self, file_path: str) -> None:
        ImageLoader.save_image(self.current_image, file_path)
        gc.collect()

    def _cleanup_images(self) -> None:
        try:
            if self.original_image is not None:
                self.original_image.close()
        except:
            pass
        
        try:
            if self.current_image is not None and self.current_image != self.original_image:
                self.current_image.close()
        except:
            pass

    def cleanup(self) -> None:
        self._cleanup_images()
        self.original_image = None
        self.current_image = None
        gc.collect()
