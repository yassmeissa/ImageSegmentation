from PIL import Image
import os
import gc


class ImageLoader:
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    MAX_IMAGE_SIZE = 1024

    @staticmethod
    def load_image(file_path: str) -> Image.Image:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if not any(file_path.lower().endswith(fmt) for fmt in ImageLoader.SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported image format. Supported: {ImageLoader.SUPPORTED_FORMATS}")
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 20:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB. Maximum: 20MB")
        
        image = Image.open(file_path)
        
        width, height = image.size
        if width > ImageLoader.MAX_IMAGE_SIZE or height > ImageLoader.MAX_IMAGE_SIZE:
            image.thumbnail((ImageLoader.MAX_IMAGE_SIZE, ImageLoader.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'P':
            rgb_image = image.convert('RGB')
            image.close()
            image = rgb_image
        elif image.mode not in ('RGB', 'L'):
            rgb_image = image.convert('RGB')
            image.close()
            image = rgb_image
        
        gc.collect()
        return image

    @staticmethod
    def save_image(image: Image.Image, file_path: str, quality: int = 75) -> None:
        try:
            if image.mode == 'RGBA':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image
            
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                rgb_image.save(file_path, 'JPEG', quality=quality, optimize=True)
            else:
                rgb_image.save(file_path, format='PNG', optimize=True)
            
            if rgb_image != image:
                rgb_image.close()
        finally:
            gc.collect()

    @staticmethod
    def get_image_dimensions(image: Image.Image) -> tuple[int, int]:
        return image.size
