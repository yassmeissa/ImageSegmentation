import logging
from datetime import datetime

class AppLogger:
    _logger = None
    
    @classmethod
    def setup(cls, log_file: str = 'app.log'):
        if cls._logger is not None:
            return cls._logger
        
        cls._logger = logging.getLogger('ImageSegmentation')
        cls._logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        cls._logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        cls._logger.addHandler(file_handler)
        
        return cls._logger
    
    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls.setup()
        return cls._logger

def get_logger():
    return AppLogger.get_logger()
