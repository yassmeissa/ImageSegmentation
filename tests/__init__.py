"""
Package de tests de validation
Contient les tests automatisés pour l'application de segmentation d'images
"""

from .test_models import TestModels
from .test_ui import TestUI

__all__ = ['TestModels', 'TestUI']
