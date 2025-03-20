from importlib.metadata import version, PackageNotFoundError

__title__ = "spaceship-classifier"
__description__ = "ML models for the Spaceship Titanic challenge"
__author__ = "Dark-Type"

try:
    __version__ = version(__title__)
except PackageNotFoundError:
    __version__ = "0.1.0"

try:
    from .model import My_Classifier_Model


    __all__ = ["My_Classifier_Model"]
except ImportError:

    pass