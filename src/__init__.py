from . import data
from . import models
from . import utils

__version__ = '0.1.0'

# src/data/__init__.py
from .data_loader import create_data_loaders
from .data_preprocessing import split_dataset

# src/models/__init__.py
from .model import ModelBuilder
from .training import ModelTrainer, hyperparameter_search
from .evaluation import ModelEvaluator, evaluate_and_save_model

# src/utils/__init__.py
from .file_operations import ensure_directory, get_class_directories, organize_files