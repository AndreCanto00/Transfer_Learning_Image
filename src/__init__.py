from . import data
from . import models
from . import utils

__version__ = '0.1.0'

# src/data/__init__.py
from src.data.data_loader import create_data_loaders
from src.data.data_preprocessing import split_dataset

# src/models/__init__.py
from src.models.model import ModelBuilder
from src.models.training import ModelTrainer, hyperparameter_search
from src.models.evaluation import ModelEvaluator, evaluate_and_save_model

# src/utils/__init__.py
from src.utils.file_operations import ensure_directory, get_class_directories, organize_files

# src/__init__.py
# Questo file è intenzionalmente lasciato vuoto per rendere `src` un modulo Python.