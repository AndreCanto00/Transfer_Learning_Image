import os
from os.path import join, isfile, isdir
import shutil
from typing import List

def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_class_directories(path: str) -> List[str]:
    """Get all class directories from the dataset path."""
    return [class_name for class_name in os.listdir(path) 
            if isdir(join(path, class_name))]

def organize_files(source_path: str, destination_path: str) -> None:
    """Organize files into class directories based on filename prefix."""
    for file_name in os.listdir(source_path):
        source_file_path = os.path.join(source_path, file_name)
        if os.path.isfile(source_file_path):
            prefix = file_name[:9]
            destination_folder = os.path.join(destination_path, prefix)
            ensure_directory(destination_folder)
            destination_file_path = os.path.join(destination_folder, file_name)
            shutil.move(source_file_path, destination_file_path)