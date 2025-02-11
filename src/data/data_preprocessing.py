import os
from os.path import join
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from ..utils.file_operations import ensure_directory, get_class_directories, organize_files

def create_split_directories(base_path: str) -> Tuple[str, str, str]:
    """Create train, validation and test directories."""
    split_dataset_path = os.path.join(base_path, 'split_dataset')
    train_path = os.path.join(split_dataset_path, 'train')
    val_path = os.path.join(split_dataset_path, 'val')
    test_path = os.path.join(split_dataset_path, 'test')
    
    for path in [train_path, val_path, test_path]:
        ensure_directory(path)
    
    return train_path, val_path, test_path

def split_dataset(dataset_path: str) -> None:
    """Split dataset into train, validation and test sets."""
    train_path, val_path, test_path = create_split_directories(dataset_path)
    classes = get_class_directories(dataset_path)
    
    for class_name in classes:
        class_folder = os.path.join(dataset_path, class_name)
        images = [img for img in os.listdir(class_folder) 
                 if isfile(join(class_folder, img))]
        
        # Split into train (70%), validation (15%), test (15%)
        train_images, test_val_images = train_test_split(
            images, test_size=0.3, random_state=42)
        val_images, test_images = train_test_split(
            test_val_images, test_size=0.5, random_state=42)
        
        # Copy images to respective directories
        for img, dest_path in [
            (train_images, train_path),
            (val_images, val_path),
            (test_images, test_path)
        ]:
            for image in img:
                shutil.copy(
                    os.path.join(class_folder, image),
                    os.path.join(dest_path, image)
                )
    
    # Organize files in their respective directories
    for path in [train_path, val_path, test_path]:
        organize_files(path, path)