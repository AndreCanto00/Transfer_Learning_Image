import os
from os.path import join, isfile
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from src.utils.file_operations import ensure_directory, get_class_directories, organize_files

import os
import shutil
from os import listdir
from os.path import isfile, isdir, join
from sklearn.model_selection import train_test_split
from src.utils.file_operations import ensure_directory

def split_dataset(dataset_path: str) -> None:
    """Split dataset into train, validation, and test sets while preserving class structure."""

    # Creazione della cartella principale per i dati suddivisi
    split_dataset_path = os.path.join(dataset_path, 'split_dataset')
    os.makedirs(split_dataset_path, exist_ok=True)

    # Creazione delle sottocartelle per train, val e test
    train_path = os.path.join(split_dataset_path, 'train')
    val_path = os.path.join(split_dataset_path, 'val')
    test_path = os.path.join(split_dataset_path, 'test')
    
    for path in [train_path, val_path, test_path]:
        os.makedirs(path, exist_ok=True)

    # Recupera le classi presenti nel dataset
    classes = [class_name for class_name in listdir(dataset_path) if isdir(join(dataset_path, class_name))]

    for class_name in classes:
        class_folder = os.path.join(dataset_path, class_name)
        images = [img for img in listdir(class_folder) if isfile(join(class_folder, img))]

        # Suddivisione train (70%), val (15%), test (15%)
        train_images, test_val_images = train_test_split(images, test_size=0.3, random_state=42)
        val_images, test_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

        # Creazione delle cartelle per ogni classe nelle tre divisioni
        class_train_path = os.path.join(train_path, class_name)
        class_val_path = os.path.join(val_path, class_name)
        class_test_path = os.path.join(test_path, class_name)

        for path in [class_train_path, class_val_path, class_test_path]:
            os.makedirs(path, exist_ok=True)

        # Copia delle immagini nelle rispettive cartelle
        for img in train_images:
            shutil.copy2(os.path.join(class_folder, img), os.path.join(class_train_path, img))

        for img in val_images:
            shutil.copy2(os.path.join(class_folder, img), os.path.join(class_val_path, img))

        for img in test_images:
            shutil.copy2(os.path.join(class_folder, img), os.path.join(class_test_path, img))

    print("✅ Dataset split into train, validation, and test sets.")
