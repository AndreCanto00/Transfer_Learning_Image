import os
import shutil
import random
from pathlib import Path

def create_test_dataset(
    source_dir: str = "Dataset/miniImageNet",
    test_dir: str = "Dataset/test_miniImageNet",
    samples_per_class: int = 10
):
    """Create a small test dataset for CI/CD."""
    # Create test directory
    os.makedirs(test_dir, exist_ok=True)
    
    # For each class directory in source
    for class_dir in Path(source_dir).iterdir():
        if class_dir.is_dir():
            # Create corresponding class directory in test
            test_class_dir = Path(test_dir) / class_dir.name
            os.makedirs(test_class_dir, exist_ok=True)
            
            # Get list of all images
            images = list(class_dir.glob('*.jpg'))
            
            # Select random samples
            selected_images = random.sample(
                images, 
                min(samples_per_class, len(images))
            )
            
            # Copy selected images
            for img in selected_images:
                shutil.copy2(img, test_class_dir)

if __name__ == "__main__":
    create_test_dataset()