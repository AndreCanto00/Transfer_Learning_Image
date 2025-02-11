import torch
try:
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError("Torch is not installed. Please install it using 'pip install torch'.") from e

from torchvision import transforms, datasets
from typing import Tuple

def get_data_transforms() -> transforms.Compose:
    """Get data transformations for the model."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation and test sets."""
    transform = get_data_transforms()
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader