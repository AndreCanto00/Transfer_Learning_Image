import pytest
import torch
import os
import shutil
from pathlib import Path

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data structure."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create sample class directories
    for class_name in ['class1', 'class2']:
        class_dir = data_dir / class_name
        class_dir.mkdir()
        
        # Create dummy image files
        for i in range(5):
            (class_dir / f"img_{i}.jpg").touch()
    
    return data_dir

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )

@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")