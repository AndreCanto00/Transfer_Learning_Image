import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_preprocessing import split_dataset
from src.data.data_loader import create_data_loaders

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    class_dir = data_dir / "class_a"
    class_dir.mkdir()
    # Aggiungi alcuni file di esempio nella directory
    for i in range(5):
        with open(class_dir / f"image_{i}.jpg", "w") as f:
            f.write("test image data")
    return data_dir

def test_split_dataset(sample_data_dir):
    """Test dataset splitting functionality."""
    # Assicurati che la directory non sia vuota
    assert len(os.listdir(sample_data_dir)) > 0, "La directory dei dati di esempio è vuota"
    split_dataset(str(sample_data_dir))
    
    # Check if split directories are created
    split_dir = sample_data_dir / "split_dataset"
    assert (split_dir / "train").exists()
    assert (split_dir / "val").exists()
    assert (split_dir / "test").exists()

def test_create_data_loaders(sample_data_dir):
    """Test data loader creation."""
    # Assicurati che la directory non sia vuota
    assert len(os.listdir(sample_data_dir)) > 0, "La directory dei dati di esempio è vuota"
    # First split the dataset
    split_dataset(str(sample_data_dir))
    split_dir = sample_data_dir / "split_dataset"
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=str(split_dir / "train"),
        val_path=str(split_dir / "val"),
        test_path=str(split_dir / "test"),
        batch_size=2,
        num_workers=0  # Use 0 for testing
    )
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None