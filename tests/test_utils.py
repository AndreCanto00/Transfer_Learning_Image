import pytest
import os
from src.utils.file_operations import (
    ensure_directory,
    get_class_directories,
    organize_files
)

def test_ensure_directory(tmp_path):
    """Test directory creation."""
    test_dir = tmp_path / "test_dir"
    ensure_directory(str(test_dir))
    assert test_dir.exists()

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    class1_dir = data_dir / "class1"
    class1_dir.mkdir()
    class2_dir = data_dir / "class2"
    class2_dir.mkdir()
    return data_dir

def test_get_class_directories(sample_data_dir):
    """Test getting class directories."""
    classes = get_class_directories(str(sample_data_dir))
    assert len(classes) == 2
    assert 'class1' in classes
    assert 'class2' in classes

def test_organize_files(tmp_path):
    """Test file organization."""
    # Create test files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    
    # Create test files with specific prefixes
    (source_dir / "prefix1_file1.txt").touch()
    (source_dir / "prefix1_file2.txt").touch()
    (source_dir / "prefix2_file1.txt").touch()
    
    organize_files(str(source_dir), str(source_dir))
    
    assert (source_dir / "prefix1").exists()
    assert (source_dir / "prefix2").exists()
    assert (source_dir / "prefix1" / "prefix1_file1.txt").exists()
    assert (source_dir / "prefix1" / "prefix1_file2.txt").exists()
    assert (source_dir / "prefix2" / "prefix2_file1.txt").exists()
    
    # Verifica che i file siano stati spostati correttamente
    assert not (source_dir / "prefix1_file1.txt").exists()
    assert not (source_dir / "prefix1_file2.txt").exists()
    assert not (source_dir / "prefix2_file1.txt").exists()