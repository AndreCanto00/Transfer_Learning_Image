import pytest
import yaml
import os
import sys
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import main

def test_config_loading():
    """Test configuration loading."""
    assert os.path.exists("config/config.yaml")
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'data' in config
    assert 'model' in config
    assert 'training' in config
    assert 'paths' in config

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    class_dir = data_dir / "class_a"
    class_dir.mkdir()
    # Aggiungi alcune immagini di esempio valide nella directory
    for i in range(5):
        img = Image.new('RGB', (60, 30), color = (73, 109, 137))
        img.save(class_dir / f"image_{i}.jpg")
    return data_dir

@pytest.mark.slow
def test_main_execution(sample_data_dir, monkeypatch):
    """Test main script execution with minimal dataset."""
    # Mock configuration
    config = {
        'data': {
            'base_path': str(sample_data_dir),
            'batch_size': 2,  # Aggiungi batch_size
            'num_workers': 0  # Aggiungi num_workers
        },
        'training': {
            'num_epochs': 1,
            'learning_rates': [0.001],
            'weight_decays': [0.0001],
            'optimizer_names': ['Adam']
        },
        'paths': {
            'output': str(sample_data_dir / "output"),
            'logs': str(sample_data_dir / "logs")
        }
    }
    
    # Mock config loading
    def mock_load(*args, **kwargs):
        return config
    
    monkeypatch.setattr(yaml, 'safe_load', mock_load)
    
    # Execute main
    try:
        main()
    except Exception as e:
        pytest.fail(f"Main execution failed: {str(e)}")