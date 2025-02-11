import pytest
import yaml
import os
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

@pytest.mark.slow
def test_main_execution(sample_data_dir, monkeypatch):
    """Test main script execution with minimal dataset."""
    # Mock configuration
    config = {
        'data': {'base_path': str(sample_data_dir)},
        'training': {
            'num_epochs': 1,
            'learning_rates': [0.001],
            'weight_decays': [0.0001],
            'optimizer_names': ['Adam']
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