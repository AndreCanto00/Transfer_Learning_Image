import pytest
import torch
from src.models.model import ModelBuilder
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator

def test_model_builder():
    """Test model builder functionality."""
    builder = ModelBuilder()
    model = builder.get_model(num_classes=2)
    
    assert isinstance(model, torch.nn.Module)
    assert model.fc.out_features == 2

def test_model_trainer(mock_model, device):
    """Test model trainer functionality."""
    trainer = ModelTrainer(
        model=mock_model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(mock_model.parameters()),
        device=device,
        num_epochs=1
    )
    
    assert trainer.model is not None
    assert trainer.criterion is not None
    assert trainer.optimizer is not None

def test_model_evaluator(mock_model, device):
    """Test model evaluator functionality."""
    # Create dummy data loader
    x = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    evaluator = ModelEvaluator(
        model=mock_model,
        device=device,
        test_loader=loader
    )
    
    metrics = evaluator.evaluate()
    assert 'test_accuracy' in metrics
    assert 'total_samples' in metrics
    assert 'correct_predictions' in metrics