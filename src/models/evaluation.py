import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm
import os
import yaml

class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        test_loader: DataLoader
    ):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.model.to(device)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the test set."""
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        metrics = {
            'test_accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        return metrics

def save_model(model: nn.Module, save_path: str, metadata: Dict = None):
    """Save the model and its metadata."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    # Save the model
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

def evaluate_and_save_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config_path: str = "config/config.yaml"
):
    """Evaluate the model and save if performance is good."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, test_loader)
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Print results
    print("\nTest Results:")
    print(f"Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    
    # Save model if accuracy is above threshold (e.g., 70%)
    if metrics['test_accuracy'] > 0.70:
        save_path = os.path.join(
            config['paths']['model_save_dir'],
            f"model_acc_{metrics['test_accuracy']:.4f}.pth"
        )
        save_model(model, save_path, metadata=metrics)
    
    return metrics

def load_pretrained_model(
    model: nn.Module,
    model_path: str
) -> Tuple[nn.Module, Dict]:
    """Load a pretrained model and its metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    metadata = checkpoint.get('metadata', {})
    
    return model, metadata