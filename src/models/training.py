import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int = 10
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.best_accuracy = 0.0
        self.model.to(device)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Complete training process."""
        best_accuracy = 0.0
        history = {
            'train_loss': [],
            'val_accuracy': []
        }

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            accuracy = self.validate(val_loader)
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Accuracy: {accuracy*100:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_accuracy = best_accuracy
                
        return history

def hyperparameter_search(
    model_builder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rates: list,
    weight_decays: list,
    optimizer_names: list,
    device: torch.device
) -> Tuple[Dict, float]:
    """Search for the best hyperparameters."""
    best_accuracy = 0.0
    best_hyperparameters = {}

    for lr in learning_rates:
        for weight_decay in weight_decays:
            for optimizer_name in optimizer_names:
                print(f"\nTesting: LR={lr}, Weight Decay={weight_decay}, "
                      f"Optimizer={optimizer_name}")
                
                # Initialize model and training components
                model = model_builder.get_model()
                criterion = model_builder.get_criterion()
                optimizer = model_builder.get_optimizer(
                    optimizer_name, model, lr, weight_decay
                )
                
                # Train model
                trainer = ModelTrainer(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device
                )
                trainer.train(train_loader, val_loader)
                
                # Update best results
                if trainer.best_accuracy > best_accuracy:
                    best_accuracy = trainer.best_accuracy
                    best_hyperparameters = {
                        'lr': lr,
                        'weight_decay': weight_decay,
                        'optimizer': optimizer_name
                    }

    return best_hyperparameters, best_accuracy