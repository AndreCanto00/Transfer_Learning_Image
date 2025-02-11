import torch
import yaml
import os
from src.data.data_preprocessing import split_dataset
from src.data.data_loader import create_data_loaders
from src.models.model import ModelBuilder
from src.models.training import hyperparameter_search
from src.models.evaluation import evaluate_and_save_model

def main():
    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    for dir_path in config['paths'].values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset
    print("Preparing dataset...")
    split_dataset(config['data']['base_path'])
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=f"{config['data']['base_path']}/split_dataset/train",
        val_path=f"{config['data']['base_path']}/split_dataset/val",
        test_path=f"{config['data']['base_path']}/split_dataset/test",
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model builder
    model_builder = ModelBuilder()
    
    # Perform hyperparameter search
    print("Starting hyperparameter search...")
    best_hyperparameters, best_accuracy = hyperparameter_search(
        model_builder=model_builder,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rates=config['training']['learning_rates'],
        weight_decays=config['training']['weight_decays'],
        optimizer_names=config['training']['optimizer_names'],
        device=device
    )
    
    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    final_model = model_builder.get_model()
    criterion = model_builder.get_criterion()
    optimizer = model_builder.get_optimizer(
        best_hyperparameters['optimizer'],
        final_model,
        best_hyperparameters['lr'],
        best_hyperparameters['weight_decay']
    )
    
    # Evaluate and save model
    print("\nEvaluating final model...")
    metrics = evaluate_and_save_model(
        model=final_model,
        test_loader=test_loader,
        device=device,
        config_path="config/config.yaml"
    )

if __name__ == "__main__":
    main()