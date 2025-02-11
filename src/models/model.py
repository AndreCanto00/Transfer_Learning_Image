import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Union, Dict

class ModelBuilder:
    @staticmethod
    def get_model(num_classes: int = 64, pretrained: bool = False) -> nn.Module:
        """Initialize the ResNet model."""
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    @staticmethod
    def get_optimizer(
        optimizer_name: str,
        model: nn.Module,
        lr: float,
        weight_decay: float
    ) -> Union[optim.Adam, optim.SGD]:
        """Get the specified optimizer."""
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @staticmethod
    def get_criterion() -> nn.Module:
        """Get the loss criterion."""
        return nn.CrossEntropyLoss()