from .base import get_model
from .resnet import resnet18, resnet34, resnet50
from .simple_cnn import SimpleCNN

__all__ = ["get_model", "resnet18", "resnet34", "resnet50", "SimpleCNN"]
