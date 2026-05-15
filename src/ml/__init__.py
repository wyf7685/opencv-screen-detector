"""机器学习模块"""

from src.ml.predict import load_model, predict
from src.ml.train import train_model

__all__ = ["load_model", "predict", "train_model"]
