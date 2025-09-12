"""
Core ML Modules - Módulos principais do sistema ML
"""

from .ml_features import FeatureExtractor
from .ml_classifiers import ClassicalMLWeedDetector
from .ml_training import MLTrainingPipeline, TrainingConfig

__all__ = ['FeatureExtractor', 'ClassicalMLWeedDetector', 'MLTrainingPipeline', 'TrainingConfig']