"""Model backends used by the evaluation framework."""
from .base import GenerationConfig, ModelBackend
from .deepseek_backend import DeepSeekBackend
from .transformers_backend import TransformersBackend
__all__ = [
    "DeepSeekBackend", "GenerationConfig", "ModelBackend", "TransformersBackend"
]
