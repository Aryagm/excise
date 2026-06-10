"""excise: extract one capability from an open LLM into a smaller,
deployable model. One command, no labels."""

from .config import ExtractConfig
from .extractor import ExtractionResult, extract
from .export import slice_model, param_count

__version__ = "0.1.0"
__all__ = ["extract", "ExtractConfig", "ExtractionResult", "slice_model",
           "param_count"]
