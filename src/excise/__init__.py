"""excise: extract one capability from an open LLM into a smaller,
deployable model. One command, no labels."""

from .config import ExtractConfig
from .extractor import ExtractionResult, extract
from .export import load_sliced, param_count, prune_vocab, slice_model

__version__ = "0.2.0"
__all__ = ["extract", "ExtractConfig", "ExtractionResult", "slice_model",
           "prune_vocab", "load_sliced", "param_count"]
