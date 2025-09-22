"""
Meeting Summarizer Package

A robust abstraction layer for meeting transcription summarization with support for:
- Quantized local models (Llama, etc.)
- Pipeline-based approaches (DistilBART, etc.)
- Configurable summarization strategies
"""

from .base import BaseSummarizer, SummarizerConfig
from .factory import SummarizerFactory
from .summarizers import (
    QuantizedModelSummarizer,
    PipelineSummarizer
)

__version__ = "0.1.0"
__all__ = [
    "BaseSummarizer",
    "SummarizerConfig", 
    "SummarizerFactory",
    "QuantizedModelSummarizer",
    "PipelineSummarizer"
]
