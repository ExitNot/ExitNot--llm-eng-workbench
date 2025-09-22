"""
Configuration management for meeting summarizers.
"""

from .presets import PresetManager
from .loader import ConfigLoader

__all__ = [
    "PresetManager",
    "ConfigLoader"
]
