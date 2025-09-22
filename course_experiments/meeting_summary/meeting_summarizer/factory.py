"""
Factory pattern for creating summarizer instances.

Provides easy instantiation of different summarizer types with sensible defaults.
"""

import logging
from typing import Dict, Any, Optional, Union

from .base import BaseSummarizer, SummarizerConfig, SummarizerType, DeviceType
from .summarizers import QuantizedModelSummarizer, PipelineSummarizer
from .config import PresetManager

logger = logging.getLogger(__name__)


class SummarizerFactory:
    """
    Factory class for creating summarizer instances.
    
    Provides convenient methods to create different types of summarizers
    with sensible defaults and easy configuration.
    """
    
    @classmethod
    def create_summarizer(
        cls,
        model_name_or_preset: str,
        summarizer_type: Optional[Union[str, SummarizerType]] = None,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        **kwargs
    ) -> BaseSummarizer:
        """
        Create a summarizer instance.
        
        Args:
            model_name_or_preset: Model name or preset configuration key
            summarizer_type: Type of summarizer (auto-detected if not provided)
            device: Device to run the model on
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured summarizer instance
        """
        # Check if it's a preset configuration using PresetManager
        try:
            preset_info = PresetManager.get_preset(model_name_or_preset)
            logger.info(f"Using preset configuration: {model_name_or_preset}")
            config_dict = preset_info["config"].copy()
            config_dict.update(kwargs)  # Override with user-provided kwargs
            config_dict["device"] = device
        except KeyError:
            # Create configuration from scratch
            config_dict = {
                "model_name": model_name_or_preset,
                "summarizer_type": summarizer_type or cls._detect_summarizer_type(model_name_or_preset),
                "device": device,
                **kwargs
            }
        
        # Create configuration object
        config = SummarizerConfig(**config_dict)
        
        # Create appropriate summarizer instance
        return cls._create_from_config(config)
    
    @classmethod
    def create_quantized_summarizer(
        cls,
        model_name: str,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        quantization_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QuantizedModelSummarizer:
        """
        Create a quantized model summarizer.
        
        Args:
            model_name: Name of the model to load
            device: Device to run the model on
            quantization_config: Quantization configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            QuantizedModelSummarizer instance
        """
        config_dict = {
            "model_name": model_name,
            "summarizer_type": SummarizerType.QUANTIZED_MODEL,
            "device": device,
            "quantization_config": quantization_config or {"weights": "int8"},
            **kwargs
        }
        
        config = SummarizerConfig(**config_dict)
        return QuantizedModelSummarizer(config)
    
    @classmethod
    def create_pipeline_summarizer(
        cls,
        model_name: str,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        max_chunk_tokens: int = 800,
        **kwargs
    ) -> PipelineSummarizer:
        """
        Create a pipeline-based summarizer.
        
        Args:
            model_name: Name of the model to load
            device: Device to run the model on
            max_chunk_tokens: Maximum tokens per chunk
            **kwargs: Additional configuration parameters
            
        Returns:
            PipelineSummarizer instance
        """
        config_dict = {
            "model_name": model_name,
            "summarizer_type": SummarizerType.PIPELINE,
            "device": device,
            "max_chunk_tokens": max_chunk_tokens,
            **kwargs
        }
        
        config = SummarizerConfig(**config_dict)
        return PipelineSummarizer(config)
    
    @classmethod
    def create_from_config(cls, config: SummarizerConfig) -> BaseSummarizer:
        """
        Create a summarizer from a configuration object.
        
        Args:
            config: SummarizerConfig instance
            
        Returns:
            Appropriate summarizer instance
        """
        return cls._create_from_config(config)
    
    @classmethod
    def _create_from_config(cls, config: SummarizerConfig) -> BaseSummarizer:
        """Internal method to create summarizer from config."""
        if config.summarizer_type == SummarizerType.QUANTIZED_MODEL:
            return QuantizedModelSummarizer(config)
        elif config.summarizer_type == SummarizerType.PIPELINE:
            return PipelineSummarizer(config)
        else:
            raise ValueError(f"Unknown summarizer type: {config.summarizer_type}")
    
    @classmethod
    def _detect_summarizer_type(cls, model_name: str) -> SummarizerType:
        """
        Auto-detect summarizer type based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Detected summarizer type
        """
        model_name_lower = model_name.lower()
        
        # Pipeline models (specialized for summarization)
        pipeline_indicators = [
            "distilbart", "bart", "t5", "pegasus", "led", "bigbird",
            "summarization", "summary"
        ]
        
        # Quantized models (large language models)
        quantized_indicators = [
            "llama", "mistral", "falcon", "gpt", "opt", "bloom",
            "alpaca", "vicuna", "claude"
        ]
        
        for indicator in pipeline_indicators:
            if indicator in model_name_lower:
                return SummarizerType.PIPELINE
        
        for indicator in quantized_indicators:
            if indicator in model_name_lower:
                return SummarizerType.QUANTIZED_MODEL
        
        # Default to pipeline for unknown models
        logger.warning(f"Could not detect summarizer type for {model_name}, defaulting to pipeline")
        return SummarizerType.PIPELINE
    
    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        List available preset configurations.
        
        Returns:
            Dictionary of preset configurations
        """
        return PresetManager.get_all_presets()
    
    @classmethod
    def get_preset_info(cls, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific preset.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Preset configuration or None if not found
        """
        try:
            return PresetManager.get_preset(preset_name)
        except KeyError:
            return None
