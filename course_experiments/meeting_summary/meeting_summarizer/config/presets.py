"""
Preset configurations for different use cases and models.
"""

from typing import Dict, Any, List
from ..base import SummarizerType, DeviceType


class PresetManager:
    """
    Manages preset configurations for different summarization scenarios.
    """
    
    # Production-ready configurations
    PRODUCTION_PRESETS = {
        "fast_summary": {
            "name": "Fast Summary",
            "description": "Quick summarization using DistilBART, optimized for speed",
            "config": {
                "model_name": "sshleifer/distilbart-cnn-12-6",
                "summarizer_type": SummarizerType.PIPELINE,
                "device": DeviceType.AUTO,
                "max_chunk_tokens": 600,
                "chunk_overlap": 50,
                "num_beams": 2,
                "do_sample": False,
                "max_new_tokens": 300
            }
        },
        
        "quality_summary": {
            "name": "Quality Summary",
            "description": "High-quality summarization using DistilBART with better parameters",
            "config": {
                "model_name": "sshleifer/distilbart-cnn-12-6",
                "summarizer_type": SummarizerType.PIPELINE,
                "device": DeviceType.AUTO,
                "max_chunk_tokens": 800,
                "chunk_overlap": 100,
                "num_beams": 4,
                "do_sample": False,
                "max_new_tokens": 512
            }
        },
        
        "llama3.1_8b": {
            "name": "Llama 3.1 8B",
            "description": "Llama 3.1 8B with 8-bit quantization for balanced speed/quality",
            "config": {
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "summarizer_type": SummarizerType.QUANTIZED_MODEL,
                "device": DeviceType.AUTO,
                "quantization_config": {"weights": "int8"},
                "torch_dtype": "float32",  # Use float32 for stability
                "max_new_tokens": 2000,
                "temperature": 1.0,
                "top_p": 0.9,
                "do_sample": False,  # Greedy decoding for stability
                "num_beams": 1,
                "generation_kwargs": {
                    "repetition_penalty": 1.05,
                    "min_new_tokens": 50
                }
            }
        },
        
    }
    
    # Development and experimental configurations
    EXPERIMENTAL_PRESETS = {
        "debug_fast": {
            "name": "Debug Fast",
            "description": "Minimal configuration for quick testing",
            "config": {
                "model_name": "sshleifer/distilbart-cnn-12-6",
                "summarizer_type": SummarizerType.PIPELINE,
                "device": DeviceType.CPU,
                "max_chunk_tokens": 200,
                "chunk_overlap": 20,
                "num_beams": 1,
                "do_sample": False,
                "max_new_tokens": 100
            }
        },
        
        "cpu_only": {
            "name": "CPU Only",
            "description": "CPU-optimized configuration for systems without GPU",
            "config": {
                "model_name": "sshleifer/distilbart-cnn-12-6",
                "summarizer_type": SummarizerType.PIPELINE,
                "device": DeviceType.CPU,
                "max_chunk_tokens": 400,
                "chunk_overlap": 50,
                "num_beams": 2,
                "do_sample": False,
                "max_new_tokens": 256
            }
        }
    }
    
    # Device-specific optimized configurations
    DEVICE_OPTIMIZED_PRESETS = {
        "apple_silicon": {
            "name": "Apple Silicon Optimized",
            "description": "Optimized for M1/M2/M3 Macs with MPS support",
            "config": {
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "summarizer_type": SummarizerType.QUANTIZED_MODEL,
                "device": DeviceType.MPS,
                "quantization_config": {"weights": "int8"},
                "torch_dtype": "float16",
                "max_new_tokens": 1500,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        },
        
        "cuda_optimized": {
            "name": "CUDA Optimized",
            "description": "Optimized for NVIDIA GPUs with CUDA support using optimum-quanto",
            "config": {
                "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "summarizer_type": SummarizerType.QUANTIZED_MODEL,
                "device": DeviceType.CUDA,
                "quantization_config": {"weights": "int8"},
                "torch_dtype": "float16",
                "max_new_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
    }
    
    @classmethod
    def get_all_presets(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available presets."""
        all_presets = {}
        all_presets.update(cls.PRODUCTION_PRESETS)
        all_presets.update(cls.EXPERIMENTAL_PRESETS)
        all_presets.update(cls.DEVICE_OPTIMIZED_PRESETS)
        return all_presets
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """
        Get a specific preset configuration.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Preset configuration
            
        Raises:
            KeyError: If preset not found
        """
        all_presets = cls.get_all_presets()
        if preset_name not in all_presets:
            available = list(all_presets.keys())
            raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")
        
        return all_presets[preset_name]
    
    @classmethod
    def list_presets(cls, category: str = "all") -> List[Dict[str, Any]]:
        """
        List presets with their descriptions.
        
        Args:
            category: Category to filter by ("production", "experimental", "device", "all")
            
        Returns:
            List of preset information
        """
        if category == "production":
            presets = cls.PRODUCTION_PRESETS
        elif category == "experimental":
            presets = cls.EXPERIMENTAL_PRESETS
        elif category == "device":
            presets = cls.DEVICE_OPTIMIZED_PRESETS
        elif category == "all":
            presets = cls.get_all_presets()
        else:
            raise ValueError(f"Unknown category: {category}")
        
        return [
            {
                "name": key,
                "display_name": info["name"],
                "description": info["description"],
                "model": info["config"]["model_name"],
                "type": info["config"]["summarizer_type"].value
            }
            for key, info in presets.items()
        ]
    
    @classmethod
    def get_preset_config(cls, preset_name: str) -> Dict[str, Any]:
        """
        Get just the configuration part of a preset.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Configuration dictionary
        """
        preset = cls.get_preset(preset_name)
        return preset["config"]
    
    @classmethod
    def create_custom_preset(
        cls,
        name: str,
        display_name: str,
        description: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a custom preset configuration.
        
        Args:
            name: Internal name for the preset
            display_name: Human-readable name
            description: Description of the preset
            config: Configuration dictionary
            
        Returns:
            Formatted preset dictionary
        """
        return {
            "name": display_name,
            "description": description,
            "config": config
        }
