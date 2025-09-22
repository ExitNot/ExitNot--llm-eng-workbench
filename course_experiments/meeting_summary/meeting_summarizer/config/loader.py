"""
Configuration loader for external configuration files.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

from ..base import SummarizerConfig, SummarizerType, DeviceType

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and validates summarizer configurations from files.
    
    Supports JSON and YAML configuration files with validation and defaults.
    """
    
    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> SummarizerConfig:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
            
        Returns:
            SummarizerConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() in ['.json']:
            return cls._load_json(config_path)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            return cls._load_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    @classmethod
    def _load_json(cls, config_path: Path) -> SummarizerConfig:
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls._create_config_from_dict(config_dict, config_path)
    
    @classmethod
    def _load_yaml(cls, config_path: Path) -> SummarizerConfig:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls._create_config_from_dict(config_dict, config_path)
    
    @classmethod
    def _create_config_from_dict(cls, config_dict: Dict[str, Any], source_path: Path) -> SummarizerConfig:
        """Create SummarizerConfig from dictionary with validation."""
        try:
            # Validate required fields
            if 'model_name' not in config_dict:
                raise ValueError("'model_name' is required in configuration")
            
            # Convert string enums to enum instances
            if 'summarizer_type' in config_dict and isinstance(config_dict['summarizer_type'], str):
                config_dict['summarizer_type'] = SummarizerType(config_dict['summarizer_type'])
            
            if 'device' in config_dict and isinstance(config_dict['device'], str):
                try:
                    config_dict['device'] = DeviceType(config_dict['device'])
                except ValueError:
                    # Keep as string if not in enum (e.g., "cuda:0")
                    pass
            
            return SummarizerConfig(**config_dict)
            
        except Exception as e:
            raise ValueError(f"Invalid configuration in {source_path}: {e}")
    
    @classmethod
    def save_config(cls, config: SummarizerConfig, output_path: Union[str, Path], format: str = "json") -> None:
        """
        Save configuration to a file.
        
        Args:
            config: SummarizerConfig to save
            output_path: Output file path
            format: Output format ("json" or "yaml")
        """
        output_path = Path(output_path)
        
        # Convert config to dictionary
        config_dict = cls._config_to_dict(config)
        
        if format.lower() == "json":
            cls._save_json(config_dict, output_path)
        elif format.lower() in ["yaml", "yml"]:
            cls._save_yaml(config_dict, output_path)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Configuration saved to: {output_path}")
    
    @classmethod
    def _config_to_dict(cls, config: SummarizerConfig) -> Dict[str, Any]:
        """Convert SummarizerConfig to dictionary for serialization."""
        config_dict = {}
        
        # Convert all fields to serializable format
        for field_name, field_value in config.__dict__.items():
            if isinstance(field_value, (SummarizerType, DeviceType)):
                config_dict[field_name] = field_value.value
            elif field_value is not None:
                config_dict[field_name] = field_value
        
        return config_dict
    
    @classmethod
    def _save_json(cls, config_dict: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def _save_yaml(cls, config_dict: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML output. Install with: pip install pyyaml")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def create_template_config(cls, output_path: Union[str, Path], format: str = "json") -> None:
        """
        Create a template configuration file with examples and comments.
        
        Args:
            output_path: Output file path
            format: Output format ("json" or "yaml")
        """
        template_config = {
            "_comment": "Meeting Summarizer Configuration Template",
            "model_name": "sshleifer/distilbart-cnn-12-6",
            "summarizer_type": "pipeline",
            "device": "auto",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "num_beams": 4,
            "max_chunk_tokens": 800,
            "chunk_overlap": 100,
            "system_message": "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown.",
            "user_prompt": "Below is an extract transcript of council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\\n{transcription}",
            "quantization_config": {
                "_comment": "For quantized models only - uses optimum-quanto",
                "weights": "int8"
            },
            "model_kwargs": {
                "_comment": "Additional model loading parameters"
            },
            "generation_kwargs": {
                "_comment": "Additional generation parameters"
            }
        }
        
        output_path = Path(output_path)
        
        if format.lower() == "json":
            cls._save_json(template_config, output_path)
        elif format.lower() in ["yaml", "yml"]:
            cls._save_yaml(template_config, output_path)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Template configuration created: {output_path}")
    
    @classmethod
    def validate_config_file(cls, config_path: Union[str, Path]) -> bool:
        """
        Validate a configuration file without loading it.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            cls.load_from_file(config_path)
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
