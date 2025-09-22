"""
Base abstract classes and configuration for meeting summarizers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from enum import Enum


class SummarizerType(Enum):
    """Types of available summarizers."""
    QUANTIZED_MODEL = "quantized_model"
    PIPELINE = "pipeline"


class DeviceType(Enum):
    """Supported device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class SummarizerConfig:
    """Configuration for summarizers."""
    
    # Model configuration
    model_name: str
    summarizer_type: SummarizerType
    device: Union[str, DeviceType] = DeviceType.AUTO
    
    # Generation parameters
    max_new_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 4
    
    # Quantization settings (for quantized models)
    quantization_config: Optional[Dict[str, Any]] = None
    torch_dtype: Optional[str] = None
    
    # Chunking settings (for pipeline models)
    max_chunk_tokens: int = 800
    chunk_overlap: int = 100
    
    # Prompts
    system_message: str = (
        "You are an assistant that produces minutes of meetings from transcripts, "
        "with summary, key discussion points, takeaways and action items with owners, in markdown."
    )
    user_prompt: str = (
        "Below is an extract transcript of council meeting. Please write minutes in markdown, "
        "including a summary with attendees, location and date; discussion points; takeaways; "
        "and action items with owners.\n{transcription}"
    )
    
    # Additional model-specific parameters
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and processing."""
        # Convert string enums to enum instances
        if isinstance(self.summarizer_type, str):
            self.summarizer_type = SummarizerType(self.summarizer_type)
        if isinstance(self.device, str):
            try:
                self.device = DeviceType(self.device)
            except ValueError:
                # Keep as string if not in enum (e.g., "cuda:0")
                pass


class BaseSummarizer(ABC):
    """
    Abstract base class for all meeting summarizers.
    
    This class defines the interface that all summarizer implementations must follow.
    """
    
    def __init__(self, config: SummarizerConfig):
        """
        Initialize the summarizer with configuration.
        
        Args:
            config: SummarizerConfig instance containing all necessary parameters
        """
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None
        
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass
    
    @abstractmethod
    def summarize(self, transcription: str) -> str:
        """
        Generate a summary from the meeting transcription.
        
        Args:
            transcription: The meeting transcription text
            
        Returns:
            Generated summary as a string
        """
        pass
    
    def summarize_with_metadata(self, transcription: str) -> Dict[str, Any]:
        """
        Generate a summary with additional metadata.
        
        Args:
            transcription: The meeting transcription text
            
        Returns:
            Dictionary containing summary and metadata
        """
        summary = self.summarize(transcription)
        
        return {
            "summary": summary,
            "metadata": {
                "model_name": self.config.model_name,
                "summarizer_type": self.config.summarizer_type.value,
                "device": str(self.config.device),
                "input_length": len(transcription),
                "summary_length": len(summary),
                "config": self.config
            }
        }
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
    
    def _get_device_string(self) -> str:
        """Get device string for model loading."""
        if isinstance(self.config.device, DeviceType):
            if self.config.device == DeviceType.AUTO:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            else:
                return self.config.device.value
        else:
            return str(self.config.device)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count from text."""
        # Simple word-based estimation (approximately 0.75 words per token)
        return int(len(text.split()) * 0.75)
    
    def _build_messages(self, transcription: str) -> list:
        """Build chat messages for chat-based models."""
        formatted_prompt = self.config.user_prompt.format(transcription=transcription)
        return [
            {"role": "system", "content": self.config.system_message},
            {"role": "user", "content": formatted_prompt}
        ]
