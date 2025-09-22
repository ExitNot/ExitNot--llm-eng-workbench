"""
Quantized model summarizer implementation for large language models.

Supports models like Llama 3.1 with quantization for memory efficiency.
"""

import logging
from typing import Optional, Any, Dict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextStreamer,
    QuantoConfig
)

from ..base import BaseSummarizer, SummarizerConfig

logger = logging.getLogger(__name__)


class QuantizedModelSummarizer(BaseSummarizer):
    """
    Summarizer for quantized large language models.
    
    Optimized for models like Llama 3.1 with quantization support for memory efficiency.
    Uses optimum-quanto for quantization support across different devices.
    """
    
    def __init__(self, config: SummarizerConfig):
        """Initialize the quantized model summarizer."""
        super().__init__(config)
        self._streamer = None
        
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None and self._tokenizer is not None
    
    def load_model(self) -> None:
        """Load the quantized model and tokenizer."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading quantized model: {self.config.model_name}")
        
        # Initialize tokenizer
        self._load_tokenizer()
        
        # Initialize model with quantization
        self._load_quantized_model()
        
        # Initialize text streamer for real-time output
        self._streamer = TextStreamer(self._tokenizer, skip_prompt=True)
        
        logger.info("✅ Quantized model loaded successfully")
    
    def _load_tokenizer(self) -> None:
        """Load and configure the tokenizer."""
        logger.info("Loading tokenizer...")

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set pad token
        # This is required for llama models to resolve this issue: https://github.com/meta-llama/llama/issues/380
        self._tokenizer.pad_token = "[PAD]"
        self._tokenizer.padding_side = "left"

        logger.info("✅ Tokenizer loaded")
    
    def _load_quantized_model(self) -> None:
        """Load the model with quantization configuration."""
        device_str = self._get_device_string()
        self._device = device_str
        
        # Prepare quantization config
        quant_config = self._prepare_quantization_config()
        
        # Determine torch dtype
        torch_dtype = self._get_torch_dtype()
        
        # Model loading parameters
        model_kwargs = {
            "quantization_config": quant_config,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            **self.config.model_kwargs
        }
        
        try:
            # Try with device_map="auto" first (requires accelerate)
            logger.info("Attempting to load with device_map='auto'...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                **model_kwargs
            )
            logger.info("✅ Model loaded with device_map='auto'")
            
        except Exception as e:
            logger.warning(f"device_map='auto' failed: {e}")
            logger.info("🔄 Falling back to manual device placement...")
            
            # Fallback: load without device_map and manually move to device
            model_kwargs.pop("device_map", None)  # Remove if present
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Move to device if not CPU
            if device_str != "cpu":
                self._model = self._model.to(device_str)
                
            logger.info(f"✅ Model loaded and moved to {device_str}")
    
    def _prepare_quantization_config(self) -> Optional[Any]:
        """Prepare quantization configuration based on config."""
        if not self.config.quantization_config:
            # Default quantization config using optimum-quanto
            return QuantoConfig(weights="int8")
        
        quant_config = self.config.quantization_config
        
        # Handle optimum-quanto configuration
        if "weights" in quant_config:
            return QuantoConfig(**quant_config)
        else:
            logger.warning("Unknown quantization config format, using default optimum-quanto config")
            return QuantoConfig(weights="int8")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on device and config."""
        if self.config.torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16
            }
            return dtype_map.get(self.config.torch_dtype, torch.float16)
        
        device_str = self._get_device_string()
        if device_str == "cpu":
            return torch.float32
        else:
            return torch.float16
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        self._streamer = None
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model unloaded and memory cleared")
    
    def summarize(self, transcription: str) -> str:
        """
        Generate a summary from the meeting transcription using quantized model.
        
        Args:
            transcription: The meeting transcription text
            
        Returns:
            Generated summary as a string
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info("🎯 Generating meeting summary with quantized model...")
        
        # Build chat messages
        messages = self._build_messages(transcription)
        
        # Tokenize input
        inputs = self._tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self._device)
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": max(0.1, min(2.0, self.config.temperature)),  # Clamp temperature
            "top_p": max(0.1, min(1.0, self.config.top_p)),  # Clamp top_p
            "streamer": self._streamer,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            # Add stability parameters
            "renormalize_logits": True,
            "output_attentions": False,
            "output_hidden_states": False,
            **self.config.generation_kwargs
        }
        
        # Additional safety checks for sampling
        if self.config.do_sample:
            # Ensure we have valid sampling parameters
            if generation_kwargs["temperature"] <= 0.0:
                logger.warning("Invalid temperature for sampling, switching to greedy decoding")
                generation_kwargs["do_sample"] = False
            elif generation_kwargs["top_p"] <= 0.0 or generation_kwargs["top_p"] >= 1.0:
                logger.warning("Invalid top_p for sampling, adjusting to safe range")
                generation_kwargs["top_p"] = 0.9
        
        # Generate summary with error handling for numerical instability
        try:
            with torch.no_grad():
                outputs = self._model.generate(inputs, **generation_kwargs)
        except RuntimeError as e:
            if "probability tensor contains" in str(e) or "inf" in str(e) or "nan" in str(e):
                logger.warning(f"Numerical instability detected: {e}")
                logger.info("🔄 Retrying with safer generation parameters...")
                
                # Fallback to greedy decoding with safer parameters
                safe_kwargs = {
                    "max_new_tokens": self.config.max_new_tokens,
                    "do_sample": False,  # Disable sampling
                    "num_beams": 1,
                    "pad_token_id": self._tokenizer.eos_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                    "renormalize_logits": True,
                    "output_attentions": False,
                    "output_hidden_states": False,
                }
                
                with torch.no_grad():
                    outputs = self._model.generate(inputs, **safe_kwargs)
            else:
                raise  # Re-raise if it's a different error
        
        # Decode response (excluding input prompt)
        response = self._tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        logger.info("✅ Summary generated successfully")
        return response.strip()
    
    def summarize_streaming(self, transcription: str) -> str:
        """
        Generate a summary with real-time streaming output.
        
        Args:
            transcription: The meeting transcription text
            
        Returns:
            Generated summary as a string
        """
        logger.info("🎯 Generating meeting summary with streaming...")
        logger.info("=" * 50)
        
        summary = self.summarize(transcription)
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ Summary generated!")
        
        return summary
