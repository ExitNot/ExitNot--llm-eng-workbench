"""
Pipeline-based summarizer implementation for specialized summarization models.

Supports models like DistilBART with automatic chunking for long transcriptions.
"""

import logging
from typing import List, Optional, Any, Dict
from transformers import pipeline, Pipeline

from ..base import BaseSummarizer, SummarizerConfig

logger = logging.getLogger(__name__)


class PipelineSummarizer(BaseSummarizer):
    """
    Summarizer using HuggingFace pipelines for specialized summarization models.
    
    Optimized for models like DistilBART with automatic chunking support for long texts.
    Handles context length limitations through intelligent text segmentation.
    """
    
    def __init__(self, config: SummarizerConfig):
        """Initialize the pipeline summarizer."""
        super().__init__(config)
        self._pipeline: Optional[Pipeline] = None
        
    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded and ready."""
        return self._pipeline is not None
    
    def load_model(self) -> None:
        """Load the summarization pipeline."""
        if self.is_loaded:
            logger.info("Pipeline already loaded")
            return
            
        logger.info(f"Loading summarization pipeline: {self.config.model_name}")
        
        device_str = self._get_device_string()
        self._device = device_str
        
        # Convert device string for pipeline (pipeline expects different format)
        pipeline_device = self._convert_device_for_pipeline(device_str)
        
        # Initialize pipeline
        self._pipeline = pipeline(
            "summarization",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            device=pipeline_device,
            **self.config.model_kwargs
        )
        
        logger.info(f"✅ Pipeline loaded on device: {pipeline_device}")
    
    def _convert_device_for_pipeline(self, device_str: str) -> int:
        """Convert device string to format expected by pipeline."""
        if device_str == "cpu":
            return -1
        elif device_str.startswith("cuda"):
            # Extract GPU index if specified (e.g., "cuda:0" -> 0)
            if ":" in device_str:
                return int(device_str.split(":")[1])
            return 0
        elif device_str == "mps":
            # MPS support varies by pipeline, fallback to CPU for compatibility
            logger.warning("MPS device detected, falling back to CPU for pipeline compatibility")
            return -1
        else:
            logger.warning(f"Unknown device {device_str}, falling back to CPU")
            return -1
    
    def unload_model(self) -> None:
        """Unload the pipeline to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
        logger.info("Pipeline unloaded")
    
    def summarize(self, transcription: str) -> str:
        """
        Generate a summary from the meeting transcription using pipeline.
        
        Args:
            transcription: The meeting transcription text
            
        Returns:
            Generated summary as a string
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load_model() first.")
        
        logger.info("🎯 Generating meeting summary with pipeline...")
        
        # Check if text needs chunking
        estimated_tokens = self._estimate_tokens(transcription)
        prompt_overhead = self._estimate_prompt_overhead()
        available_tokens = self.config.max_chunk_tokens - prompt_overhead - 50  # Safety buffer
        
        logger.info(f"📊 Transcription: ~{estimated_tokens} tokens")
        logger.info(f"📊 Available space: ~{available_tokens} tokens")
        
        if estimated_tokens <= available_tokens:
            # Text fits in available space, process directly
            logger.info("✅ Text fits in context window, processing directly")
            return self._summarize_direct(transcription)
        else:
            # Text too long, use chunking
            logger.info("⚠️  Text exceeds context window, using chunking strategy")
            return self._summarize_with_chunking(transcription)
    
    def _estimate_prompt_overhead(self) -> int:
        """Estimate token overhead from system and user prompts."""
        prompt_text = f"{self.config.system_message} {self.config.user_prompt}"
        return self._estimate_tokens(prompt_text)
    
    def _summarize_direct(self, transcription: str) -> str:
        """Summarize text directly without chunking."""
        # Create input with system and user prompts
        input_text = self._format_input(transcription)
        
        # Generate summary
        result = self._pipeline(
            input_text,
            max_length=min(512, self.config.max_new_tokens),
            min_length=50,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
            num_beams=self.config.num_beams,
            **self.config.generation_kwargs
        )
        
        return result[0]['summary_text']
    
    def _summarize_with_chunking(self, transcription: str) -> str:
        """Summarize long text using chunking strategy."""
        # Calculate optimal chunk size
        prompt_overhead = self._estimate_prompt_overhead()
        chunk_size = max(200, self.config.max_chunk_tokens - prompt_overhead - 50)
        
        logger.info(f"📄 Using chunk size: ~{chunk_size} tokens")
        
        # Split text into chunks
        chunks = self._chunk_text(transcription, chunk_size)
        logger.info(f"📄 Split into {len(chunks)} chunks")
        
        if len(chunks) == 1:
            return self._summarize_direct(chunks[0])
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"🔄 Processing chunk {i}/{len(chunks)}...")
            
            # Verify chunk length and truncate if necessary
            chunk_with_prompts = self._format_input(chunk)
            chunk_tokens = self._estimate_tokens(chunk_with_prompts)
            
            if chunk_tokens > self.config.max_chunk_tokens:
                logger.warning(f"   ⚠️  Chunk {i} still too long ({chunk_tokens} tokens), truncating...")
                chunk = self._truncate_chunk(chunk, chunk_size)
                chunk_with_prompts = self._format_input(chunk)
            
            # Summarize chunk
            result = self._pipeline(
                chunk_with_prompts,
                max_length=150,  # Shorter summaries for chunks
                min_length=30,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                num_beams=self.config.num_beams,
                **self.config.generation_kwargs
            )
            
            chunk_summaries.append(result[0]['summary_text'])
            logger.info(f"✅ Chunk {i} summarized")
        
        # Combine chunk summaries
        return self._combine_summaries(chunk_summaries)
    
    def _format_input(self, transcription: str) -> str:
        """Format input text with system and user prompts."""
        formatted_prompt = self.config.user_prompt.format(transcription=transcription)
        return f"System: {self.config.system_message}\\nUser: {formatted_prompt}"
    
    def _chunk_text(self, text: str, max_chunk_tokens: int) -> List[str]:
        """
        Split text into overlapping chunks suitable for model processing.
        
        Args:
            text: Input text to chunk
            max_chunk_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        # Convert token estimates to word counts (rough estimate: ~1.3 words per token)
        max_words = int(max_chunk_tokens * 1.3)
        overlap_words = int(self.config.chunk_overlap * 1.3)
        
        start = 0
        while start < len(words):
            # Calculate end position
            end = min(start + max_words, len(words))
            
            # Create chunk
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start position (with overlap)
            if end >= len(words):
                break
            start = end - overlap_words
            
            # Ensure we make progress
            if start <= 0:
                start = max_words
        
        return chunks
    
    def _truncate_chunk(self, chunk: str, target_tokens: int) -> str:
        """Truncate chunk to fit within token limit."""
        words = chunk.split()
        max_words = int(target_tokens * 1.3)  # Conservative estimate
        
        if len(words) <= max_words:
            return chunk
            
        return ' '.join(words[:max_words])
    
    def _combine_summaries(self, chunk_summaries: List[str]) -> str:
        """Combine multiple chunk summaries into a final summary."""
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        
        logger.info("🔗 Combining chunk summaries...")
        
        # Create combined text with section labels
        combined = "\\n\\n".join([
            f"Section {i+1}: {summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        # Check if combined result needs final summarization
        combined_tokens = self._estimate_tokens(combined)
        if combined_tokens > self.config.max_chunk_tokens:
            logger.info("📋 Final summarization step...")
            
            # Final summarization of combined chunks
            final_input = f"Summarize this meeting: {combined}"
            result = self._pipeline(
                final_input,
                max_length=min(512, self.config.max_new_tokens),
                min_length=100,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                num_beams=self.config.num_beams,
                **self.config.generation_kwargs
            )
            return result[0]['summary_text']
        
        return combined
