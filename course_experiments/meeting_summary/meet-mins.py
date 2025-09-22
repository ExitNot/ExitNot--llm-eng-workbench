#!/usr/bin/env python3
"""
Meeting Minutes Generator - Gradio UI Application

A web interface for testing meeting summarization with different models and configurations.
Supports both local and remote deployment.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import gradio as gr
from dotenv import load_dotenv
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from meeting_summarizer import SummarizerFactory
from meeting_summarizer.config import PresetManager
from meeting_summarizer.base import SummarizerConfig, SummarizerType, DeviceType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for caching loaded models
_current_summarizer = None
_current_config = None
_asr_pipeline = None


def load_sample_transcriptions() -> Dict[str, str]:
    """Load sample transcriptions for testing."""
    return {
        "Short Team Meeting": """
Good morning everyone, welcome to our weekly team standup. Let's go around and share updates.

Sarah: This week I completed the user authentication module. All tests are passing and it's ready for review. Next week I'll start working on the dashboard components.

Mike: I finished the database migrations and optimized the queries we discussed last week. Performance improved by about 40%. I'm planning to work on the API documentation next.

Lisa: I've been working on the mobile responsive design. The main pages are done, but I need another day or two for the settings page. I also found a few bugs in the checkout flow that I'll fix this week.

Manager: Great progress everyone. Sarah, can you coordinate with Lisa on the dashboard design? And Mike, let's schedule a code review for the auth module by Thursday.

Sarah: Absolutely, I'll set up a meeting with Lisa for tomorrow.

Mike: Sounds good, I'll prepare the review materials.

Manager: Perfect. Any blockers or concerns? No? Alright, let's wrap up. Same time next week.
        """,
        
        "Budget Planning Meeting": """
Good afternoon everyone. We're here to discuss the Q4 budget allocation and planning for next year.

CFO: Let me start with our current financial position. We're 12% over budget for Q3, primarily due to increased marketing spend, but revenue is up 18%, so ROI is positive.

Marketing Director: The increased spend was for the product launch campaign. We saw 45% more qualified leads and conversion rates improved by 22%. I'd like to request a 15% budget increase for Q4 to capitalize on this momentum.

Engineering Manager: From the tech side, we need additional resources for infrastructure scaling. Current servers are at 85% capacity. I'm requesting $50K for cloud infrastructure upgrades.

HR Director: We have three critical hires planned for Q4 - two senior developers and a product manager. Total cost including benefits would be approximately $180K for the quarter.

CFO: Let me run the numbers... With current revenue projections, we can accommodate the marketing increase and infrastructure costs. The hiring budget will need to be spread across Q4 and Q1 next year.

CEO: Agreed. Marketing gets their increase, engineering gets the infrastructure budget approved immediately. For hiring, let's prioritize the product manager and one senior developer for Q4, with the second developer starting in January.

Marketing Director: That works for us. I'll adjust the campaign timeline accordingly.

Engineering Manager: Perfect, I'll start the infrastructure procurement process this week.

HR Director: I'll update the hiring timeline and communicate with the candidates.

CEO: Excellent. Let's reconvene in two weeks to review progress. Meeting adjourned.
        """,
        
        "Project Review Meeting": """
Welcome to the quarterly project review. We'll be evaluating the status of our three major initiatives.

Project Manager: Let me start with Project Alpha - the customer portal redesign. We're currently at 75% completion, slightly behind our original timeline due to scope changes requested by stakeholders.

Lead Developer: The technical implementation is solid, but we had to rebuild the authentication system to support SSO integration. This added two weeks to our timeline, but it's a crucial feature for enterprise clients.

UX Designer: User testing results are very positive. We're seeing a 35% improvement in task completion rates and user satisfaction scores are up significantly.

Project Manager: Project Beta - the mobile app - is on track. We're at 60% completion with the MVP features implemented. Beta testing starts next week with 100 internal users.

Mobile Developer: The core functionality is working well. We've resolved the performance issues we had with the data sync feature. The app is now responding within acceptable parameters.

Project Manager: Finally, Project Gamma - the analytics dashboard. This one is ahead of schedule at 85% completion. We should be ready for production deployment by month-end.

Data Analyst: The dashboard is providing insights we never had before. Management can now see real-time KPIs and the automated reporting is saving us hours of manual work each week.

VP Engineering: Great progress across all projects. For Project Alpha, let's prioritize the SSO integration completion. For Beta, make sure the beta feedback loop is tight - I want weekly summaries. And for Gamma, let's start planning the rollout to all departments.

Project Manager: Understood. I'll coordinate with all teams and send updated timelines by Friday.

VP Engineering: Perfect. Any other concerns or resource needs? No? Great work everyone. Let's keep this momentum going.
        """
    }


def get_available_presets() -> Dict[str, str]:
    """Get available preset configurations with descriptions."""
    presets = {}
    for preset in PresetManager.list_presets():
        key = preset['name']
        description = f"{preset['display_name']} - {preset['description']}"
        presets[key] = description
    return presets


def load_asr_pipeline() -> bool:
    """Load the ASR pipeline for audio-to-text conversion."""
    global _asr_pipeline
    
    if _asr_pipeline is not None:
        logger.info("ASR pipeline already loaded")
        return True
    
    try:
        logger.info("Loading ASR pipeline...")
        
        # ASR model configuration (from notebook)
        ASR_MODEL = "distil-whisper/distil-medium.en"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(ASR_MODEL)
        
        # Create pipeline
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,  # Enable Long-Form Transcription
            torch_dtype=torch_dtype,
            device=device,
        )
        
        logger.info(f"✅ ASR pipeline loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading ASR pipeline: {e}")
        return False


def transcribe_audio(audio_file, progress=gr.Progress()) -> Tuple[str, str]:
    """
    Transcribe audio file to text using ASR pipeline.
    
    Args:
        audio_file: Path to audio file or tuple (sample_rate, audio_data)
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (transcription_text, status_message)
    """
    global _asr_pipeline
    
    if audio_file is None:
        return "", "❌ No audio file provided"
    
    try:
        progress(0.1, desc="Loading ASR model...")
        
        # Load ASR pipeline if not already loaded
        if _asr_pipeline is None:
            if not load_asr_pipeline():
                return "", "❌ Failed to load ASR pipeline"
        
        progress(0.3, desc="Processing audio file...")
        
        # Handle different input types
        if isinstance(audio_file, tuple):
            # Gradio microphone input: (sample_rate, audio_data)
            sample_rate, audio_data = audio_file
            logger.info(f"Processing microphone input: {sample_rate}Hz, {len(audio_data)} samples")
        else:
            # File path
            logger.info(f"Processing audio file: {audio_file}")
        
        progress(0.5, desc="Transcribing audio...")
        
        # Transcribe using the pipeline
        result = _asr_pipeline(audio_file)
        transcription = result['text']
        
        progress(1.0, desc="Transcription complete!")
        
        # Create status message
        word_count = len(transcription.split())
        status_msg = f"✅ Transcription complete! ({word_count:,} words)"
        
        logger.info(f"Transcription completed: {word_count} words")
        return transcription, status_msg
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        error_msg = f"❌ Error transcribing audio: {str(e)}"
        return "", error_msg


def load_summarizer(preset_name: str, custom_config: Optional[str] = None) -> Tuple[bool, str]:
    """Load a summarizer with the specified configuration."""
    global _current_summarizer, _current_config
    
    try:
        # Unload current model if any
        if _current_summarizer is not None:
            _current_summarizer.unload_model()
            _current_summarizer = None
        
        # Create new summarizer
        if custom_config and custom_config.strip():
            # TODO: Implement custom config parsing
            return False, "Custom configuration not yet implemented"
        else:
            logger.info(f"Loading preset: {preset_name}")
            _current_summarizer = SummarizerFactory.create_summarizer(preset_name)
            _current_config = preset_name
        
        # Load the model
        _current_summarizer.load_model()
        
        return True, f"✅ Successfully loaded {preset_name}"
        
    except Exception as e:
        logger.error(f"Error loading summarizer: {e}")
        return False, f"❌ Error loading summarizer: {str(e)}"


def generate_summary(
    transcription: str, 
    preset_name: str, 
    custom_config: str = "",
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate a meeting summary from transcription."""
    global _current_summarizer, _current_config
    
    if not transcription.strip():
        return "❌ Please provide a transcription to summarize.", ""
    
    try:
        progress(0.1, desc="Checking model...")
        
        # Check if we need to load/reload the model
        if _current_summarizer is None or _current_config != preset_name:
            progress(0.2, desc="Loading model...")
            success, message = load_summarizer(preset_name, custom_config)
            if not success:
                return message, ""
        
        progress(0.4, desc="Generating summary...")
        
        # Generate summary with metadata
        result = _current_summarizer.summarize_with_metadata(transcription)
        summary = result["summary"]
        metadata = result["metadata"]
        
        progress(0.9, desc="Formatting output...")
        
        # Format metadata info
        metadata_info = f"""
            **Model Information:**
            - Model: {metadata['model_name']}
            - Type: {metadata['summarizer_type']}
            - Device: {metadata['device']}
            - Input length: {metadata['input_length']:,} characters
            - Summary length: {metadata['summary_length']:,} characters
            - Compression ratio: {metadata['input_length'] / max(metadata['summary_length'], 1):.1f}:1
        """
        
        progress(1.0, desc="Complete!")
        
        return summary, metadata_info
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        error_msg = f"❌ Error generating summary: {str(e)}"
        return error_msg, ""


def create_gradio_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    # Load sample transcriptions and presets
    samples = load_sample_transcriptions()
    presets = get_available_presets()
    
    with gr.Blocks(
        title="Meeting Minutes Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .summary-box {
            min-height: 300px;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🎯 Meeting Minutes Generator
        
        Generate professional meeting summaries using advanced AI models. Upload audio files or record directly,
        then choose from different summarization approaches including quantized local models and specialized pipeline models.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Audio Input Section
                gr.Markdown("## 🎤 Audio Input")
                
                with gr.Tabs():
                    with gr.TabItem("🎵 Upload Audio File"):
                        audio_file_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        
                        transcribe_file_btn = gr.Button(
                            "🔄 Transcribe Audio File", 
                            variant="secondary"
                        )
                    
                    with gr.TabItem("🎙️ Record Audio"):
                        audio_mic_input = gr.Audio(
                            label="Record Audio",
                            type="numpy",
                            sources=["microphone"]
                        )
                        
                        transcribe_mic_btn = gr.Button(
                            "🔄 Transcribe Recording", 
                            variant="secondary"
                        )
                
                # Transcription status
                transcription_status = gr.Markdown("", visible=False)
                
                # Text Input Section
                gr.Markdown("## 📝 Meeting Transcription")
                
                transcription_input = gr.Textbox(
                    label="Meeting Transcription",
                    placeholder="Upload audio above, or paste your meeting transcription here...",
                    lines=10,
                    max_lines=20
                )
                
                with gr.Row():
                    gr.Markdown("**Quick Start - Sample Transcriptions:**")
                
                with gr.Row():
                    for sample_name, sample_text in samples.items():
                        sample_btn = gr.Button(sample_name, size="sm")
                        sample_btn.click(
                            fn=lambda text=sample_text: text,
                            outputs=transcription_input
                        )
            
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Configuration")
                
                preset_dropdown = gr.Dropdown(
                    label="Model Preset",
                    choices=list(presets.keys()),
                    value=list(presets.keys())[0] if presets else "fast_summary",
                    info="Choose a pre-configured model setup"
                )
                
                # Show preset description
                preset_info = gr.Markdown(
                    value=list(presets.values())[0] if presets else "Fast Summary - Quick summarization using DistilBART, optimized for speed",
                    elem_classes=["preset-info"]
                )
                
                # Update preset info when selection changes
                preset_dropdown.change(
                    fn=lambda x: presets.get(x, ""),
                    inputs=preset_dropdown,
                    outputs=preset_info
                )
                
                with gr.Accordion("🔧 Advanced Configuration", open=False):
                    custom_config = gr.Textbox(
                        label="Custom Configuration (JSON)",
                        placeholder="Optional: Provide custom model configuration...",
                        lines=5,
                        info="Leave empty to use preset configuration"
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate Summary",
                    variant="primary",
                    size="lg"
                )
        
        gr.Markdown("## 📋 Generated Summary")
        
        with gr.Row():
            with gr.Column(scale=3):
                summary_output = gr.Markdown(
                    label="Meeting Summary",
                    elem_classes=["summary-box"]
                )
            
            with gr.Column(scale=1):
                metadata_output = gr.Markdown(
                    label="Model Information",
                    elem_classes=["metadata-box"]
                )
        
        # Connect the transcription buttons
        transcribe_file_btn.click(
            fn=transcribe_audio,
            inputs=[audio_file_input],
            outputs=[transcription_input, transcription_status],
            show_progress=True
        ).then(
            fn=lambda status: gr.update(visible=bool(status.strip())),
            inputs=[transcription_status],
            outputs=[transcription_status]
        )
        
        transcribe_mic_btn.click(
            fn=transcribe_audio,
            inputs=[audio_mic_input],
            outputs=[transcription_input, transcription_status],
            show_progress=True
        ).then(
            fn=lambda status: gr.update(visible=bool(status.strip())),
            inputs=[transcription_status],
            outputs=[transcription_status]
        )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_summary,
            inputs=[transcription_input, preset_dropdown, custom_config],
            outputs=[summary_output, metadata_output],
            show_progress=True
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 💡 Tips:
        
        **Audio Processing:**
        - **Upload Files**: Supports MP3, WAV, M4A and other common formats
        - **Record Live**: Use your microphone to record meetings directly
        - **Long Audio**: Automatically handles long recordings (>30 seconds) with chunked processing
        - **Quality**: Uses Distil-Whisper for fast, accurate transcription
        
        **Summarization:**
        - **Fast Summary**: Quick results with DistilBART (good for testing)
        - **Quality Summary**: Better results with optimized parameters  
        - **Llama Models**: High-quality results but require more resources
        - **Apple Silicon**: Optimized for M1/M2/M3 Macs
        - **CUDA Optimized**: Best performance on NVIDIA GPUs
        
        ### 🔧 Model Types:
        - **ASR Model**: Distil-Whisper (distil-medium.en) for audio-to-text conversion
        - **Pipeline Models**: Specialized summarization models (DistilBART)
        - **Quantized Models**: Large language models with memory optimization (Llama 3.1)
        """)
    
    return interface


def main():
    """Main function to run the Gradio application."""
    parser = argparse.ArgumentParser(description="Meeting Minutes Generator - Gradio UI")
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port to bind to (default: 7860)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Check for HuggingFace token if using Llama models
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.warning("HF_TOKEN not found. Llama models will not be available.")
        logger.info("Set HF_TOKEN environment variable to use Llama models.")
    
    # Create and launch the interface
    logger.info("Creating Gradio interface...")
    interface = create_gradio_interface()
    
    logger.info(f"Starting server on {args.server_name}:{args.server_port}")
    if args.share:
        logger.info("Creating public shareable link...")
    
    try:
        interface.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            quiet=not args.debug
        )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        global _current_summarizer, _asr_pipeline
        if _current_summarizer is not None:
            logger.info("Cleaning up summarizer model...")
            _current_summarizer.unload_model()
        if _asr_pipeline is not None:
            logger.info("Cleaning up ASR pipeline...")
            del _asr_pipeline
            _asr_pipeline = None
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
