# Meeting Summary - AI-Powered Meeting Minutes Generator

A Python-based project to research quantized models and pipeline approaches, featuring a user-friendly Gradio web interface. Main focus is meeting summarization

## Overview

This project provides a comprehensive meeting summarization system that:
- **Quantized Models**: Memory-efficient large language models (Llama 3.1 8B) with 8-bit quantization
- **Pipeline Models**: Specialized summarization models (DistilBART) with automatic chunking
- **Web Interface**: User-friendly Gradio UI for testing and production use
- **Flexible Configuration**: Preset configurations and custom model settings

The system is designed to handle various meeting transcription lengths and can automatically chunk long texts for optimal processing.

## Available Models

### Quantized Models (using optimum-quanto)
- **Llama 3.1 8B Instruct**: High-quality summaries with 8-bit quantization for memory efficiency
- Automatic device detection (CPU, CUDA, MPS)
- Context-aware chat-based summarization

### Pipeline Models  
- **DistilBART-CNN**: Specialized summarization model with automatic chunking
- Fast inference optimized for meeting transcriptions
- Handles long documents through intelligent text segmentation

### ASR Models (for reference)
- **Distil-Whisper (distil-medium.en)**: For audio-to-text conversion (49% smaller, 6x faster than Whisper)

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- HuggingFace account and token (for model access)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd meeting_summary
   ```

2. **Install dependencies using uv**:
   ```bash
   # Install the main dependencies
   uv sync
   
   # Install with Jupyter notebook support
   uv sync --group jupyter
   ```

3. **Set up environment variables**:
   Copy the example environment file and add your HuggingFace token:
   ```bash
   cp .env.example .env
   # Edit .env and add your HuggingFace token
   ```

## Usage

### 🚀 Web Interface (Recommended)

Launch the Gradio web interface for an intuitive experience:

```bash
# Start local web interface
python meet-mins.py

# Or with custom settings
python meet-mins.py --server-port 8080 --share --debug
```

**Command line options:**
- `--share`: Create a public shareable link
- `--server-name`: Server IP to bind to (default: 127.0.0.1)  
- `--server-port`: Server port (default: 7860)
- `--debug`: Enable debug mode

Navigate to `http://localhost:7860` in your browser and:
1. **Audio Input** (Optional): Upload audio files or record directly using your microphone
2. **Transcribe Audio**: Convert audio to text using Distil-Whisper ASR model
3. **Choose Model Preset**: Select from fast summary, quality summary, or Llama models
4. **Generate Summary**: Click the generate button and wait for results
5. **Review Output**: Get formatted meeting minutes with metadata

### 💻 Programmatic Usage

```python
from meeting_summarizer import SummarizerFactory

# Quick start with preset
summarizer = SummarizerFactory.create_summarizer("fast_summary")

with summarizer:
    summary = summarizer.summarize(transcription_text)
    print(summary)

# Custom configuration
from meeting_summarizer import SummarizerConfig, SummarizerType

config = SummarizerConfig(
    model_name="sshleifer/distilbart-cnn-12-6",
    summarizer_type=SummarizerType.PIPELINE,
    max_chunk_tokens=800
)

summarizer = SummarizerFactory.create_from_config(config)
```

### 📓 Jupyter Notebooks

Explore detailed examples and the original pipeline:

```bash
# Start Jupyter with the project dependencies
uv run --group jupyter jupyter lab notebooks/meeting-minutes.ipynb
```
The notebook contains the original pipeline implementation and detailed explanations.

## 🎯 Available Presets

| Preset | Model | Description | Best For |
|--------|-------|-------------|----------|
| `fast_summary` | DistilBART | Quick summarization, optimized for speed | Testing, rapid prototyping |
| `quality_summary` | DistilBART | High-quality with better parameters | Production use, balanced speed/quality |
| `llama_8b` | Llama 3.1 8B | 8-bit quantized for balanced performance | High-quality summaries with good performance |
| `apple_silicon` | Llama 3.1 8B | Optimized for M1/M2/M3 Macs | Apple Silicon devices |
| `cuda_optimized` | Llama 3.1 8B | Optimized for NVIDIA GPUs | CUDA-enabled systems |

## 🔧 Features

- **Complete Audio-to-Summary Pipeline**: Upload audio files or record directly in the web interface
- **Advanced ASR**: Distil-Whisper model for fast, accurate speech-to-text conversion
- **Long Audio Support**: Automatic chunked processing for recordings longer than 30 seconds
- **Abstraction Layer**: Clean interface supporting multiple summarization approaches
- **Memory Efficient**: 8-bit quantization using optimum-quanto (cross-platform compatible)
- **Automatic Chunking**: Handles long transcriptions through intelligent text segmentation
- **Device Optimization**: Auto-detects and optimizes for CPU, CUDA, and Apple Silicon
- **Web Interface**: User-friendly Gradio UI with audio upload/recording and sample transcriptions
- **Flexible Configuration**: JSON/YAML config files and preset management
- **Context Management**: Automatic resource cleanup and model management

### Processing Your Own Audio Files

**Option 1: Web Interface (Recommended)**
1. Launch the web interface: `python meet-mins.py`
2. Upload your audio file or record directly using the microphone
3. Click "Transcribe Audio" to convert speech to text
4. Choose a summarization preset and generate meeting minutes

**Option 2: Jupyter Notebook**
1. Place your audio file in the `extra/` directory
2. Update the `audio_file_name` variable in the notebook
3. Run the pipeline to get transcription, then use the web interface or package for summarization

**Option 3: Programmatic**
```python
# Complete pipeline example
from meeting_summarizer import SummarizerFactory

# For audio files, use the web interface or notebook
# For existing transcriptions:
summarizer = SummarizerFactory.create_summarizer("fast_summary")
with summarizer:
    summary = summarizer.summarize(transcription_text)
```

Supported audio formats include MP3, WAV, M4A, and other common formats.

### Example Usage

TBD

## Project Structure

```
meeting_summary/
├── extra/                      # Audio files for processing
│   └── denver_meeting_rec.mp3   # Sample meeting recording
├── notebooks/                  # Jupyter notebooks
│   └── meeting-minutes.ipynb   # Main pipeline notebook
├── main.py                     # Command-line entry point
├── pyproject.toml             # Project dependencies and configuration
├── uv.lock                    # Locked dependency versions
└── README.md                  # This file
```

## Performance Notes

- **CPU vs GPU**: The pipeline automatically detects CUDA availability and uses GPU acceleration when possible
- **Memory Usage**: Distil-Whisper uses low CPU memory usage optimizations
- **Processing Speed**: Long-form transcription is ~9x faster than sequential processing
- **Chunk Length**: Default chunk length is set to 15 seconds for optimal performance

## Troubleshooting

### Common Issues

1. **HuggingFace Token Error**: Ensure your `HF_TOKEN` is set in the `.env` file
2. **CUDA Out of Memory**: Reduce batch size or use CPU processing
3. **Audio Format Issues**: Convert audio to a supported format (MP3, WAV, etc.)

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed: `uv sync --all-groups`
2. Verify your HuggingFace token has access to the required models
3. Ensure your audio file is in a supported format
