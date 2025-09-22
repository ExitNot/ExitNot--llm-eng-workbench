# Meeting Summary Pipeline

A Python-based pipeline that transforms audio recordings into structured meeting summaries using advanced speech-to-text and natural language processing models.

## Overview

This project implements an example of meeting summary pipeline that:
- Converts audio files to text using Distil-Whisper (a distilled version of OpenAI's Whisper model)
- Processes long-form audio recordings (>30 seconds) efficiently
- Generates structured meeting summaries from transcribed text

The pipeline is designed to work with various audio formats and can handle lengthy meeting recordings through chunked processing.

## Models Used

- **Distil-Whisper (distil-medium.en)**: A distilled version of Whisper that's 49% smaller and 6x faster while maintaining accuracy

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

### Running the Pipeline

#### Option 1: Using Jupyter Notebook (Recommended)
```bash
# Start Jupyter with the project dependencies
uv run --group jupyter jupyter lab notebooks/meeting-minutes.ipynb
```
The notebook contains explanation of pipeline

#### Option 2: Using the Agent **Not Implemented**

### Processing Your Own Audio Files

1. Place your audio file in the `extra/` directory
2. Update the `audio_file_name` variable in the notebook
3. Run the pipeline

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
