# Ollama Integration Setup Guide

This guide explains how to use Ollama (local LLM) with your PageIndex codebase.

## Prerequisites

- NVIDIA RTX 4090 GPU (or any CUDA-compatible GPU)
- CUDA drivers installed
- Python environment set up

## Installation Steps

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or download from: https://ollama.com/download

### 2. Pull a Model

For keyword similarity and document processing, we recommend:

```bash
# Fast and efficient (recommended)
ollama pull llama3.1:8b

# Or for higher quality (slower)
ollama pull llama3.1:70b

# Or very fast option
ollama pull mistral:7b
```

### 3. Verify Ollama is Running

```bash
# Start Ollama (if not already running)
ollama serve

# Test in another terminal
ollama run llama3.1:8b "Hello, test"
```

### 4. Configure Environment Variables

Create or update your `.env` file:

```bash
# Set API provider to Ollama
API_PROVIDER=ollama

# Optional: Customize Ollama settings
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1:8b
```

## Usage

Once configured, your code will automatically use Ollama instead of external APIs. No code changes needed!

### Switching Between Providers

You can switch providers by changing the `API_PROVIDER` environment variable:

- `API_PROVIDER=ollama` - Use local Ollama (default)
- `API_PROVIDER=openai` - Use OpenAI API
- `API_PROVIDER=huggingface` - Use HuggingFace API

### Model Selection

For Ollama, you can specify different models:

- `OLLAMA_MODEL=llama3.1:8b` - Fast, good quality (recommended)
- `OLLAMA_MODEL=llama3.1:70b` - Higher quality, slower
- `OLLAMA_MODEL=mistral:7b` - Very fast
- `OLLAMA_MODEL=phi3` - Extremely fast, smaller model

## Benefits

✅ **No API costs** - Runs entirely locally  
✅ **No usage limits** - Process unlimited requests  
✅ **Privacy** - Data never leaves your machine  
✅ **Fast** - RTX 4090 provides excellent performance  
✅ **Offline** - Works without internet connection  

## Troubleshooting

### Ollama not found
Make sure Ollama is installed and in your PATH:
```bash
which ollama
```

### Connection refused
Make sure Ollama server is running:
```bash
ollama serve
```

### Out of memory
If you get OOM errors, try a smaller model:
```bash
OLLAMA_MODEL=llama3.1:8b  # Instead of 70b
```

### Slow performance
- Make sure CUDA is properly configured
- Check GPU utilization: `nvidia-smi`
- Try a smaller model if needed

## Performance Tips

With RTX 4090:
- **llama3.1:8b**: ~50-100 tokens/sec
- **llama3.1:70b**: ~10-20 tokens/sec (if fits in VRAM)
- **mistral:7b**: ~80-150 tokens/sec

For batch processing, the 8b model is usually the best balance.
