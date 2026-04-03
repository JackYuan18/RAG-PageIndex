# PageIndex: Vectorless, Reasoning-based RAG

<div align="center">

**Reasoning-based RAG • No Vector DB • No Chunking • Human-like Retrieval**

[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE)

</div>

## Overview

> **Note**: This is a modified version of [PageIndex](https://github.com/VectifyAI/PageIndex) with enhanced local LLM support such that the code can run without API and improved error handling. The original PageIndex framework is developed by [VectifyAI](https://github.com/VectifyAI).

PageIndex is a **reasoning-based**, **vectorless RAG** framework that performs retrieval through document structure analysis and LLM reasoning, rather than traditional vector similarity search. Unlike conventional RAG systems, PageIndex:

- **No Vectors Needed**: Uses document structure and LLM reasoning for retrieval
- **No Chunking Needed**: Documents are organized into natural sections rather than artificial chunks
- **Human-like Retrieval**: Simulates how human experts navigate and extract knowledge from complex documents
- **Transparent Retrieval Process**: Retrieval based on reasoning — say goodbye to approximate semantic search

## Key Features

### 📚 Document Processing
- **PDF Processing**: Extract and structure table of contents (TOC) from PDF documents
- **Markdown Support**: Process markdown documents with hierarchical structure
- **Automatic TOC Detection**: Intelligently detects and extracts table of contents
- **Page Index Mapping**: Maps logical page numbers to physical page indices

### 🤖 LLM Integration
- **Multi-Provider Support**: Works with OpenAI, Ollama (local), and HuggingFace
- **Ollama Integration**: Run entirely locally with no API costs or usage limits
- **Flexible Model Selection**: Switch between providers via environment variables
- **Async Processing**: Concurrent processing for improved performance

### 🔍 Document Structure Generation
- **Hierarchical Tree Structure**: Creates tree-like document structure with sections and subsections
- **Node Summarization**: Generates summaries for document nodes
- **Keyword Extraction**: Extracts and manages keywords for document indexing
- **DocIndex Management**: Maintains keyword-to-document mapping for efficient retrieval

### ✅ Quality Assurance
- **TOC Verification**: Validates extracted table of contents accuracy
- **Title Matching**: Fuzzy matching for section titles across pages
- **Error Handling**: Robust error handling and retry mechanisms
- **JSON Parsing**: Improved JSON extraction with control character handling

## Recent Changes

### 🆕 Ollama Integration (Major Update)
- **Local LLM Support**: Added full support for Ollama, enabling local execution without API costs
- **Provider Abstraction**: Unified API interface for OpenAI, Ollama, and HuggingFace
- **Environment Configuration**: Easy switching between providers via `API_PROVIDER` environment variable
- **Performance Optimized**: Optimized prompts specifically for Ollama models

### 🔧 Improvements
- **Enhanced JSON Parsing**: Improved `extract_json` function to handle control characters and malformed JSON
- **Better Prompt Engineering**: Refined prompts for better accuracy with local LLMs
- **Async/Await Fixes**: Fixed `asyncio.run()` issues in async contexts
- **Token Counting**: Added fallback for unknown tokenizer models
- **DocIndex Updates**: Improved keyword merging and document indexing logic

### 🐛 Bug Fixes
- Fixed `SyntaxError` with f-string expressions
- Fixed `ValueError` for unknown tokenizer encodings
- Fixed `RuntimeError` with nested event loops
- Improved error handling for API failures
- Better handling of empty or missing keywords

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for Ollama)

### Install Dependencies

#### Option 1: Using Conda (Recommended)

Create and activate the conda environment:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate pageindex
```

#### Option 2: Using pip

```bash
pip install -r requirements.txt
```

### Required Packages
- `openai>=1.101.0` - OpenAI API client
- `pymupdf>=1.26.4` - PDF processing
- `PyPDF2>=3.0.1` - PDF parsing
- `python-dotenv>=1.1.0` - Environment variable management
- `tiktoken>=0.11.0` - Token counting
- `pyyaml>=6.0.2` - YAML configuration
- `flask>=3.0.0` - Web framework for chatbot interface

### Optional: Ollama Setup

For local LLM support, see [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed instructions.

Quick setup:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```

## Configuration

Create a `.env` file in the project root:

```bash
# API Provider: "ollama", "openai", or "huggingface"
API_PROVIDER=ollama

# OpenAI Configuration (if using OpenAI)
CHATGPT_API_KEY=your_openai_api_key

# HuggingFace Configuration (if using HuggingFace)
HF_TOKEN=your_huggingface_token

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1:8b
```

## Usage

### Basic PDF Processing

```python
from pageindex import page_index_main
import argparse

# Create options
opt = argparse.Namespace(
    pdf_path="path/to/document.pdf",
    model="llama3.1:8b",  # or "gpt-4o-2024-11-20" for OpenAI
    # ... other options
)

# Process PDF and generate structure
toc_with_page_number = page_index_main(opt.pdf_path, opt)
```

### Command Line Usage

```bash
python run_pageindex.py --pdf_path path/to/document.pdf --model llama3.1:8b
```

### Markdown Processing

```python
from pageindex.page_index_md import md_to_tree

tree_structure = await md_to_tree(
    md_path="path/to/document.md",
    model="llama3.1:8b",
    if_add_node_summary='yes'
)
```

## Project Structure

```
PageIndex/
├── pageindex/              # Main package
│   ├── __init__.py
│   ├── page_index.py      # PDF processing and TOC extraction
│   ├── page_index_md.py   # Markdown processing
│   ├── utils.py           # Utility functions and API clients
│   └── config.yaml        # Configuration file
├── RAG/                    # RAG query scripts
│   ├── rag_query.py       # RAG query function
│   └── utils.py           # RAG utility functions
├── Interface/              # Web interface
│   ├── app.py             # Flask application
│   └── templates/         # HTML templates
├── run_pageindex.py       # Main entry point for PDF processing
├── test_similar.py        # Keyword similarity testing
├── cookbook/              # Example notebooks and tutorials
├── results/               # Generated document structures (gitignored)
├── requirements.txt       # Python dependencies (pip)
├── environment.yml        # Conda environment configuration
├── OLLAMA_SETUP.md        # Ollama setup guide
└── README.md              # This file
```

## Key Components

### `page_index.py`
- `page_index_main()`: Main entry point for PDF processing
- `process_toc_with_page_numbers()`: Processes TOC and maps to page numbers
- `generate_toc_init()`: Generates initial TOC structure from document
- `verify_toc()`: Validates TOC accuracy
- `check_title_appearance()`: Verifies if titles appear on assigned pages

### `utils.py`
- `ChatGPT_API()`: Synchronous API calls
- `ChatGPT_API_async()`: Asynchronous API calls
- `extract_json()`: Robust JSON extraction from LLM responses
- `count_tokens()`: Token counting with fallback support
- Provider abstraction for OpenAI, Ollama, and HuggingFace

### `run_pageindex.py`
- Main script for processing PDFs and markdown files
- DocIndex management and keyword extraction
- Command-line interface

## API Providers

### Ollama (Recommended for Local Use)
- **Pros**: No API costs, no usage limits, privacy, offline capable
- **Cons**: Requires local GPU, slower than cloud APIs
- **Best for**: Development, privacy-sensitive documents, high-volume processing

### OpenAI
- **Pros**: Fast, high quality, reliable
- **Cons**: API costs, usage limits
- **Best for**: Production, when quality is critical

### HuggingFace
- **Pros**: Free tier available, multiple models
- **Cons**: Rate limits, variable quality
- **Best for**: Testing, cost-sensitive applications

## Performance

With NVIDIA RTX 4090 and Ollama:
- **llama3.1:8b**: ~50-100 tokens/sec
- **llama3.1:70b**: ~10-20 tokens/sec (if fits in VRAM)
- **mistral:7b**: ~80-150 tokens/sec

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under a **Non-Commercial License** - see the [LICENSE](LICENSE) file for details.

**Important**: This license prohibits commercial use without explicit written permission. For commercial licensing inquiries, please contact the repository maintainer.

**Note**: This work is based on [PageIndex](https://github.com/VectifyAI/PageIndex), which is licensed under the MIT License by VectifyAI. This non-commercial license applies to the modifications and enhancements made in this repository.

## Acknowledgments

- **Original PageIndex Framework**: This project is based on and extends the [PageIndex](https://github.com/VectifyAI/PageIndex) framework by VectifyAI
- Built on top of the PageIndex framework
- Inspired by reasoning-based retrieval approaches
- Thanks to the Ollama team for local LLM support

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

## Credits

This project is a modified version of [PageIndex](https://github.com/VectifyAI/PageIndex) by [VectifyAI](https://github.com/VectifyAI), with the following enhancements:

- Enhanced Ollama integration for local LLM support
- Improved error handling and JSON parsing
- Better prompt engineering for local models
- Additional bug fixes and optimizations

**Original Repository**: [https://github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)  
**Original Website**: [https://pageindex.ai](https://pageindex.ai)

For the original PageIndex framework and documentation, please visit the [official repository](https://github.com/VectifyAI/PageIndex) or [pageindex.ai](https://pageindex.ai).
