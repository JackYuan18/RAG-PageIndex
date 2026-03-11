# PageIndex RAG Web Chatbot Interface

A web-based chatbot interface for querying documents using the PageIndex RAG system.

## Features

- 🎨 Modern, responsive web interface
- 💬 Real-time chat interface
- 📚 Query documents indexed by PageIndex
- 🔍 Shows matched documents and retrieved nodes
- ⚡ Fast and efficient document retrieval

## Installation

Make sure you have Flask installed:

```bash
pip install flask
```

Or install all requirements from the project root:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate DocIndex First

Before using the chatbot, make sure you have generated document structures and DocIndex:

```bash
python run_pageindex.py --pdf_path path/to/document.pdf
```

### 2. Start the Web Server

From the project root:

```bash
python Interface/app.py
```

Or from the Interface directory:

```bash
cd Interface
python app.py
```

### 3. Open in Browser

Open your web browser and navigate to:

```
http://localhost:5000
```

## Configuration

The chatbot uses the same environment variables as the RAG system:

- `API_PROVIDER`: Set to "ollama", "openai", or "huggingface" (default: "ollama")
- `OLLAMA_MODEL`: Model name for Ollama (default: "llama3.1:8b")
- `CHATGPT_API_KEY`: OpenAI API key (if using OpenAI)
- `HF_TOKEN`: HuggingFace token (if using HuggingFace)

Create a `.env` file in the project root with your configuration.

## API Endpoints

### `GET /`
Main chatbot interface page.

### `POST /api/query`
Submit a query to the RAG system.

**Request Body:**
```json
{
    "query": "Your question here",
    "model": "llama3.1:8b"  // Optional, defaults to environment setting
}
```

**Response:**
```json
{
    "success": true,
    "query": "Your question",
    "answer": "Generated answer...",
    "matched_documents": ["path/to/doc1.json", "path/to/doc2.json"],
    "retrieved_nodes": [
        {
            "path": "path/to/doc.json",
            "node_ids": ["0001", "0002"],
            "thinking": "Reasoning process..."
        }
    ],
    "context_length": 1234
}
```

### `GET /api/status`
Check if DocIndex is available and get system status.

**Response:**
```json
{
    "docindex_available": true,
    "docindex_path": "/path/to/DocIndex",
    "default_model": "llama3.1:8b"
}
```

## Troubleshooting

### DocIndex Not Found

If you see "DocIndex Not Found" in the status:
1. Make sure you've run `run_pageindex.py` to generate document structures
2. Check that `./results/DocIndex` exists
3. Verify the path in the status message

### Port Already in Use

If port 5000 is already in use, modify `app.py` to use a different port:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Model Not Responding

- Check that your API provider is correctly configured
- For Ollama: Make sure Ollama is running (`ollama serve`)
- For OpenAI: Verify your API key is set correctly
- Check the terminal output for error messages

## Customization

### Change Port

Edit `app.py` and modify the port in the `app.run()` call.

### Modify UI

Edit `templates/chatbot.html` to customize the appearance and behavior.

### Add Features

The Flask app structure makes it easy to add:
- Chat history persistence
- Multiple document selection
- Export conversations
- User authentication

## License

Same as the main PageIndex project.
