#!/usr/bin/env python3
"""
Web-based Chatbot Interface for PageIndex RAG

This Flask application provides a web interface for querying documents
using the PageIndex RAG system.
"""

import os
import sys
import asyncio
import json
import re
import webbrowser
import threading
import time
import queue
from urllib.parse import quote
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory, abort
from dotenv import load_dotenv

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RAG query function
from RAG.rag_query import rag_query

load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Support non-ASCII characters

# Configuration
RESULTS_DIR = os.path.join(project_root, 'results')
DATABASE_DIR = os.path.join(project_root, 'Database')
DOCINDEX_PATH = os.path.join(RESULTS_DIR, 'DocIndex.json')

@app.route('/docs/<path:filename>')
def serve_doc(filename: str):
    """
    Serve documents referenced by sources links.

    - PDFs and original documents: served from `Database/`
    - Generated artifacts (e.g., *_structure.json): served from `results/`
    """
    # send_from_directory guards against path traversal.
    if not filename or filename.strip() == "":
        abort(404)

    # Prefer original documents when present (e.g., PDFs).
    db_candidate = os.path.join(DATABASE_DIR, filename)
    if os.path.isfile(db_candidate):
        return send_from_directory(DATABASE_DIR, filename, as_attachment=False)

    results_candidate = os.path.join(RESULTS_DIR, filename)
    if os.path.isfile(results_candidate):
        return send_from_directory(RESULTS_DIR, filename, as_attachment=False)

    abort(404)


@app.route('/')
def index():
    """Render the main chatbot interface."""
    return render_template('chatbot.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests with progress streaming."""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        model_choice = data.get('model', 'qwen')  # Value from dropdown

        # Map dropdown choices to actual model strings / providers
        if model_choice == 'ollama':
            # Use default Ollama model from environment (handled in utils)
            model = 'ollama'
        elif model_choice == 'qwen':
            # Use a sensible default Qwen model (served via Ollama or remote)
            model = 'qwen'
        elif model_choice == 'huggingface':
            # Use HuggingFace router (handled in utils.get_model_name)
            model = 'huggingface'
        elif model_choice == 'openai gpt':
            # Default OpenAI GPT model
            model = 'openai'
        else:
            # Fallback to raw value
            model = model_choice
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Create a queue to collect progress messages
        progress_queue = queue.Queue()
        result_container = {'result': None, 'error': None}
        
        def progress_callback(message):
            """Callback to send progress updates."""
            # Detect if this is a new step (contains "Step X:" pattern)
            is_new_step = bool(re.search(r'Step\s+\d+:', message))
            progress_queue.put({
                'type': 'progress', 
                'message': message,
                'clear_previous': is_new_step
            })
        
        def run_rag_query():
            """Run RAG query in a separate thread."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        rag_query(query_text, model=model, doc_index_path=DOCINDEX_PATH, progress_callback=progress_callback)
                    )
                    result_container['result'] = result
                    progress_queue.put({'type': 'complete'})
                finally:
                    loop.close()
            except Exception as e:
                result_container['error'] = str(e)
                progress_queue.put({'type': 'error', 'message': str(e)})
        
        # Start RAG query in background thread
        query_thread = threading.Thread(target=run_rag_query)
        query_thread.daemon = True
        query_thread.start()
        
        def generate():
            """Generate Server-Sent Events stream."""
            while True:
                try:
                    # Get progress update with timeout
                    item = progress_queue.get(timeout=0.1)
                    
                    if item['type'] == 'complete':
                        # Send final result
                        result = result_container['result']
                        if result and 'error' in result:
                            yield f"data: {json.dumps({'type': 'error', 'error': result['error']})}\n\n"
                        else:
                            # Attach clickable sources for UI rendering (for inline citations + Sources section).
                            sources = result.get("sources") or []
                            if not sources:
                                try:
                                    retrieved = result.get("retrieved_contexts") or []
                                    paths = [r.get("doc_path") for r in retrieved if isinstance(r, dict)]
                                    if not any(paths):
                                        paths = result.get("matched_documents") or []
                                    seen = set()
                                    for p in paths:
                                        if not p:
                                            continue
                                        base = os.path.basename(str(p))
                                        if not base or base in seen:
                                            continue
                                        seen.add(base)
                                        sources.append({"name": base, "url": f"/docs/{quote(base)}"})
                                except Exception:
                                    pass
                            result["sources"] = sources
                            yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
                        break
                    elif item['type'] == 'error':
                        yield f"data: {json.dumps({'type': 'error', 'error': item['message']})}\n\n"
                        break
                    else:
                        # Send progress update
                        yield f"data: {json.dumps({'type': 'progress', 'message': item['message'], 'clear_previous': item.get('clear_previous', False)})}\n\n"
                        
                except queue.Empty:
                    # Check if thread is still alive
                    if not query_thread.is_alive():
                        # Thread finished but no completion message - check for error
                        if result_container['error']:
                            yield f"data: {json.dumps({'type': 'error', 'error': result_container['error']})}\n\n"
                        elif result_container['result']:
                            result = result_container['result']
                            if result and 'error' in result:
                                yield f"data: {json.dumps({'type': 'error', 'error': result['error']})}\n\n"
                            else:
                                yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
                        break
                    continue
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if DocIndex is available."""
    docindex_exists = os.path.exists(DOCINDEX_PATH)
    return jsonify({
        'docindex_available': docindex_exists,
        'docindex_path': DOCINDEX_PATH
    })


if __name__ == '__main__':
    # Check if DocIndex exists
    if not os.path.exists(DOCINDEX_PATH):
        print(f"Warning: DocIndex not found at {DOCINDEX_PATH}")
        print("Please run run_pageindex.py first to generate document structures and DocIndex.")
    
    # Get port from environment variable or use default
    port = int(os.getenv('FLASK_PORT', 5001))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    # Determine the URL to open
    if host == '0.0.0.0':
        url = f'http://localhost:{port}'
    else:
        url = f'http://{host}:{port}'
    
    # Function to open browser after a short delay
    def open_browser():
        time.sleep(1.5)  # Wait for Flask to start
        print(f"Opening browser at {url}...")
        webbrowser.open(url)
    
    # Start browser opening in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the Flask app
    print(f"Starting chatbot interface...")
    print(f"DocIndex path: {DOCINDEX_PATH}")
    print(f"Server will open at {url}")
    
    app.run(debug=True, host=host, port=port)
