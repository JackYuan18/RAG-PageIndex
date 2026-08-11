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
from RAG.utils import load_docindex, warm_docindex_keyword_embeddings, _default_ollama_embed_model
from pageindex.model_registry import get_rag_chat_model, get_rag_embed_model

load_dotenv()

_warm_lock = threading.Lock()
_warm_started = False


def _warm_keyword_embeddings_background():
    """Pre-embed DocIndex keywords so Step 1 only embeds the user query."""
    global _warm_started
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True
    try:
        if not os.path.exists(DOCINDEX_PATH):
            return
        doc_index = load_docindex(DOCINDEX_PATH, quiet=True)
        n = warm_docindex_keyword_embeddings(
            doc_index,
            embed_model=get_rag_embed_model(),
        )
        print(
            f"DocIndex keyword embedding warm-up complete "
            f"({n} new embedding(s), model={get_rag_embed_model()})."
        )
    except Exception as e:
        print(f"Warning: DocIndex keyword warm-up failed: {e}")

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
        conversation_history = data.get('history') or data.get('conversation_history')
        model = get_rag_chat_model()

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
                        rag_query(
                            query_text,
                            model=model,
                            doc_index_path=DOCINDEX_PATH,
                            progress_callback=progress_callback,
                            conversation_history=conversation_history,
                        )
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
    else:
        threading.Thread(target=_warm_keyword_embeddings_background, daemon=True).start()
    
    # Get port from environment variable or use default
    port = int(os.getenv('FLASK_PORT', 5001))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    # Determine the URL to open
    if host == '0.0.0.0':
        url = f'http://localhost:{port}'
    else:
        url = f'http://{host}:{port}'

    def _open_browser_opt_in() -> bool:
        return os.getenv("FLASK_OPEN_BROWSER", "").strip().lower() in ("1", "true", "yes", "on")

    def open_browser():
        if not _open_browser_opt_in():
            return
        time.sleep(1.5)  # Wait for Flask to start
        print(f"Opening browser at {url}...")
        # Avoid Linux GTK atk-bridge warning when spawning the default browser.
        env = os.environ.copy()
        env.setdefault("NO_AT_BRIDGE", "1")
        try:
            import subprocess
            subprocess.Popen(["xdg-open", url], env=env, start_new_session=True)
        except (FileNotFoundError, OSError):
            os.environ.setdefault("NO_AT_BRIDGE", "1")
            webbrowser.open(url)

    if _open_browser_opt_in():
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the Flask app
    print(f"Starting chatbot interface...")
    print(f"DocIndex path: {DOCINDEX_PATH}")
    print(f"Open in browser: {url}")
    if not _open_browser_opt_in():
        print("Tip: set FLASK_OPEN_BROWSER=1 to auto-open the browser on start.")
    
    app.run(debug=True, host=host, port=port, use_reloader=False)
