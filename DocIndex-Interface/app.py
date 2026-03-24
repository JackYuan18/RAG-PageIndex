#!/usr/bin/env python3
"""
Web UI to manage Database documents, DocIndex.json, and structure files.

Runs build_docindex.py (full reconstruction) and run_pageindex.py per document
(Add to DocIndex / Update Structure).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from typing import Any, Dict, List, Optional, Set

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

RESULTS_DIR = os.path.join(project_root, "results")
DATABASE_DIR = os.path.join(project_root, "Database")
DOCINDEX_PATH = os.path.join(RESULTS_DIR, "DocIndex.json")

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _real_database_dir() -> str:
    return os.path.realpath(DATABASE_DIR)


def _is_safe_database_file(filename: str) -> bool:
    if not filename or filename != os.path.basename(filename):
        return False
    if ".." in filename or filename.startswith(("/", "\\")):
        return False
    full = os.path.realpath(os.path.join(DATABASE_DIR, filename))
    return full.startswith(_real_database_dir() + os.sep) and os.path.isfile(full)


def _indexed_stems_from_docindex() -> Set[str]:
    """Map structure JSON basenames to document stems (e.g. Foo_structure.json -> Foo)."""
    stems: Set[str] = set()
    if not os.path.isfile(DOCINDEX_PATH):
        return stems
    try:
        with open(DOCINDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return stems
    if not isinstance(data, dict):
        return stems
    for _kw, paths in data.items():
        items = paths if isinstance(paths, list) else [paths]
        for p in items:
            if not p:
                continue
            base = os.path.basename(str(p).replace("\\", "/"))
            if base.endswith("_structure.json"):
                stems.add(base[: -len("_structure.json")])
    return stems


def _docindex_stats() -> Dict[str, Any]:
    exists = os.path.isfile(DOCINDEX_PATH)
    out: Dict[str, Any] = {
        "path": DOCINDEX_PATH,
        "exists": exists,
        "keyword_count": 0,
        "indexed_document_stems": [],
    }
    if not exists:
        return out
    try:
        with open(DOCINDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    out["keyword_count"] = len(data)
    stems = sorted(_indexed_stems_from_docindex())
    out["indexed_document_stems"] = stems
    return out


def _list_database_documents() -> List[Dict[str, Any]]:
    os.makedirs(DATABASE_DIR, exist_ok=True)
    indexed = _indexed_stems_from_docindex()
    rows: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(DATABASE_DIR))
    except OSError:
        names = []
    for name in names:
        lower = name.lower()
        if not (lower.endswith(".pdf") or lower.endswith(".md")):
            continue
        stem = os.path.splitext(name)[0]
        structure_name = f"{stem}_structure.json"
        structure_path = os.path.join(RESULTS_DIR, structure_name)
        rows.append(
            {
                "filename": name,
                "stem": stem,
                "kind": "pdf" if lower.endswith(".pdf") else "md",
                "structure_filename": structure_name,
                "structure_exists": os.path.isfile(structure_path),
                "in_docindex": stem in indexed,
                "open_url": f"/database/{name}",
            }
        )
    return rows


def _start_job(cmd: List[str], label: str) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "label": label,
            "status": "running",
            "cmd": cmd,
            "output": "",
            "returncode": None,
            "error": None,
        }

    def run() -> None:
        proc: Optional[subprocess.Popen[str]] = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            acc: List[str] = []
            if proc.stdout is not None:
                for line in proc.stdout:
                    acc.append(line)
                    blob = "".join(acc)
                    if len(blob) > 200_000:
                        blob = blob[-200_000:]
                        acc = [blob]
                    with _jobs_lock:
                        j = _jobs.get(job_id)
                        if j:
                            j["output"] = "".join(acc)
            rc = proc.wait()
            with _jobs_lock:
                j = _jobs.get(job_id)
                if j:
                    j["returncode"] = rc
                    j["status"] = "done" if rc == 0 else "failed"
        except Exception as e:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            with _jobs_lock:
                j = _jobs.get(job_id)
                if j:
                    j["status"] = "failed"
                    j["error"] = str(e)

    threading.Thread(target=run, daemon=True).start()
    return job_id


@app.route("/")
def index():
    return render_template("docindex.html")


@app.route("/database/<path:filename>")
def serve_database_file(filename: str):
    if not _is_safe_database_file(filename):
        abort(404)
    return send_from_directory(DATABASE_DIR, filename, as_attachment=False)


@app.route("/api/overview", methods=["GET"])
def api_overview():
    return jsonify(
        {
            "project_root": project_root,
            "database_dir": DATABASE_DIR,
            "results_dir": RESULTS_DIR,
            "docindex": _docindex_stats(),
            "documents": _list_database_documents(),
        }
    )


@app.route("/api/reconstruct", methods=["POST"])
def api_reconstruct():
    py = sys.executable
    build_script = os.path.join(project_root, "build_docindex.py")
    cmd = [
        py,
        build_script,
        "--database-dir",
        DATABASE_DIR,
        "--output-dir",
        RESULTS_DIR,
    ]
    job_id = _start_job(cmd, "DocIndex reconstruction (build_docindex.py)")
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/process-document", methods=["POST"])
def api_process_document():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or ""
    mode = data.get("mode") or ""
    model = (data.get("model") or "qwen").strip() or "qwen"

    if mode not in ("add_to_docindex", "update_structure"):
        return jsonify({"success": False, "error": "Invalid mode"}), 400

    if not _is_safe_database_file(filename):
        return jsonify({"success": False, "error": "File not found or invalid"}), 400

    lower = filename.lower()
    if not lower.endswith(".pdf"):
        return jsonify(
            {
                "success": False,
                "error": "Only PDF files are supported for indexing in this project.",
            }
        ), 400

    abs_path = os.path.realpath(os.path.join(DATABASE_DIR, filename))
    py = sys.executable
    run_script = os.path.join(project_root, "run_pageindex.py")
    cmd = [
        py,
        run_script,
        "--pdf_path",
        abs_path,
        "--model",
        model,
    ]
    if mode == "add_to_docindex":
        cmd.append("--update_docindex")

    label = (
        "Add to DocIndex"
        if mode == "add_to_docindex"
        else "Update structure (no DocIndex merge)"
    )
    job_id = _start_job(cmd, f"{label}: {filename}")
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET"])
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Unknown job"}), 404
    return jsonify({"success": True, "job": job})


if __name__ == "__main__":
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    port = int(os.environ.get("DOCINDEX_FLASK_PORT", "5002"))
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    open_url = f"http://localhost:{port}" if host == "0.0.0.0" else f"http://{host}:{port}"

    def open_browser() -> None:
        time.sleep(1.5)
        print(f"Opening browser at {open_url}...")
        webbrowser.open(open_url)

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"DocIndex interface: {open_url}")
    print(f"Database: {DATABASE_DIR}")
    print(f"DocIndex: {DOCINDEX_PATH}")
    app.run(debug=True, host=host, port=port)
