#!/usr/bin/env python3
"""
Web UI to manage Database documents, DocIndex.json, and structure files.

Runs build_docindex.py (full reconstruction) and run_pageindex.py per document
with --update_docindex (structure + DocIndex).
"""

from __future__ import annotations

import json
import os
import re
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
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pageindex.model_registry import get_indexing_chat_model

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("DOCINDEX_MAX_UPLOAD_MB", "100")) * 1024 * 1024

RESULTS_DIR = os.path.join(project_root, "results")
DATABASE_DIR = os.path.join(project_root, "Database")
DOCINDEX_PATH = os.path.join(RESULTS_DIR, "DocIndex.json")

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".md"}

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _real_database_dir() -> str:
    return os.path.realpath(DATABASE_DIR)


def _is_safe_database_filename(filename: str) -> bool:
    """Check basename is safe for a path under Database/ (file may not exist yet)."""
    if not filename or filename != os.path.basename(filename):
        return False
    if ".." in filename or filename.startswith(("/", "\\")):
        return False
    full = os.path.realpath(os.path.join(DATABASE_DIR, filename))
    return full.startswith(_real_database_dir() + os.sep)


def _is_safe_database_file(filename: str) -> bool:
    return _is_safe_database_filename(filename) and os.path.isfile(
        os.path.join(DATABASE_DIR, filename)
    )


def _sanitize_upload_filename(original: str) -> Optional[str]:
    """Return a safe basename for Database/ or None if invalid."""
    if not original:
        return None
    name = os.path.basename(original.strip())
    if not name or ".." in name or name.startswith(("/", "\\")):
        return None
    base, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return None
    safe_base = re.sub(r"[^\w\s.\-()]", "", base).strip().strip(".")
    if not safe_base:
        safe_base = "document"
    safe_name = f"{safe_base}{ext}"
    if safe_name != os.path.basename(safe_name):
        return None
    return safe_name


def _unique_database_path(filename: str) -> str:
    """Path under Database/; suffix _1, _2, … if the name already exists."""
    dest = os.path.join(DATABASE_DIR, filename)
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(filename)
    for n in range(1, 1000):
        candidate = f"{base}_{n}{ext}"
        dest = os.path.join(DATABASE_DIR, candidate)
        if not os.path.exists(dest):
            return dest
    raise ValueError(f"Could not find unique name for {filename!r}")


@app.errorhandler(413)
def _request_entity_too_large(_e):
    max_mb = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
    return jsonify(
        {"success": False, "error": f"File too large (max {max_mb} MB)"}
    ), 413


def _structure_basename(stem: str) -> str:
    return f"{stem}_structure.json"


def _docindex_path_matches_stem(path_str: str, stem: str) -> bool:
    if not path_str:
        return False
    return os.path.basename(str(path_str).replace("\\", "/")) == _structure_basename(stem)


def _load_docindex_as_sets() -> Dict[str, Set[str]]:
    if not os.path.isfile(DOCINDEX_PATH):
        return {}
    try:
        with open(DOCINDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Set[str]] = {}
    for kw, value in data.items():
        if isinstance(value, set):
            out[kw] = {str(v) for v in value if v}
        elif isinstance(value, list):
            out[kw] = {str(v) for v in value if v}
        elif isinstance(value, str) and value:
            out[kw] = {value}
        else:
            out[kw] = set()
    return out


def _save_docindex_from_sets(doc_index: Dict[str, Set[str]]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    serializable = {kw: sorted(vals) for kw, vals in sorted(doc_index.items())}
    with open(DOCINDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def _remove_stem_from_docindex(stem: str) -> Dict[str, int]:
    """Drop all DocIndex keyword entries pointing at this document's structure file."""
    stats = {
        "keywords_removed": 0,
        "keyword_entries_updated": 0,
        "structure_refs_removed": 0,
    }
    doc_index = _load_docindex_as_sets()
    if not doc_index:
        return stats

    keys_to_remove: List[str] = []
    for key, val_set in list(doc_index.items()):
        before = len(val_set)
        val_set = {p for p in val_set if not _docindex_path_matches_stem(p, stem)}
        removed = before - len(val_set)
        if removed:
            stats["structure_refs_removed"] += removed
            stats["keyword_entries_updated"] += 1
            doc_index[key] = val_set
        if not doc_index[key]:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del doc_index[key]
        stats["keywords_removed"] += 1

    if stats["structure_refs_removed"] or keys_to_remove:
        _save_docindex_from_sets(doc_index)

    return stats


def _job_running_for_filename(filename: str) -> bool:
    with _jobs_lock:
        for job in _jobs.values():
            if job.get("status") != "running":
                continue
            haystack = " ".join(
                [str(job.get("label") or "")] + [str(c) for c in (job.get("cmd") or [])]
            )
            if filename in haystack:
                return True
    return False


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


@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith("/api/"):
        return jsonify(
            {
                "success": False,
                "error": "API endpoint not found. Restart the DocIndex server to load the latest code.",
            }
        ), 404
    return e


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


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Save an uploaded PDF or Markdown file into Database/."""
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"success": False, "error": "No file provided"}), 400
    if not upload.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    safe_name = _sanitize_upload_filename(upload.filename)
    if not safe_name:
        return jsonify(
            {
                "success": False,
                "error": "Only PDF (.pdf) and Markdown (.md) files are allowed",
            }
        ), 400

    os.makedirs(DATABASE_DIR, exist_ok=True)
    try:
        dest_path = _unique_database_path(safe_name)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    saved_name = os.path.basename(dest_path)
    if not _is_safe_database_filename(saved_name):
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    try:
        upload.save(dest_path)
    except OSError as e:
        return jsonify({"success": False, "error": f"Could not save file: {e}"}), 500

    renamed = saved_name != safe_name
    return jsonify(
        {
            "success": True,
            "filename": saved_name,
            "original_filename": upload.filename,
            "renamed": renamed,
            "message": (
                f"Saved as {saved_name} (name already existed)"
                if renamed
                else f"Uploaded {saved_name}"
            ),
        }
    )


@app.route("/api/remove-document", methods=["POST"])
def api_remove_document():
    """Remove a document from Database/, results/, and DocIndex.json."""
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()
    delete_file = bool(data.get("delete_file", True))
    delete_structure = bool(data.get("delete_structure", True))

    if not filename or not _is_safe_database_filename(filename):
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    if _job_running_for_filename(filename):
        return jsonify(
            {"success": False, "error": "A job is running for this document. Wait for it to finish."}
        ), 409

    stem = os.path.splitext(filename)[0]
    db_path = os.path.join(DATABASE_DIR, filename)
    structure_path = os.path.join(RESULTS_DIR, _structure_basename(stem))

    file_exists = os.path.isfile(db_path)
    structure_exists = os.path.isfile(structure_path)
    in_docindex = stem in _indexed_stems_from_docindex()

    if not (file_exists or structure_exists or in_docindex):
        return jsonify({"success": False, "error": "Document not found"}), 404

    removed: Dict[str, Any] = {
        "filename": filename,
        "stem": stem,
        "database_file_deleted": False,
        "structure_file_deleted": False,
        "docindex": {},
    }

    removed["docindex"] = _remove_stem_from_docindex(stem)

    if delete_structure and structure_exists:
        try:
            os.remove(structure_path)
            removed["structure_file_deleted"] = True
        except OSError as e:
            return jsonify(
                {"success": False, "error": f"Could not delete structure file: {e}"}
            ), 500

    if delete_file and file_exists:
        try:
            os.remove(db_path)
            removed["database_file_deleted"] = True
        except OSError as e:
            return jsonify(
                {"success": False, "error": f"Could not delete database file: {e}"}
            ), 500

    return jsonify(
        {
            "success": True,
            "message": f"Removed {filename}",
            "removed": removed,
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
    mode = (data.get("mode") or "update_structure").strip()
    model = (data.get("model") or get_indexing_chat_model()).strip() or get_indexing_chat_model()

    if mode != "update_structure":
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
        "--update_docindex",
    ]

    job_id = _start_job(cmd, f"run_pageindex --update_docindex: {filename}")
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

    def _open_browser_opt_in() -> bool:
        return os.environ.get("FLASK_OPEN_BROWSER", "").strip().lower() in ("1", "true", "yes", "on")

    def open_browser() -> None:
        if not _open_browser_opt_in():
            return
        time.sleep(1.5)
        print(f"Opening browser at {open_url}...")
        env = os.environ.copy()
        env.setdefault("NO_AT_BRIDGE", "1")
        try:
            import subprocess
            subprocess.Popen(["xdg-open", open_url], env=env, start_new_session=True)
        except (FileNotFoundError, OSError):
            os.environ.setdefault("NO_AT_BRIDGE", "1")
            webbrowser.open(open_url)

    if _open_browser_opt_in():
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"DocIndex interface: {open_url}")
    if not _open_browser_opt_in():
        print("Tip: set FLASK_OPEN_BROWSER=1 to auto-open the browser on start.")
    print(f"Database: {DATABASE_DIR}")
    print(f"DocIndex: {DOCINDEX_PATH}")
    app.run(debug=True, host=host, port=port, use_reloader=False)
