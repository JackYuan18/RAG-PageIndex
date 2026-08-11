"""
Load chat / embedding models from llm_models.yaml (project root).

All indexing and RAG entry points should resolve models through this module.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "llm_models.yaml"


def _registry_path() -> Path:
    raw = os.getenv("LLM_MODELS_PATH", "").strip()
    return Path(raw) if raw else _DEFAULT_PATH


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    path = _registry_path()
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def reload_registry() -> None:
    """Clear cached registry (e.g. after editing llm_models.yaml in tests)."""
    _load_raw.cache_clear()


def get_ollama_base_url() -> str:
    raw = _load_raw()
    providers = raw.get("providers") or {}
    return os.getenv(
        "OLLAMA_BASE_URL",
        str(providers.get("ollama_base_url") or "http://localhost:11434/v1"),
    ).strip()


def get_aliases() -> dict[str, str]:
    raw = _load_raw()
    file_aliases = raw.get("aliases") or {}
    aliases: dict[str, str] = {}
    if isinstance(file_aliases, dict):
        for key, val in file_aliases.items():
            if key and val:
                aliases[str(key).strip()] = str(val).strip()

    # Env wins for common short names.
    if os.getenv("QWEN_MODEL"):
        aliases["qwen"] = os.getenv("QWEN_MODEL", "").strip()
    elif "qwen" not in aliases:
        aliases["qwen"] = "qwen2.5:14b"

    if os.getenv("OLLAMA_MODEL"):
        aliases["ollama"] = os.getenv("OLLAMA_MODEL", "").strip()
    elif "ollama" not in aliases:
        aliases["ollama"] = "llama3.1:8b"

    if "huggingface" not in aliases:
        aliases["huggingface"] = "openai/gpt-oss-120b:groq"

    return aliases


def resolve_chat_model(model: Optional[str]) -> str:
    """Resolve alias (qwen, ollama) or pass through raw API model name."""
    if not model:
        return ""
    name = str(model).strip()
    if not name:
        return ""
    return get_aliases().get(name, name)


def get_indexing_chat_model() -> str:
    env = os.getenv("PAGEINDEX_CHAT_MODEL", "").strip()
    if env:
        return env
    indexing = _load_raw().get("indexing") or {}
    return str(indexing.get("chat_model") or "qwen").strip()


def get_indexing_embed_model() -> str:
    env = os.getenv("PAGEINDEX_EMBED_MODEL", "").strip()
    if env:
        return resolve_ollama_tag(env)
    indexing = _load_raw().get("indexing") or {}
    return resolve_ollama_tag(str(indexing.get("embed_model") or "mxbai-embed-large").strip())


def get_rag_chat_model() -> str:
    env = os.getenv("RAG_CHAT_MODEL", "").strip()
    if env:
        return env
    rag = _load_raw().get("rag") or {}
    return str(rag.get("chat_model") or get_indexing_chat_model()).strip()


def get_rag_tree_search_model() -> Optional[str]:
    env = os.getenv("RAG_TREE_SEARCH_MODEL", "").strip()
    if env:
        return env
    rag = _load_raw().get("rag") or {}
    ts = rag.get("tree_search_model")
    if ts is None:
        return None
    s = str(ts).strip()
    if not s or s.lower() in {"null", "none", "~"}:
        return None
    return s


def get_rag_embed_model() -> str:
    env = os.getenv("RAG_EMBED_MODEL", "").strip()
    if env:
        return resolve_ollama_tag(env)
    rag = _load_raw().get("rag") or {}
    val = rag.get("keyword_embed_model")
    if val:
        return resolve_ollama_tag(str(val).strip())
    return get_indexing_embed_model()


def get_rag_max_tree_search_docs() -> Optional[int]:
    env = os.getenv("RAG_MAX_TREE_SEARCH_DOCS", "").strip()
    if env:
        try:
            n = int(env)
            return n if n > 0 else None
        except ValueError:
            return None
    rag = _load_raw().get("rag") or {}
    val = rag.get("max_tree_search_docs")
    if val is None:
        return None
    try:
        n = int(val)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def registry_path() -> str:
    return str(_registry_path())


@lru_cache(maxsize=1)
def get_installed_ollama_models() -> tuple[str, ...]:
    """Return Ollama model names from GET /api/tags (empty if Ollama unreachable)."""
    base = get_ollama_base_url().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/api/tags"
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8")) or {}
        models = data.get("models") or []
        names = []
        for item in models:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]).strip())
        return tuple(sorted(set(names)))
    except Exception:
        return tuple()


def resolve_ollama_tag(tag: Optional[str]) -> str:
    """
    Map a configured Ollama tag to an installed model name.
    Handles missing :latest suffix and exact tag matches.
    """
    if not tag:
        return ""
    want = str(tag).strip()
    if not want:
        return ""
    installed = get_installed_ollama_models()
    if not installed:
        return want
    if want in installed:
        return want
    latest = f"{want}:latest"
    if latest in installed:
        return latest
    for name in installed:
        if name.split(":")[0] == want.split(":")[0]:
            return name
    return want


def _is_ollama_chat_alias(model: Optional[str], resolved: str) -> bool:
    if resolved.startswith("gpt-") or "/" in resolved:
        return False
    if model in ("ollama", "qwen"):
        return True
    aliases = get_aliases()
    if model and str(model).strip() in aliases:
        return True
    if resolved in aliases.values():
        return True
    if "/" in resolved or resolved.startswith("gpt-"):
        return False
    return bool(get_installed_ollama_models())


def resolve_chat_model_for_api(model: Optional[str]) -> str:
    """Resolve alias then map to an installed Ollama tag when applicable."""
    resolved = resolve_chat_model(model)
    if _is_ollama_chat_alias(model, resolved):
        return resolve_ollama_tag(resolved)
    return resolved


def resolve_embed_model_for_api(tag: Optional[str]) -> str:
    """Resolve embedding model tag to installed Ollama name."""
    name = str(tag or "").strip() or get_indexing_embed_model()
    return resolve_ollama_tag(name)


def uses_ollama_chat_provider(model: Optional[str]) -> bool:
    """True when chat completions should go to Ollama (local aliases/tags)."""
    if model == "huggingface":
        return False
    m = str(model or "").strip()
    if not m:
        return True
    resolved = resolve_chat_model(m)
    if resolved.startswith("gpt-") or "/" in resolved:
        return False
    if m in ("ollama", "qwen") or m in get_aliases():
        return True
    # Raw Ollama tag passed on CLI (e.g. qwen2.5:14b)
    if ":" in m:
        return True
    return False


def uses_openai_chat_provider(model: Optional[str]) -> bool:
    """True when chat completions should use OpenAI API (CHATGPT_API_KEY)."""
    if model == "huggingface":
        return False
    m = str(model or "").strip()
    if not m:
        return False
    resolved = resolve_chat_model(m)
    return resolved.startswith("gpt-")


def validate_configured_models() -> list[str]:
    """Return warning strings for chat/embed models not found in local Ollama."""
    warnings: list[str] = []
    installed = get_installed_ollama_models()
    if not installed:
        return warnings

    def _check(label: str, raw: str, resolved_api: str) -> None:
        if resolved_api in installed:
            return
        base = resolved_api.split(":")[0]
        if any(n.split(":")[0] == base for n in installed):
            return
        if "/" in resolved_api or resolved_api.startswith("gpt-"):
            return  # remote API, not Ollama
        warnings.append(
            f"{label}: configured '{raw}' -> '{resolved_api}' not in Ollama "
            f"(installed: {', '.join(installed)})"
        )

    idx_chat = get_indexing_chat_model()
    _check("indexing.chat_model", idx_chat, resolve_chat_model_for_api(idx_chat))

    rag_chat = get_rag_chat_model()
    _check("rag.chat_model", rag_chat, resolve_chat_model_for_api(rag_chat))

    rag_ts = get_rag_tree_search_model()
    if rag_ts:
        _check("rag.tree_search_model", rag_ts, resolve_chat_model_for_api(rag_ts))

    idx_emb = get_indexing_embed_model()
    _check("indexing.embed_model", idx_emb, resolve_embed_model_for_api(idx_emb))

    rag_emb = get_rag_embed_model()
    _check("rag.keyword_embed_model", rag_emb, resolve_embed_model_for_api(rag_emb))

    return warnings
