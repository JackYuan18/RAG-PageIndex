import tiktoken
import openai
import logging
import os
import hashlib
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config
import re
from difflib import SequenceMatcher
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
# API Provider: "ollama", "openai", "huggingface", or None (auto-detect)
# API_PROVIDER = os.getenv("API_PROVIDER", "ollama").lower()  # Default to Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # Default model for Ollama
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:70b")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5:14b")

# Defaults for Ollama embedding chunking (overridden by env PAGEINDEX_EMBED_* when set).
PAGEINDEX_EMBED_MAX_CHARS = 512
PAGEINDEX_EMBED_BATCH = 16
def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip()
    return s if s != "" else default

def _normalize_choice(value: str | None, allowed: set[str], default: str) -> str:
    if not value:
        return default
    v = str(value).strip().lower()
    return v if v in allowed else default

def get_summary_method(opt=None) -> str:
    """
    Leaf-node summarization backend: read opt.summary_method (set via config, CLI, or ConfigLoader env merge).
    """
    if opt is None:
        return "llm"
    return _normalize_choice(getattr(opt, "summary_method", None), {"llm", "mmr"}, "llm")

def get_parent_summary_method(opt=None) -> str:
    """Parent-node summarization backend: read opt.parent_summary_method."""
    if opt is None:
        return "llm"
    return _normalize_choice(getattr(opt, "parent_summary_method", None), {"llm", "mmr"}, "llm")


def get_abstract_method(opt=None) -> str:
    """Document abstract backend: read opt.abstract_method."""
    if opt is None:
        return "llm"
    return _normalize_choice(getattr(opt, "abstract_method", None), {"llm", "mmr"}, "llm")


def get_keyword_method(opt=None) -> str:
    """
    Document keyword generation backend.
    - llm: generate keywords from doc_abstract text (existing behavior)
    - rich: generate keywords from richer context (title + section titles + snippets)
    """
    if opt is None:
        return "llm"
    return _normalize_choice(getattr(opt, "keyword_method", None), {"llm", "rich", "embed_mmr"}, "rich")


def _apply_method_env_overrides_to_merged(
    merged: dict, explicit_user_keys: set
) -> dict:
    """
    After merging defaults + user dict, apply PAGEINDEX_* env vars only for keys the user
    did not set explicitly (CLI / programmatic user_dict).
    """
    out = dict(merged)
    if "summary_method" not in explicit_user_keys:
        v = _env_str("PAGEINDEX_NODE_SUMMARY_METHOD") or _env_str("PAGEINDEX_SUMMARY_METHOD")
        if v:
            out["summary_method"] = v
    if "parent_summary_method" not in explicit_user_keys:
        v = _env_str("PAGEINDEX_PARENT_SUMMARY_METHOD")
        if v:
            out["parent_summary_method"] = v
    if "abstract_method" not in explicit_user_keys:
        v = _env_str("PAGEINDEX_ABSTRACT_METHOD")
        if v:
            out["abstract_method"] = v
    if "keyword_method" not in explicit_user_keys:
        v = _env_str("PAGEINDEX_KEYWORD_METHOD")
        if v:
            out["keyword_method"] = v
    return out


def _mmr_near_dup_ratio_value(opt) -> float:
    """SequenceMatcher ratio threshold in [0.5, 1.0]; higher = stricter (fewer merges)."""
    env = _env_str("PAGEINDEX_MMR_NEAR_DUP_RATIO")
    if env is not None:
        try:
            r = float(env)
        except ValueError:
            r = 0.92
    else:
        try:
            r = float(getattr(opt, "mmr_near_dup_ratio", 0.92) or 0.92)
        except (TypeError, ValueError):
            r = 0.92
    return max(0.5, min(1.0, r))


def merge_near_duplicate_strings(strings: list[str], opt=None) -> list[str]:
    """
    Drop near-duplicate strings using difflib similarity (longest-first as canonical).
    Exact duplicates are already removed earlier; this catches minor wording/spacing variants.
    """
    if not strings:
        return strings
    ratio = _mmr_near_dup_ratio_value(opt)
    sorted_s = sorted(
        [str(s).strip() for s in strings if s and str(s).strip()],
        key=lambda x: len(x),
        reverse=True,
    )
    kept: list[str] = []
    for s in sorted_s:
        dup = False
        for k in kept:
            if SequenceMatcher(None, s, k).ratio() >= ratio:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


def merge_near_duplicate_candidates(candidates: list[dict], opt=None) -> list[dict]:
    """Same as merge_near_duplicate_strings but preserves dict metadata for the kept canonical row."""
    if not candidates:
        return candidates
    ratio = _mmr_near_dup_ratio_value(opt)
    items = [
        c
        for c in candidates
        if isinstance(c, dict) and str(c.get("snippet", "")).strip()
    ]
    items.sort(key=lambda c: len(str(c.get("snippet", ""))), reverse=True)
    kept: list[dict] = []
    for cand in items:
        s = str(cand.get("snippet", "")).strip()
        dup = False
        for k in kept:
            ks = str(k.get("snippet", "")).strip()
            if SequenceMatcher(None, s, ks).ratio() >= ratio:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def get_openai_client(model):
    """Get OpenAI client configured for the selected provider."""
        
    if model == "ollama":
        # Ollama uses OpenAI-compatible API, no API key needed
        return openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # Ollama doesn't require a real API key
        )
    elif model == "qwen":
        return openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="qwen"
        )
    elif model == "huggingface":
        return openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key= HF_TOKEN
        )
    else:  # Default to OpenAI
        return openai.OpenAI(api_key= CHATGPT_API_KEY)

def get_async_openai_client(model):
    """Get async OpenAI client configured for the selected provider."""
    if model == "ollama":
        return openai.AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"
        )
    elif model == "qwen":
        return openai.AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="qwen"
        )
    elif model == "huggingface":
        return openai.AsyncOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )
    else:  # Default to OpenAI
        return openai.AsyncOpenAI(api_key= CHATGPT_API_KEY)

def _embedding_cache_dir(embed_model: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", embed_model or "unknown")
    return os.path.join("cache", "embeddings", safe)

def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def _load_cached_embedding(embed_model: str, text: str) -> list[float] | None:
    try:
        d = _embedding_cache_dir(embed_model)
        h = _sha256_text(text)
        path = os.path.join(d, f"{h}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj.get("model") == embed_model and isinstance(obj.get("embedding"), list):
            return obj["embedding"]
    except Exception:
        return None
    return None

def _save_cached_embedding(embed_model: str, text: str, embedding: list[float]) -> None:
    try:
        d = _embedding_cache_dir(embed_model)
        os.makedirs(d, exist_ok=True)
        h = _sha256_text(text)
        path = os.path.join(d, f"{h}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": embed_model, "embedding": embedding}, f)
    except Exception:
        return


def _embed_max_chars() -> int:
    """Max characters per embedding request chunk (mxbai-embed-large token limit is enforced by Ollama)."""
    raw = os.getenv("PAGEINDEX_EMBED_MAX_CHARS")
    if raw is None or str(raw).strip() == "":
        try:
            v = int(PAGEINDEX_EMBED_MAX_CHARS)
        except (TypeError, ValueError):
            v = 512
    else:
        try:
            v = int(raw)
        except ValueError:
            v = int(PAGEINDEX_EMBED_MAX_CHARS) if isinstance(PAGEINDEX_EMBED_MAX_CHARS, int) else 512
    return max(128, v)


def _split_text_for_embedding(s: str, max_chars: int) -> list[str]:
    """Split long text into chunks that fit embedding context; prefer breaking at whitespace."""
    s = (s or "").strip()
    if not s:
        return []
    if max_chars <= 0:
        return [s]
    if len(s) <= max_chars:
        return [s]
    chunks: list[str] = []
    i = 0
    while i < len(s):
        end = min(i + max_chars, len(s))
        if end < len(s):
            window = s[i:end]
            sp = window.rfind(" ")
            if sp > max_chars // 4:
                end = i + sp
        piece = s[i:end].strip()
        if piece:
            chunks.append(piece)
        i = end
        while i < len(s) and s[i].isspace():
            i += 1
    return chunks if chunks else [s[:max_chars]]


def _embed_parts_ollama(
    client,
    embed_model: str,
    parts: list[str],
) -> list[list[float]]:
    """Embed a list of short strings in small batches; each part must already be under max length."""
    if not parts:
        return []
    out: list[list[float]] = []
    raw_b = os.getenv("PAGEINDEX_EMBED_BATCH")
    if raw_b is None or str(raw_b).strip() == "":
        try:
            batch_size = max(1, int(PAGEINDEX_EMBED_BATCH))
        except (TypeError, ValueError):
            batch_size = 16
    else:
        try:
            batch_size = max(1, int(raw_b))
        except ValueError:
            batch_size = max(1, int(PAGEINDEX_EMBED_BATCH)) if isinstance(PAGEINDEX_EMBED_BATCH, int) else 16
    max_retries = 6
    for start in range(0, len(parts), batch_size):
        batch = parts[start : start + batch_size]
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=embed_model, input=batch)
                data = getattr(resp, "data", None) or []
                row = []
                for item in data:
                    row.append(getattr(item, "embedding", None) or item.get("embedding"))
                if len(row) != len(batch):
                    raise RuntimeError(
                        f"Embedding count mismatch: got {len(row)} expected {len(batch)}"
                    )
                out.extend(row)
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
        else:
            raise last_err  # type: ignore[misc]
    return out


def _embed_one_string_ollama(client, embed_model: str, text: str) -> list[float]:
    """
    One logical string -> one embedding vector. Long strings are split, embedded, mean-pooled.
    Sub-chunks are cached; full string is cached as the pooled vector.
    """
    text = "" if text is None else str(text)
    if not text.strip():
        return []

    cached = _load_cached_embedding(embed_model, text)
    if cached is not None:
        return cached

    max_c = _embed_max_chars()
    parts = _split_text_for_embedding(text, max_c)
    need_fetch: list[str] = []
    for p in parts:
        if _load_cached_embedding(embed_model, p) is None:
            need_fetch.append(p)

    if need_fetch:
        fresh = _embed_parts_ollama(client, embed_model, need_fetch)
        for p, emb in zip(need_fetch, fresh):
            _save_cached_embedding(embed_model, p, emb)

    part_embs: list[list[float]] = []
    for p in parts:
        ce = _load_cached_embedding(embed_model, p)
        if ce:
            part_embs.append(ce)

    pooled = _mean_embedding(part_embs)
    if pooled:
        _save_cached_embedding(embed_model, text, pooled)
    return pooled


def embed_texts_ollama(texts: list[str], embed_model: str) -> list[list[float]]:
    """
    Embed a list of texts using Ollama's OpenAI-compatible /v1/embeddings endpoint.
    Uses a local on-disk cache keyed by sha256(text).
    Long inputs are split, embedded, and mean-pooled so Ollama's context limit is not exceeded.
    """
    embed_model = str(embed_model or "").strip() or "mxbai-embed-large"
    texts = ["" if t is None else str(t) for t in (texts or [])]
    if not texts:
        return []

    cached: list[list[float] | None] = []
    to_fetch: list[str] = []
    fetch_idx: list[int] = []
    for i, t in enumerate(texts):
        emb = _load_cached_embedding(embed_model, t)
        cached.append(emb)
        if emb is None:
            to_fetch.append(t)
            fetch_idx.append(i)

    if to_fetch:
        client = get_openai_client("ollama")
        for j, t in enumerate(to_fetch):
            emb = _embed_one_string_ollama(client, embed_model, t)
            idx = fetch_idx[j]
            cached[idx] = emb

    # At this point, everything should be filled
    return [c or [] for c in cached]


def _sentence_split(text: str) -> list[str]:
    """
    Cheap sentence splitter (no external deps). Good enough for extractive snippets.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    # Split on sentence-ending punctuation followed by space and a capital/number/quote/paren.
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\"'\(\[])", t)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text_to_snippets(
    text: str,
    *,
    max_tokens: int,
    sentence_overlap: int,
    model=None,
) -> list[str]:
    """
    Group sentences into snippets up to ~max_tokens with N-sentence overlap.
    """
    max_tokens = int(max_tokens or 0)
    if max_tokens <= 0:
        max_tokens = 220
    sentence_overlap = max(0, int(sentence_overlap or 0))

    sents = _sentence_split(text)
    if not sents:
        return []

    snippets: list[str] = []
    i = 0
    while i < len(sents):
        j = i
        cur: list[str] = []
        while j < len(sents):
            cand = " ".join(cur + [sents[j]]).strip()
            if not cand:
                j += 1
                continue
            if count_tokens(cand, model=model) > max_tokens and cur:
                break
            cur.append(sents[j])
            j += 1
            # If single sentence is too long, allow it alone.
            if len(cur) == 1 and count_tokens(cur[0], model=model) > max_tokens:
                break

        snippet = " ".join(cur).strip()
        if snippet:
            snippets.append(snippet)

        if j <= i:
            i += 1
        else:
            i = max(i + 1, j - sentence_overlap)

    # De-dupe while preserving order
    seen = set()
    out = []
    for s in snippets:
        key = s
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return (_dot(a, a) ** 0.5) if a else 0.0


def cosine_sim(a: list[float], b: list[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def mmr_select(
    *,
    doc_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidates: list[str],
    top_k: int,
    lambda_mult: float,
) -> list[int]:
    """
    Return indices of selected candidates by Maximal Marginal Relevance (MMR).
    """
    top_k = max(0, int(top_k or 0))
    if top_k <= 0 or not candidates:
        return []
    lambda_mult = float(lambda_mult)
    if lambda_mult < 0:
        lambda_mult = 0.0
    if lambda_mult > 1:
        lambda_mult = 1.0

    # Precompute relevance sim(candidate, doc)
    rel = [cosine_sim(e, doc_embedding) for e in candidate_embeddings]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < min(top_k, len(candidates)):
        if not selected:
            best = max(remaining, key=lambda idx: rel[idx])
            selected.append(best)
            remaining.remove(best)
            continue

        def score(idx: int) -> float:
            diversity = max(cosine_sim(candidate_embeddings[idx], candidate_embeddings[s]) for s in selected)
            return lambda_mult * rel[idx] - (1.0 - lambda_mult) * diversity

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)

    return selected


def _mean_embedding(vectors: list[list[float]]) -> list[float]:
    """
    Compute element-wise mean of embeddings. Returns [] if empty.
    """
    if not vectors:
        return []
    dim = 0
    for v in vectors:
        if v:
            dim = len(v)
            break
    if dim <= 0:
        return []
    acc = [0.0] * dim
    n = 0
    for v in vectors:
        if not v or len(v) != dim:
            continue
        for i in range(dim):
            acc[i] += float(v[i])
        n += 1
    if n <= 0:
        return []
    return [x / n for x in acc]


async def generate_node_summary_mmr(node: dict, *, opt=None, model=None) -> str:
    """
    Extractive summarizer: chunk→embed (Ollama)→MMR→concat.
    Returns the concatenated text; caller is responsible for writing summary_payload.
    """
    text = (node or {}).get("text") or ""
    if not text.strip():
        return ""

    embed_model = getattr(opt, "ollama_embed_model", None) or "mxbai-embed-large"
    top_k = int(getattr(opt, "mmr_top_k", 8))
    lambda_mult = float(getattr(opt, "mmr_lambda", 0.5))
    max_candidates = int(getattr(opt, "mmr_max_candidates", 64))
    chunk_max_tokens = int(getattr(opt, "mmr_chunk_max_tokens", 220))
    overlap = int(getattr(opt, "mmr_chunk_sentence_overlap", 1))

    candidates = chunk_text_to_snippets(
        text,
        max_tokens=chunk_max_tokens,
        sentence_overlap=overlap,
        # IMPORTANT: use an Ollama-tuned token counter (fallback encoding) so chunks
        # don't depend on the chat LLM tokenizer (e.g. qwen), which can under-estimate
        # input length for embeddings.
        model="ollama",
    )
    candidates = merge_near_duplicate_strings(candidates, opt=opt)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    if not candidates:
        return text.strip()

    # Embed candidates. Use their mean embedding as a proxy for the full-document embedding
    # to avoid embedding extremely long inputs (which can exceed model context limits).
    cand_emb = embed_texts_ollama(candidates, embed_model=embed_model)
    doc_emb = _mean_embedding(cand_emb)

    sel_idx = mmr_select(
        doc_embedding=doc_emb,
        candidate_embeddings=cand_emb,
        candidates=candidates,
        top_k=top_k,
        lambda_mult=lambda_mult,
    )
    selected = [candidates[i] for i in sel_idx]
    try:
        node["_mmr_selected"] = [{"rank": r + 1, "snippet": s, "source": "text"} for r, s in enumerate(selected)]
    except Exception:
        pass
    return "\n\n".join(selected).strip()

def get_model_name(model=None):
    """Get the model name based on provider and input model."""
    if model == "ollama":
        return OLLAMA_MODEL
    elif model == "huggingface":
        return "openai/gpt-oss-120b:groq"
    elif model == "qwen":
        return QWEN_MODEL
    else:
        return model

def count_tokens(text, model=None):
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        return len(tokens)
    except:
        enc = tiktoken.get_encoding("o200k_base")
        tokens = enc.encode(text)
        return len(tokens)

def ChatGPT_API_with_finish_reason(model, prompt, chat_history=None):
    max_retries = 10
    client = get_openai_client(model)
    model_name = get_model_name(model)
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "system", "content": prompt['system_prompt']},
                            {"role": "user", "content": prompt['user_prompt']}]
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
            if response.choices[0].finish_reason == "length":
                return response.choices[0].message.content, "max_output_reached"
            else:
                return response.choices[0].message.content, "finished"

        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)  # Wait for 1秒 before retrying
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"



def ChatGPT_API(model, prompt, chat_history=None):
    max_retries = 10
    client = get_openai_client(model)
    model_name = get_model_name(model)
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "system", "content": prompt['system_prompt']},
                            {"role": "user", "content": prompt['user_prompt']}]
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
   
            return response.choices[0].message.content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)  # Wait for 1秒 before retrying
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"


async def ChatGPT_API_async(model, prompt):
    max_retries = 10
    messages = [{"role": "system", "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['user_prompt']}]
    model_name = get_model_name(model)
    for i in range(max_retries):
        try:
            async with get_async_openai_client(model) as client:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0].message.content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)  # Wait for 1s before retrying
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"  
            
            
def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
         

def extract_json(content):
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace
        # json_content = json_content.replace('\\\\"', '\\"',json_content) # Fix double-escaped quotes
        # json_content = re.sub(r',\s([}\]])', r'\1', json_content) # Remove trailing commas with any whitespace


        # Attempt to parse and return the JSON object
        # print(f'json_content: {json_content}')
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("Failed to parse JSON even after cleanup")
            
            raise e
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

def get_parent_nodes(structure):
    if isinstance(structure, dict):
        if 'nodes' in structure:
            return [structure] + get_parent_nodes(structure['nodes'])
        return []
    elif isinstance(structure, list):
        return [item for sublist in [get_parent_nodes(item) for item in structure] for item in sublist]
    
def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    ###return text not list 
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text

def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text

def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text):
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path):
    # Extract PDF name
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)
    



def list_to_tree(data):
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data



def get_page_tokens(pdf_path, model="gpt-4o-2024-11-20", pdf_parser="PyPDF2"):
    enc = tiktoken.encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



def post_processing(structure, end_physical_index):
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        ### remove appear_start 
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model='gpt-4o')
        # if num_tokens > limit:
        #     print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
        #     print("Start Index:", node['start_index'])
        #     print("End Index:", node['end_index'])
        #     print("Title:", node['title'])
        #     print("\n")


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        if not 'nodes' in node:
            node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node, model=None):
    if 'text' in node:
        prompt = {}
        prompt['system_prompt'] = f"""
        You are an expert in generating summaries for documents.
        You are given a part of a document.
        Your task is to generate a summary of the partial document about what are main points covered in the partial document.
        """
        prompt['user_prompt'] = f"""
        Partial Document Text: {node['text']}
        
        Directly return the summary, do not include any other text.
        """
        response = await ChatGPT_API_async(model, prompt)
        return response

async def generate_parent_node_summary(node, model=None):
    children = (node or {}).get("nodes") or []
    child_summaries = [
        str(ch.get("summary")).strip()
        for ch in children
        if isinstance(ch, dict) and isinstance(ch.get("summary"), str) and str(ch.get("summary")).strip()
    ]
    if not child_summaries:
        return ""

    prompt = {}
    prompt['system_prompt'] = f"""
    You are an expert in generating summaries for documents.
    You are given a part of a document.
    Your task is to generate a summary of the parent node of the partial document about what are main points covered in the children nodes using the summaries of the children nodes.
    """
    prompt['user_prompt'] = f"""
    Partial Document Text: {' '.join(child_summaries)}
    
    Directly return the summary, do not include any other text.
    """
    
    response = await ChatGPT_API_async(model, prompt)
    return response


async def generate_parent_node_summary_mmr(node: dict, *, opt=None, model=None) -> str:
    """
    Extractive parent summarizer: build candidates from children summaries/snippets,
    then chunk→embed→MMR→concat.
    """
    children = (node or {}).get("nodes") or []
    if not children:
        return ""

    # Candidate snippets from children:
    candidates: list[dict] = []
    for ch in children:
        if not isinstance(ch, dict):
            continue
        ch_id = ch.get("node_id")
        payload = ch.get("summary_payload")
        if isinstance(payload, dict) and payload.get("type") == "mmr":
            selected = payload.get("selected")
            if isinstance(selected, list):
                for item in selected:
                    if isinstance(item, dict) and str(item.get("snippet", "")).strip():
                        candidates.append(
                            {"snippet": str(item["snippet"]), "source": "child", "source_node_id": ch_id}
                        )
        # Fallback: use child summary string
        s = ch.get("summary")
        if isinstance(s, str) and s.strip():
            candidates.append({"snippet": s.strip(), "source": "child", "source_node_id": ch_id})

    # De-dupe by snippet text
    seen = set()
    dedup: list[dict] = []
    for c in candidates:
        sn = c.get("snippet") or ""
        if sn in seen:
            continue
        seen.add(sn)
        dedup.append(c)

    dedup = merge_near_duplicate_candidates(dedup, opt=opt)

    max_candidates = int(getattr(opt, "mmr_max_candidates", 64))
    if max_candidates > 0:
        dedup = dedup[:max_candidates]

    snippets = [c["snippet"] for c in dedup if isinstance(c.get("snippet"), str) and c["snippet"].strip()]
    if not snippets:
        return ""

    embed_model = getattr(opt, "ollama_embed_model", None) or "mxbai-embed-large"
    top_k = int(getattr(opt, "mmr_top_k", 8))
    lambda_mult = float(getattr(opt, "mmr_lambda", 0.5))

    cand_emb = embed_texts_ollama(snippets, embed_model=embed_model)
    doc_emb = _mean_embedding(cand_emb)

    sel_idx = mmr_select(
        doc_embedding=doc_emb,
        candidate_embeddings=cand_emb,
        candidates=snippets,
        top_k=top_k,
        lambda_mult=lambda_mult,
    )
    selected = [dedup[i] for i in sel_idx]
    try:
        node["_mmr_selected"] = [
            {
                "rank": r + 1,
                "snippet": item.get("snippet", ""),
                "source": item.get("source", "child"),
                "source_node_id": item.get("source_node_id"),
            }
            for r, item in enumerate(selected)
        ]
    except Exception:
        pass
    return "\n\n".join([s["snippet"] for s in selected]).strip()
    
        
    

async def get_node_summary(
    node: dict,
    summary_token_threshold: int = 200,
    model=None,
    opt=None,
) -> str | None:
    """
    Shortcut: if node text is short, avoid an LLM call and return the text.
    Mirrors the Markdown pipeline behavior (see page_index_md.py).
    """
    node_text = None
    if isinstance(node, dict):
        node_text = node.get("text")
    if not node_text:
        return None
    try:
        num_tokens = count_tokens(node_text, model=model)
    except Exception:
        num_tokens = 0
    if num_tokens < int(summary_token_threshold or 0):
        # Still emit payload via caller if desired; return raw text as "summary"
        return node_text

    method = get_summary_method(opt=opt)
    if method == "mmr":
        return await generate_node_summary_mmr(node, opt=opt, model=model)
    return await generate_node_summary(node, model=model)


async def _gather_bounded(tasks, max_concurrency: int):
    """
    Run awaitables with bounded concurrency.
    """
    n = int(max_concurrency or 0)
    if n <= 0:
        return await asyncio.gather(*tasks)
    sem = asyncio.Semaphore(n)

    async def _run_one(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_run_one(t) for t in tasks])


async def generate_summaries_for_structure(
    structure,
    model=None,
    summary_token_threshold: int = 200,
    max_concurrency: int = 8,
    opt=None,
):
    nodes = structure_to_list(structure)
    tasks = [
        get_node_summary(node, summary_token_threshold=summary_token_threshold, model=model, opt=opt)
        for node in nodes
    ]
    summaries = await _gather_bounded(tasks, max_concurrency)

    for node, summary in zip(nodes, summaries):
        if summary is not None:
            node['summary'] = summary
            method = get_summary_method(opt=opt)
            if method == "mmr":
                embed_model = getattr(opt, "ollama_embed_model", None) or "mxbai-embed-large"
                selected_items = []
                if isinstance(node, dict) and isinstance(node.get("_mmr_selected"), list):
                    selected_items = node.get("_mmr_selected")
                if isinstance(node, dict) and "_mmr_selected" in node:
                    node.pop("_mmr_selected", None)
                node["summary_payload"] = {
                    "type": "mmr",
                    "text": summary,
                    "embed_backend": "ollama",
                    "embed_model": embed_model,
                    "k": int(getattr(opt, "mmr_top_k", 8)),
                    "candidates": int(getattr(opt, "mmr_max_candidates", 64)),
                    "selected": selected_items,
                    "chunking": {
                        "mode": "sentences",
                        "max_tokens": int(getattr(opt, "mmr_chunk_max_tokens", 220)),
                        "sentence_overlap": int(getattr(opt, "mmr_chunk_sentence_overlap", 1)),
                    },
                }
            else:
                node["summary_payload"] = {
                    "type": "llm",
                    "text": summary,
                    "model": get_model_name(model),
                    "token_threshold": int(summary_token_threshold or 0),
                    "generated_from": "text",
                }
    return structure

async def generate_parent_node_summaries_for_structure(structure, model=None, max_concurrency: int = 8, opt=None):
    nodes = get_parent_nodes(structure)
    # To ensure parent node summaries are generated only after all children's summaries are ready,
    # we need to process from leaves upward (bottom-up). We'll organize nodes by depth and
    # process them level by level, starting with the deepest.
    def get_parent_nodes_by_depth(structure):
        """
        Returns a list of (depth, node) tuples for all parent nodes.
        """
        nodes_by_depth = []
        def traverse(node, depth=0):
            if isinstance(node, dict):
                if 'nodes' in node and node['nodes']:
                    nodes_by_depth.append((depth, node))
                    if isinstance(node['nodes'], list):
                        for child in node['nodes']:
                            traverse(child, depth+1)
            elif isinstance(node, list):
                for item in node:
                    traverse(item, depth)
        traverse(structure)
        return nodes_by_depth

    nodes_with_depth = get_parent_nodes_by_depth(structure)
    if not nodes_with_depth:
        summaries = []
    else:
        # Group nodes by their depth (deepest first)
        from collections import defaultdict
        by_depth = defaultdict(list)
        max_depth = 0
        for depth, node in nodes_with_depth:
            by_depth[depth].append(node)
            if depth > max_depth:
                max_depth = depth

        # summaries_map: node id (or repr) to summary, for child lookup if needed
        summaries_map = {}

        # Bottom-up: deepest parents first
        for depth in reversed(range(0, max_depth+1)):
            method = get_parent_summary_method(opt=opt)
            if method == "mmr":
                tasks = [generate_parent_node_summary_mmr(node, opt=opt, model=model) for node in by_depth[depth]]
            else:
                tasks = [generate_parent_node_summary(node, model=model) for node in by_depth[depth]]
            results = await _gather_bounded(tasks, max_concurrency) if tasks else []
            for node, summary in zip(by_depth[depth], results):
                node['summary'] = summary
                method = get_parent_summary_method(opt=opt)
                if method == "mmr":
                    embed_model = getattr(opt, "ollama_embed_model", None) or "mxbai-embed-large"
                    selected_items = []
                    if isinstance(node, dict) and isinstance(node.get("_mmr_selected"), list):
                        selected_items = node.get("_mmr_selected")
                    if isinstance(node, dict) and "_mmr_selected" in node:
                        node.pop("_mmr_selected", None)
                    node["summary_payload"] = {
                        "type": "mmr",
                        "text": summary,
                        "embed_backend": "ollama",
                        "embed_model": embed_model,
                        "k": int(getattr(opt, "mmr_top_k", 8)),
                        "candidates": int(getattr(opt, "mmr_max_candidates", 64)),
                        "selected": selected_items,
                    }
                else:
                    node["summary_payload"] = {
                        "type": "llm",
                        "text": summary,
                        "model": get_model_name(model),
                        "generated_from": "children",
                    }
                # Use id(node) as a unique object identity (safe since we're traversing memory graph)
                summaries_map[id(node)] = summary
        # For output (to match interface): gather in original order in nodes_with_depth
        summaries = [node.get('summary') for (_, node) in nodes_with_depth]
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure

def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure

def create_clean_structure_for_abstract(structure):
    """
    Create a clean structure for document abstract generation,
    excluding unnecessary fields like 'text'. Recurses into child nodes.
    """
    if isinstance(structure, dict):
        clean_node = {}
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_abstract(structure['nodes'])
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_abstract(item) for item in structure]
    else:
        return structure


def _collect_mmr_candidates_from_tree(structure) -> list[dict]:
    """
    Build snippet candidates from all nodes (same idea as parent MMR: prefer
    summary_payload.selected snippets, else node summary string).
    """
    candidates: list[dict] = []
    for node in structure_to_list(structure):
        if not isinstance(node, dict):
            continue
        ch_id = node.get("node_id")
        payload = node.get("summary_payload")
        if isinstance(payload, dict) and payload.get("type") == "mmr":
            selected = payload.get("selected")
            if isinstance(selected, list):
                for item in selected:
                    if isinstance(item, dict) and str(item.get("snippet", "")).strip():
                        candidates.append(
                            {
                                "snippet": str(item["snippet"]),
                                "source": "node",
                                "source_node_id": ch_id,
                            }
                        )
        s = node.get("summary")
        if isinstance(s, str) and s.strip():
            candidates.append(
                {"snippet": s.strip(), "source": "node", "source_node_id": ch_id}
            )

    seen: set[str] = set()
    dedup: list[dict] = []
    for c in candidates:
        sn = c.get("snippet") or ""
        if sn in seen:
            continue
        seen.add(sn)
        dedup.append(c)
    return dedup


def generate_doc_abstract_mmr(structure, opt=None, model=None) -> tuple[str, dict]:
    """
    Document-level extractive abstract: pool snippets from all node summaries,
    embed + MMR + concat (same spirit as parent node MMR).
    """
    dedup = _collect_mmr_candidates_from_tree(structure)
    dedup = merge_near_duplicate_candidates(dedup, opt=opt)
    max_candidates = int(getattr(opt, "mmr_max_candidates", 64))
    if max_candidates > 0:
        dedup = dedup[:max_candidates]

    snippets = [c["snippet"] for c in dedup if isinstance(c.get("snippet"), str) and c["snippet"].strip()]
    embed_model = getattr(opt, "ollama_embed_model", None) or "mxbai-embed-large"
    top_k = int(getattr(opt, "mmr_top_k", 8))
    lambda_mult = float(getattr(opt, "mmr_lambda", 0.5))

    if not snippets:
        payload = {
            "type": "mmr",
            "text": "",
            "embed_backend": "ollama",
            "embed_model": embed_model,
            "k": top_k,
            "candidates": 0,
            "selected": [],
            "generated_from": "all_node_summaries",
        }
        return "", payload

    cand_emb = embed_texts_ollama(snippets, embed_model=embed_model)
    doc_emb = _mean_embedding(cand_emb)
    sel_idx = mmr_select(
        doc_embedding=doc_emb,
        candidate_embeddings=cand_emb,
        candidates=snippets,
        top_k=top_k,
        lambda_mult=lambda_mult,
    )
    selected_rows = [dedup[i] for i in sel_idx]
    text = "\n\n".join([s["snippet"] for s in selected_rows]).strip()
    selected_items = [
        {
            "rank": r + 1,
            "snippet": item.get("snippet", ""),
            "source": item.get("source", "node"),
            "source_node_id": item.get("source_node_id"),
        }
        for r, item in enumerate(selected_rows)
    ]
    payload = {
        "type": "mmr",
        "text": text,
        "embed_backend": "ollama",
        "embed_model": embed_model,
        "k": top_k,
        "candidates": len(dedup),
        "selected": selected_items,
        "generated_from": "all_node_summaries",
    }
    return text, payload


def generate_keywords(abstract, model=None):
    prompt = {}
    prompt['system_prompt'] = f"""Your are an expert in generating keywords for a document.
    You are given a abstract of a document. Your task is to generate a list of keywords with distinct meanings for the document, 
    which makes it helpful for users to find relevant documents given related queries.

    The keywords should be short and concise, no more than three words, and should be related to the content of the document.
    THe keywords should be distinct and should not be synonyms.
    The keywords should be in English.
    The keywords should cover the main topics, methods, and results of the document.
    Try to use as less keywords as possible without losing the main information of the document.
    Do not return more than 10 keywords.
    
    Directly return the keywords, do not include any other text.
    Return the keywords as a string separated by commas in the following format: "keyword1, keyword2, keyword3, ..."
    """
    prompt['user_prompt'] = f"""
    Abstract: {abstract}
    
    Directly return the keywords, do not include any other text.
    """
    response = ChatGPT_API(model, prompt)
    return response


def _collect_section_titles_for_keywords(structure, opt=None) -> list[str]:
    max_headers = int(getattr(opt, "keywords_rich_max_headers", 40) or 40) if opt is not None else 40
    titles: list[str] = []
    for node in structure_to_list(structure):
        if not isinstance(node, dict):
            continue
        t = node.get("title")
        if isinstance(t, str) and t.strip():
            titles.append(t.strip())
    # Exact dedupe while keeping order
    seen = set()
    uniq: list[str] = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if max_headers > 0 and len(uniq) >= max_headers:
            break
    return uniq


def _collect_snippets_for_keywords(structure, opt=None) -> list[str]:
    max_snippets = int(getattr(opt, "keywords_rich_max_snippets", 40) or 40) if opt is not None else 40
    snippet_max_chars = int(getattr(opt, "keywords_rich_snippet_max_chars", 400) or 400) if opt is not None else 400
    rows = _collect_mmr_candidates_from_tree(structure)
    rows = merge_near_duplicate_candidates(rows, opt=opt)
    snippets: list[str] = []
    for r in rows:
        s = str(r.get("snippet", "")).strip()
        if not s:
            continue
        if snippet_max_chars > 0 and len(s) > snippet_max_chars:
            s = s[:snippet_max_chars].rstrip() + "…"
        snippets.append(s)
        if max_snippets > 0 and len(snippets) >= max_snippets:
            break
    return snippets


def _keywords_embed_stopwords() -> set[str]:
    # Compact stopword list for keyword candidate filtering (not for full NLP).
    return {
        "a","an","the","and","or","but","if","then","than","when","while","where","which","who","whom","whose",
        "to","of","in","on","for","with","without","as","at","by","from","into","over","under","between","within",
        "is","are","was","were","be","been","being","do","does","did","done",
        "this","that","these","those","it","its","they","them","their","we","our","you","your","i","me","my",
        "can","could","may","might","must","should","would","will",
        "not","no","yes",
        "using","use","used","via","based","approach","method","methods","results","result","paper","study","work",
    }


def _normalize_keyword_phrase(s: str, *, max_words: int = 3) -> str | None:
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    if not t:
        return None
    t = re.sub(r"[\s_]+", " ", t).strip()
    t = t.strip(" .,:;|/\\-–—()[]{}<>\"'`")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    # Keep only words/numbers; drop stray symbols.
    words = re.findall(r"[a-z0-9]+", t)
    if not words:
        return None
    if max_words > 0:
        words = words[:max_words]
    if len(words) == 1 and len(words[0]) <= 2:
        return None
    if all(w.isdigit() for w in words):
        return None
    return " ".join(words)


def _extract_candidate_ngrams(snippets: list[str], opt=None) -> list[str]:
    """
    Mine 1..N-grams from snippets and return a ranked list of phrase candidates.
    Uses simple regex tokenization + frequency filtering (no external NLP deps).
    """
    max_n = int(getattr(opt, "keywords_embed_ngram_max_n", 3) or 3) if opt is not None else 3
    max_candidates = int(getattr(opt, "keywords_embed_max_candidates", 500) or 500) if opt is not None else 500
    min_df = int(getattr(opt, "keywords_embed_min_df", 2) or 2) if opt is not None else 2

    stop = _keywords_embed_stopwords()
    df: dict[str, int] = {}

    for s in snippets or []:
        if not isinstance(s, str) or not s.strip():
            continue
        tokens = re.findall(r"[a-z0-9]+", s.lower())
        if not tokens:
            continue
        # Build set of grams present in this snippet (document frequency across snippets).
        present: set[str] = set()
        L = len(tokens)
        for n in range(1, max(1, max_n) + 1):
            for i in range(0, L - n + 1):
                gram_tokens = tokens[i : i + n]
                if any(len(w) <= 2 for w in gram_tokens):
                    continue
                if gram_tokens[0] in stop or gram_tokens[-1] in stop:
                    continue
                stop_frac = sum(1 for w in gram_tokens if w in stop) / float(n)
                if stop_frac >= 0.67:
                    continue
                gram = " ".join(gram_tokens)
                present.add(gram)
        for g in present:
            df[g] = df.get(g, 0) + 1

    # Filter by min_df and keep top by df
    items = [(g, c) for g, c in df.items() if c >= max(1, min_df)]
    items.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    grams = [g for g, _c in items[: max(1, max_candidates)]]
    return grams


def generate_keywords_embed_mmr(structure, *, doc_title: str | None = None, opt=None, model=None) -> tuple[list[str], dict]:
    """
    LLM-free keyword generation: mine candidates (headers + n-grams), embed + MMR select.
    Returns (keywords_list, payload).
    """
    section_titles = _collect_section_titles_for_keywords(structure, opt=opt)
    snippets = _collect_snippets_for_keywords(structure, opt=opt)
    grams = _extract_candidate_ngrams(snippets, opt=opt)

    candidates: list[str] = []
    if isinstance(doc_title, str) and doc_title.strip():
        candidates.append(doc_title.strip())
    candidates.extend(section_titles)
    candidates.extend(grams)

    norm: list[str] = []
    seen = set()
    for c in candidates:
        p = _normalize_keyword_phrase(c, max_words=3)
        if not p or p in seen:
            continue
        seen.add(p)
        norm.append(p)

    # Merge near-dups (always on) and cap candidate count for embedding cost.
    norm = merge_near_duplicate_strings(norm, opt=opt)
    cap = int(getattr(opt, "keywords_embed_max_candidates", 500) or 500) if opt is not None else 500
    if cap > 0:
        norm = norm[:cap]

    embed_model = getattr(opt, "ollama_embed_model", None) or "mxbai-embed-large"
    top_k = int(getattr(opt, "keywords_embed_top_k", 10) or 10) if opt is not None else 10
    lambda_mult = float(getattr(opt, "keywords_embed_lambda", 0.5) or 0.5) if opt is not None else 0.5

    if not norm:
        return [], {
            "type": "embed_mmr",
            "text": "",
            "embed_backend": "ollama",
            "embed_model": embed_model,
            "k": top_k,
            "candidates": 0,
            "selected": [],
            "generated_from": "headers+ngrams",
        }

    cand_emb = embed_texts_ollama(norm, embed_model=embed_model)
    doc_emb = _mean_embedding(cand_emb)
    sel_idx = mmr_select(
        doc_embedding=doc_emb,
        candidate_embeddings=cand_emb,
        candidates=norm,
        top_k=top_k,
        lambda_mult=lambda_mult,
    )
    selected = [norm[i] for i in sel_idx]
    payload = {
        "type": "embed_mmr",
        "text": ", ".join(selected),
        "embed_backend": "ollama",
        "embed_model": embed_model,
        "k": top_k,
        "lambda": float(lambda_mult),
        "candidates": len(norm),
        "selected": [{"rank": r + 1, "keyword": kw} for r, kw in enumerate(selected)],
        "generated_from": "headers+ngrams",
    }
    return selected, payload


def generate_keywords_rich(structure, *, doc_title: str | None = None, model=None, opt=None) -> str:
    """
    Generate keywords using a richer candidate set than the doc_abstract string:
    document title + section titles + many representative snippets.
    """
    section_titles = _collect_section_titles_for_keywords(structure, opt=opt)
    snippets = _collect_snippets_for_keywords(structure, opt=opt)

    prompt = {}
    prompt["system_prompt"] = f"""Your are an expert in generating keywords for a document.
You are given a document title (optional), a list of section titles, and a set of representative snippets from the document.
Your task is to generate a list of keywords with distinct meanings for the document, which makes it helpful for users to find relevant documents given related queries.

Rules:
- Keywords should be short and concise (no more than three words).
- Keywords must be distinct and should not be synonyms.
- Keywords must be in English.
- Cover main topics, methods, and results.
- Do not return more than 10 keywords.
- Directly return the keywords as a string separated by commas in the following format: "keyword1, keyword2, keyword3, ..."
"""
    prompt["user_prompt"] = f"""
Title: {doc_title or ""}

Section titles:
{json.dumps(section_titles, ensure_ascii=False, indent=2)}

Representative snippets:
{json.dumps(snippets, ensure_ascii=False, indent=2)}

Directly return the keywords, do not include any other text.
"""
    response = ChatGPT_API(model, prompt)
    return response


def generate_doc_keywords(structure, doc_abstract: str, *, doc_title: str | None = None, model=None, opt=None) -> tuple[str, list[str] | None, dict]:
    """
    Keyword generation entrypoint with a switchable backend.
    Returns (keywords_string, keywords_payload).
    """
    method = get_keyword_method(opt=opt)
    if method == "embed_mmr":
        keywords_list, payload = generate_keywords_embed_mmr(structure, doc_title=doc_title, model=model, opt=opt)
        return (", ".join(keywords_list), keywords_list, payload)
    if method == "rich":
        text = generate_keywords_rich(structure, doc_title=doc_title, model=model, opt=opt)
        return (
            text,
            None,
            {
                "type": "llm",
                "method": "rich",
                "model": get_model_name(model),
                "generated_from": "title+section_titles+snippets",
            },
        )

    text = generate_keywords(doc_abstract, model=model)
    return (
        text,
        None,
        {
            "type": "llm",
            "method": "llm",
            "model": get_model_name(model),
            "generated_from": "doc_abstract",
        },
    )

def generate_doc_description(structure, model=None):
    prompt = {}
    prompt['system_prompt'] = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
    """
    prompt['user_prompt'] = f"""
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    response = ChatGPT_API(model, prompt)
    return response

def generate_doc_abstract(structure, model=None, opt=None) -> tuple[str, dict]:
    """
    Document abstract: LLM (abstractive) or MMR over pooled node summaries (extractive).
    Returns (abstract_text, abstract_payload) for downstream RAG.
    """
    method = get_abstract_method(opt=opt)
    if method == "mmr":
        return generate_doc_abstract_mmr(structure, opt=opt, model=model)

    clean_structure = create_clean_structure_for_abstract(structure)
    prompt = {}
    prompt['system_prompt'] = f"""Your are an expert in generating abstracts for a document.
    You are given a structure of summaries of a document. Your task is to generate a concise abstract for the document, which makes it easy to have a quick idea of the content of the document.

    The abstract should contain: 
    1. The background or motivation of the document. 
    2. The major issues or challenges addressed by the document. 
    3. The major methods or approaches used to address the issues.
    4. The major findings or results of the document.
    5. The major conclusions, limitations, or implications of the document.
    The abstract should be no more than 300 words.

    Directly return the description, do not include any other text."""

    prompt['user_prompt'] = f"""
    Document Structure: {clean_structure}

    Directly return the abstract, do not include any other text.
    """
    response = ChatGPT_API(model, prompt)
    text = response if isinstance(response, str) else str(response)
    payload = {
        "type": "llm",
        "text": text,
        "model": get_model_name(model),
        "generated_from": "structure_summaries",
    }
    return text, payload

def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")
        self._validate_keys(user_dict)
        explicit_keys = set(user_dict.keys())
        merged = {**self._default_dict, **user_dict}
        merged = _apply_method_env_overrides_to_merged(merged, explicit_keys)
        return config(**merged)
