import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken


def compute_mdhash_id(text: str, prefix: str) -> str:
    return f"{prefix}{hashlib.md5(text.encode('utf-8')).hexdigest()}"


def generate_cache_key(
    file_path: str,
    parse_method: str,
    kwargs: Dict[str, Any],
) -> str:
    path = str(Path(file_path).resolve())
    mtime = os.path.getmtime(path)
    payload = {
        "path": path,
        "mtime": mtime,
        "parse_method": parse_method,
        "kwargs": kwargs,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def generate_content_based_doc_id(content_list: List[Dict[str, Any]]) -> str:
    signature_parts: List[str] = []
    for item in content_list:
        item_type = item.get("type", "text")
        page_idx = item.get("page_idx", -1)
        if item_type == "text":
            signature_parts.append(
                f"text::{page_idx}::{item.get('text', '')[:4000]}"
            )
        elif item_type == "image":
            signature_parts.append(
                f"image::{page_idx}::{item.get('img_path', '')}::{item.get('image_caption', '')}"
            )
        elif item_type == "table":
            signature_parts.append(
                f"table::{page_idx}::{item.get('table_caption', '')}::{item.get('table_body', '')}"
            )
        elif item_type == "equation":
            signature_parts.append(
                f"equation::{page_idx}::{item.get('text', '')}"
            )
        else:
            signature_parts.append(
                f"{item_type}::{page_idx}::{json.dumps(item, sort_keys=True, default=str)[:4000]}"
            )

    return compute_mdhash_id("\n".join(signature_parts), "doc-")


def separate_content(content_list: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    multimodal_items: List[Dict[str, Any]] = []

    for item in content_list:
        item_type = item.get("type", "text")
        if item_type == "text":
            text = str(item.get("text", "")).strip()
            if text:
                text_parts.append(text)
        else:
            multimodal_items.append(item)

    return "\n\n".join(text_parts), multimodal_items


def get_encoder() -> Any:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    split_by_character: Optional[str] = None,
    split_by_character_only: bool = False,
) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if split_by_character:
        parts = [p.strip() for p in text.split(split_by_character) if p.strip()]
        if split_by_character_only:
            return parts
        text = "\n".join(parts)

    enc = get_encoder()
    if enc is None:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    tokens = enc.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def extract_run_text(run_output: Any) -> str:
    if run_output is None:
        return ""
    if isinstance(run_output, str):
        return run_output
    if hasattr(run_output, "content") and run_output.content is not None:
        return str(run_output.content)
    return str(run_output)


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}