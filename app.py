import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.rag_agno import RAGAnythingAgno, Settings


# =========================
# App + Path Setup
# =========================
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "ui_uploads"
REGISTRY_FILE = DATA_DIR / "ui_registry.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(APP_ROOT / ".env", override=True)


# =========================
# Async Helper
# =========================
def run_async(awaitable):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


# =========================
# Persistence Helpers
# =========================
def load_registry() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_registry_entry(
    doc_id: str,
    file_path: str,
    original_name: str,
    result_dict: Dict[str, Any],
) -> None:
    registry = load_registry()
    registry[doc_id] = {
        "doc_id": doc_id,
        "file_path": file_path,
        "original_name": original_name,
        "status": result_dict.get("status"),
        "parse_cache_hit": result_dict.get("parse_cache_hit"),
        "text_chunks": result_dict.get("text_chunks", 0),
        "multimodal_items": result_dict.get("multimodal_items", 0),
        "error": result_dict.get("error"),
        "last_ingested_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_registry(registry)


def get_sorted_registry_items() -> List[Dict[str, Any]]:
    registry = load_registry()
    items = list(registry.values())
    items.sort(key=lambda x: x.get("last_ingested_at", ""), reverse=True)
    return items


# =========================
# Backend Singleton
# =========================
@st.cache_resource(show_spinner=False)
def get_rag_backend() -> RAGAnythingAgno:
    settings = Settings()
    settings.ensure_dirs()
    return RAGAnythingAgno(settings)


# =========================
# File Helpers
# =========================
def safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", ".", " "}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip().replace(" ", "_")


def persist_uploaded_file(uploaded_file) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    clean_name = safe_filename(uploaded_file.name)
    target = UPLOAD_DIR / f"{timestamp}_{clean_name}"
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


# =========================
# UI State
# =========================
def init_session_state() -> None:
    if "selected_doc_id" not in st.session_state:
        st.session_state.selected_doc_id = None

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if "last_query_answer" not in st.session_state:
        st.session_state.last_query_answer = ""

    if "last_ingest_results" not in st.session_state:
        st.session_state.last_ingest_results = []


def get_chat_history(doc_id: str) -> List[Dict[str, str]]:
    if doc_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[doc_id] = []
    return st.session_state.chat_histories[doc_id]


# =========================
# Render Helpers
# =========================
def format_doc_option(item: Dict[str, Any]) -> str:
    status = item.get("status", "UNKNOWN")
    name = item.get("original_name", item.get("doc_id", "document"))
    chunks = item.get("text_chunks", 0)
    mm = item.get("multimodal_items", 0)
    return f"{name} | {status} | text={chunks}, mm={mm}"


def render_doc_details(doc_meta: Dict[str, Any]) -> None:
    st.markdown("#### Selected document")
    st.json(doc_meta)


def render_chat(doc_id: str) -> None:
    history = get_chat_history(doc_id)
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# =========================
# Ingestion Actions
# =========================
def ingest_uploaded_files(
    uploaded_files,
    backend: str,
    parse_method: str,
) -> None:
    rag = get_rag_backend()
    results = []

    for uploaded_file in uploaded_files:
        local_path = persist_uploaded_file(uploaded_file)

        with st.spinner(f"Ingesting {uploaded_file.name} ..."):
            try:
                result = run_async(
                    rag.process_document_complete(
                        str(local_path),
                        backend=backend,
                        parse_method=parse_method,
                    )
                )
                result_dict = result.model_dump()
                upsert_registry_entry(
                    doc_id=result_dict["doc_id"],
                    file_path=str(local_path),
                    original_name=uploaded_file.name,
                    result_dict=result_dict,
                )
                results.append(result_dict)

                if result_dict["status"] == "PROCESSED":
                    st.success(
                        f"{uploaded_file.name} ingested successfully. "
                        f"doc_id={result_dict['doc_id']}"
                    )
                    st.session_state.selected_doc_id = result_dict["doc_id"]
                else:
                    st.error(
                        f"{uploaded_file.name} failed: {result_dict.get('error')}"
                    )

            except Exception as exc:
                st.exception(exc)

    st.session_state.last_ingest_results = results


def refresh_selected_doc_status(doc_id: str) -> Dict[str, Any]:
    rag = get_rag_backend()
    return rag.get_doc_status(doc_id)


# =========================
# Query Action
# =========================
def ask_question(doc_id: str, question: str) -> str:
    rag = get_rag_backend()
    answer = run_async(rag.aquery(question=question, doc_id=doc_id))
    st.session_state.last_query_answer = answer
    return answer


# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="Agno RAG Anything UI",
    page_icon="📚",
    layout="wide",
)

init_session_state()

st.title("📚 Agno RAG Anything")
st.caption("Upload documents, ingest them, inspect status, and chat with a selected document.")

with st.sidebar:
    st.header("Settings")

    backend = st.selectbox(
        "MinerU backend",
        options=["pipeline", "auto"],
        index=0,
        help="Use pipeline for CPU / low-GPU-memory setups.",
    )

    parse_method = st.selectbox(
        "Parse method",
        options=["auto", "ocr", "txt"],
        index=0,
    )

    st.divider()
    st.subheader("Indexed documents")

    registry_items = get_sorted_registry_items()

    if registry_items:
        selected_item = st.selectbox(
            "Choose a document",
            options=registry_items,
            format_func=format_doc_option,
            index=0 if st.session_state.selected_doc_id is None else next(
                (
                    i
                    for i, item in enumerate(registry_items)
                    if item["doc_id"] == st.session_state.selected_doc_id
                ),
                0,
            ),
        )
        st.session_state.selected_doc_id = selected_item["doc_id"]
    else:
        st.info("No documents ingested yet.")

    if st.button("Reload registry"):
        st.rerun()

    if st.session_state.selected_doc_id:
        if st.button("Clear current chat"):
            st.session_state.chat_histories[st.session_state.selected_doc_id] = []
            st.rerun()

tab_ingest, tab_chat, tab_status = st.tabs(["Ingest", "Chat", "Status"])

with tab_ingest:
    st.subheader("Upload and ingest")
    uploaded_files = st.file_uploader(
        "Choose one or more documents",
        type=[
            "pdf", "txt", "md", "doc", "docx", "ppt", "pptx",
            "xls", "xlsx", "png", "jpg", "jpeg", "webp", "gif",
            "bmp", "tif", "tiff", "html", "htm", "xhtml"
        ],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Start ingestion", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                ingest_uploaded_files(
                    uploaded_files=uploaded_files,
                    backend=backend,
                    parse_method=parse_method,
                )

    with col2:
        st.info(
            "This calls your existing `RAGAnythingAgno.process_document_complete(...)` "
            "pipeline and stores the result in the local UI registry."
        )

    if st.session_state.last_ingest_results:
        st.markdown("#### Last ingestion results")
        st.json(st.session_state.last_ingest_results)

with tab_chat:
    st.subheader("Ask questions")

    selected_doc_id = st.session_state.selected_doc_id
    registry = load_registry()
    selected_meta = registry.get(selected_doc_id) if selected_doc_id else None

    if not selected_doc_id or not selected_meta:
        st.info("Please ingest and select a document first.")
    else:
        st.markdown(
            f"**Current document:** `{selected_meta.get('original_name')}`  \n"
            f"**doc_id:** `{selected_doc_id}`"
        )

        render_chat(selected_doc_id)

        question = st.chat_input("Ask something about the selected document")
        if question:
            history = get_chat_history(selected_doc_id)
            history.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(selected_doc_id, question)
                    except Exception as exc:
                        answer = f"Query failed:\n\n{exc}"
                    st.markdown(answer)

            history.append({"role": "assistant", "content": answer})

with tab_status:
    st.subheader("Document status and metadata")

    selected_doc_id = st.session_state.selected_doc_id
    registry = load_registry()

    if not selected_doc_id or selected_doc_id not in registry:
        st.info("No selected document.")
    else:
        doc_meta = registry[selected_doc_id]
        render_doc_details(doc_meta)

        if st.button("Refresh backend status"):
            try:
                live_status = refresh_selected_doc_status(selected_doc_id)
                st.markdown("#### Live backend status")
                st.json(live_status)
            except Exception as exc:
                st.exception(exc)

        file_path = doc_meta.get("file_path")
        if file_path and Path(file_path).exists():
            st.markdown("#### Stored file")
            st.code(file_path)
        else:
            st.warning("Stored file path no longer exists on disk.")