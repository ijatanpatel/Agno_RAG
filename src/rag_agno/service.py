import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings
from .context import ContextExtractor
from .knowledge import AgnoKnowledgeIndex
from .processors import (
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
    TextChunkEnricher,
)
from .schemas import IngestResult
from .stores import SQLiteStateStore
from .utils import (
    chunk_text,
    compute_mdhash_id,
    generate_cache_key,
    generate_content_based_doc_id,
    separate_content,
)

# Keep this file unchanged from your current RAG-Anything repo:
from .vendor.parser import DoclingParser, MineruParser, Parser


class RAGAnythingAgno:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_dirs()

        self.state_store = SQLiteStateStore(self.settings.state_db_path)

        # Keep parsing behavior aligned with parser.py
        self.mineru_parser = MineruParser()
        self.docling_parser = DoclingParser()

        self.context_extractor = ContextExtractor(
            window_pages=self.settings.context_window_pages,
            max_context_chars=self.settings.max_context_chars,
        )
        self.text_enricher = TextChunkEnricher(self.settings)
        self.knowledge_index = AgnoKnowledgeIndex(self.settings, self.state_store)

        self.modal_processors = {
            "image": ImageModalProcessor(self.settings, self.context_extractor),
            "table": TableModalProcessor(self.settings, self.context_extractor),
            "equation": EquationModalProcessor(self.settings, self.context_extractor),
            "generic": GenericModalProcessor(self.settings, self.context_extractor),
        }

    def _file_ref(self, file_path: str) -> str:
        return file_path if self.settings.store_full_path_references else Path(file_path).name

    def _pick_modal_processor(self, item_type: str):
        return self.modal_processors.get(item_type, self.modal_processors["generic"])

    async def parse_document(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        parse_method: str = "auto",
        display_stats: bool = False,
        backend: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        del display_stats  # kept only for API parity

        output_dir = output_dir or str(self.settings.parser_output_dir)
        cache_key = generate_cache_key(file_path, parse_method, kwargs)
        cached = self.state_store.get_parse_cache(cache_key)
        current_mtime = os.path.getmtime(file_path)

        if cached and cached.get("mtime") == current_mtime:
            return cached["content_list"], cached["doc_id"], True

        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            content_list = await asyncio.to_thread(
                self.mineru_parser.parse_pdf,
                file_path,
                output_dir,
                parse_method,
                kwargs.get("lang"),
                backend,
                **kwargs,
            )
        elif suffix in Parser.IMAGE_FORMATS:
            content_list = await asyncio.to_thread(
                self.mineru_parser.parse_image,
                file_path,
                output_dir,
                kwargs.get("lang"),
                kwargs.get("backend"),
                kwargs.get("device"),
                kwargs.get("source"),
            )
        elif suffix in Parser.OFFICE_FORMATS:
            content_list = await asyncio.to_thread(
                self.docling_parser.parse_office_doc,
                file_path,
                output_dir,
                parse_method,
                kwargs.get("lang"),
                **kwargs,
            )
        elif suffix in getattr(DoclingParser, "HTML_FORMATS", {".html", ".htm", ".xhtml"}):
            content_list = await asyncio.to_thread(
                self.docling_parser.parse_html,
                file_path,
                output_dir,
                kwargs.get("lang"),
                **kwargs,
            )
        elif suffix in Parser.TEXT_FORMATS:
            content_list = await asyncio.to_thread(
                self.mineru_parser.parse_text_file,
                file_path,
                output_dir,
                kwargs.get("lang"),
            )
        else:
            content_list = await asyncio.to_thread(
                self.mineru_parser.parse_document,
                file_path,
                output_dir,
                parse_method,
                kwargs.get("lang"),
                **kwargs,
            )

        if not content_list:
            raise ValueError(f"No content extracted from {file_path}")

        doc_id = generate_content_based_doc_id(content_list)

        self.state_store.upsert_parse_cache(
            cache_key,
            {
                "mtime": current_mtime,
                "doc_id": doc_id,
                "content_list": content_list,
            },
        )
        return content_list, doc_id, False

    def set_content_source_for_context(
        self,
        content_source: Any,
        content_format: str = "mineru",
    ) -> None:
        for processor in self.modal_processors.values():
            processor.set_content_source(content_source, content_format)

    async def _insert_text_content(
        self,
        text_content: str,
        file_path: str,
        doc_id: str,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
    ) -> int:
        chunks = chunk_text(
            text=text_content,
            chunk_size=self.settings.text_chunk_size,
            overlap=self.settings.text_chunk_overlap,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
        )
        if not chunks:
            return 0

        semaphore = asyncio.Semaphore(self.settings.max_workers)
        file_ref = self._file_ref(file_path)

        async def process_chunk(idx: int, chunk: str) -> None:
            async with semaphore:
                chunk_id = compute_mdhash_id(f"{doc_id}:{idx}:{chunk}", "chunk-")
                enrichment = await self.text_enricher.process(
                    text=chunk,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )

                searchable_text = "\n".join(
                    [
                        f"[doc_id] {doc_id}",
                        f"[chunk_id] {chunk_id}",
                        f"[source] text",
                        f"[file] {file_ref}",
                        f"[summary] {enrichment.summary}",
                        f"[keywords] {', '.join(enrichment.keywords)}",
                        "",
                        chunk,
                        "",
                        enrichment.indexable_text,
                    ]
                ).strip()

                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source_type": "text",
                    "file_path": file_ref,
                    "chunk_index": idx,
                }

                await self.knowledge_index.insert_text(
                    name=f"{doc_id}-text-{idx}",
                    text=searchable_text,
                    metadata=metadata,
                )

                self.state_store.add_triples(
                    doc_id=doc_id,
                    source_id=chunk_id,
                    source_type="text",
                    page_idx=None,
                    triples=[t.model_dump() for t in enrichment.triples],
                )

        await asyncio.gather(*(process_chunk(i, c) for i, c in enumerate(chunks)))
        return len(chunks)

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
    ) -> int:
        if not multimodal_items:
            status = self.state_store.get_doc_status(doc_id) or {}
            status["multimodal_processed"] = True
            self.state_store.upsert_doc_status(doc_id, status)
            return 0

        existing = self.state_store.get_doc_status(doc_id) or {}
        if existing.get("multimodal_processed"):
            return 0

        semaphore = asyncio.Semaphore(self.settings.max_workers)
        file_ref = self._file_ref(file_path)

        async def process_item(index: int, item: Dict[str, Any]) -> None:
            async with semaphore:
                item_type = item.get("type", "generic")
                source_id = compute_mdhash_id(
                    f"{doc_id}:{item_type}:{index}:{str(item)}",
                    "mm-",
                )
                processor = self._pick_modal_processor(item_type)
                enrichment = await processor.process(item, doc_id, source_id)

                page_idx = item.get("page_idx")
                searchable_text = "\n".join(
                    [
                        f"[doc_id] {doc_id}",
                        f"[source_id] {source_id}",
                        f"[source] {item_type}",
                        f"[page] {page_idx}",
                        f"[file] {file_ref}",
                        f"[summary] {enrichment.summary}",
                        f"[keywords] {', '.join(enrichment.keywords)}",
                        f"[entities] {', '.join(enrichment.entities)}",
                        "",
                        enrichment.indexable_text,
                    ]
                ).strip()

                metadata = {
                    "doc_id": doc_id,
                    "source_id": source_id,
                    "source_type": item_type,
                    "file_path": file_ref,
                    "page_idx": page_idx,
                    "item_index": index,
                }

                await self.knowledge_index.insert_text(
                    name=f"{doc_id}-{item_type}-{index}",
                    text=searchable_text,
                    metadata=metadata,
                )

                self.state_store.add_triples(
                    doc_id=doc_id,
                    source_id=source_id,
                    source_type=item_type,
                    page_idx=page_idx,
                    triples=[t.model_dump() for t in enrichment.triples],
                )

        try:
            await asyncio.gather(
                *(process_item(i, item) for i, item in enumerate(multimodal_items))
            )
        finally:
            status = self.state_store.get_doc_status(doc_id) or {}
            status["multimodal_processed"] = True
            self.state_store.upsert_doc_status(doc_id, status)

        return len(multimodal_items)

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        doc_id: Optional[str] = None,
        display_stats: bool = False,
    ) -> IngestResult:
        del display_stats  # kept only for API parity

        doc_id = doc_id or generate_content_based_doc_id(content_list)
        self.state_store.upsert_doc_status(
            doc_id,
            {
                "status": "HANDLING",
                "file_path": file_path,
                "multimodal_processed": False,
            },
        )

        try:
            text_content, multimodal_items = separate_content(content_list)
            self.set_content_source_for_context(content_list, self.settings.content_format)

            text_chunks = 0
            if text_content.strip():
                text_chunks = await self._insert_text_content(
                    text_content=text_content,
                    file_path=file_path,
                    doc_id=doc_id,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                )

            multimodal_count = await self._process_multimodal_content(
                multimodal_items=multimodal_items,
                file_path=file_path,
                doc_id=doc_id,
            )

            self.state_store.upsert_doc_status(
                doc_id,
                {
                    "status": "PROCESSED",
                    "file_path": file_path,
                    "multimodal_processed": True,
                    "text_chunks": text_chunks,
                    "multimodal_items": multimodal_count,
                },
            )

            return IngestResult(
                doc_id=doc_id,
                file_path=file_path,
                status="PROCESSED",
                text_chunks=text_chunks,
                multimodal_items=multimodal_count,
            )
        except Exception as exc:
            self.state_store.upsert_doc_status(
                doc_id,
                {
                    "status": "FAILED",
                    "file_path": file_path,
                    "error": str(exc),
                    "multimodal_processed": False,
                },
            )
            return IngestResult(
                doc_id=doc_id,
                file_path=file_path,
                status="FAILED",
                error=str(exc),
            )

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        parse_method: str = "auto",
        display_stats: bool = False,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        doc_id: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ) -> IngestResult:
        pre_doc_id = doc_id or f"doc-pre-{self._file_ref(file_path)}"
        self.state_store.upsert_doc_status(
            pre_doc_id,
            {"status": "HANDLING", "file_path": file_path, "multimodal_processed": False},
        )

        try:
            content_list, parsed_doc_id, cache_hit = await self.parse_document(
                file_path=file_path,
                output_dir=output_dir,
                parse_method=parse_method,
                display_stats=display_stats,
                backend=backend,                                                                                                    
                **kwargs,
            )
            final_doc_id = doc_id or parsed_doc_id

            result = await self.insert_content_list(
                content_list=content_list,
                file_path=file_path,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                doc_id=final_doc_id,
            )
            result.parse_cache_hit = cache_hit
            return result

        except Exception as exc:
            self.state_store.upsert_doc_status(
                pre_doc_id,
                {
                    "status": "FAILED",
                    "file_path": file_path,
                    "error": str(exc),
                    "multimodal_processed": False,
                },
            )
            return IngestResult(
                doc_id=pre_doc_id,
                file_path=file_path,
                status="FAILED",
                error=str(exc),
            )

    async def process_document_complete_lightrag_api(
        self,
        *args,
        **kwargs,
    ) -> IngestResult:
        return await self.process_document_complete(*args, **kwargs)

    async def process_documents_batch(
        self,
        file_paths: List[str],
        max_workers: Optional[int] = None,
    ) -> List[IngestResult]:
        semaphore = asyncio.Semaphore(max_workers or self.settings.max_workers)

        async def _one(path: str) -> IngestResult:
            async with semaphore:
                return await self.process_document_complete(path)

        return await asyncio.gather(*(_one(p) for p in file_paths))

    async def process_folder_complete(
        self,
        folder_path: str,
        recursive: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[IngestResult]:
        folder = Path(folder_path)
        supported = (
            {".pdf"}
            | Parser.OFFICE_FORMATS
            | Parser.IMAGE_FORMATS
            | Parser.TEXT_FORMATS
            | getattr(DoclingParser, "HTML_FORMATS", {".html", ".htm", ".xhtml"})
        )

        pattern = "**/*" if recursive else "*"
        file_paths = [
            str(p)
            for p in folder.glob(pattern)
            if p.is_file() and p.suffix.lower() in supported
        ]
        return await self.process_documents_batch(file_paths, max_workers=max_workers)

    async def aquery(
        self,
        question: str,
        doc_id: Optional[str] = None,
    ) -> str:
        agent = self.knowledge_index.build_query_agent(doc_id=doc_id)

        def _run() -> str:
            result = agent.run(question)
            return str(getattr(result, "content", result))

        return await asyncio.to_thread(_run)

    def get_doc_status(self, doc_id: str) -> Dict[str, Any]:
        return self.state_store.get_doc_status(doc_id) or {}