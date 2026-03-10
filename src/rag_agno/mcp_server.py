import json
import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .config import Settings
from .service import RAGAnythingAgno

settings = Settings()
service = RAGAnythingAgno(settings)
mcp = FastMCP("rag_anything_agno")


@mcp.tool()
async def ingest_file(path: str, backend: str = "pipeline", parse_method: str = "auto") -> Dict[str, Any]:
    result = await service.process_document_complete(
        path,
        backend=backend,
        parse_method=parse_method,
    )
    return result.model_dump()


@mcp.tool()
async def ingest_folder(
    path: str,
    recursive: bool = True,
    max_workers: int = 4,
) -> Dict[str, Any]:
    results = await service.process_folder_complete(
        folder_path=path,
        recursive=recursive,
        max_workers=max_workers,
    )
    return {
        "total": len(results),
        "processed": sum(1 for r in results if r.status == "PROCESSED"),
        "failed": sum(1 for r in results if r.status == "FAILED"),
        "results": [r.model_dump() for r in results],
    }


@mcp.tool()
async def ingest_content_list(file_path: str, content_list_json: str) -> Dict[str, Any]:
    content_list = json.loads(content_list_json)
    result = await service.insert_content_list(content_list=content_list, file_path=file_path)
    return result.model_dump()


# @mcp.tool()
# async def query_documents(question: str, doc_id: Optional[str] = None) -> str:
#     return await service.aquery(question=question, doc_id=doc_id)


@mcp.tool()
def get_document_status(doc_id: str) -> Dict[str, Any]:
    return service.get_doc_status(doc_id)


@mcp.tool()
def list_indexed_documents(limit: int = 50) -> Dict[str, Any]:
    items = service.state_store.list_doc_statuses(limit=limit)
    return {"documents": items, "count": len(items)}


@mcp.tool()
def search_document_graph(
    query: str,
    doc_id: Optional[str] = None,
    limit: int = 12,
) -> Dict[str, Any]:
    hits = service.state_store.search_triples(query=query, doc_id=doc_id, limit=limit)
    return {"hits": hits, "count": len(hits)}


@mcp.tool()
def get_document_overview(doc_id: str) -> Dict[str, Any]:
    status = service.get_doc_status(doc_id)
    graph_preview = service.state_store.search_triples(query=doc_id, doc_id=doc_id, limit=5)
    return {
        "doc_id": doc_id,
        "status": status,
        "graph_preview": graph_preview,
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)