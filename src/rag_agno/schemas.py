from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class EnrichmentResult(BaseModel):
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    triples: List[Triple] = Field(default_factory=list)
    indexable_text: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    doc_id: str
    file_path: str
    status: str
    parse_cache_hit: bool = False
    text_chunks: int = 0
    multimodal_items: int = 0
    error: Optional[str] = None