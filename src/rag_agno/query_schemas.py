from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    normalized_question: str
    intent: str = "fact_lookup"
    search_focus: List[str] = Field(default_factory=list)
    needs_graph_lookup: bool = False
    needs_multimodal_focus: bool = False
    answer_style: str = "concise"
    notes: str = ""


class QueryResult(BaseModel):
    question: str
    doc_id: Optional[str] = None
    plan: Dict[str, Any]
    draft_answer: str
    final_answer: str