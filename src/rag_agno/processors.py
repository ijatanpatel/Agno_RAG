import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIResponses

from .config import Settings
from .context import ContextExtractor
from .schemas import EnrichmentResult, Triple
from .utils import extract_run_text, safe_json_loads, truncate_text


class JsonEnricher:
    def __init__(
        self,
        model_id: str,
        instructions: List[str],
        api_key: Optional[str] = None,
    ) -> None:
        self.agent = Agent(
            model=OpenAIResponses(id=model_id, api_key=api_key),
            instructions=instructions,
            markdown=False,
        )

    async def run_json(
        self,
        prompt: str,
        images: Optional[List[Image]] = None,
    ) -> Dict[str, Any]:
        def _run() -> Any:
            kwargs: Dict[str, Any] = {}
            if images:
                kwargs["images"] = images
            return self.agent.run(prompt, **kwargs)

        result = await asyncio.to_thread(_run)
        return safe_json_loads(extract_run_text(result))


class TextChunkEnricher:
    def __init__(self, settings: Settings) -> None:
        self.runner = JsonEnricher(
            model_id=settings.llm_model,
            api_key=settings.openai_api_key,
            instructions=[
                "You enrich document chunks for a multimodal RAG system.",
                "Return strict JSON only.",
                "Extract a concise summary, keywords, entities, and knowledge triples.",
            ],
        )

    async def process(self, text: str, doc_id: str, chunk_id: str) -> EnrichmentResult:
        prompt = f"""
Return strict JSON only with this schema:
{{
  "summary": "short summary",
  "keywords": ["k1", "k2"],
  "entities": ["e1", "e2"],
  "triples": [
    {{"subject": "A", "predicate": "related_to", "object": "B"}}
  ],
  "indexable_text": "compact searchable enrichment text"
}}

Document ID: {doc_id}
Chunk ID: {chunk_id}

TEXT:
{text}
""".strip()

        data = await self.runner.run_json(prompt)
        return self._normalize(data, text)

    def _normalize(self, data: Dict[str, Any], text: str) -> EnrichmentResult:
        triples = []
        for item in data.get("triples", []) or []:
            if all(k in item for k in ("subject", "predicate", "object")):
                triples.append(Triple(**item))

        summary = str(data.get("summary", "")).strip() or truncate_text(text, 400)
        keywords = [str(x).strip() for x in (data.get("keywords") or []) if str(x).strip()]
        entities = [str(x).strip() for x in (data.get("entities") or []) if str(x).strip()]
        indexable_text = str(data.get("indexable_text", "")).strip()

        if not indexable_text:
            indexable_text = "\n".join(
                [
                    f"Summary: {summary}",
                    f"Keywords: {', '.join(keywords)}",
                    f"Entities: {', '.join(entities)}",
                ]
            ).strip()

        return EnrichmentResult(
            summary=summary,
            keywords=keywords,
            entities=entities,
            triples=triples,
            indexable_text=indexable_text,
            raw=data,
        )


class BaseModalProcessor:
    item_type = "generic"

    def __init__(self, settings: Settings, context_extractor: ContextExtractor) -> None:
        self.settings = settings
        self.context_extractor = context_extractor
        self.content_source: Any = None
        self.content_format: str = "mineru"
        self.runner = JsonEnricher(
            model_id=settings.vision_model,
            api_key=settings.openai_api_key,
            instructions=[
                "You analyze multimodal document items for a multimodal RAG system.",
                "Return strict JSON only.",
                "Extract summary, keywords, entities, triples, and compact searchable text.",
            ],
        )

    def set_content_source(self, content_source: Any, content_format: str = "mineru") -> None:
        self.content_source = content_source
        self.content_format = content_format

    def get_context(self, item: Dict[str, Any]) -> str:
        return self.context_extractor.extract_context(
            self.content_source,
            item,
            self.content_format,
        )

    async def process(self, item: Dict[str, Any], doc_id: str, source_id: str) -> EnrichmentResult:
        prompt = self.build_prompt(item, doc_id, source_id)
        images = self.build_images(item)
        data = await self.runner.run_json(prompt, images=images)
        return self.normalize(item, data)

    def build_images(self, item: Dict[str, Any]) -> Optional[List[Image]]:
        img_path = item.get("img_path")
        if img_path and Path(img_path).exists():
            return [Image(filepath=Path(img_path))]
        return None

    def build_prompt(self, item: Dict[str, Any], doc_id: str, source_id: str) -> str:
        raise NotImplementedError

    def normalize(self, item: Dict[str, Any], data: Dict[str, Any]) -> EnrichmentResult:
        triples = []
        for row in data.get("triples", []) or []:
            if all(k in row for k in ("subject", "predicate", "object")):
                triples.append(Triple(**row))

        summary = str(data.get("summary", "")).strip()
        if not summary:
            summary = f"{item.get('type', 'item')} on page {item.get('page_idx', 'unknown')}"

        keywords = [str(x).strip() for x in (data.get("keywords") or []) if str(x).strip()]
        entities = [str(x).strip() for x in (data.get("entities") or []) if str(x).strip()]
        indexable_text = str(data.get("indexable_text", "")).strip() or summary

        return EnrichmentResult(
            summary=summary,
            keywords=keywords,
            entities=entities,
            triples=triples,
            indexable_text=indexable_text,
            raw=data,
        )


class ImageModalProcessor(BaseModalProcessor):
    item_type = "image"

    def build_prompt(self, item: Dict[str, Any], doc_id: str, source_id: str) -> str:
        context = self.get_context(item)
        caption = item.get("image_caption", "")
        footnote = item.get("image_footnote", "")
        return f"""
Return strict JSON only with this schema:
{{
  "summary": "what the image conveys in the document context",
  "keywords": ["k1", "k2"],
  "entities": ["e1", "e2"],
  "triples": [{{"subject": "A", "predicate": "shows", "object": "B"}}],
  "indexable_text": "compact searchable image description"
}}

Document ID: {doc_id}
Source ID: {source_id}
Page: {item.get("page_idx", -1)}
Caption: {caption}
Footnote: {footnote}

NEARBY TEXT CONTEXT:
{context}
""".strip()


class TableModalProcessor(BaseModalProcessor):
    item_type = "table"

    def build_prompt(self, item: Dict[str, Any], doc_id: str, source_id: str) -> str:
        context = self.get_context(item)
        table_body = item.get("table_body", "")
        table_text = json.dumps(table_body, ensure_ascii=False) if not isinstance(table_body, str) else table_body
        return f"""
Return strict JSON only with this schema:
{{
  "summary": "what the table means",
  "keywords": ["k1", "k2"],
  "entities": ["e1", "e2"],
  "triples": [{{"subject": "A", "predicate": "has_value", "object": "B"}}],
  "indexable_text": "compact searchable table description"
}}

Document ID: {doc_id}
Source ID: {source_id}
Page: {item.get("page_idx", -1)}
Caption: {item.get("table_caption", "")}
Footnote: {item.get("table_footnote", "")}

TABLE BODY:
{table_text}

NEARBY TEXT CONTEXT:
{context}
""".strip()


class EquationModalProcessor(BaseModalProcessor):
    item_type = "equation"

    def build_prompt(self, item: Dict[str, Any], doc_id: str, source_id: str) -> str:
        context = self.get_context(item)
        return f"""
Return strict JSON only with this schema:
{{
  "summary": "plain-language meaning of the equation",
  "keywords": ["k1", "k2"],
  "entities": ["e1", "e2"],
  "triples": [{{"subject": "A", "predicate": "depends_on", "object": "B"}}],
  "indexable_text": "compact searchable equation description"
}}

Document ID: {doc_id}
Source ID: {source_id}
Page: {item.get("page_idx", -1)}

EQUATION / FORMULA TEXT:
{item.get("text", "")}

NEARBY TEXT CONTEXT:
{context}
""".strip()


class GenericModalProcessor(BaseModalProcessor):
    item_type = "generic"

    def build_prompt(self, item: Dict[str, Any], doc_id: str, source_id: str) -> str:
        context = self.get_context(item)
        return f"""
Return strict JSON only with this schema:
{{
  "summary": "what this item means",
  "keywords": ["k1", "k2"],
  "entities": ["e1", "e2"],
  "triples": [{{"subject": "A", "predicate": "related_to", "object": "B"}}],
  "indexable_text": "compact searchable item description"
}}

Document ID: {doc_id}
Source ID: {source_id}
Item type: {item.get("type", "unknown")}
Page: {item.get("page_idx", -1)}

RAW ITEM:
{json.dumps(item, ensure_ascii=False, default=str)}

NEARBY TEXT CONTEXT:
{context}
""".strip()