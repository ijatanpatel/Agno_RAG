from typing import Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.vectordb.lancedb import LanceDb, SearchType

from .config import Settings
from .stores import SQLiteStateStore


class AgnoKnowledgeIndex:
    def __init__(self, settings: Settings, state_store: SQLiteStateStore) -> None:
        self.settings = settings
        self.state_store = state_store
        self.contents_db = SqliteDb(db_file=str(settings.contents_db_path))
        self.vector_db = LanceDb(
            uri=str(settings.vector_db_uri),
            table_name="rag_anything_docs",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(
                id=settings.embedding_model,
                api_key=settings.openai_api_key,
            ),
        )
        self.knowledge = Knowledge(
            name="RAGAnythingAgnoKnowledge",
            description="Indexed text chunks and multimodal summaries",
            contents_db=self.contents_db,
            vector_db=self.vector_db,
        )

    async def insert_text(
        self,
        name: str,
        text: str,
        metadata: dict,
    ) -> None:
        await self.knowledge.ainsert(
            name=name,
            text_content=text,
            metadata=metadata,
        )

    def build_query_agent(
        self,
        doc_id: Optional[str] = None,
    ) -> Agent:
        def search_document_graph(query: str) -> str:
            hits = self.state_store.search_triples(query=query, doc_id=doc_id, limit=12)
            if not hits:
                return "No graph hits found."
            lines = []
            for hit in hits:
                page = hit.get("page_idx")
                page_str = "unknown" if page is None else str(page)
                lines.append(
                    f"[{hit['source_type']}][page={page_str}] "
                    f"{hit['subject']} --{hit['predicate']}--> {hit['object']}"
                )
            return "\n".join(lines)

        search_document_graph.__name__ = "search_document_graph"
        search_document_graph.__doc__ = (
            "Search the document knowledge graph for entity relations and semantic triples."
        )

        instructions = [
            "Answer only from the indexed document knowledge when possible.",
            "First search the knowledge base.",
            "Also use search_document_graph when entity relations matter.",
            "If the answer is not grounded in retrieved evidence, say so clearly.",
        ]

        if doc_id:
            instructions.append(f"Prefer evidence connected to document id: {doc_id}")

        return Agent(
            name="RAGAnythingAgnoQueryAgent",
            model=OpenAIResponses(
                id=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
            ),
            knowledge=self.knowledge,
            search_knowledge=True,
            tools=[search_document_graph],
            markdown=True,
            instructions=instructions,
        )