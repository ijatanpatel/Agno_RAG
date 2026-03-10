import asyncio
import json
import sys
import os
from typing import Any, Dict, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.knowledge import KnowledgeTools
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

from .config import Settings
from .query_schemas import QueryPlan, QueryResult
from .service import RAGAnythingAgno
from .utils import extract_run_text, safe_json_loads


class AgnoRAGQueryPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        backend: Optional[RAGAnythingAgno] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_dirs()

        self.backend = backend or RAGAnythingAgno(self.settings)
        self._mcp_tools: Optional[MCPTools] = None

        self._planner_agent = Agent(
            model=OpenAIResponses(
                id=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
            ),
            markdown=False,
            instructions=[
                "You are a query planner for a multimodal RAG system.",
                "Return strict JSON only.",
                "Decide the retrieval strategy before answer generation.",
            ],
        )

        self._verifier_agent = Agent(
            model=OpenAIResponses(
                id=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
            ),
            markdown=True,
            instructions=[
                "You are a grounding verifier for a multimodal RAG system.",
                "Rewrite the draft answer so it is precise, clean, and only based on provided evidence.",
                "Remove speculation, repetition, and unsupported claims.",
                "Keep useful structure and preserve technical accuracy.",
            ],
        )

    async def connect(self) -> None:
        """
        Use stdio MCP locally and force the subprocess to run in stdio mode.
        """
        if self._mcp_tools is None:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "src.rag_agno.mcp_server"],
                env={
                    **os.environ,
                    "MCP_TRANSPORT": "stdio",
                },
            )
            self._mcp_tools = MCPTools(server_params=server_params)
            await self._mcp_tools.connect()

    async def close(self) -> None:
        if self._mcp_tools is not None:
            await self._mcp_tools.close()
            self._mcp_tools = None

    async def plan(self, question: str, doc_id: Optional[str] = None) -> QueryPlan:
        prompt = f"""
Return strict JSON only with this schema:
{{
  "normalized_question": "rewritten clear question",
  "intent": "summary | fact_lookup | compare | multimodal | timeline | troubleshooting",
  "search_focus": ["text", "table", "image", "equation", "graph"],
  "needs_graph_lookup": true,
  "needs_multimodal_focus": false,
  "answer_style": "concise | detailed | step_by_step",
  "notes": "short planning note"
}}

Document ID: {doc_id}
USER QUESTION:
{question}
""".strip()

        response = await self._planner_agent.arun(prompt)
        data = safe_json_loads(extract_run_text(response))

        if not data:
            data = {
                "normalized_question": question,
                "intent": "fact_lookup",
                "search_focus": ["text"],
                "needs_graph_lookup": False,
                "needs_multimodal_focus": False,
                "answer_style": "concise",
                "notes": "",
            }

        return QueryPlan.model_validate(data)

    def _build_research_agent(self) -> Agent:
        if self._mcp_tools is None:
            raise RuntimeError("MCP tools not connected. Call connect() first.")

        knowledge_tools = KnowledgeTools(
            knowledge=self.backend.knowledge_index.knowledge,
            enable_think=True,
            enable_search=True,
            enable_analyze=True,
        )

        return Agent(
            model=OpenAIResponses(
                id=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
            ),
            knowledge=self.backend.knowledge_index.knowledge,
            search_knowledge=True,
            tools=[knowledge_tools, self._mcp_tools],
            markdown=True,
            instructions=[
                "You are the retrieval and reasoning agent for a multimodal RAG system.",
                "Use the knowledge base for the main semantic retrieval.",
                "Use MCP helper tools only for document status, doc listing, and graph lookup.",
                "Do NOT try to call a full query tool through MCP.",
                "Prefer real text chunks and meaningful multimodal summaries.",
                "Ignore discarded, empty, or clearly noisy artifacts.",
                "When the question is about tables, images, equations, or figures, actively search for multimodal evidence.",
                "Write a grounded draft answer with these sections:",
                "1. Draft Answer",
                "2. Evidence Used",
                "3. Remaining Uncertainty",
            ],
        )

    async def _run_research(
        self,
        question: str,
        plan: QueryPlan,
        doc_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        agent = self._build_research_agent()

        filters: Optional[Dict[str, Any]] = None
        if doc_id:
            filters = {"doc_id": doc_id}

        session_state = {
            "active_doc_id": doc_id,
            "query_plan": plan.model_dump(),
        }

        prompt = f"""
USER QUESTION:
{question}

QUERY PLAN:
{json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)}

TARGET DOC ID:
{doc_id}

Instructions:
- If doc_id is provided, stay within that document unless the evidence is missing.
- Call get_document_status first when doc_id is provided.
- Use KnowledgeTools to search and analyze the knowledge base.
- If the plan says graph lookup is needed, call search_document_graph.
- If doc_id is missing, you may call list_indexed_documents to identify candidate docs.
- Do not call any MCP tool that runs another full query.
- Produce a grounded draft answer only after retrieval.
""".strip()

        response = await agent.arun(
            prompt,
            knowledge_filters=filters,
            session_id=session_id,
            user_id=user_id,
            session_state=session_state,
        )
        return extract_run_text(response)

    async def verify(
        self,
        question: str,
        plan: QueryPlan,
        draft_answer: str,
        doc_id: Optional[str] = None,
    ) -> str:
        prompt = f"""
QUESTION:
{question}

DOC ID:
{doc_id}

PLAN:
{json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)}

DRAFT ANSWER:
{draft_answer}

Rewrite this into a final answer that is:
- grounded
- concise but complete
- cleanly structured
- free of unsupported claims
""".strip()

        response = await self._verifier_agent.arun(prompt)
        return extract_run_text(response)

    async def answer(
        self,
        question: str,
        doc_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> QueryResult:
        plan = await self.plan(question=question, doc_id=doc_id)

        try:
            await self.connect()
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to the local MCP helper server. "
                "Check your Python environment and package imports."
            ) from exc

        draft = await self._run_research(
            question=question,
            plan=plan,
            doc_id=doc_id,
            session_id=session_id,
            user_id=user_id,
        )
        final = await self.verify(
            question=question,
            plan=plan,
            draft_answer=draft,
            doc_id=doc_id,
        )

        return QueryResult(
            question=question,
            doc_id=doc_id,
            plan=plan.model_dump(),
            draft_answer=draft,
            final_answer=final,
        )