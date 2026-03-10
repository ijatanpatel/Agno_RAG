import asyncio
from src.rag_agno.query_pipeline import AgnoRAGQueryPipeline

DOC_ID = "doc-28667f788e5b12cb2fa6829ae1084e1e"

async def main():
    pipeline = AgnoRAGQueryPipeline()
    try:
        result = await pipeline.answer(
            question="Explain me deferred tax computation and its impact and defereed tax liability (net), I want exact figures showing the exact values. Use the tables for the exact values and calculations",
            doc_id=DOC_ID,
            session_id="demo-session-1",
            user_id="demo-user",
        )

        print("\n=== PLAN ===\n")
        print(result.plan)

        print("\n=== DRAFT ANSWER ===\n")
        print(result.draft_answer)

        print("\n=== FINAL ANSWER ===\n")
        print(result.final_answer)
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())