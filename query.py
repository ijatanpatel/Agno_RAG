
import asyncio
from src.rag_agno import RAGAnythingAgno, Settings
from test_direct_ingest import main_result

async def main():
    settings = Settings()
    print("settings.openai_api_key loaded:", bool(settings.openai_api_key))

    rag = RAGAnythingAgno(settings)

    
    answer1 = await rag.aquery(
                "What is the format of Balance sheet?",
                doc_id=main_result.doc_id,
            )
    print("\nQUERY ANSWER:")
    print(answer1)


if __name__ == "__main__":
    asyncio.run(main())