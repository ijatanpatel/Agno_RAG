import asyncio
from src.rag_agno import RAGAnythingAgno, Settings

import nest_asyncio
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv() 
langfuse = get_client()

 
# # Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")

# import openlit
# openlit.init(tracer=langfuse._otel_tracer, disable_batch=True)

FILE_PATH = r"C:\Udemy\agno_rag\test\Chapter-2.pdf"

main_result = None
async def main():
    settings = Settings()
    print("settings.openai_api_key loaded:", bool(settings.openai_api_key))

    rag = RAGAnythingAgno(settings)

    print(f"\nTesting file: {FILE_PATH}\n")

    result1 = await rag.process_document_complete(
        FILE_PATH,
        backend="pipeline",
    )
    # print("FIRST INGEST RESULT:")
    # print(result1.model_dump())

    status1 = rag.get_doc_status(result1.doc_id)
    print("\nDOC STATUS AFTER FIRST INGEST:")
    print(status1)

    # if result1.status == "PROCESSED":
    #     answer1 = await rag.aquery(
    #         "Explain me the Balance Sheet Format.",
    #         doc_id=result1.doc_id,
    #     )
    #     print("\nQUERY ANSWER:")
    #     print(answer1)

    # result2 = await rag.process_document_complete(
    #     FILE_PATH,
    #     backend="pipeline",
    # )
    # print("\nSECOND INGEST RESULT:")
    # print(result2.model_dump())

    # print("\nCACHE CHECK:")
    # print("First run parse_cache_hit:", result1.parse_cache_hit)
    # print("Second run parse_cache_hit:", result2.parse_cache_hit)
    return result1


if __name__ == "__main__":
    main_result = asyncio.run(main())