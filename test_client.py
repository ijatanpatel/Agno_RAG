import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

print("Loading .env from:", ENV_PATH)
print(".env exists:", ENV_PATH.exists())

load_dotenv(dotenv_path=ENV_PATH, override=True)

print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"OPENAI_API_KEY not found. Checked: {ENV_PATH}"
        )

    tools = MCPTools(
        transport="streamable-http",
        url="http://127.0.0.1:8000/mcp",
    )
    await tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(
                id="gpt-4.1-mini",
                api_key=api_key,   # pass explicitly
            ),
            tools=[tools],
        )

        result = await agent.arun("List all available MCP tools.")
        print(result.content)

    finally:
        await tools.close()

if __name__ == "__main__":
    asyncio.run(main())