import asyncio
import os
from dotenv import load_dotenv

from src.mcp_research_assistant import MCPResearchAssistant


async def main():
    load_dotenv()
    openai_key = os.environ["OPENAI_API_KEY"]
    tavily_key = os.environ.get("TAVILY_API_KEY", "")

    assistant = MCPResearchAssistant(openai_key, tavily_key)

    query = "what is the best conversational AI assistant"
    #query1 = "Best techniques for LLM optimization?"
    result = await assistant.answer_query(query, max_papers=8)

    print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
