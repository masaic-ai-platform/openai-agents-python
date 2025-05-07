import os
import asyncio
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.models.openai_responses import OpenAIResponsesModel

BASE_URL = os.getenv("OPEN_RESPONSES_URL") or "http://localhost:8080/v1" #Either set OPEN_RESPONSES_URL in environment variable or put it directly here.
API_KEY = os.getenv("GROQ_API_KEY") or "" #Either set GROQ_API_KEY in environment variable or put it directly here.
MODEL_NAME = "groq@qwen-2.5-32b"
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a humorous person who can tell funny jokes.",
        model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client)
    )

    result = await Runner.run(agent, "Tell me a joke")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
