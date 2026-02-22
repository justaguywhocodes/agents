import asyncio
import os
from dotenv import load_dotenv
from picoagents import Agent, OpenAIChatCompletionClient

load_dotenv()

def get_weather(location: str) -> str:
    """Get current weather for a given location."""
    return f"The weather in {location} is sunny, 75°F"

async def main():

    agent = Agent(
        name="assistant",
        description="You are a helpful assistant.",
        instructions="You are helpful. Use tools when appropriate.",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        tools=[get_weather]
    )

    async for event in agent.run_stream("What's the weather in Paris?"):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())