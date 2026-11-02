import os
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

# 1. Disable OpenAI tracing (since we aren't using an OpenAI API key)
set_tracing_disabled(True)

# 2. Point the OpenAI client to Google's Gemini API
gemini_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# 3. Create the agent, explicitly assigning the Gemini model and custom client
agent = Agent(
    name="Research Assistant",
    instructions="You are a concise research assistant. Answer clearly and directly.",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=gemini_client
    ),
)

# 4. Run the agent
print("🤖 Running agent...\n")
result = Runner.run_sync(
    agent,
    "What are the latest developments in Python?"
)

print("--- Result ---")
print(result.final_output)
