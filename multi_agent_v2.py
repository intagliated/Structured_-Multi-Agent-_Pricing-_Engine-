import os
import sys

from ddgs import DDGS
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import (
    Agent, Runner, SQLiteSession, OpenAIChatCompletionsModel,
    RunContextWrapper, function_tool, handoff, set_tracing_disabled,
)

set_tracing_disabled(True)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_SEARCHES = 5
SEARCH_STATE = {"count": 0}

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("Error: set the GEMINI_API_KEY environment variable first.")

client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
    timeout=60,
)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)


class PriceRecord(BaseModel):
    item: str = Field(description="What is being priced, e.g. 'lily stem'.")
    unit_price: float = Field(
        description="Price for ONE unit, exactly as shown in a search result."
    )
    currency: str = Field(description="Currency code or symbol, e.g. 'INR' or 'Rs'.")
    unit: str = Field(
        description="What the price covers: 'stem', 'bouquet', 'GPU-hour', etc."
    )
    quantity: float = Field(description="How many units the user wants.")
    source_url: str = Field(description="URL of the page where the price was found.")
    is_estimate: bool = Field(
        description="True if the price is NOT stated exactly in a search result."
    )


async def on_calculator_handoff(ctx: RunContextWrapper, record: PriceRecord):
    print(f"[record] {record.model_dump()}")
    if not record.source_url.startswith("http"):
        print("[warning] source_url does not look like a URL")


@function_tool
def calculate_total(price: float, quantity: float) -> float:
    """Multiply a price by a quantity (units, hours, etc.)."""
    print(f"[tool] calculate_total(price={price}, quantity={quantity})")
    if price < 0 or quantity <= 0:
        return "Error: price must be >= 0 and quantity must be > 0."
    return round(price * quantity, 2)


@function_tool
def web_search(query: str) -> str:
    """Search the web for current prices, news or facts."""
    SEARCH_STATE["count"] += 1
    if SEARCH_STATE["count"] > MAX_SEARCHES:
        print("[search limit reached]")
        return "Search limit reached. Do NOT search again. Use the results you have."
    print(f"[tool] web_search(query={query!r})")
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


calculator = Agent(
    name="Calculator",
    handoff_description="Computes order totals from a price and quantity.",
    instructions=(
        "You compute costs. If you received a price record in the handoff, use its "
        "unit_price and quantity EXACTLY; do not substitute other numbers. "
        "Use calculate_total for EVERY multiplication; never calculate in your head. "
        "If a price or quantity is missing, just ask the user for it. Do NOT search the web and Do NOT transfer to any other agent. "
        "Write the answer as: item, unit price and unit, quantity, total. "
        "If is_estimate is true, say the price is an estimate. "
        "Compare against any maximum price the user gave and say whether it fits. "
        "End with 'Source:' and the record's source_url when one was provided."
    ),
    tools=[calculate_total],
    model=model,
)

calculator_handoff = handoff(
    agent=calculator,
    tool_description_override=(
        "Hand off to the Calculator. You MUST fill in every field of the price "
        "record from your search results first."
    ),
    on_handoff=on_calculator_handoff,
    input_type=PriceRecord,
)

researcher = Agent(
    name="Researcher",
    handoff_description="Finds current prices, news and facts on the web.",
    instructions=(
        "You research current information. Use web_search at most 3 times. "
        "Base answers only on search results and use the exact price a result shows. "
        "Never invent an average. If no result states a price, say so. "
        "When the user needs a total, hand off to the Calculator and fill the price "
        "record with the single price you chose, the user's quantity and the URL of "
        "the page it came from. If the request is ambiguous (for example whether a "
        "limit is per item or in total), ask one short question first. "
        "If no total is needed, answer directly and end with 'Sources:' and the URLs."
    ),
    tools=[web_search],
    handoffs=[calculator_handoff],
    model=model,
)

triage = Agent(
    name="Triage",
    instructions=(
        "You route requests. Do not answer them yourself. "
        "If the user needs current information (prices, news, facts), hand off "
        "to the Researcher. If they already gave a price and quantity and only "
        "need a total, hand off to the Calculator. "
        "If key details are missing, ask one short question first."
    ),
    handoffs=[researcher, calculator],
    model=model,
)


def main() -> None:
    session = SQLiteSession("multi_v2_user", "multi_v2_conversations.db")
    print("Structured multi-agent chat. Type 'quit' to exit.\n")
    try:
        while True:
            try:
                message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message.lower() in {"quit", "exit"}:
                break

            SEARCH_STATE["count"] = 0
            try:
                result = Runner.run_sync(
                    triage, message, session=session, max_turns=25
                )
            except Exception as exc:
                print(f"Agent error ({type(exc).__name__}): {getattr(exc, 'body', None) or exc}\n")
                continue

            print(f"[answered by: {result.last_agent.name}]")
            print(f"Agent: {result.final_output}\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
