import os
import sys

from ddgs import DDGS
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, SQLiteSession, OpenAIChatCompletionsModel,
    function_tool, set_tracing_disabled,
)

set_tracing_disabled(True)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_SEARCHES = 3
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


# ---------- Tools ----------
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


# ---------- Agents ----------
calculator = Agent(
    name="Calculator",
    handoff_description="Computes order totals from a price and quantity.",
    instructions=(
        "You compute costs. Use calculate_total for EVERY multiplication, "
        "including hours and alternative scenarios; never calculate in your head. "
        "If a price or quantity is missing, ask the user for it. "
        "Write the final answer: first the result with the inputs used, "
        "then, only if sources were given to you, a 'Sources:' list."
    ),
    tools=[calculate_total, web_search],
    model=model,
)

researcher = Agent(
    name="Researcher",
    handoff_description="Finds current prices, news and facts on the web.",
    instructions=(
        "You research current information. Use web_search at most 3 times. "
        "Base answers only on search results and use exact prices they show. "
        "Label anything not in the results as an assumption. "
        "If the user also needs a total or any multiplication, hand off to the "
        "Calculator and include in your handoff the price(s) you found, "
        "the quantity, and the source URLs. "
        "If a quantity was given, ALWAYS hand off to the Calculator after researching. If the request is ambiguous (for example whether a limit is per item or in total), ask one short question first. Otherwise answer directly and end with a 'Sources:' list of URLs used."
    ),
    tools=[web_search],
    handoffs=[calculator],
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


# ---------- Chat loop ----------
def main() -> None:
    session = SQLiteSession("multi_user_1", "multi_conversations.db")
    print("Multi-agent chat. Type 'quit' to exit.\n")
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
                print(f"Agent error ({type(exc).__name__}): {exc}\n")
                continue

            print(f"[answered by: {result.last_agent.name}]")
            print(f"Agent: {result.final_output}\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
