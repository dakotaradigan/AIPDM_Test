import os
from typing import TypedDict, Optional

from openai import OpenAI
from langgraph.graph import StateGraph

# Simple prompts for each node
TRIAGE_PROMPT = (
    "Classify the user's request as either 'minimum' or 'alternative'. "
    "Respond with only one of these two words."
)
MINIMUM_PROMPT = "You answer questions about benchmark account minimums concisely."
ALTERNATIVE_PROMPT = (
    "You suggest alternative benchmarks when the requested option isn't suitable."
)

TRIAGE_MODEL = "gpt-3.5-turbo-0125"
CHAT_MODEL = "gpt-3.5-turbo"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))


class AgentState(TypedDict, total=False):
    query: str
    category: Optional[str]
    answer: Optional[str]


def triage(state: AgentState) -> AgentState:
    """Classify the query as minimum or alternative."""
    resp = client.chat.completions.create(
        model=TRIAGE_MODEL,
        messages=[
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
    )
    label = resp.choices[0].message.content.strip().lower()
    if "min" in label:
        state["category"] = "minimum"
    else:
        state["category"] = "alternative"
    return state


def minimum(state: AgentState) -> AgentState:
    """Handle minimum-related requests."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": MINIMUM_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
    )
    state["answer"] = resp.choices[0].message.content.strip()
    return state


def alternative(state: AgentState) -> AgentState:
    """Handle requests for benchmark alternatives."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ALTERNATIVE_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
    )
    state["answer"] = resp.choices[0].message.content.strip()
    return state


def route_category(state: AgentState) -> str:
    return state.get("category", "alternative")


workflow = StateGraph(AgentState)
workflow.add_node("triage", triage)
workflow.add_node("minimum", minimum)
workflow.add_node("alternative", alternative)
workflow.add_conditional_edges("triage", route_category)
workflow.set_entry_point("triage")
workflow.add_edge("minimum", workflow.END)
workflow.add_edge("alternative", workflow.END)

graph = workflow.compile()


def run(query: str) -> str:
    state = {"query": query}
    result = graph.invoke(state)
    return result.get("answer", "")


if __name__ == "__main__":
    while True:
        user = input("User: ")
        if user.lower() in {"exit", "quit"}:
            break
        print("Assistant:", run(user))
