# LangGraph-based chatbot for benchmark guidance
# This script sets up a simple LangGraph with triage, minimum check,
# and benchmark alternative nodes. Each node calls the corresponding
# agent logic defined in chatbot.py. The final node appends the
# disclaimer from the system prompt.

from __future__ import annotations

from typing import Any, Dict
import json

from langgraph.graph import Graph

import chatbot

# Use the same system prompt and functions defined in chatbot.py
SYSTEM_PROMPT = chatbot.SYSTEM_PROMPT
DISCLAIMER_TEXT = chatbot.DISCLAIMER_TEXT
DISCLAIMER_FREQUENCY = chatbot.DISCLAIMER_FREQUENCY


# Keep a simple interaction counter for the disclaimer logic
class InteractionCounter:
    def __init__(self) -> None:
        self.count = 0

    def increment(self) -> bool:
        self.count += 1
        return self.count % DISCLAIMER_FREQUENCY == 0


counter = InteractionCounter()

# Helper to build a chat completion request using OpenAI
client = chatbot.client


def openai_chat(messages: list[dict[str, str]]) -> str:
    response = chatbot._with_retry(
        model=chatbot.CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


# Node: Triage
# Determine whether the user request is about minimum checks or
# benchmark alternatives. The node outputs a string label.
def triage_node(state: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["input"]},
        {
            "role": "system",
            "content": (
                "Classify the request as 'minimum' if it asks about account "
                "minimums, otherwise classify as 'alternative'. Respond with "
                "just the label."
            ),
        },
    ]
    result = openai_chat(messages).strip().lower()
    if "minimum" in result:
        return "minimum"
    return "alternative"


# Node: Minimum checks
# Use chatbot.get_minimum or blend_minimum depending on the request.
def minimum_node(state: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["input"]},
    ]
    response = chatbot.client.chat.completions.create(
        model=chatbot.CHAT_MODEL,
        messages=messages,
        tools=[{"type": "function", "function": f} for f in chatbot.FUNCTIONS],
        tool_choice="auto",
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": msg.tool_calls}
        )
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            result = chatbot.call_function(func_name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )
        follow = chatbot.client.chat.completions.create(
            model=chatbot.CHAT_MODEL,
            messages=messages,
        )
        return follow.choices[0].message.content or ""
    return msg.content or ""


# Node: Benchmark alternatives
# Use chatbot.search_benchmarks to recommend alternatives
def alternative_node(state: Dict[str, Any]) -> str:
    prompt = (
        "The user request may require recommending alternative benchmarks."
        " Use your data and search_benchmarks function to provide guidance."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["input"]},
        {"role": "system", "content": prompt},
    ]
    response = chatbot.client.chat.completions.create(
        model=chatbot.CHAT_MODEL,
        messages=messages,
        tools=[{"type": "function", "function": f} for f in chatbot.FUNCTIONS],
        tool_choice="auto",
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": msg.tool_calls}
        )
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            result = chatbot.call_function(func_name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )
        follow = chatbot.client.chat.completions.create(
            model=chatbot.CHAT_MODEL,
            messages=messages,
        )
        return follow.choices[0].message.content or ""
    return msg.content or ""


# Node: Final output with disclaimer handling


def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state["result"]
    if counter.increment():
        text = f"{text}\n\n{DISCLAIMER_TEXT}"
    return {"output": text}


# Build the graph
def build_graph() -> Graph:
    graph = Graph()
    graph.add_node("triage", triage_node)
    graph.add_node("minimum", minimum_node)
    graph.add_node("alternative", alternative_node)
    graph.add_node("output", output_node)

    # Edges
    graph.add_edge("triage", "minimum", condition=lambda x: x == "minimum")
    graph.add_edge("triage", "alternative", condition=lambda x: x == "alternative")
    graph.add_edge("minimum", "output")
    graph.add_edge("alternative", "output")
    graph.set_entry_point("triage")
    return graph


def run_chat(query: str) -> str:
    graph = build_graph()
    result = graph.invoke({"input": query})
    return result["output"]


if __name__ == "__main__":
    while True:
        try:
            q = input("User: ")
        except EOFError:
            break
        if not q:
            continue
        print(run_chat(q))
