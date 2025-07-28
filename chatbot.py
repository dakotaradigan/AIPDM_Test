import json
import os
from typing import List, Dict, Any

import tiktoken

import logging
import time
from pinecone import Pinecone
from openai import OpenAI

# Load the large system prompt from an external file for readability
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Extract the disclaimer text from the system prompt so it can be
# appended to responses in the chat loop. The line in the prompt looks
# like:
# "- Provide disclaimer every 3-4 interactions: *\u201cThis assistant ... authority.\u201d*"
import re
match = re.search(
    r"disclaimer every 3-4 interactions:\s*\*?[\"\u201c](.+?)[\"\u201d]\*?",
    SYSTEM_PROMPT,
    flags=re.IGNORECASE,
)
DISCLAIMER_TEXT = (
    match.group(1).strip()
    if match
    else "This assistant provides benchmark eligibility guidance only. "
    "No investment advice or account approval authority."
)

# How often to append the disclaimer to assistant responses
DISCLAIMER_FREQUENCY = 3

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

EMBEDDING_MODEL = "text-embedding-3-small"

# Chat completion model and token limits
CHAT_MODEL = "gpt-3.5-turbo"
MAX_MODEL_TOKENS = 16000
TOKEN_MARGIN = 1000

# Prompt and model for routing queries
TRIAGE_MODEL = "gpt-3.5-turbo-0125"
TRIAGE_PROMPT = (
    "Classify the user's request as either 'minimum' or 'alternative'. "
    "Respond with only one of these two words."
)

# Prompts for task-specific handling
MINIMUM_PROMPT = (
    "You answer questions about benchmark account minimums concisely."
)
ALTERNATIVE_PROMPT = (
    "You suggest alternative benchmarks when the requested option isn't suitable."
)


def _with_retry(*, max_attempts: int = 3, **kwargs):
    """Call the OpenAI chat API with simple exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                logging.error("OpenAI request failed after retries", exc_info=True)
                raise
            logging.warning(
                "OpenAI request failed (attempt %s/%s): %s",
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(2 ** attempt)

def embed(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = CHAT_MODEL) -> int:
    """Return total token count for a list of chat messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            total_tokens += len(encoding.encode(str(value)))
            if key == "name":
                total_tokens += tokens_per_name
    total_tokens += 3
    return total_tokens


def trim_history(messages: List[Dict[str, Any]], limit: int = MAX_MODEL_TOKENS - TOKEN_MARGIN) -> bool:
    """Trim oldest user/assistant pairs until total tokens are under limit."""
    truncated = False
    while num_tokens_from_messages(messages) > limit and len(messages) > 1:
        # Find first user message after the system prompt
        start = next((i for i, m in enumerate(messages) if i > 0 and m["role"] == "user"), None)
        if start is None:
            break
        end = start + 1
        # remove messages until just before next user message
        while end < len(messages) and messages[end]["role"] != "user":
            end += 1
        del messages[start:end]
        truncated = True
    return truncated

INDEX_NAME = "benchmark-index"
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(
        f"Pinecone index '{INDEX_NAME}' does not exist. Run build_index.py first."
    )
index = pc.Index(INDEX_NAME)

with open("benchmarks.json", "r") as f:
    DATA = json.load(f)["benchmarks"]

# Build a mapping from lowercase benchmark name to the benchmark data for
# constant-time lookup when retrieving a benchmark by name.
BENCHMARK_MAP = {bench["name"].lower(): bench for bench in DATA}


def get_benchmark(name: str) -> Dict[str, Any] | None:
    """Return benchmark data by name using a pre-built map."""
    return BENCHMARK_MAP.get(name.lower())


def search_benchmarks(
    query: str,
    top_k: int = 3,
    filters: Dict[str, Any] | None = None,
    include_dividend: bool = False,
) -> List[Dict[str, Any]]:
    vec = embed(query)

    pinecone_filter: Dict[str, Any] | None = None
    if filters:
        pinecone_filter = {}
        for key, value in filters.items():
            if isinstance(value, dict) and any(k.startswith("$") for k in value):
                pinecone_filter[key] = value
            else:
                pinecone_filter[key] = {"$eq": value}

    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        **({"filter": pinecone_filter} if pinecone_filter else {}),
    )
    results = []
    for match in res.matches:
        bench = match.metadata
        item = {
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            "score": match.score,
        }
        if include_dividend:
            item["dividend_yield"] = bench.get("dividend_yield")
        results.append(item)
    return results


def get_minimum(name: str, include_dividend: bool = False) -> Dict[str, Any]:
    bench = get_benchmark(name)
    if bench:
        result = {
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
        }
        if include_dividend:
            result["dividend_yield"] = bench.get("fundamentals", {}).get("dividend_yield")
        return result
    return {"error": f"Benchmark '{name}' not found"}


def blend_minimum(allocations: List[Dict[str, Any]], include_dividend: bool = False) -> Dict[str, Any]:
    total_weight = sum(a.get("weight", 0) for a in allocations)
    if abs(total_weight - 1.0) > 1e-6:
        return {"error": f"weights sum to {total_weight}"}
    total = 0.0
    weighted_yield = 0.0
    has_yield = True
    for a in allocations:
        bench = get_benchmark(a.get("name", ""))
        if not bench:
            return {"error": f"Benchmark '{a.get('name')}' not found"}
        total += bench["account_minimum"] * a["weight"]
        dy = bench.get("fundamentals", {}).get("dividend_yield")
        if dy is None:
            has_yield = False
        else:
            weighted_yield += dy * a["weight"]
    result = {"blend_minimum": total}
    if include_dividend and has_yield:
        result["dividend_yield"] = weighted_yield
    return result


FUNCTIONS = [
    {
        "name": "search_benchmarks",
        "description": "Search for similar benchmarks in the dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters. Example: {\"pe_ratio\": {\"$gt\": 20}, \"region\": \"US\"}",
                },
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in results",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_minimum",
        "description": "Get minimum for a specific benchmark",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in result",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "blend_minimum",
        "description": "Calculate minimum for a blend of benchmarks",
        "parameters": {
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["name", "weight"],
                    },
                },
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in result",
                },
            },
            "required": ["allocations"],
        },
    },
]


def call_function(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_benchmarks":
        return {
            "results": search_benchmarks(
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 3),
                filters=arguments.get("filters"),
                include_dividend=arguments.get("include_dividend", False),
            )
        }
    if name == "get_minimum":
        return get_minimum(
            name=arguments.get("name", ""),
            include_dividend=arguments.get("include_dividend", False),
        )
    if name == "blend_minimum":
        return blend_minimum(
            allocations=arguments.get("allocations", []),
            include_dividend=arguments.get("include_dividend", False),
        )
    return {"error": f"Unknown function {name}"}


def triage_query(query: str) -> str:
    """Classify a user query as minimum or alternative."""
    resp = client.chat.completions.create(
        model=TRIAGE_MODEL,
        messages=[
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    label = resp.choices[0].message.content.strip().lower()
    if "min" in label:
        return "minimum"
    return "alternative"


def handle_minimum(query: str) -> str:
    """Respond to minimum-related queries using a dedicated prompt."""
    resp = _with_retry(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": MINIMUM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content.strip()


def handle_alternative(query: str) -> str:
    """Respond to requests for alternative benchmarks."""
    resp = _with_retry(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": ALTERNATIVE_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content.strip()


def chat():
    print("Hello! I'm here to assist with benchmark eligibility questions. How can I help you today?")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    resp_count = 0
    while True:
        user = input("\nUser: ")
        if user.lower() in {"exit", "quit"}:
            break

        category = triage_query(user)
        if category == "minimum":
            final = handle_minimum(user)
            resp_count += 1
            if resp_count % DISCLAIMER_FREQUENCY == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            print(f"\nAssistant: {final}")
            continue
        if category == "alternative":
            final = handle_alternative(user)
            resp_count += 1
            if resp_count % DISCLAIMER_FREQUENCY == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            print(f"\nAssistant: {final}")
            continue

        messages.append({"role": "user", "content": user})
        if trim_history(messages):
            print("[Notice: conversation history truncated to fit token limit]")
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=[{"type": "function", "function": func} for func in FUNCTIONS],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": msg.tool_calls})
            if trim_history(messages):
                print("[Notice: conversation history truncated to fit token limit]")

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = call_function(func_name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
                if trim_history(messages):
                    print("[Notice: conversation history truncated to fit token limit]")

            follow = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
            )
            final = follow.choices[0].message.content
            resp_count += 1
            if resp_count % DISCLAIMER_FREQUENCY == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")
        else:
            final = msg.content or ""
            resp_count += 1
            if resp_count % DISCLAIMER_FREQUENCY == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")


if __name__ == "__main__":
    chat()
