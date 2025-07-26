import json
import os
import time
import logging
from typing import List, Dict, Any

from pinecone import Pinecone
from pinecone.core.client.exceptions import ApiException
from openai import OpenAI, OpenAIError

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

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

EMBEDDING_MODEL = "text-embedding-3-small"

# Configure basic logging
logging.basicConfig(level=logging.INFO)


def _with_retry(func, *args, **kwargs):
    """Helper to retry API calls with exponential backoff."""
    delay = 1
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except (OpenAIError, ApiException) as exc:
            logging.warning("Attempt %d failed: %s", attempt + 1, exc)
            if attempt == 2:
                logging.error("Operation failed after retries: %s", exc)
                raise
            time.sleep(delay)
            delay *= 2


def embed(text: str) -> List[float]:
    try:
        resp = _with_retry(client.embeddings.create, model=EMBEDDING_MODEL, input=text)
        return resp.data[0].embedding
    except Exception:
        logging.error("Failed to create embedding for query '%s'", text)
        return []

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
    if not vec:
        return [{"error": "Failed to generate embeddings. Please try again later."}]

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


def chat():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Hello! I'm here to assist with benchmark eligibility questions. How can I help you today?")
    resp_count = 0
    while True:
        user = input("\nUser: ")
        if user.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user})
        try:
            response = _with_retry(
                client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=messages,
                tools=[{"type": "function", "function": func} for func in FUNCTIONS],
                tool_choice="auto",
            )
        except Exception:
            print("\nAssistant: Sorry, I'm having trouble right now. Please try again later.")
            logging.error("Initial chat completion failed", exc_info=True)
            continue
        msg = response.choices[0].message
        if msg.tool_calls:
            # Add the assistant message with tool calls first
            messages.append({"role": "assistant", "content": None, "tool_calls": msg.tool_calls})

            # Handle each tool call individually
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = call_function(func_name, args)

                # Add individual tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Now get the final response
            try:
                follow = _with_retry(
                    client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=messages,
                )
            except Exception:
                final = "Sorry, I'm having trouble processing that request right now."
                messages.append({"role": "assistant", "content": final})
                print(f"\nAssistant: {final}")
                logging.error("Follow-up chat completion failed", exc_info=True)
                continue
            final = follow.choices[0].message.content
            resp_count += 1
            if resp_count % 4 == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")
        else:
            final = msg.content or ""
            resp_count += 1
            if resp_count % 4 == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")


if __name__ == "__main__":
    chat()
