import json
import os
import time
import logging
from typing import List

from openai import OpenAI, OpenAIError
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import ApiException

EMBEDDING_MODEL = "text-embedding-3-small"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "benchmark-index"
DIMENSION = 1536

# Configure logging
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
        logging.error("Failed to create embedding for '%s'", text)
        return []


def main() -> None:
    with open("benchmarks.json", "r") as f:
        data = json.load(f)["benchmarks"]

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    index = pc.Index(INDEX_NAME)

    if index.describe_index_stats().get("total_vector_count"):
        print(f"Index '{INDEX_NAME}' already populated.")
        return

    items = []
    for bench in data:
        vec = embed(bench["name"])
        if not vec:
            logging.error("Skipping '%s' due to embedding failure", bench["name"])
            continue

        # Flatten the metadata to simple key-value pairs
        metadata = {
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            # Flatten tags
            "region": ",".join(bench["tags"]["region"]),
            "asset_class": ",".join(bench["tags"]["asset_class"]),
            "style": ",".join(bench["tags"]["style"]),
            "factor_tilts": ",".join(bench["tags"]["factor_tilts"]),
            "esg": bench["tags"]["esg"],
            "weighting_method": bench["tags"]["weighting_method"],
            "sector_focus": ",".join(bench["tags"]["sector_focus"]),
            # Flatten fundamentals
            "num_constituents": bench["fundamentals"]["num_constituents"],
            "rebalance_frequency": bench["fundamentals"]["rebalance_frequency"],
            "rebalance_dates": ",".join(bench["fundamentals"]["rebalance_dates"]),
            "pe_ratio": bench["fundamentals"]["pe_ratio"],
            "dividend_yield": bench["fundamentals"].get("dividend_yield"),
        }

        items.append((bench["name"], vec, metadata))

    for i in range(0, len(items), 100):
        index.upsert(items[i:i + 100])

    print(f"Upserted {len(items)} vectors to '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
