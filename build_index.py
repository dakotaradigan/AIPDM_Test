import json
import os
from typing import List

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-3-small"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "benchmark-index"
DIMENSION = 1536


def embed(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


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
        index.upsert(items[i : i + 100])

    print(f"Upserted {len(items)} vectors to '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
