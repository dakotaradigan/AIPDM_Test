# AIPDM_Test

## Build the Pinecone Index

Install dependencies:
```bash
pip install pinecone-client openai
```

Run the index builder once to load benchmark data (it will skip if the index already contains vectors):
```bash
python build_index.py
```

## Chatbot Usage

After the index is built, start the chatbot:
```bash
python chatbot.py
```

`search_benchmarks` accepts an optional `filters` dictionary to narrow results
by metadata fields (e.g. `{ "region": "US", "pe_ratio": {"$gt": 20} }`).

Set the following environment variables before running either script: `PINECONE_API_KEY`, `PINECONE_ENV`, and `OPENAI_API_KEY`.
