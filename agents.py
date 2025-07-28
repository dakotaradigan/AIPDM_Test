"""Load prompt files used by the chatbot."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load(name: str) -> str:
    path = BASE_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT = _load("system_prompt.txt")
TRIAGE_PROMPT = _load("triage_prompt.txt")
MINIMUM_PROMPT = _load("minimum_prompt.txt")
ALTERNATIVE_PROMPT = _load("alternative_prompt.txt")
