"""Persistent local memory for AGENT. No API or cloud service is used."""
import json
from pathlib import Path

MEMORY_PATH = Path("memory.json")
ALIASES = {
    "favorite language": "favorite computing language",
    "programming language": "favorite computing language",
    "name": "name",
}


def _canonical(key):
    key = " ".join(key.lower().strip().split())
    return ALIASES.get(key, key)


def load_memory():
    if not MEMORY_PATH.exists():
        return {}
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(memory):
    MEMORY_PATH.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")


def remember(key, value):
    memory = load_memory()
    key = _canonical(key)
    memory[key] = value.strip()
    save_memory(memory)
    return f"I'll remember that your {key} is {value.strip()}."


def recall(key):
    return load_memory().get(_canonical(key))


def all_memory():
    return load_memory()
