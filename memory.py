"""Small local persistent memory for AGENT.

No API or cloud service is used. Memories are stored in memory.json.
"""
import json
from pathlib import Path

MEMORY_PATH = Path("memory.json")


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
    memory[key] = value
    save_memory(memory)
    return f"I'll remember that {key} is {value}."


def recall(key):
    memory = load_memory()
    return memory.get(key)


def all_memory():
    return load_memory()
