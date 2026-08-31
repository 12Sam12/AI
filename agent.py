from pathlib import Path
import re
import torch
from model import AGENT, AGENTConfig
from tools import choose_tool, open_website, open_application
from memory import remember, recall, all_memory

MODEL_PATH = Path("agent_model.pt")


def load_agent():
    if not MODEL_PATH.exists():
        print("AGENT has not been trained yet.")
        print("Run: python train.py")
        raise SystemExit(1)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    config = AGENTConfig(checkpoint["config"]["vocab_size"])
    model = AGENT(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["stoi"], checkpoint["itos"]


def generate(model, stoi, itos, prompt, max_new_tokens=150):
    unknown = next(iter(stoi.values()))
    ids = [stoi.get(ch, unknown) for ch in prompt]
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = idx[:, -model.config.block_size:]
            logits, _ = model(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
    return "".join(itos[int(i)] for i in idx[0])


def clean_key(key):
    key = key.strip().lower()
    key = re.sub(r"^(?:my|the)\s+", "", key)
    key = re.sub(r"[?!.]+$", "", key)
    return key


def memory_command(text):
    t = text.strip()
    low = t.lower()
    if low in {"what do you remember", "what do you remember?", "show my memories", "show memories"}:
        memories = all_memory()
        if not memories:
            return "I don't have any saved memories yet."
        return "I remember: " + "; ".join(f"{k}: {v}" for k, v in memories.items())

    patterns = [
        r"^remember\s+(?:that\s+)?my\s+(.+?)\s+is\s+(.+)$",
        r"^remember\s+(?:that\s+)?(.+?)\s+is\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, t, re.IGNORECASE)
        if match:
            key, value = match.groups()
            return remember(clean_key(key), value)

    match = re.match(r"^(?:what(?:'s| is)\s+my|what\s+is\s+my)\s+(.+?)[?]?$", t, re.IGNORECASE)
    if match:
        key = clean_key(match.group(1))
        value = recall(key)
        if value:
            return f"Your {key} is {value}."
        return f"I don't have a memory for your {key} yet."
    return None


def website_target(target):
    target = target.strip()
    lower = target.lower()
    known = {
        "toddle": "https://toddleapp.com",
        "my toddle": "https://toddleapp.com",
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
    }
    if lower in known:
        return known[lower]
    if target.startswith(("http://", "https://")):
        return target
    if "." in target and " " not in target:
        return "https://" + target
    return "https://www.google.com/search?q=" + target.replace(" ", "+")


def action_command(text):
    t = text.strip()
    low = t.lower()

    # Compound action: open/launch an app and then a destination.
    chrome_destination = re.search(r"(?:open|go to|launch)\s+(?:my\s+)?(.+?)\s+in\s+(?:google\s+)?chrome\b", t, re.IGNORECASE)
    if chrome_destination:
        target = chrome_destination.group(1).strip()
        open_application("chrome")
        return open_website(website_target(target))

    direct_destination = re.match(r"^(?:open|go to|launch)\s+(?:my\s+)?(.+?)\s*$", t, re.IGNORECASE)
    if direct_destination:
        target = direct_destination.group(1).strip()
        if target.lower() in {"chrome", "google chrome"}:
            return open_application("chrome")
        if target.lower() in {"vscode", "vs code", "visual studio code"}:
            return open_application("vscode")
        return open_website(website_target(target))

    if low in {"open web", "open the web", "open browser", "open google"}:
        return open_website("https://www.google.com")
    return None


def main():
    model, stoi, itos = load_agent()
    print("AGENT 0.6 online. Type 'exit' to shut down.")
    print("Memory + intent parser + safe tools enabled.")
    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            print("AGENT: Goodbye.")
            break
        if not user:
            continue

        result = memory_command(user)
        if result:
            print(f"AGENT: {result}")
            continue

        result = action_command(user)
        if result:
            print(f"AGENT: {result}")
            continue

        tool = choose_tool(user)
        if tool:
            function, args = tool
            print(f"AGENT: {function(*args)}")
            continue

        prompt = f"USER: {user}\nAGENT:"
        output = generate(model, stoi, itos, prompt)
        response = output.split("AGENT:", 1)[-1]
        if "USER:" in response:
            response = response.split("USER:", 1)[0]
        print("AGENT:" + response)


if __name__ == "__main__":
    main()
