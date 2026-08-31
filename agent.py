from pathlib import Path
import re
import torch
from model import AGENT, AGENTConfig
from tools import choose_tool, open_website
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


def memory_command(text):
    t = text.strip()
    low = t.lower()
    if low in {"what do you remember", "what do you remember?", "show my memories"}:
        memories = all_memory()
        if not memories:
            return "I don't have any saved memories yet."
        return "I remember: " + "; ".join(f"{k}: {v}" for k, v in memories.items())

    patterns = [
        r"^remember (?:that )?my (.+?) is (.+)$",
        r"^remember (?:that )?my (.+?) are (.+)$",
        r"^remember (?:that )?(.+?) is (.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, low, re.IGNORECASE)
        if match:
            key, value = match.groups()
            return remember(key, value)

    match = re.match(r"^(?:what(?:'s| is) my|what is my) (.+?)[?]?$", low)
    if match:
        key = match.group(1).strip()
        value = recall(key)
        if value:
            return f"Your {key} is {value}."
        return f"I don't have a memory for your {key} yet."
    return None


def multi_action(text):
    """Handle simple chains such as 'open Chrome and open YouTube'."""
    parts = re.split(r"\s+(?:and then|then|and)\s+", text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    results = []
    handled = 0
    for part in parts:
        tool = choose_tool(part)
        if tool:
            function, args = tool
            results.append(function(*args))
            handled += 1
            continue
        match = re.match(r"open (?:the )?(?:website |site )?(.+?)(?: in chrome)?$", part.strip(), re.IGNORECASE)
        if match and not part.lower().strip() == "open chrome":
            target = match.group(1).strip()
            if target.lower() in {"my toddle", "toddle"}:
                target = "https://toddleapp.com"
            elif not target.startswith(("http://", "https://")):
                target = "https://www.google.com/search?q=" + target.replace(" ", "+")
            results.append(open_website(target))
            handled += 1
    return " ".join(results) if handled else None


def main():
    model, stoi, itos = load_agent()
    print("AGENT 0.5 online. Type 'exit' to shut down.")
    print("Memory + tools enabled. Memories stay local in memory.json.")
    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            print("AGENT: Goodbye.")
            break
        if not user:
            continue

        memory_result = memory_command(user)
        if memory_result:
            print(f"AGENT: {memory_result}")
            continue

        action_result = multi_action(user)
        if action_result:
            print(f"AGENT: {action_result}")
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
