from pathlib import Path
import torch
from model import AGENT, AGENTConfig

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


if __name__ == "__main__":
    model, stoi, itos = load_agent()
    print("AGENT online. Type 'exit' to shut down.")

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            print("AGENT: Goodbye.")
            break
        if not user:
            continue

        prompt = f"USER: {user}\nAGENT:"
        output = generate(model, stoi, itos, prompt)
        response = output.split("AGENT:", 1)[-1]
        if "USER:" in response:
            response = response.split("USER:", 1)[0]
        print("AGENT:" + response)
