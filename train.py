from pathlib import Path
import torch
from model import AGENT, AGENTConfig

DATA_PATH = Path("data/training.txt")
MODEL_PATH = Path("agent_model.pt")

text = DATA_PATH.read_text(encoding="utf-8")

# Character-level tokenizer: no external tokenizer or API is required.
chars = sorted(set(text))
itos = {i: ch for i, ch in enumerate(chars)}
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

config = AGENTConfig(len(chars))
model = AGENT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

batch_size = 16
steps = 1000
block_size = min(config.block_size, max(8, len(data) - 1))

for step in range(steps):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])

    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

checkpoint = {
    "model": model.state_dict(),
    "config": vars(config),
    "stoi": stoi,
    "itos": itos,
}
torch.save(checkpoint, MODEL_PATH)
print(f"Saved AGENT to {MODEL_PATH}")
