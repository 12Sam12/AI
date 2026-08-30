# AGENT

AGENT is a personal AI assistant being built from scratch.

The goal is to eventually make AGENT able to:

- understand natural language
- remember conversations
- learn from training data
- speak and listen
- use computer tools
- see and understand the screen/camera
- run tasks autonomously

## Important

AGENT does **not** use an AI API or API key. The language model in this repository is implemented and trained locally from scratch using PyTorch.

The first version is intentionally small. A randomly initialized model cannot magically understand English; it needs training data. The included dataset is only a starting point.

## Requirements

Python 3.10+

Install:

```bash
pip install -r requirements.txt
```

## Train AGENT

```bash
python train.py
```

The trained model will be saved as `agent_model.pt`.

## Talk to AGENT

```bash
python agent.py
```

This first version is text-based. Voice input/output will be added as a separate local subsystem later so the core AGENT brain remains independent of any API.

## Roadmap

- [x] Project structure
- [x] Local tokenizer
- [x] Transformer language model from scratch
- [x] Local training loop
- [x] Local conversation loop
- [ ] Better tokenizer
- [ ] Larger training dataset
- [ ] Long-term memory
- [ ] Local speech-to-text
- [ ] Local text-to-speech
- [ ] Tool/function calling
- [ ] Computer control
- [ ] Vision
- [ ] Autonomous task planning
