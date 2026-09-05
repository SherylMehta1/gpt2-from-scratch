from pathlib import Path

import torch

from dataset import ShakespeareDataset
from model import GPTLanguageModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0

BATCH_SIZE = 16
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

@torch.no_grad()
def estimate_loss(model, dataset):
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):
            X, Y = dataset.get_batch(split)

            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            _, loss = model(X, Y)

            split_losses[k] = loss.item()

        losses[split] = split_losses.mean().item()

    model.train()

    return losses


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():
    torch.manual_seed(1337)

    dataset = ShakespeareDataset(
        Path("data/input.txt"),
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE
    )

    model = GPTLanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Device: {DEVICE}")

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{num_parameters / 1e6:.3f}M"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for step in range(MAX_ITERS):

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, dataset)

            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        X, Y = dataset.get_batch("train")

        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

    # --------------------------------------------------
    # Generate text
    # --------------------------------------------------

    context = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=DEVICE
    )

    generated = model.generate(
        context,
        max_new_tokens=500
    )

    text = dataset.tokenizer.decode(
        generated[0].tolist()
    )

    print("\nGenerated text:\n")
    print(text)


if __name__ == "__main__":
    train()