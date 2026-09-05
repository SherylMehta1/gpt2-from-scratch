from pathlib import Path

import torch

from dataset import ShakespeareDataset
from model import GPTLanguageModel


BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0
BATCH_SIZE = 16


def test_model_forward():
    dataset = ShakespeareDataset(
        Path("data/input.txt"),
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    model = GPTLanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )

    xb, yb = dataset.get_batch("train")

    logits, loss = model(xb, yb)

    B, T = xb.shape
    vocab_size = dataset.tokenizer.vocab_size

    assert logits.shape == (B * T, vocab_size)
    assert loss is not None
    assert loss.ndim == 0

    print("Forward pass test passed.")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")


def test_model_generation():
    dataset = ShakespeareDataset(
        Path("data/input.txt"),
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    model = GPTLanguageModel(
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )

    context = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(
        context,
        max_new_tokens=20,
    )

    assert generated.shape == (1, 21)

    print("Generation test passed.")
    print(f"Generated shape: {generated.shape}")


if __name__ == "__main__":
    test_model_forward()
    test_model_generation()