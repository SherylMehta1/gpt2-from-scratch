from pathlib import Path
import torch

from dataset import ShakespeareDataset
from model import BigramLanguageModel

dataset = ShakespeareDataset(Path("data/input.txt"))

xb, yb = dataset.get_batch("train")

model = BigramLanguageModel(
    dataset.tokenizer.vocab_size
)

logits, loss = model(xb, yb)

print(logits.shape)
print(loss)