from pathlib import Path

from dataset import ShakespeareDataset

dataset = ShakespeareDataset(
    Path("../data/input.txt")
)

xb, yb = dataset.get_batch("train")

print(xb.shape)
print(yb.shape)