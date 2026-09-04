import torch

from feed_forward import FeedForward


# Model configuration
n_embd = 64
dropout = 0.0

# Dummy input:
# 2 sequences, 10 tokens per sequence,
# 64-dimensional representation per token.
x = torch.randn(2, 10, n_embd)

ffwd = FeedForward(
    n_embd=n_embd,
    dropout=dropout,
)

output = ffwd(x)

print("Input shape: ", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

print("\nFeed-forward network test passed.")