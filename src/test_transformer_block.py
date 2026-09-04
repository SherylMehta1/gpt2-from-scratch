import torch

from transformer_block import TransformerBlock


# Model configuration
n_embd = 64
n_head = 4
block_size = 32
dropout = 0.0

# Dummy input:
# 2 sequences × 10 tokens × 64 features
x = torch.randn(2, 10, n_embd)

block = TransformerBlock(
    n_embd=n_embd,
    n_head=n_head,
    block_size=block_size,
    dropout=dropout,
)

output = block(x)

print("Input shape: ", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

print("\nTransformer block test passed.")