import torch

from attention import Head, MultiHeadAttention


# Model configuration
n_embd = 64
n_head = 4
block_size = 32
dropout = 0.0

head_size = n_embd // n_head


# --------------------------------------------------
# Test 1: Single attention head
# --------------------------------------------------

B, T = 2, 10

x = torch.randn(B, T, n_embd)

head = Head(
    n_embd=n_embd,
    head_size=head_size,
    block_size=block_size,
    dropout=dropout,
)

head_output = head(x)

print("Single Head")
print("Input shape: ", x.shape)
print("Output shape:", head_output.shape)

assert head_output.shape == (B, T, head_size)


# --------------------------------------------------
# Test 2: Multi-head attention
# --------------------------------------------------

multi_head = MultiHeadAttention(
    n_embd=n_embd,
    n_head=n_head,
    block_size=block_size,
    dropout=dropout,
)

multi_head_output = multi_head(x)

print("\nMulti-Head Attention")
print("Input shape: ", x.shape)
print("Output shape:", multi_head_output.shape)

assert multi_head_output.shape == (B, T, n_embd)


# --------------------------------------------------
# Test 3: Verify all heads are registered
# --------------------------------------------------

assert len(multi_head.heads) == n_head

print("\nNumber of attention heads:", len(multi_head.heads))
print("Head size:", head_size)

print("\nAll attention tests passed.")