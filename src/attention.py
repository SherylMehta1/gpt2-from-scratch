import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Single causal self-attention head.

    Each token produces a query, key, and value vector.
    Queries and keys determine attention weights, while
    values provide the information that gets aggregated.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()

        # Learned projections from token embeddings to
        # query, key, and value representations.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: token at position t can only attend
        # to positions <= t.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: token representations of shape (B, T, C)

        Returns:
            output: context-aware representations of shape
                    (B, T, head_size)
        """

        B, T, C = x.shape

        # Project each token into query, key, and value vectors.
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Compute scaled dot-product attention scores.
        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        # Prevent tokens from attending to future positions.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into attention probabilities.
        wei = F.softmax(wei, dim=-1)

        # Optional dropout on attention weights.
        wei = self.dropout(wei)

        # Weighted aggregation of value vectors.
        out = wei @ v

        return out