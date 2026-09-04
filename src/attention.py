import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    A single head of causal self-attention.
    """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        # Independent learned projections for Query, Key, and Value.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Fixed causal mask.
        # register_buffer ensures the mask moves with the model
        # when model.to(device) is called, but is not trainable.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        # Project token representations into Key, Query, and Value spaces.
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # Compute scaled dot-product attention scores.
        # (B, T, head_size) @ (B, head_size, T)
        # -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5

        # Prevent tokens from attending to future positions.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into attention probabilities.
        wei = F.softmax(wei, dim=-1)

        # Regularize attention weights during training.
        wei = self.dropout(wei)

        # Weighted aggregation of Value vectors.
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple independent self-attention heads running in parallel.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        # Each head operates on a smaller portion of the embedding.
        assert n_embd % n_head == 0, (
            "n_embd must be divisible by n_head"
        )

        head_size = n_embd // n_head

        # Run multiple independent attention heads.
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd=n_embd,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_head)
            ]
        )

        # Mix information from all heads after concatenation.
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Each head produces:
        # (B, T, head_size)
        #
        # Concatenating n_head heads gives:
        # (B, T, n_head * head_size)
        # = (B, T, n_embd)
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        # Learned projection mixes information across heads.
        out = self.proj(out)

        # Apply dropout after the projection.
        out = self.dropout(out)

        return out