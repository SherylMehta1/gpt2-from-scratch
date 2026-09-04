import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single pre-norm Transformer block.

    Each block performs:
        1. LayerNorm -> Multi-Head Self-Attention -> Residual
        2. LayerNorm -> Feed-Forward Network -> Residual
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):
        # Pre-norm self-attention with residual connection.
        x = x + self.sa(self.ln1(x))

        # Pre-norm feed-forward network with residual connection.
        x = x + self.ffwd(self.ln2(x))

        return x