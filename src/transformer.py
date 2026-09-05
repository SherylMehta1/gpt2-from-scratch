import torch.nn as nn

from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside a Transformer block.
    """

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One pre-norm Transformer block.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )

        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x