import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer import Block


class GPTLanguageModel(nn.Module):
    """
    Decoder-only Transformer language model.
    """

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
    ):
        super().__init__()

        self.block_size = block_size

        # Learned token embeddings.
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd,
        )

        # Learned positional embeddings.
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd,
        )

        # Stack Transformer blocks.
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        # Final normalization.
        self.ln_f = nn.LayerNorm(n_embd)

        # Convert final representations into vocabulary logits.
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Convert token IDs into learned vectors.
        tok_emb = self.token_embedding_table(idx)

        # Create position IDs: 0, 1, ..., T-1.
        pos = torch.arange(
            T,
            device=idx.device,
        )

        # Convert position IDs into learned vectors.
        pos_emb = self.position_embedding_table(pos)

        # Combine token identity and position.
        x = tok_emb + pos_emb

        # Process through stacked Transformer blocks.
        x = self.blocks(x)

        # Final normalization.
        x = self.ln_f(x)

        # Predict next-token logits.
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # Flatten all token predictions into one dimension.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Keep only the most recent block_size tokens.
            idx_cond = idx[:, -self.block_size:]

            # Get predictions.
            logits, _ = self(idx_cond)

            # We only need the prediction for the final token.
            logits = logits[:, -1, :]

            # Convert logits into probabilities.
            probs = F.softmax(logits, dim=-1)

            # Sample the next token.
            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            # Append it to the sequence.
            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        return idx