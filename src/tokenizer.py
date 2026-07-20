from pathlib import Path
import torch


class CharacterTokenizer:
    """
    Character-level tokenizer for Tiny Shakespeare.
    """

    def __init__(self, text: str):
        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Character <-> Integer mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str):
        """Convert text into token IDs."""
        return [self.stoi[c] for c in text]

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        return "".join(self.itos[i] for i in token_ids)

    def encode_tensor(self, text: str):
        """Return encoded text as a PyTorch tensor."""
        return torch.tensor(self.encode(text), dtype=torch.long)