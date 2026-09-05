# GPT-2 From Scratch

A small GPT-style decoder-only Transformer language model built from scratch in PyTorch, following Andrej Karpathy's "Let's build GPT" approach.

The goal of this project is not to reproduce the scale of GPT-2, but to understand how a modern autoregressive Transformer language model works by implementing its components from the ground up.

The model is trained on the Tiny Shakespeare dataset using character-level tokenization.

---

## What This Project Covers

The implementation is built step-by-step, starting from a simple Bigram language model and progressively introducing the components of a Transformer:

- Character-level tokenization
- Dataset creation and batching
- Bigram language modeling
- Gradient-based optimization with AdamW
- Causal self-attention
- Query, Key, and Value projections
- Scaled dot-product attention
- Causal masking
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Transformer blocks
- Learned positional embeddings
- Stacked Transformer blocks
- Next-token prediction
- Autoregressive text generation
- Training and validation loss estimation

---

## Architecture

The final model follows a GPT-style decoder-only Transformer architecture:

```text
Input Token IDs
      │
      ▼
Token Embeddings
      │
      ├───────────────┐
      │               │
      ▼               ▼
Position Embeddings
      │
      ▼
Token + Position Embeddings
      │
      ▼
┌─────────────────────────────┐
│      Transformer Block      │
│                             │
│  LayerNorm                  │
│      ↓                      │
│  Multi-Head Self-Attention  │
│      ↓                      │
│  Residual Connection        │
│      ↓                      │
│  LayerNorm                  │
│      ↓                      │
│  Feed-Forward Network       │
│      ↓                      │
│  Residual Connection        │
└─────────────────────────────┘
      │
      ▼
   × N Layers
      │
      ▼
Final LayerNorm
      │
      ▼
Linear Language Model Head
      │
      ▼
Vocabulary Logits
      │
      ▼
Next-Token Prediction