import torch
from attention import Head


def test_attention_head():
    torch.manual_seed(1337)

    B = 4
    T = 8
    n_embd = 32
    head_size = 16

    x = torch.randn(B, T, n_embd)

    head = Head(
        n_embd=n_embd,
        head_size=head_size,
        block_size=T
    )

    out = head(x)

    assert out.shape == (B, T, head_size)

    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
    print("Attention head test passed.")


if __name__ == "__main__":
    test_attention_head()