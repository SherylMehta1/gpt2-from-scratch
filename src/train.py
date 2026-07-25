from pathlib import Path

import torch

from dataset import ShakespeareDataset
from model import BigramLanguageModel

def train():
    
    dataset = ShakespeareDataset( Path("data/input.txt") )
    
    model = BigramLanguageModel(dataset.tokenizer.vocab_size)
    
    optimizer = torch.optim.AdamW( model.parameters(), lr=1e-3 )
    
    max_steps = 100
    
    for step in range(max_steps):
        xb, yb = dataset.get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print( f"Step {step:3d} | Loss: {loss.item():.4f}")
    
    start = torch.zeros((1,1), dtype=torch.long)
    generated = model.generate( start, max_new_tokens=500 )
    text = dataset.tokenizer.decode( generated[0].tolist() )
    print(text)

if __name__ == "__main__":
    train()