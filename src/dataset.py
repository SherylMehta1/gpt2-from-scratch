from pathlib import Path

import torch

from tokenizer import CharacterTokenizer

class ShakespeareDataset:

    def __init__(self, data_path, train_split=0.9):
        with open(data_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.tokenizer = CharacterTokenizer(self.text)
        self.data = self.tokenizer.encode_tensor(self.text)

        n = int(train_split * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
        self.batch_size = 4
        self.block_size = 8

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        return x, y