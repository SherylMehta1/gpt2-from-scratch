from pathlib import Path
from tokenizer import CharacterTokenizer

DATA_PATH = Path("../data/input.txt")

text = DATA_PATH.read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)

print("Vocabulary size:", tokenizer.vocab_size)

sample = "hii there"

encoded = tokenizer.encode(sample)

decoded = tokenizer.decode(encoded)

print(encoded)
print(decoded)