from pathlib import Path

DATA_PATH = Path("data/input.txt")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

print("\nFirst 500 characters:\n")
print(text[:500])